from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING
import numpy as np
import ufl
from dolfinx import fem
from dolfinx.fem import petsc
from petsc4py import PETSc
from slepc4py import SLEPc
from mpi4py import MPI


from simulation.solver_base import SolverBase

if TYPE_CHECKING:
    from simulation.ligamentassembler import LigamentSpringSystem

# Use centralized logging - no custom handlers
logger = logging.getLogger(__name__)


class ElasticEigenSolver(SolverBase):
    """Solve ``K x = λ M x`` with mapped geometry and optional ligaments.

    Builds the consistent mass with ``J = det(I + ∇φ)``, reuses the stiffness
    from :class:`SolverBase`, and drives a SLEPc shift-invert solve while
    filtering rigid modes below ``rigid_mode_tolerance``.
    """

    # ---------------------------------------------------------------- init

    def __init__(
        self,
        V: fem.FunctionSpace,
        E: fem.Function,
        nu: fem.Function,
        phi: fem.Function,
        config: dict,
        ligament_system: LigamentSpringSystem | None = None,
        comm: MPI.Comm | None = None,
    ) -> None:
        self.rigid_mode_tolerance = float(config.get("RIGID_MODE_TOLERANCE"))
        super().__init__(
            V=V,
            E=E,
            nu=nu,
            phi=phi,
            config=config,
            ligament_system=ligament_system,
            comm=comm,
        )

        # Mass matrices (allocated in _assemble_mass, destroyed before reassembly on update)
        self.M: PETSc.Mat | None = None
        self.M0: PETSc.Mat | None = None

        if self.is_root:
            t0 = time.perf_counter()
            logger.info("Init ElasticEigenSolver")

        # Build forms and assemble consistent mass with J (F-mapping enabled)
        self._build_forms()
        self._assemble_mass()
        self._allocate_stiffness()
        self._assemble_stiffness()

        if self.is_root:
            t1 = time.perf_counter()
            logger.info("ElasticEigenSolver init done (%.4f s)", t1 - t0)

    # --------------------------------------------------------------- internals

    # Form build and stiffness assembly are provided by SolverBase

    def _assemble_mass(self) -> None:
        """Assemble ``M = ∫ J u·v dx`` and cache the unmapped reference matrix."""
        u = ufl.TrialFunction(self.V)
        v = ufl.TestFunction(self.V)
        gdim = self.V.mesh.geometry.dim
        I = ufl.Identity(gdim)
        F = I + ufl.grad(self.phi)
        J = ufl.det(F)
        dx = ufl.Measure("dx", domain=self.V.mesh, metadata={"quadrature_degree": self.quadrature_degree})
        m_form = fem.form(ufl.inner(u, v) * J * dx)

        # Release previous mass matrices (e.g., on update() after geometry change)
        if self.M is not None:
            self.M.destroy()
        if self.M0 is not None:
            self.M0.destroy()

        self.M = petsc.create_matrix(m_form)
        self.M.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)
        petsc.assemble_matrix(self.M, m_form)
        self.M.assemble()

        # Unmapped reference mass matrix (φ = 0)
        ref_mass_form = fem.form(ufl.inner(u, v) * dx)
        self.M0 = petsc.create_matrix(ref_mass_form)
        petsc.assemble_matrix(self.M0, ref_mass_form)
        self.M0.assemble()

    # --------------------------------------------------------------- SLEPc solver methods

    def _monitor_eps_short(self, eps: SLEPc.EPS, it: int, nconv: int, eig: list, err: list, it_skip: int = 5):
        """Log sparse SLEPc progress every ``it_skip`` iterations."""
        if self.is_root:
            if it == 1:
                logger.debug("SLEPc eigenvalue solver iterations:")
            
            if it == 1 or not it % it_skip:
                max_error = max(err) if err else 0.0
                logger.debug("  Iteration %3d: %3d converged, max error: %1.1e", it, nconv, max_error)

    def _print_eps_results(self, eps: SLEPc.EPS):
        """Summarize convergence counts and optionally list eigenvalues."""
        if self.is_root:
            its = eps.getIterationNumber()
            nconv = eps.getConverged()
            logger.info("SLEPc solver completed: %d iterations, %d converged eigenpairs", its, nconv)

            if nconv > 0:
                vr, _ = eps.getOperators()[0].createVecs()
                
                # Collect all eigenvalues for summary
                all_eigvals = []
                for i in range(nconv):
                    k = eps.getEigenpair(i, vr)
                    all_eigvals.append(k)
                
                # Only show detailed eigenvalues at DEBUG level
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug("Detailed eigenvalue results:")
                    logger.debug("-" * 40)
                    logger.debug("   Eigenvalue         |    Error")
                    logger.debug("-" * 40)
                    
                    for i in range(min(10, nconv)):  # Limit to first 10 for brevity
                        k = all_eigvals[i]
                        error = eps.computeError(i)
                        if k.imag != 0.0:
                            logger.debug(" %2.2e + %2.2ej | %1.1e", k.real, k.imag, error)
                        else:
                            logger.debug(" %2.2e             | %1.1e", k.real, error)
                    if nconv > 10:
                        logger.debug("  ... and %d more eigenvalues", nconv - 10)
                
                # Always show concise summary at INFO level
                real_parts = np.array([eig.real for eig in all_eigvals])
                logger.info("Eigenvalue range: [%.4e, %.4e]", np.min(real_parts), np.max(real_parts))

    def _get_eps_spectrum(self, eps: SLEPc.EPS) -> tuple[list[complex], list[PETSc.Vec], list[PETSc.Vec]]:
        """Retrieve sorted eigenvalues and eigenvectors from a solved SLEPc EPS object."""
        eigvals, eigvecs_r, eigvecs_i = [], [], []
        nconv = eps.getConverged()
        
        if nconv == 0:
            return eigvals, eigvecs_r, eigvecs_i

        vr_template, vi_template = eps.getOperators()[0].createVecs()
        
        for i in range(nconv):
            vr_i, vi_i = vr_template.copy(), vi_template.copy()
            eigval = eps.getEigenpair(i, vr_i, vi_i)
            eigvals.append(eigval)
            eigvecs_r.append(vr_i)
            eigvecs_i.append(vi_i)

        idx = np.argsort(np.abs(np.array(eigvals)))
        sorted_eigvals = [eigvals[i] for i in idx]
        sorted_eigvecs_r = [eigvecs_r[i] for i in idx]
        sorted_eigvecs_i = [eigvecs_i[i] for i in idx]
        
        return sorted_eigvals, sorted_eigvecs_r, sorted_eigvecs_i

    def _create_eps_solver(
        self,
        nev: int,
        target: float,
        initial_space: list[PETSc.Vec] | None = None,
    ) -> SLEPc.EPS:
        """Configure SLEPc EPS solver for the generalized eigenvalue problem."""
        if self.is_root:
            logger.info("Configuring SLEPc solver: %d eigenvalues, target=%s", nev, target)

        eps = SLEPc.EPS()
        eps.create(comm=self.comm)
        eps.setOperators(self.K, self.M)

        if self.is_root:
            logger.debug("Problem type: GHEP, Solver: KRYLOVSCHUR, Transform: SINVERT")
        
        eps.setProblemType(SLEPc.EPS.ProblemType.GHEP)
        eps.setDimensions(nev=nev)
        eps.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
        eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_REAL)
        eps.setTarget(target)

        if initial_space:
            if self.is_root:
                logger.info("Using %d initial vectors for warm start", len(initial_space))
            eps.setInitialSpace(initial_space)

        # Configure spectral transformation
        st = eps.getST()
        st.setType(SLEPc.ST.Type.SINVERT)
        st.setShift(target)
        
        ksp = st.getKSP()
        ksp.setType('preonly')
        # Apply tolerances if provided (useful for iterative PC/KSP types)
        if self._solver_rtol is not None or self._solver_atol is not None or self._solver_max_iters is not None:
            ksp.setTolerances(
                rtol=self._solver_rtol if self._solver_rtol is not None else None,
                atol=self._solver_atol if self._solver_atol is not None else None,
                max_it=self._solver_max_iters if self._solver_max_iters is not None else None,
            )
        pc = ksp.getPC()
        pc.setType('lu')
        pc.setFactorSolverType('mumps')

        # Set monitor
        eps.setMonitor(
            lambda eps, it, nconv, eig, err: self._monitor_eps_short(eps, it, nconv, eig, err)
        )

        eps.setFromOptions()
        return eps

    # --------------------------------------------------------------- public API

    def _on_geometry_updated(self) -> None:
        """Reassemble mass matrix when geometry mapping changes."""
        self._assemble_mass()


    def _detect_rigid_modes(self, eigenvalues: np.ndarray, eigenvectors: np.ndarray) -> tuple[int, np.ndarray, np.ndarray]:
        """
        Detect rigid body modes based on eigenvalue magnitude.
        
        Returns:
            num_rigid: Number of detected rigid modes
            filtered_eigenvalues: Eigenvalues with rigid modes removed
            filtered_eigenvectors: Eigenvectors with rigid modes removed
        """
        # Find modes with eigenvalues below tolerance
        rigid_mask = np.abs(eigenvalues) < self.rigid_mode_tolerance
        num_rigid = np.sum(rigid_mask)
        
        if self.is_root:
            logger.info("Detected %d rigid modes (tol=%g)", num_rigid, self.rigid_mode_tolerance)
            if num_rigid > 0:
                rigid_eigenvals = eigenvalues[rigid_mask]
                logger.debug("Rigid mode eigenvalues: %s", rigid_eigenvals)
        
        # Filter out rigid modes
        elastic_mask = ~rigid_mask
        filtered_eigenvalues = eigenvalues[elastic_mask]
        filtered_eigenvectors = eigenvectors[elastic_mask]
        
        return num_rigid, filtered_eigenvalues, filtered_eigenvectors

    def solve(
        self, 
        *, 
        nev: int, 
        target: float | None = None, 
        initial_space_vecs: np.ndarray | None = None,
        return_rigid_modes: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return eigen‑pairs as (*values*, *vectors*).
        
        Args:
            nev: Number of eigenvalues to compute (including rigid modes)
            target: Target eigenvalue for shift-invert
            initial_space_vecs: Initial guess vectors
            return_rigid_modes: If True, return all modes; if False, filter rigid modes
        
        Returns:
            Tuple of (eigenvalues, eigenvectors) with rigid modes filtered unless return_rigid_modes=True
        """
        t0 = time.perf_counter() if self.is_root else None
        if self.is_root:
            logger.info("Solve eigenproblem nev=%d target=%s", nev, target)
        
        # Prepare initial space vectors if provided
        petsc_initial_space = []
        if initial_space_vecs is not None:
            num_initial_vectors = min(nev, initial_space_vecs.shape[0])
            for i in range(num_initial_vectors):
                petsc_vec = self.K.createVecRight()
                petsc_vec.setArray(initial_space_vecs[i].flatten())
                petsc_initial_space.append(petsc_vec)
            if self.is_root:
                logger.info("Using %d initial vectors for warm start", num_initial_vectors)

        # Create and configure solver
        eps = self._create_eps_solver(
            nev=nev,
            target=target if target else 0.0,
            initial_space=petsc_initial_space if petsc_initial_space else None
        )
        
        # Solve eigenvalue problem
        if self.is_root:
            logger.info("Start eigen solve (monitor=DEBUG)")
        
        eps.solve()
        self._print_eps_results(eps)
        
        # Extract results
        vals, xr, _ = self._get_eps_spectrum(eps)
        lam = np.asarray([v.real for v in vals])
        
        vecs = np.asarray([vec.array for vec in xr]).reshape(len(lam), -1, self.V.mesh.geometry.dim)
        
        # Detect and filter rigid modes unless explicitly requested
        if not return_rigid_modes:
            num_rigid, lam, vecs = self._detect_rigid_modes(lam, vecs)
            if self.is_root and num_rigid > 0:
                logger.info("Filtered %d rigid body modes from results", num_rigid)
        
        if self.is_root and t0 is not None:
            t1 = time.perf_counter()
            logger.info("Eigen solve done (elastic modes=%d, %.4f s)", len(lam), t1 - t0)

        return lam, vecs

    def eigenvalue_sensitivities(self, eigenvectors: np.ndarray, mass_tol: float = 1e-12) -> np.ndarray:
        """Compute ∂λ/∂EAᵢ sensitivities for each ligament bundle."""
        if self.ligament_system is None or self.ligament_system.num_bundles == 0:
            return np.zeros((eigenvectors.shape[0], 0), dtype=float)

        mass_norms = self._mass_norms(eigenvectors)
        quad_forms = self.ligament_system.mode_bundle_quadratic_forms(eigenvectors)

        sensitivities = np.zeros_like(quad_forms)
        valid = mass_norms > mass_tol
        if np.any(valid):
            sensitivities[valid] = quad_forms[valid] / mass_norms[valid, None]
        return sensitivities
    
    def eigenvalue_pretension_sensitivities(self, eigenvectors: np.ndarray, mass_tol: float = 1e-12) -> np.ndarray:
        """
        Return matrix dλ/dT_b, shape = (num_modes, num_bundles).

        Uses xᵀ (∂K/∂T_b) x / (xᵀ M x), where ∂K/∂T_b is provided by the ligament
        system as bundle-wise geometric quadratic forms.
        """
        if self.ligament_system is None or self.ligament_system.num_bundles == 0:
            return np.zeros((eigenvectors.shape[0], 0), dtype=float)

        mass_norms = self._mass_norms(eigenvectors)
        quad_forms = self.ligament_system.mode_bundle_geometric_quadratic_forms(eigenvectors)

        out = np.zeros_like(quad_forms)
        valid = mass_norms > mass_tol
        if np.any(valid):
            out[valid] = quad_forms[valid] / mass_norms[valid, None]
        return out


    def _mass_norms(self, eigenvectors: np.ndarray) -> np.ndarray:
        """Evaluate xᵀ M x for each mode."""
        num_modes = eigenvectors.shape[0]
        norms = np.zeros(num_modes, dtype=float)
        if num_modes == 0:
            return norms

        work = self.M.createVecRight()
        for idx, vec in enumerate(eigenvectors):
            arr = np.ascontiguousarray(vec, dtype=float).reshape(-1)
            petsc_vec = PETSc.Vec().createWithArray(arr, comm=self.comm)
            self.M.mult(petsc_vec, work)
            norms[idx] = petsc_vec.dot(work).real
            petsc_vec.destroy()
        work.destroy()
        return norms

    def mode_region_energies(
        self,
        eigenvectors: np.ndarray,
        cell_tags,
        region_ids: list[int],
    ) -> np.ndarray:
        """Compute elastic energy contribution per region for each mode."""
        if not region_ids:
            return np.zeros((eigenvectors.shape[0], 0), dtype=float)

        dx = ufl.Measure(
            "dx",
            domain=self.V.mesh,
            subdomain_data=cell_tags,
            metadata={"quadrature_degree": self.quadrature_degree},
        )
        gdim = self.V.mesh.geometry.dim
        u_fn = fem.Function(self.V)

        F = ufl.Identity(gdim) + ufl.grad(self.phi)
        F_inv = ufl.inv(F)
        J = ufl.det(F)
        mu = self.E / (2.0 * (1.0 + self.nu))
        lmbda = self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))

        eps_u = lambda w: ufl.sym(ufl.grad(w) * F_inv)
        sigma_u = lambda w: lmbda * ufl.tr(eps_u(w)) * ufl.Identity(gdim) + 2.0 * mu * eps_u(w)

        energy_density = ufl.inner(sigma_u(u_fn), eps_u(u_fn)) * J
        forms = {rid: fem.form(energy_density * dx(int(rid))) for rid in region_ids}

        num_modes = eigenvectors.shape[0]
        energies = np.zeros((num_modes, len(region_ids)), dtype=float)
        if num_modes == 0 or len(region_ids) == 0:
            return energies

        for mode_idx, vec in enumerate(eigenvectors):
            flat = np.ascontiguousarray(vec, dtype=float).reshape(-1)
            u_fn.x.array[:] = flat
            u_fn.x.scatter_forward()
            row = energies[mode_idx]
            for j, rid in enumerate(region_ids):
                row[j] = float(fem.assemble_scalar(forms[rid]))

        return energies

    def mode_weighted_energies(self, eigenvectors: np.ndarray, cell_weights: np.ndarray) -> np.ndarray:
        """Compute ∫ w(x) E_density(u_i) dx for each mode.

        Parameters
        ----------
        eigenvectors : np.ndarray
            Array of eigenmodes with shape (num_modes, num_blocks, gdim).
        cell_weights : np.ndarray
            DG0-aligned per-cell weights (shape matches self.E.x.array) to multiply
            the elastic energy density prior to integration.

        Returns
        -------
        np.ndarray
            Weighted energies per mode, shape (num_modes,).
        """
        num_modes = eigenvectors.shape[0]
        if num_modes == 0:
            return np.zeros(0, dtype=float)

        # Build DG0 weighting function on the fly
        W_space = fem.functionspace(self.V.mesh, ("DG", 0))
        W_fn = fem.Function(W_space)
        if W_fn.x.array.shape != np.asarray(cell_weights).shape:
            raise ValueError(
                f"cell_weights has shape {np.asarray(cell_weights).shape}, expected {W_fn.x.array.shape}"
            )
        W_fn.x.array[:] = np.asarray(cell_weights, dtype=float)
        W_fn.x.scatter_forward()

        # Set up energy density form multiplied by W
        gdim = self.V.mesh.geometry.dim
        u_fn = fem.Function(self.V)
        F = ufl.Identity(gdim) + ufl.grad(self.phi)
        F_inv = ufl.inv(F)
        J = ufl.det(F)
        mu = self.E / (2.0 * (1.0 + self.nu))
        lmbda = self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))

        eps_u = lambda w: ufl.sym(ufl.grad(w) * F_inv)
        sigma_u = lambda w: lmbda * ufl.tr(eps_u(w)) * ufl.Identity(gdim) + 2.0 * mu * eps_u(w)
        dx = ufl.Measure("dx", domain=self.V.mesh, metadata={"quadrature_degree": self.quadrature_degree})
        energy_density = ufl.inner(sigma_u(u_fn), eps_u(u_fn)) * J * W_fn
        form = fem.form(energy_density * dx)

        out = np.zeros(num_modes, dtype=float)
        for i, vec in enumerate(eigenvectors):
            u_fn.x.array[:] = np.ascontiguousarray(vec, dtype=float).reshape(-1)
            u_fn.x.scatter_forward()
            out[i] = float(fem.assemble_scalar(form))
        return out
