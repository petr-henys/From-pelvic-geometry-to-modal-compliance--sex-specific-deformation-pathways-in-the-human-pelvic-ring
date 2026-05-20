from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import ufl
from dolfinx import fem
from dolfinx.fem import petsc
from mpi4py import MPI
from petsc4py import PETSc
from scipy.interpolate import RBFInterpolator

if TYPE_CHECKING:
    from simulation.ligamentassembler import LigamentSpringSystem


logger = logging.getLogger(__name__)


class SolverBase:
    """Shared base for elastic operators (linear and eigen solves).

    Responsibilities
    - Hold common fields (space ``V``, material fields ``E``, ``nu``, geometry map ``phi``, optional ligaments)
    - Build the small‑strain isotropic elastic bilinear form with deformation mapping φ(X)
      using the standard pullback with ``F = I + ∇φ``, ``F⁻¹`` and ``J = det(F)``
    - Allocate and assemble the global stiffness matrix ``K`` (adds ligament contributions if present)
    - Apply a penalty‑based fixed Dirichlet constraint on facets (reference or mapped surface)
    - Provide an ``update()`` that rebuilds geometry/material‑dependent quantities and reassembles matrices

    Notes
    -----
    - An RBF interpolator of the geometry map, ``_fu(X) ≈ φ(X)``, is built for evaluating mapped
      coordinates outside the FE DoFs (used, e.g., when deforming point clouds).
    - This class defines trial/test functions ``self.u``/``self.v`` for form construction. Subclasses may
      later reuse the attribute name ``self.u`` for a computed solution function; this is intentional and
      confined to subclass methods.
    """

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
        self.comm = comm or V.mesh.comm
        self.is_root = self.comm.rank == 0

        self.V = V
        self.E = E
        self.nu = nu
        self.phi = phi
        self.config = config
        self.ligament_system = ligament_system
        self.quadrature_degree = int(config["QUADRATURE_DEGREE"])
        self._solver_rtol = float(config["SOLVER_RTOL"])
        self._solver_atol = float(config["SOLVER_ATOL"])
        self._solver_max_iters = int(config["SOLVER_MAX_ITERS"])

        # Members set by _build_forms()
        self._dx = None
        self._F_inv = None
        self._J = None
        self._a_form = None
        self.u = None
        self.v = None
        self.K = None

        # Build RBF interpolator for phi
        self._fu = self._build_fu_interpolator()

    # --------------------------------------------------------------- form build

    def _build_fu_interpolator(self) -> RBFInterpolator:
        """Build RBF interpolator from current phi values."""
        X = self.V.tabulate_dof_coordinates()
        gdim = self.V.mesh.geometry.dim
        phi_vals = self.phi.x.array.reshape((-1, gdim))
        smoothing = float(self.config["RBF_SOLVER_SMOOTHING"])
        neighbors = self.config["RBF_SOLVER_NEIGHBORS"]
        return RBFInterpolator(X, phi_vals, neighbors=neighbors, smoothing=smoothing)

    def _build_forms(self) -> None:
        """Build common UFL forms for linearized isotropic elasticity with mapping.

        - F = I + ∇φ, J = det(F), F_inv = F⁻¹
        - ε(u) = sym(∇u · F_inv)
        - σ(u) = λ tr(ε) I + 2 μ ε, with μ, λ from (E, ν)
        - Bilinear form: a(u, v) = ∫ J ⟨σ(u), ε(v)⟩ dx on the reference domain

        Stores on ``self``: trial/test functions (``u``, ``v``), measure (``_dx``),
        geometric factors (``_F_inv``, ``_J``) and the UFL bilinear form (``_a_form``).
        """
        if self.is_root:
            logger.debug("Build UFL forms (base)")

        gdim = self.V.mesh.geometry.dim
        I = ufl.Identity(gdim)
        self.u = ufl.TrialFunction(self.V)
        self.v = ufl.TestFunction(self.V)
        self._dx = ufl.Measure("dx", domain=self.V.mesh, metadata={"quadrature_degree": self.quadrature_degree})

        F = I + ufl.grad(self.phi)
        F_inv = ufl.inv(F)
        J = ufl.det(F)
        self._F_inv = F_inv
        self._J = J

        mu = self.E / (2.0 * (1.0 + self.nu))
        lmbda = self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))

        def eps(w):
            return ufl.sym(ufl.grad(w) * F_inv)

        def sigma(w):
            return lmbda * ufl.tr(eps(w)) * ufl.Identity(gdim) + 2.0 * mu * eps(w)

        self._a_form = ufl.inner(sigma(self.u), eps(self.v)) * self._J * self._dx

    # ----------------------------- geometric helpers (facet pullbacks)

    def _facet_normal(self):
        """Return facet normal expression for current mesh."""
        return ufl.FacetNormal(self.V.mesh)

    def _F_inv_T(self):
        """Return F^{-T} built from stored F^{-1}."""
        return ufl.transpose(self._F_inv)

    def _surface_jacobian(self):
        """J_surf = J ||F^{-T} n|| for pullback of surface measure to reference."""
        n = self._facet_normal()
        F_inv_T = self._F_inv_T()
        return self._J * ufl.sqrt(ufl.inner(F_inv_T * n, F_inv_T * n))

    def _pressure_pullback_vector(self):
        """Vector J F^{-T} n used in pressure load pullback."""
        n = self._facet_normal()
        return self._J * self._F_inv_T() * n

    # -------------------------------------------------------- stiffness assembly

    def _allocate_stiffness(self) -> None:
        """Allocate PETSc matrix with sparsity from bilinear form."""
        if self.is_root:
            logger.debug("Allocate stiffness sparsity (base)")
        form = fem.form(self._a_form)
        self.K = petsc.create_matrix(form)
        self.K.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)
        self.K.setOption(PETSc.Mat.Option.IGNORE_ZERO_ENTRIES, False)

    def _assemble_stiffness(self) -> None:
        """Assemble K and add ligament contributions if present."""
        if self.is_root:
            t0 = time.perf_counter()
            logger.debug("Assemble K (base)")

        self.K.zeroEntries()
        form = fem.form(self._a_form)
        petsc.assemble_matrix(self.K, form)
        self.K.assemble()

        if self.ligament_system is not None:
            self.K.axpy(1.0, self.ligament_system.K, structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)
            self.K.assemble()

        if self.is_root:
            t1 = time.perf_counter()
            logger.debug("K assembly done in %.4f s", t1 - t0)

    # -------------------------------------------------------------- BC penalties

    def fixed_dirichlet(self, gamma: float, ds=None, defined_on: str = "sample") -> None:
        """Penalty term γ ∫ ⟨u, v⟩ dS on boundary (optionally on mapped surface).

        Args:
            gamma: Penalty parameter.
            ds: Facet measure. If None, uses global boundary `ufl.ds`.
            defined_on: 'sample' or synonyms ('sam', 'physical') for mapped surface; anything else treated as reference surface.
        """
        if ds is None:
            ds = ufl.ds(domain=self.V.mesh)

        # Use the stored trial/test functions from base form build
        u = self.u
        v = self.v

        # Normalize mode string
        mode = (defined_on or "sample").strip().lower()
        is_sample = mode.startswith("sam") or mode.startswith("phys")

        if is_sample:
            J_surf = self._surface_jacobian()
            self._a_form += float(gamma) * ufl.inner(u, v) * J_surf * ds
        else:
            self._a_form += float(gamma) * ufl.inner(u, v) * ds
        # Reassemble with added penalty
        self._assemble_stiffness()

    # --------------------------------------------------------------- linear solver

    def _create_ksp_solver(self, *, options_prefix: str = "elas_") -> None:
        """Create and configure a PETSc KSP for the current K."""
        if self.is_root:
            logger.info("Creating KSP solver (prefix '%s')", options_prefix)

        ksp = PETSc.KSP().create(self.comm)
        ksp.setOperators(self.K)
        if options_prefix:
            ksp.setOptionsPrefix(options_prefix)
        ksp.setType('preonly')
        pc = ksp.getPC()
        pc.setType('lu')
        pc.setFactorSolverType('mumps')
        if self._solver_rtol is not None or self._solver_atol is not None or self._solver_max_iters is not None:
            ksp.setTolerances(
                rtol=self._solver_rtol,
                atol=self._solver_atol,
                max_it=self._solver_max_iters,
            )
        self.ksp = ksp

    # ------------------------------------------------------------------- update

    def _on_geometry_updated(self) -> None:
        """Hook for subclasses (e.g., reassemble mass when geometry changes)."""
        pass

    def update(self) -> None:
        """Rebuild internal operators after in-place changes to E/nu/phi."""
        if self.is_root:
            t0 = time.perf_counter()
            logger.info("Update solver state (materials/geometry)")
        
        # Rebuild RBF interpolator for updated phi
        self._fu = self._build_fu_interpolator()
        
        # Geometry-dependent hooks (e.g., recompute F^{-1}, J, normals...)
        self._on_geometry_updated()
        
        # Reassemble stiffness with current E, nu, phi (referenced functions)
        self._assemble_stiffness()
        
        if self.is_root:
            t1 = time.perf_counter()
            logger.info("Update finished in %.4f s", t1 - t0)
    
