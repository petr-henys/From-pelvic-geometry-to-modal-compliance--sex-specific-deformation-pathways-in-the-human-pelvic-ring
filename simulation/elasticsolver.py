from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING
import numpy as np
import ufl
from dolfinx import fem
from dolfinx.fem import petsc
from petsc4py import PETSc
from mpi4py import MPI
from scipy.interpolate import RBFInterpolator


from simulation.solver_base import SolverBase

if TYPE_CHECKING:
    from simulation.ligamentassembler import LigamentSpringSystem

# Use centralized logging - no custom handlers
logger = logging.getLogger(__name__)

dtype = PETSc.ScalarType  # type: ignore


class ElasticSolver(SolverBase):
    """Solve Ku = b with optional ligament coupling and mapping φ.

    Accumulates volumetric and surface loads, assembles the PETSc RHS, and
    reuses the stiffness assembled in :class:`SolverBase`. The solution is
    interpolated with an RBF to evaluate deformed coordinates via ``deform``.
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
        super().__init__(
            V=V,
            E=E,
            nu=nu,
            phi=phi,
            config=config,
            ligament_system=ligament_system,
            comm=comm,
        )

        if self.is_root:
            t0 = time.perf_counter()
            logger.info("Init ElasticSolver")

        # load-form accumulator (list of UFL terms)
        self._L_terms: list[ufl.form.Form] = []
        self._L_form: ufl.form.Form | None = None

        self._build_forms()
        self._allocate_stiffness()
        self._assemble_stiffness()

        if self.is_root:
            t1 = time.perf_counter()
            logger.info("ElasticSolver initialization finished in %.4f seconds.", t1 - t0)

    # --------------------------------------------------------------- internals
    # Form build and stiffness assembly are provided by SolverBase

    # ------------------------------ RHS helpers (accumulation & assembly)

    def _as_scalar_constant(self, val: float | complex | fem.Constant | ufl.core.expr.Expr):
        if isinstance(val, (fem.Constant, ufl.core.expr.Expr)):
            return val
        return fem.Constant(self.V.mesh, dtype(val))  # type: ignore[arg-type]

    def _as_vector_expr(self, vec: list[float] | np.ndarray | fem.Constant | ufl.core.expr.Expr):
        """Coerce to a vector-valued UFL expression of shape (gdim,)."""
        if isinstance(vec, (fem.Constant, ufl.core.expr.Expr)):
            return vec
        arr = np.asarray(vec, dtype=float).ravel()
        gdim = self.V.mesh.geometry.dim
        if arr.size != gdim:
            raise ValueError(f"Vector must have length {gdim}, got {arr.size}")
        # fem.Constant accepts tuple for vector constants
        return fem.Constant(self.V.mesh, tuple(dtype(x) for x in arr))  # type: ignore[arg-type]

    def _append_load_term(self, term: ufl.form.Form):
        """Append UFL load term to RHS accumulator and reassemble b vector."""
        self._L_terms.append(term)
        self._assemble_rhs()

    # ------------------------------ Load term builders (clarity helpers)

    def _pressure_intensity_term(self, p_expr, ds, defined_on: str):
        v = self.v
        n = self._facet_normal()
        if defined_on == "template":
            return -ufl.inner(p_expr * n, v) * ds
        else:
            JFn = self._pressure_pullback_vector()
            return -ufl.inner(p_expr * JFn, v) * ds

    def _pressure_total_force_term(self, F_mag: float, ds, defined_on: str):
        gdim = self.V.mesh.geometry.dim
        n = self._facet_normal()
        if defined_on == "template":
            comps = [self.comm.allreduce(fem.assemble_scalar(fem.form(n[i] * ds)), op=MPI.SUM) for i in range(gdim)]
        else:
            JFn = self._pressure_pullback_vector()
            comps = [self.comm.allreduce(fem.assemble_scalar(fem.form(JFn[i] * ds)), op=MPI.SUM) for i in range(gdim)]
        N_vec = np.array(comps, dtype=float)
        N_norm = float(np.linalg.norm(N_vec))
        if N_norm <= 1e-14:
            raise RuntimeError(
                "apply_load (pressure/total_force): net normal is ~0; uniform pressure cannot produce nonzero resultant on this ROI."
            )
        p_val = abs(float(F_mag)) / N_norm
        p_const = fem.Constant(self.V.mesh, dtype(p_val))
        if self.is_root:
            logger.info("apply_load: pressure total_force |F|=%.6g, |N|=%.6g -> p=%.6g Pa", F_mag, N_norm, p_val)
        return self._pressure_intensity_term(p_const, ds, defined_on)

    def _traction_intensity_term(self, t_expr, ds, defined_on: str):
        v = self.v
        if defined_on == "template":
            return ufl.inner(t_expr, v) * ds
        else:
            J_surf = self._surface_jacobian()
            return ufl.inner(t_expr, v) * J_surf * ds

    def _traction_total_force_term(self, F_vec: np.ndarray, ds, defined_on: str):
        gdim = self.V.mesh.geometry.dim
        F_vec = np.asarray(F_vec, dtype=float).ravel()
        if F_vec.size != gdim:
            raise ValueError(f"apply_load: traction total_force expects length-{gdim} vector, got {F_vec.shape}.")
        v = self.v
        if defined_on == "template":
            A_ref = self.comm.allreduce(fem.assemble_scalar(fem.form(1 * ds)), op=MPI.SUM)
            if A_ref <= 0:
                raise RuntimeError("apply_load (traction/total_force): reference area is zero/undefined.")
            t_const = fem.Constant(self.V.mesh, tuple((F_vec / A_ref).tolist()))
            if self.is_root:
                logger.info("apply_load: traction total_force on template: F=%s, A_ref=%.6g", F_vec, A_ref)
            return ufl.inner(t_const, v) * ds
        else:
            J_surf = self._surface_jacobian()
            A_phys = self.comm.allreduce(fem.assemble_scalar(fem.form(J_surf * ds)), op=MPI.SUM)
            if A_phys <= 0:
                raise RuntimeError("apply_load (traction/total_force): physical area is zero/undefined (via J_surf).")
            t_const = fem.Constant(self.V.mesh, tuple((F_vec / A_phys).tolist()))
            if self.is_root:
                logger.info("apply_load: traction total_force on sample: F=%s, A_phys=%.6g", F_vec, A_phys)
            return ufl.inner(t_const, v) * J_surf * ds

    def _assemble_rhs(self):
        """Assemble RHS vector b from accumulated UFL load terms plus ligament pretension."""
        if len(self._L_terms) == 0:
            self.b = self.K.createVecRight()
            self.b.set(0.0)
            if self.ligament_system is not None and self.ligament_system.fpret is not None:
                self.b.axpy(1.0, self.ligament_system.fpret)
            return

        # Sum UFL terms
        L_form = self._L_terms[0]
        for t in self._L_terms[1:]:
            L_form = L_form + t
        self._L_form = L_form

        L = fem.form(self._L_form)
        test_space = L.function_spaces[0]
        self.b = petsc.create_vector(test_space)
        self.b.set(0.0)
        petsc.assemble_vector(self.b, L)
        self.b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        
        if self.ligament_system is not None and self.ligament_system.fpret is not None:
            self.b.axpy(1.0, self.ligament_system.fpret)

    def clear_loads(self):
        """Reset all accumulated loads from RHS."""
        if self.is_root:
            logger.info("Reset all loads")
        self._L_terms.clear()
        self._L_form = None
        self._assemble_rhs()

    # ------------------------------------------------------------- solver/RHS

    # ------------------------------ Load application methods

    def apply_load(
        self,
        q: float | list[float] | fem.Constant | ufl.core.expr.Expr,
        ds,
        *,
        defined_on: str = "sample",
        input_type: str = "intensity",
    ):
        """
        Unified loading:
        - scalar q  -> pressure (−p n)
        - vector q  -> traction (t)

        Parameters
        ----------
        q          : pressure intensity (Pa) or traction intensity (N/m^2) if input_type='intensity';
                    total force if input_type='total_force' (scalar |F| for pressure, vector F for traction).
        ds         : UFL boundary measure marking the ROI.
        defined_on : 'template' | 'sample'
        input_type : 'intensity' | 'total_force'

        Notes
        -----
        Pressure (sample):  −∫_Γs p n_x · v dS_x = −∫_Γref [p J F^{-T} n_ref] · v̂ dS_X
        Traction (sample):   ∫_Γs t_phys · v dS_x =  ∫_Γref [J_surf t_phys] · v̂ dS_X,
                            J_surf = J ||F^{-T} n_ref||.
        """
        if ds is None:
            raise ValueError("Boundary measure 'ds' must be provided to apply_load().")
        if defined_on not in ("template", "sample"):
            raise ValueError("defined_on must be 'template' or 'sample'.")
        if input_type not in ("intensity", "total_force"):
            raise ValueError("input_type must be 'intensity' or 'total_force'.")

        gdim = self.V.mesh.geometry.dim

        # --- Distinguish scalar (pressure) vs. vector (traction) intent
        is_scalar = False
        is_vector = False

        if isinstance(q, (fem.Constant, ufl.core.expr.Expr)):
            shp = ufl.shape(q)
            if shp == ():
                is_scalar = True
            elif shp == (gdim,):
                is_vector = True
            else:
                raise ValueError(f"Unsupported UFL shape for q: {shp}, expected scalar or length-{gdim} vector.")
        else:
            arr = np.asarray(q, dtype=float).ravel()
            if arr.size == 1:
                is_scalar = True
            elif arr.size == gdim:
                is_vector = True
            else:
                raise ValueError(f"q must be scalar (pressure) or length-{gdim} vector (traction); got shape {arr.shape}")

        # =========================================
        # PRESSURE BRANCH (scalar)
        # =========================================
        if is_scalar:
            if input_type == "intensity":
                p_expr = self._as_scalar_constant(q)
                term = self._pressure_intensity_term(p_expr, ds, defined_on)
                if self.is_root:
                    logger.info(
                        "apply_load: pressure intensity on %s", "template" if defined_on == "template" else "sample"
                    )
                self._append_load_term(term)
                return

            # input_type == 'total_force' for PRESSURE:
            F_mag = float(np.asarray(q, dtype=float).ravel()[0])
            term = self._pressure_total_force_term(F_mag, ds, defined_on)
            self._append_load_term(term)
            return

        # =========================================
        # TRACTION BRANCH (vector)
        # =========================================
        if is_vector:
            if input_type == "intensity":
                t_expr = self._as_vector_expr(q)
                term = self._traction_intensity_term(t_expr, ds, defined_on)
                if self.is_root:
                    logger.info(
                        "apply_load: traction intensity on %s", "template" if defined_on == "template" else "sample"
                    )
                self._append_load_term(term)
                return

            # input_type == 'total_force' for TRACTION:
            F_vec = np.asarray(q, dtype=float).ravel()
            term = self._traction_total_force_term(F_vec, ds, defined_on)
            self._append_load_term(term)
            return

        # Should never get here
        raise RuntimeError("apply_load: unreachable branch.")



    def apply_gravity(
        self,
        rho: float | fem.Constant | ufl.core.expr.Expr,
        g: list[float] | np.ndarray | fem.Constant | ufl.core.expr.Expr,
        dx=None,
    ):
        """Add body force due to gravity: ∫_Ω ρ g · v J dx.

        Parameters
        ----------
        rho : float | Constant | UFL Expr
            Mass density (kg/m^3). Can be scalar Constant or spatially varying expression/function.
        g : Sequence[float] | Constant | UFL Expr
            Gravitational acceleration vector (m/s^2), e.g., (0, 0, -9.81).
        dx : Measure
            Cell measure for (sub)domains; defaults to global ufl.dx.
        """
        v = self.v
        J = self._J
        rho_expr = self._as_scalar_constant(rho)
        g_expr = self._as_vector_expr(g)
        if dx is None:
            dx = self._dx
        term = ufl.inner(rho_expr * g_expr, v) * J * dx
        if self.is_root:
            logger.info("Adding gravity body force to RHS (accumulating).")
        self._append_load_term(term)

    def fixed_dirichlet(self, ds, gamma: float, defined_on: str = "sample"):
        """Backward-compatible wrapper: original signature (ds, gamma, defined_on).

        Delegates to base implementation which expects (gamma, ds, defined_on).
        """
        super().fixed_dirichlet(gamma=gamma, ds=ds, defined_on=defined_on)

    def update(self) -> None:
        """Refresh operators from current E/nu/phi."""
        super().update()

    def solve(self) -> fem.Function:
        """Solve ``K u = b`` and return the FEM displacement field."""
        if self.is_root:
            logger.info("Solve linear system")

        if not hasattr(self, "ksp"):
            self._create_ksp_solver()

        if not hasattr(self, "b"):
            self._assemble_rhs()

        self.u = fem.Function(self.V, name="u")
        x = self.u.x.petsc_vec
        x.set(0.0)

        self.ksp.setOperators(self.K)
        self.ksp.solve(self.b, x)
        its = self.ksp.getIterationNumber()
        reason = self.ksp.getConvergedReason()

        self.u.x.scatter_forward()

        if self.is_root:
            logger.info("KSP iters: %d, reason: %s", its, str(reason))

        self._u_rbf = RBFInterpolator(
            self.V.tabulate_dof_coordinates(),
            self.u.x.array.reshape(-1, self.V.mesh.geometry.dim),
            neighbors=self.config.get("RBF_SOLVER_NEIGHBORS"),
            smoothing=self.config.get("RBF_SOLVER_SMOOTHING"),
        )
        return self.u

    def compute_strain_stress(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute strain and stress tensors projected to DG0 (per-cell constant).

        Uses the current displacement solution and geometry mapping φ.
        Returns full 3×3 tensors (not Voigt).

        Returns
        -------
        strain : ndarray shape (num_cells, 3, 3)
            Symmetric strain tensor ε = sym(∇u · F⁻¹).
        stress : ndarray shape (num_cells, 3, 3)
            Cauchy-type stress tensor σ = λ tr(ε) I + 2μ ε.
        """
        if not isinstance(self.u, fem.Function):
            raise RuntimeError("Call solve() before computing strain/stress.")

        mesh = self.V.mesh
        gdim = mesh.geometry.dim
        I = ufl.Identity(gdim)

        # DG0 tensor function space (gdim × gdim)
        T = fem.functionspace(mesh, ("DG", 0, (gdim, gdim)))

        # UFL expressions using solved displacement (fem.Function)
        F = I + ufl.grad(self.phi)
        F_inv = ufl.inv(F)
        eps_expr = ufl.sym(ufl.grad(self.u) * F_inv)

        mu = self.E / (2.0 * (1.0 + self.nu))
        lmbda = self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))
        sig_expr = lmbda * ufl.tr(eps_expr) * I + 2.0 * mu * eps_expr

        # Interpolate into DG0
        pts = T.element.interpolation_points
        strain_func = fem.Function(T, name="strain")
        stress_func = fem.Function(T, name="stress")
        strain_func.interpolate(fem.Expression(eps_expr, pts))
        stress_func.interpolate(fem.Expression(sig_expr, pts))

        num_cells = mesh.topology.index_map(mesh.topology.dim).size_local
        strain = strain_func.x.array.reshape(num_cells, gdim, gdim).copy()
        stress = stress_func.x.array.reshape(num_cells, gdim, gdim).copy()

        if self.is_root:
            logger.info(
                "Strain/stress computed (DG0): %d cells, max|ε|=%.4e, max|σ|=%.4e",
                num_cells,
                np.max(np.abs(strain)),
                np.max(np.abs(stress)),
            )

        return strain, stress

    def deform(self, points: np.ndarray) -> np.ndarray:
        """Evaluate deformed coordinates X' at arbitrary points.

        Returns ``X' = X + u(X) + φ(X)``, where ``u`` (the FE solution) is
        interpolated to ``points`` via an RBF built on the FE DoF coordinates,
        and ``φ`` is interpolated by the geometry RBF constructed in the base class.
        """
        if not hasattr(self, "_u_rbf"):
            raise RuntimeError("Call solve() before requesting deformed coordinates.")
        
        return self._u_rbf(points) + points + self._fu(points)
