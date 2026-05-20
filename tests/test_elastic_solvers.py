from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from mpi4py import MPI
from dolfinx import mesh, fem
from dolfinx.fem import petsc
from dolfinx.mesh import locate_entities_boundary, meshtags
import ufl
from petsc4py import PETSc

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from conftest import _require_single_rank, test_config, create_zero_phi, make_test_config
from simulation.elasticsolver import ElasticSolver


pytestmark = pytest.mark.simulation





# ------------------------ fixtures ------------------------

@pytest.fixture(scope="session")
def cube_with_tags():
    """Unit cube + facet tags: 1:left(x=0), 2:right(x=1), 3:bottom(y=0), 4:top(y=1), 5:back(z=0), 6:front(z=1)."""
    _require_single_rank()

    from conftest import create_unit_cube
    domain = create_unit_cube()  # tetra by default
    tdim = domain.topology.dim
    fdim = tdim - 1

    # Markers
    def is_left(x):   return np.isclose(x[0], 0.0)
    def is_right(x):  return np.isclose(x[0], 1.0)
    def is_bottom(x): return np.isclose(x[1], 0.0)
    def is_top(x):    return np.isclose(x[1], 1.0)
    def is_back(x):   return np.isclose(x[2], 0.0)
    def is_front(x):  return np.isclose(x[2], 1.0)

    tags = []
    vals = []

    for val, marker in [
        (1, is_left), (2, is_right),
        (3, is_bottom), (4, is_top),
        (5, is_back), (6, is_front)
    ]:
        facets = locate_entities_boundary(domain, fdim, marker)
        if facets.size > 0:
            tags.append(facets)
            vals.append(np.full(facets.size, val, dtype=np.int32))

    if len(tags) == 0:
        pytest.skip("No boundary facets found (unexpected).")
    facets_all = np.hstack(tags).astype(np.int32)
    values_all = np.hstack(vals).astype(np.int32)

    facet_tags = meshtags(domain, fdim, facets_all, values_all)

    # FE spaces + materials
    V = fem.functionspace(domain, ("Lagrange", 1, (domain.geometry.dim,)))
    W = fem.functionspace(domain, ("DG", 0))
    E = fem.Function(W); E.x.array[:] = 900.0
    nu = fem.Function(W); nu.x.array[:] = 0.28

    # Measures (use tagged ds everywhere)
    ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)
    dx = ufl.dx(domain=domain)

    return domain, V, W, E, nu, ds, dx, facet_tags

def _random_mode(V, seed=11):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(V.dofmap.index_map.size_local, V.mesh.geometry.dim))

# ------------------------ tag sanity ------------------------

def test_facet_tagging_area_and_normals(cube_with_tags, test_config):
    """Check areas of unit cube faces and net outward normal vectors."""
    _require_single_rank()
    import ufl
    domain, V, W, E, nu, ds, dx, facet_tags = cube_with_tags

    n = ufl.FacetNormal(domain)
    # Each face area is 1.0 for the unit cube
    for tag, expected_normal in [
        (1, np.array([-1.0, 0.0,  0.0])),  # left
        (2, np.array([ 1.0, 0.0,  0.0])),  # right
        (3, np.array([ 0.0,-1.0,  0.0])),  # bottom
        (4, np.array([ 0.0, 1.0,  0.0])),  # top
        (5, np.array([ 0.0, 0.0, -1.0])),  # back
        (6, np.array([ 0.0, 0.0,  1.0])),  # front
    ]:
        area = fem.assemble_scalar(fem.form(1 * ds(tag)))
        assert np.isclose(area, 1.0, rtol=1e-12, atol=1e-14)

        N = np.array([
            fem.assemble_scalar(fem.form(n[i] * ds(tag)))
            for i in range(3)
        ], dtype=float)
        assert np.allclose(N, expected_normal * area, atol=1e-12)

# ------------------------ unified load API tests ------------------------

def test_apply_load_traction_template_intensity_with_tag(cube_with_tags, test_config):
    """t_ref intensity on TEMPLATE: b = ∫ t_ref · v dS_ref over 'front' (z=1, tag=6)."""
    _require_single_rank()
    import ufl
    from simulation.elasticsolver import ElasticSolver

    domain, V, _, E, nu, ds, _, _ = cube_with_tags
    phi = create_zero_phi(V); solver = ElasticSolver(V, E, nu, phi, test_config)

    t = np.array([0.3, -0.1, 2.5])  # N/m^2
    solver.clear_loads()
    solver.apply_load(t, ds(6), defined_on="template", input_type="intensity")

    v = solver.v
    t_expr = fem.Constant(domain, tuple(map(float, t)))
    expected_form = fem.form(ufl.inner(t_expr, v) * ds(6))
    expected_vec = petsc.create_vector(expected_form.function_spaces[0])
    expected_vec.set(0.0)
    petsc.assemble_vector(expected_vec, expected_form)
    expected_vec.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)

    assert np.allclose(solver.b.getArray(), expected_vec.getArray(), atol=1e-12)

    expected_vec.destroy()
    solver.K.destroy()

def test_apply_load_pressure_sample_intensity_with_tag_and_mapping(cube_with_tags, test_config):
    """p intensity on SAMPLE with affine map φ=sX: b = -∫ p (J F^{-T} n) · v dS_ref (tag=6, front)."""
    _require_single_rank()
    import ufl
    from simulation.elasticsolver import ElasticSolver

    domain, V, _, E, nu, ds, _, _ = cube_with_tags
    phi = create_zero_phi(V); solver = ElasticSolver(V, E, nu, phi, test_config)

    # Affine scaling map: F = I + s I -> J=(1+s)^3, F^{-T}=(1+s)^{-1} I
    s = 0.15
    X = V.tabulate_dof_coordinates()
    phi.x.array[:] = (s * X).reshape(-1)
    solver.update()

    p = 0.2  # Pa
    solver.clear_loads()
    solver.apply_load(p, ds(6), defined_on="sample", input_type="intensity")

    v = solver.v
    n = ufl.FacetNormal(domain)

    phi = solver.phi
    F = ufl.Identity(domain.geometry.dim) + ufl.grad(phi)
    J = ufl.det(F)
    JFn = J * ufl.transpose(ufl.inv(F)) * n

    p_expr = fem.Constant(domain, float(p))
    expected_form = fem.form(-ufl.inner(p_expr * JFn, v) * ds(6))
    expected_vec = petsc.create_vector(expected_form.function_spaces[0])
    expected_vec.set(0.0)
    petsc.assemble_vector(expected_vec, expected_form)
    expected_vec.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)

    assert np.allclose(solver.b.getArray(), expected_vec.getArray(), atol=1e-12)

    expected_vec.destroy()
    solver.K.destroy()

def test_apply_load_traction_sample_total_force_uniform_intensity_with_tag(cube_with_tags, test_config):
    """F vector on SAMPLE → uniform t_phys = F/A_phys over 'left' (x=0, tag=1)."""
    _require_single_rank()
    import ufl
    from simulation.elasticsolver import ElasticSolver

    domain, V, _, E, nu, ds, _, _ = cube_with_tags
    phi = create_zero_phi(V); solver = ElasticSolver(V, E, nu, phi, test_config)

    # Mild affine map so J_surf != 1
    s = 0.07
    X = V.tabulate_dof_coordinates()
    phi.x.array[:] = (s * X).reshape(-1)
    solver.update()

    F_vec = np.array([3.0, 0.5, -1.2])  # N
    solver.clear_loads()
    solver.apply_load(F_vec, ds(1), defined_on="sample", input_type="total_force")

    # Expected: A_phys = ∫ J_surf dS_ref, t_phys = F / A_phys
    n = ufl.FacetNormal(domain)
    phi = solver.phi
    Fmap = ufl.Identity(domain.geometry.dim) + ufl.grad(phi)
    J = ufl.det(Fmap)
    J_surf = J * ufl.sqrt(ufl.inner(ufl.transpose(ufl.inv(Fmap)) * n,
                                    ufl.transpose(ufl.inv(Fmap)) * n))

    A_phys = fem.assemble_scalar(fem.form(J_surf * ds(1)))
    t_const = fem.Constant(domain, tuple((F_vec / A_phys).tolist()))

    v = solver.v
    expected_form = fem.form(ufl.inner(t_const, v) * J_surf * ds(1))
    expected_vec = petsc.create_vector(expected_form.function_spaces[0])
    expected_vec.set(0.0)
    petsc.assemble_vector(expected_vec, expected_form)
    expected_vec.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)

    assert np.allclose(solver.b.getArray(), expected_vec.getArray(), atol=1e-12)

    expected_vec.destroy()
    solver.K.destroy()

def test_apply_load_pressure_sample_total_force_with_tag(cube_with_tags, test_config):
    """|F| scalar on SAMPLE (pressure): p = |F| / ||∫ J F^{-T} n dS|| over 'top' (y=1, tag=4)."""
    _require_single_rank()
    import ufl
    from simulation.elasticsolver import ElasticSolver

    domain, V, _, E, nu, ds, _, _ = cube_with_tags
    phi = create_zero_phi(V); solver = ElasticSolver(V, E, nu, phi, test_config)

    Fmag = 7.5  # N (scalar)
    solver.clear_loads()
    solver.apply_load(Fmag, ds(4), defined_on="sample", input_type="total_force")

    n = ufl.FacetNormal(domain)
    N = np.array([fem.assemble_scalar(fem.form(n[i] * ds(4))) for i in range(3)], dtype=float)
    N_norm = float(np.linalg.norm(N))
    p_val = Fmag / N_norm
    p_const = fem.Constant(domain, float(p_val))

    v = solver.v
    expected_form = fem.form(-ufl.inner(p_const * n, v) * ds(4))
    expected_vec = petsc.create_vector(expected_form.function_spaces[0])
    expected_vec.set(0.0)
    petsc.assemble_vector(expected_vec, expected_form)
    expected_vec.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)

    assert np.allclose(solver.b.getArray(), expected_vec.getArray(), atol=1e-12)

    expected_vec.destroy()
    solver.K.destroy()

def test_apply_load_pressure_template_total_force_with_tag1(cube_with_tags, test_config):
    """|F| scalar on TEMPLATE (pressure): p_ref = |F| / ||∫ n_ref dS|| over 'left' (x=0, tag=1)."""
    _require_single_rank()
    import ufl
    from simulation.elasticsolver import ElasticSolver

    domain, V, _, E, nu, ds, _, _ = cube_with_tags
    phi = create_zero_phi(V); solver = ElasticSolver(V, E, nu, phi, test_config)

    Fmag = 4.2  # N
    solver.clear_loads()
    solver.apply_load(Fmag, ds(1), defined_on="template", input_type="total_force")

    n = ufl.FacetNormal(domain)
    N = np.array([fem.assemble_scalar(fem.form(n[i] * ds(1))) for i in range(3)], dtype=float)
    N_norm = float(np.linalg.norm(N))
    p_ref = Fmag / N_norm
    p_const = fem.Constant(domain, float(p_ref))

    v = solver.v
    expected_form = fem.form(-ufl.inner(p_const * n, v) * ds(1))
    expected_vec = petsc.create_vector(expected_form.function_spaces[0])
    expected_vec.set(0.0)
    petsc.assemble_vector(expected_vec, expected_form)
    expected_vec.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)

    assert np.allclose(solver.b.getArray(), expected_vec.getArray(), atol=1e-12)

    expected_vec.destroy()
    solver.K.destroy()


def test_apply_load_invalid_inputs_raise(cube_with_tags, test_config):
    """Invalid defined_on/input_type arguments should raise ValueError."""
    _require_single_rank()
    domain, V, _, E, nu, ds, _, _ = cube_with_tags
    phi = create_zero_phi(V); solver = ElasticSolver(V, E, nu, phi, test_config)

    with pytest.raises(ValueError, match="defined_on"):
        solver.apply_load(q=1.0, ds=ds(6), defined_on="invalid", input_type="intensity")

    with pytest.raises(ValueError, match="input_type"):
        solver.apply_load(q=1.0, ds=ds(6), defined_on="template", input_type="bad")

    solver.K.destroy()


def test_apply_load_total_force_zero_area_tag_raises(cube_with_tags, test_config):
    """Total-force loads on non-existent tag should raise due to zero area/norm."""
    _require_single_rank()
    import ufl
    domain, V, _, E, nu, ds_tagged, _, _ = cube_with_tags
    phi = create_zero_phi(V); solver = ElasticSolver(V, E, nu, phi, test_config)

    # Use a tag that doesn't exist in ds_tagged
    bad_tag = 99

    with pytest.raises(RuntimeError):
        solver.apply_load(q=(1.0, 0.0, 0.0), ds=ds_tagged(bad_tag), defined_on="template", input_type="total_force")

    with pytest.raises(RuntimeError):
        solver.apply_load(q=1.0, ds=ds_tagged(bad_tag), defined_on="sample", input_type="total_force")

    solver.K.destroy()

# ------------------------ gravity loads ------------------------

def test_apply_gravity_matches_form_no_mapping(cube_with_tags, test_config):
    """Gravity: ∫ ρ g · v J dx; with φ=0 => J=1."""
    _require_single_rank()
    import ufl
    from simulation.elasticsolver import ElasticSolver

    domain, V, _, E, nu, ds, dx, _ = cube_with_tags
    phi = create_zero_phi(V); solver = ElasticSolver(V, E, nu, phi, test_config)

    rho = 1.23
    g = (0.0, 0.0, -9.81)

    solver.clear_loads()
    solver.apply_gravity(rho=rho, g=g, dx=dx)

    v = solver.v
    phi = solver.phi
    F = ufl.Identity(domain.geometry.dim) + ufl.grad(phi)
    J = ufl.det(F)
    rho_c = fem.Constant(domain, float(rho))
    g_c = fem.Constant(domain, tuple(map(float, g)))

    expected_form = fem.form(ufl.inner(rho_c * g_c, v) * J * dx)
    expected_vec = petsc.create_vector(expected_form.function_spaces[0])
    expected_vec.set(0.0)
    petsc.assemble_vector(expected_vec, expected_form)
    expected_vec.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)

    assert np.allclose(solver.b.getArray(), expected_vec.getArray(), atol=1e-12)

    expected_vec.destroy()
    solver.K.destroy()

def test_apply_load_body_force_sample_intensity_with_mapping(cube_with_tags, test_config):
    """Gravity with affine φ=sX; check J appears in RHS correctly."""
    _require_single_rank()
    import ufl
    from simulation.elasticsolver import ElasticSolver

    domain, V, _, E, nu, ds, dx, _ = cube_with_tags
    phi = create_zero_phi(V); solver = ElasticSolver(V, E, nu, phi, test_config)

    s = 0.2
    X = V.tabulate_dof_coordinates()
    phi.x.array[:] = (s * X).reshape(-1)
    solver.update()

    rho = 2.0
    g = (0.1, -0.2, 0.3)

    solver.clear_loads()
    solver.apply_gravity(rho=rho, g=g, dx=dx)

    v = solver.v
    phi = solver.phi
    F = ufl.Identity(domain.geometry.dim) + ufl.grad(phi)
    J = ufl.det(F)
    rho_c = fem.Constant(domain, float(rho))
    g_c = fem.Constant(domain, tuple(map(float, g)))

    expected_form = fem.form(ufl.inner(rho_c * g_c, v) * J * dx)
    expected_vec = petsc.create_vector(expected_form.function_spaces[0])
    expected_vec.set(0.0)
    petsc.assemble_vector(expected_vec, expected_form)
    expected_vec.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)

    assert np.allclose(solver.b.getArray(), expected_vec.getArray(), atol=1e-12)

    expected_vec.destroy()
    solver.K.destroy()

# ------------------------ deformation evaluation ------------------------

def test_deform_handles_missing_mapping(cube_with_tags, test_config):
    _require_single_rank()

    domain, V, _, E, nu, ds, dx, _ = cube_with_tags
    phi = create_zero_phi(V); solver = ElasticSolver(V, E, nu, phi, test_config)

    solver.clear_loads()
    solver.solve()
    points = np.array(
        [
            [0.15, 0.2, 0.3],
            [0.7, 0.6, 0.5],
        ],
        dtype=float,
    )
    base = solver.deform(points)
    assert np.allclose(base, points, atol=1e-12)

    solver.fu = lambda coords: None
    restored = solver.deform(points)
    assert np.allclose(restored, points, atol=1e-12)
    solver.K.destroy()


def test_deform_adds_mapping_translation(cube_with_tags, test_config):
    """Test that deform() returns points + solved displacement (no fu mapping in simplified API)."""
    _require_single_rank()

    domain, V, _, E, nu, ds, dx, _ = cube_with_tags
    phi = create_zero_phi(V); solver = ElasticSolver(V, E, nu, phi, test_config)

    translation = np.array([0.01, -0.02, 0.03], dtype=float)
    X = V.tabulate_dof_coordinates()
    phi.x.array[:] = np.broadcast_to(translation, X.shape).reshape(-1)
    solver.update()

    solver.clear_loads()
    solver.solve()

    points = np.array(
        [
            [0.05, 0.1, 0.15],
            [0.9, 0.4, 0.25],
        ],
        dtype=float,
    )
    deformed = solver.deform(points)
    # With no loads, u=0, so deform returns points + phi mapping
    expected = points + translation
    assert np.allclose(deformed, expected, atol=1e-12)
    solver.K.destroy()

# ------------------------ Dirichlet penalty tests ------------------------

def test_fixed_dirichlet_penalty_energy_matches_form(cube_with_tags, test_config):
    """Energy increment from penalty γ∫_Γ u·u dS equals vecᵀ(K_with-K_wo)vec for any vec."""
    _require_single_rank()
    import ufl
    from simulation.elasticsolver import ElasticSolver

    domain, V, W, E, nu, ds, dx, _ = cube_with_tags
    phi = create_zero_phi(V); solver = ElasticSolver(V, E, nu, phi, test_config)

    # Random displacement vector
    rng = np.random.default_rng(123)
    mode = rng.normal(size=(V.dofmap.index_map.size_local, V.mesh.geometry.dim))
    uh = fem.Function(V)
    uh.x.array[:] = mode.reshape(-1)

    vec = solver.K.createVecRight(); vec.setArray(mode.reshape(-1))
    energy_wo = vec.dot(solver.K * vec)

    gamma = 5e2
    solver.fixed_dirichlet(ds(6), gamma=gamma)
    energy_with = vec.dot(solver.K * vec)

    # Expected increment = γ ∫_Γ u·u dS on the tagged boundary (front, tag=6)
    inc_expected = gamma * fem.assemble_scalar(fem.form(ufl.inner(uh, uh) * ds(6)))

    assert np.isclose(energy_with - energy_wo, inc_expected, rtol=1e-5, atol=1e-10)

    vec.destroy()
    solver.K.destroy()

# ------------------------ Material scaling tests ------------------------

def test_update_materials_scales_response_elastic_only(cube_with_tags, test_config):
    """If no penalty is present, K scales linearly with E."""
    _require_single_rank()

    domain, V, W, E, nu, ds, dx, _ = cube_with_tags
    phi = create_zero_phi(V); solver = ElasticSolver(V, E, nu, phi, test_config)

    rng = np.random.default_rng(7)
    mode = rng.normal(size=(V.dofmap.index_map.size_local, V.mesh.geometry.dim))
    vec = solver.K.createVecRight()
    vec.setArray(mode.reshape(-1))

    baseline = vec.dot(solver.K * vec)

    scale = 1.7
    E_scaled = fem.Function(W); E_scaled.x.array[:] = scale * E.x.array
    solver.E.x.array[:] = E_scaled.x.array
    solver.update()
    scaled = vec.dot(solver.K * vec)

    assert np.isclose(scaled, scale * baseline, rtol=1e-5, atol=1e-8)

    vec.destroy()
    solver.K.destroy()

def test_update_materials_scales_response_with_penalty_two_solvers(cube_with_tags, test_config):
    """
    With penalty present, K_total = K_elastic(E) + K_penalty(gamma).
    Validate scaling using two separate solvers to avoid matrix-state coupling:
      scaled_total - penalty_energy  ≈  scale * baseline_elastic
    """
    _require_single_rank()
    import ufl

    domain, V, W, E, nu, ds, dx, _ = cube_with_tags

    # Solver with penalty
    phi = create_zero_phi(V); solver_pen = ElasticSolver(V, E, nu, phi, test_config)
    gamma = 1e3
    solver_pen.fixed_dirichlet(ds(6), gamma=gamma)

    # Solver without penalty
    solver_el = ElasticSolver(V, E, nu, phi, test_config)

    rng = np.random.default_rng(17)
    mode = rng.normal(size=(V.dofmap.index_map.size_local, V.mesh.geometry.dim))

    vec = solver_pen.K.createVecRight(); vec.setArray(mode.reshape(-1))
    vec_el = solver_el.K.createVecRight(); vec_el.setArray(mode.reshape(-1))

    # Energies
    baseline_total = vec.dot(solver_pen.K * vec)
    baseline_elastic = vec_el.dot(solver_el.K * vec_el)

    # Penalty energy computed directly from form γ ∫ u·u dS on tag=6
    uh = fem.Function(V); uh.x.array[:] = mode.reshape(-1)
    penalty_energy = gamma * fem.assemble_scalar(fem.form(ufl.inner(uh, uh) * ds(6)))

    # Sanity: baseline_total - penalty ≈ elastic
    assert np.isclose(baseline_total - penalty_energy, baseline_elastic, rtol=1e-5, atol=1e-8)

    # Scale E in both solvers
    scale = 1.7
    E_scaled = fem.Function(W); E_scaled.x.array[:] = scale * E.x.array
    solver_pen.E.x.array[:] = E_scaled.x.array
    solver_pen.update()
    solver_el.E.x.array[:] = E_scaled.x.array
    solver_el.update()

    scaled_total = vec.dot(solver_pen.K * vec)
    scaled_elastic = vec_el.dot(solver_el.K * vec_el)

    # Check scaling on the elastic part and the combined operator
    assert np.isclose(scaled_elastic, scale * baseline_elastic, rtol=1e-5, atol=1e-8)
    assert np.isclose(scaled_total - penalty_energy, scale * baseline_elastic, rtol=1e-5, atol=1e-8)

    vec.destroy(); vec_el.destroy()
    solver_pen.K.destroy(); solver_el.K.destroy()

# ------------------------ Linear superposition ------------------------

def test_load_superposition_with_tags(cube_with_tags, test_config):
    from simulation.elasticsolver import ElasticSolver

    domain, V, W, E, nu, ds, dx, _ = cube_with_tags
    phi = create_zero_phi(V); solver = ElasticSolver(V, E, nu, phi, test_config)

    t1 = np.array([1.5, -0.4, 0.0])
    t2 = np.array([-0.7, 0.2, 1.1])

    solver.clear_loads()
    solver.apply_load(t1, ds(5), defined_on="template", input_type="intensity")
    rhs1 = solver.b.copy()

    solver.clear_loads()
    solver.apply_load(t2, ds(5), defined_on="template", input_type="intensity")
    rhs2 = solver.b.copy()

    solver.clear_loads()
    solver.apply_load(t1, ds(5), defined_on="template", input_type="intensity")
    solver.apply_load(t2, ds(5), defined_on="template", input_type="intensity")
    rhs_sum = solver.b.copy()

    rhs1.axpy(1.0, rhs2)
    assert np.allclose(rhs1.getArray(), rhs_sum.getArray(), atol=1e-12)

    rhs1.destroy(); rhs2.destroy(); rhs_sum.destroy()
    solver.K.destroy()

if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))

# ======================== Penalty mapping invariance tests ========================

from simulation.elasticsolver import ElasticSolver

pytestmark = pytest.mark.simulation

def _cube(n=8): return mesh.create_unit_cube(MPI.COMM_WORLD, n, n, n)

def test_dirichlet_penalty_template_vs_sample_scaling(test_config):
    """'template' penalty integrates on reference surface; 'sample' uses J_surf.
    With φ ≠ 0 (nontrivial mapping), their energies differ by the surface Jacobian factor.
    """
    _require_single_rank()
    msh = _cube(10)
    V = fem.functionspace(msh, ("Lagrange", 1, (msh.geometry.dim,)))
    W = fem.functionspace(msh, ("DG", 0))
    E = fem.Function(W); E.x.array[:] = 2000.0
    nu = fem.Function(W); nu.x.array[:] = 0.25
    cfg = make_test_config({})

    # tag a plane x=0
    from dolfinx.mesh import locate_entities_boundary, meshtags
    fdim = msh.topology.dim - 1
    facets = locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[0], 0.0))
    ft = meshtags(msh, fdim, facets, np.full(facets.size, 1, np.int32))
    ds = ufl.Measure("ds", domain=msh, subdomain_data=ft)

    # Build two solvers with the same φ
    phi = create_zero_phi(V)
    X = V.tabulate_dof_coordinates()
    A = np.diag([1.06, 0.98, 1.02]); b = np.array([0.01, -0.005, 0.0])
    u = (A @ X.T).T + b - X
    phi.x.array[:] = u.reshape(-1)

    sol_t = ElasticSolver(V, E, nu, phi, cfg)  # template penalty (no J_surf)
    sol_s = ElasticSolver(V, E, nu, phi, cfg)  # sample penalty (with J_surf)

    gamma = 1e6
    sol_t.fixed_dirichlet(ds(1), gamma, defined_on="template")
    sol_s.fixed_dirichlet(ds(1), gamma, defined_on="sample")
    # Use the same random mode to probe penalty energy
    rng = np.random.default_rng(0)
    mode = rng.normal(size=(V.dofmap.index_map.size_local, V.mesh.geometry.dim))

    vt = sol_t.K.createVecRight(); vt.setArray(mode.reshape(-1))
    vs = sol_s.K.createVecRight(); vs.setArray(mode.reshape(-1))

    Et = float(vt.dot(sol_t.K * vt))
    Es = float(vs.dot(sol_s.K * vs))

    # Both solvers share the same elastic K; only the penalty term differs.
    # template: γ ∫ u·u dS_ref ;  sample: γ ∫ u·u J_surf dS_ref
    # => Es - Et = γ ∫ u·u (J_surf - 1) dS_ref
    uh = fem.Function(V); uh.x.array[:] = mode.reshape(-1)
    n = ufl.FacetNormal(msh)
    Fmap = ufl.Identity(msh.geometry.dim) + ufl.grad(sol_s.phi)
    Jdet = ufl.det(Fmap)
    J_surf = Jdet * ufl.sqrt(ufl.inner(ufl.transpose(ufl.inv(Fmap)) * n,
                                       ufl.transpose(ufl.inv(Fmap)) * n))
    delta_expected = gamma * fem.assemble_scalar(
        fem.form(ufl.inner(uh, uh) * (J_surf - 1.0) * ds(1))
    )
    assert np.isclose(Es - Et, delta_expected, rtol=1e-6, atol=1e-8), (
        f"Penalty template/sample mismatch: ΔE={Es - Et:.6e}, expected={delta_expected:.6e}"
    )

    # Sanity: with diag(A)=(1.06,0.98,1.02) on x=0 face, J_surf = 0.98*1.02 ≈ 0.9996
    expected_J_surf = 0.98 * 1.02
    area = fem.assemble_scalar(fem.form(1.0 * ds(1)))
    mean_J_surf = fem.assemble_scalar(fem.form(J_surf * ds(1))) / area
    assert np.isclose(mean_J_surf, expected_J_surf, rtol=1e-10)

    vt.destroy(); vs.destroy()
    sol_t.K.destroy(); sol_s.K.destroy()
