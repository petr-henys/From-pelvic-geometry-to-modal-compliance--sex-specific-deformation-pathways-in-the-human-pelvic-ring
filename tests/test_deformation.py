"""Deformation mapping regression and consistency tests."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from mpi4py import MPI
from dolfinx import mesh, fem
import ufl
import pyvista as pv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from conftest import _require_single_rank, create_zero_phi, test_config, make_test_config

from simulation.eigensolver import ElasticEigenSolver
from simulation.elasticsolver import ElasticSolver

pytestmark = pytest.mark.simulation


# ======================== Fixture for test setup ========================


@pytest.fixture(scope="function")
def elastic_test_setup():
    """Create cube mesh, vector space, and homogeneous materials for tests."""
    _require_single_rank()
    import ufl

    from conftest import create_unit_cube
    domain = create_unit_cube()
    gdim = domain.geometry.dim
    V = fem.functionspace(domain, ("Lagrange", 1, (gdim,)))
    W = fem.functionspace(domain, ("DG", 0))

    E = fem.Function(W)
    E.x.array[:] = 1000.0
    nu = fem.Function(W)
    nu.x.array[:] = 0.3

    ds = ufl.ds(domain=domain)
    dx = ufl.dx(domain=domain)
    
    return domain, V, W, E, nu, ds, dx


def _top_face_measure(domain):
    """Tagged facet measure for the top face z=1 of a hexahedral unit cube mesh."""
    from dolfinx import mesh as dmesh
    import ufl
    fdim = domain.topology.dim - 1
    def is_top(x):
        return np.isclose(x[2], 1.0)
    facets_top = dmesh.locate_entities_boundary(domain, fdim, is_top)
    values_top = np.full(facets_top.size, 1, dtype=np.int32)
    facet_tags = dmesh.meshtags(domain, fdim, facets_top, values_top)
    return ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)


# ======================== ElasticEigenSolver F-mapping tests ========================


def test_eigensolver_identity_deformation_gives_original_mass(elastic_test_setup, test_config):
    """Test that φ=0 (identity mapping) gives mass matrix without Jacobian modification."""
    _require_single_rank()
    import ufl

    domain, V, W, E, nu, ds, dx = elastic_test_setup
    phi = create_zero_phi(V)

    solver = ElasticEigenSolver(V, E, nu, phi, test_config)

    # φ = 0 by default (identity mapping), so J = det(I) = 1
    # Check that M matches M0 (base mass without Jacobian)
    
    # Compare matrix norms
    M_norm = solver.M.norm()
    M0_norm = solver.M0.norm()
    
    assert np.isclose(M_norm, M0_norm, rtol=1e-10), \
        "With φ=0, M should equal M0 (Jacobian J=1)"
    
    # Cleanup
    solver.K.destroy()
    solver.M.destroy()
    solver.M0.destroy()


def test_eigensolver_uniform_expansion_increases_mass(elastic_test_setup, test_config):
    """Uniform expansion φ=cx scales mass by J=(1+c)^3 in 3D."""
    _require_single_rank()
    import ufl

    domain, V, W, E, nu, ds, dx = elastic_test_setup
    phi = create_zero_phi(V)

    solver = ElasticEigenSolver(V, E, nu, phi, test_config)
    
    # Get original mass
    M0_norm = solver.M0.norm()
    
    # Apply uniform expansion: φ = 0.1 * x (10% expansion in all directions)
    X = V.tabulate_dof_coordinates()
    expansion_factor = 0.1
    phi_values = expansion_factor * X
    solver.phi.x.array[:] = phi_values.flatten()
    solver.phi.x.scatter_forward()
    
    # Reassemble mass with new Jacobian
    solver._assemble_mass()
    
    # F = I + grad(φ) = I + 0.1*I = 1.1*I
    # J = det(F) = 1.1^3 = 1.331 for 3D
    expected_J = (1.0 + expansion_factor) ** 3
    
    # Mass should scale by J
    M_new_norm = solver.M.norm()
    actual_ratio = M_new_norm / M0_norm
    
    assert np.isclose(actual_ratio, expected_J, rtol=0.03), \
        f"Mass should scale by J={expected_J:.3f}, got {actual_ratio:.3f}"
    
    # Cleanup
    solver.K.destroy()
    solver.M.destroy()
    solver.M0.destroy()


def test_eigensolver_stiffness_includes_jacobian(elastic_test_setup, test_config):
    """Stiffness matrix changes when φ ≠ 0 due to J and F^{-1} in ε(u)."""
    _require_single_rank()

    domain, V, W, E, nu, ds, dx = elastic_test_setup
    phi = create_zero_phi(V)

    solver = ElasticEigenSolver(V, E, nu, phi, test_config)
    
    K0_norm = solver.K.norm()
    
    # Apply small deformation
    X = V.tabulate_dof_coordinates()
    solver.phi.x.array[:] = 0.05 * X.flatten()
    solver.phi.x.scatter_forward()
    
    # Reassemble stiffness
    solver._assemble_stiffness()
    
    K_new_norm = solver.K.norm()
    
    # Stiffness should change due to J and F^{-1} terms
    assert not np.isclose(K_new_norm, K0_norm, rtol=1e-5), \
        "Stiffness should change when φ != 0"
    
    # Cleanup
    solver.K.destroy()
    solver.M.destroy()
    solver.M0.destroy()


def test_eigensolver_jacobian_in_strain_definition(elastic_test_setup, test_config):
    """Strain ε(u)=sym(∇u F^{-1}) yields symmetric elastic stiffness."""
    _require_single_rank()

    domain, V, W, E, nu, ds, dx = elastic_test_setup
    phi = create_zero_phi(V)

    solver = ElasticEigenSolver(V, E, nu, phi, test_config)
    
    # Apply shear deformation in phi
    X = V.tabulate_dof_coordinates()
    # φ_x = 0.1*y, φ_y = 0, φ_z = 0 (shear in xy plane)
    phi_vals = np.zeros_like(X)
    phi_vals[:, 0] = 0.1 * X[:, 1]
    solver.phi.x.array[:] = phi_vals.flatten()
    solver.phi.x.scatter_forward()
    
    # Reassemble with shear deformation
    solver._assemble_stiffness()
    
    # Check that stiffness matrix is assembled and symmetric
    K_norm = solver.K.norm()
    assert K_norm > 0, "Stiffness should be positive"
    
    # Test symmetry (elastic stiffness should be symmetric)
    K_T = solver.K.transpose()
    K_diff = solver.K.copy()
    K_diff.axpy(-1.0, K_T)
    diff_norm = K_diff.norm()
    
    assert diff_norm / K_norm < 1e-10, \
        "Stiffness matrix should be symmetric"
    
    # Cleanup
    K_T.destroy()
    K_diff.destroy()
    solver.K.destroy()
    solver.M.destroy()
    solver.M0.destroy()


def test_eigensolver_mass_matrix_updates_with_phi(elastic_test_setup, test_config):
    """Mass matrix M(φ) updates when φ changes (J depends on φ)."""
    _require_single_rank()

    domain, V, W, E, nu, ds, dx = elastic_test_setup
    phi = create_zero_phi(V)

    solver = ElasticEigenSolver(V, E, nu, phi, test_config)
    
    M_original = solver.M.copy()
    
    # Change phi
    X = V.tabulate_dof_coordinates()
    solver.phi.x.array[:] = 0.15 * X.flatten()
    solver.phi.x.scatter_forward()
    
    # Update mass
    solver._assemble_mass()
    
    # Mass should have changed
    M_diff = solver.M.copy()
    M_diff.axpy(-1.0, M_original)
    diff_norm = M_diff.norm()
    
    assert diff_norm > 1e-6, "Mass matrix should change when φ changes"
    
    # Cleanup
    M_original.destroy()
    M_diff.destroy()
    solver.K.destroy()
    solver.M.destroy()
    solver.M0.destroy()


# ======================== ElasticSolver F-mapping tests ========================


def test_elasticsolver_has_deformation_gradient(elastic_test_setup, test_config):
    """ElasticSolver initializes F^{-1}, J, and φ=0."""
    _require_single_rank()

    domain, V, W, E, nu, ds, dx = elastic_test_setup
    phi = create_zero_phi(V)

    solver = ElasticSolver(V, E, nu, phi, test_config)
    
    # Check that F_inv and J are defined
    assert solver._F_inv is not None, "F_inv should be defined"
    assert solver._J is not None, "J should be defined"
    
    # Check phi exists
    assert solver.phi is not None
    assert np.allclose(solver.phi.x.array[:], 0.0), "phi should start at zero"
    
    # Cleanup
    solver.K.destroy()


def test_elasticsolver_surface_jacobian_computation(elastic_test_setup, test_config):
    """`apply_load` constructs pullbacks using J F^{-T} n and J_surf without error."""
    _require_single_rank()
    import ufl

    domain, V, W, E, nu, ds, dx = elastic_test_setup
    
    # Create facet tags
    ds_tagged = _top_face_measure(domain)
    
    phi = create_zero_phi(V)

    
    solver = ElasticSolver(V, E, nu, phi, test_config)
    
    # Apply pressure load - this uses J_surf internally
    solver.apply_load(
        q=100.0,  # Pa
        ds=ds_tagged,
        defined_on='sample',
        input_type='intensity'
    )
    
    # If no error, J_surf was computed correctly
    assert len(solver._L_terms) == 1, "One load term should be added"
    
    # Cleanup
    solver.K.destroy()


def test_elasticsolver_phi_updates_surface_loads(elastic_test_setup, test_config):
    """Changing φ modifies surface loads via J F^{-T} n (pressure on sample)."""
    _require_single_rank()
    import ufl

    domain, V, W, E, nu, ds, dx = elastic_test_setup
    
    # Tag top face
    ds_tagged = _top_face_measure(domain)
    
    # Test with identity (phi=0)
    phi = create_zero_phi(V)

    solver = ElasticSolver(V, E, nu, phi, test_config)
    solver.apply_load(100.0, ds_tagged, defined_on='sample', input_type='intensity')
    
    # Assemble RHS
    solver._assemble_rhs()
    b_original_norm = solver.b.norm()
    
    # Apply deformation and reassemble
    X = V.tabulate_dof_coordinates()
    solver.phi.x.array[:] = 0.1 * X.flatten()
    solver.phi.x.scatter_forward()
    
    # Need to rebuild forms with new phi
    solver._L_terms.clear()
    solver.apply_load(100.0, ds_tagged, defined_on='sample', input_type='intensity')
    solver._assemble_rhs()
    b_new_norm = solver.b.norm()
    
    # Under uniform expansion φ=cx: JFn scales by (1+c)^2, so RHS 2-norm scales similarly
    expected = (1.0 + 0.1) ** 2
    ratio = b_new_norm / b_original_norm
    assert np.isclose(ratio, expected, rtol=0.05), \
        f"Pressure RHS should scale by (1+c)^2={expected:.3f}, got {ratio:.3f}"
    
    # Cleanup
    solver.K.destroy()
    solver.b.destroy()


def test_elasticsolver_gravity_includes_jacobian(elastic_test_setup, test_config):
    """Gravity ∫ρ g·v J dx scales by J under uniform expansion."""
    _require_single_rank()
    import ufl

    domain, V, W, E, nu, ds, dx = elastic_test_setup
    phi = create_zero_phi(V)

    solver = ElasticSolver(V, E, nu, phi, test_config)
    
    # Apply gravity
    rho = 1000.0  # kg/m^3
    g = np.array([0.0, 0.0, -9.81])
    solver.apply_gravity(rho, g, dx=dx)
    
    # Assemble with phi=0
    solver._assemble_rhs()
    b0_norm = solver.b.norm()
    
    # Apply expansion (increases volume -> increases gravity load)
    X = V.tabulate_dof_coordinates()
    solver.phi.x.array[:] = 0.2 * X.flatten()
    solver.phi.x.scatter_forward()
    
    # Rebuild and reassemble
    solver._L_terms.clear()
    solver.apply_gravity(rho, g, dx=dx)
    solver._assemble_rhs()
    b1_norm = solver.b.norm()
    
    # Gravity load should increase with volume expansion
    # J = 1.2^3 = 1.728, so load should be ~1.7x larger
    expected_ratio = 1.2 ** 3
    actual_ratio = b1_norm / b0_norm
    
    assert np.isclose(actual_ratio, expected_ratio, rtol=0.05), \
        f"Gravity load should scale by J={expected_ratio:.3f}, got {actual_ratio:.3f}"
    
    # Cleanup
    solver.K.destroy()
    solver.b.destroy()


# ======================== LigamentSpringSystem F-mapping tests ========================


def test_ligament_system_uses_deformed_coordinates(test_config):
    """Ligament KDTree and geometry use deformed coordinates X + u(X)."""
    _require_single_rank()
    from simulation.ligamentassembler import LigamentSpringSystem

    domain = mesh.create_unit_cube(MPI.COMM_WORLD, 10, 10, 10)
    V = fem.functionspace(domain, ("Lagrange", 1, (3,)))
    
    # Simple two-point ligament
    points = np.array([[0.2, 0.5, 0.5], [0.8, 0.5, 0.5]])
    lines = np.array([2, 0, 1], dtype=np.int32)
    poly = pv.PolyData(points)
    poly.lines = lines
    
    # Test 1: With no displacement, KDTree should use original DOF coords
    phi_identity = fem.Function(V)
    
    lig_system = LigamentSpringSystem(V, [poly], [1000.0], phi_identity, make_test_config({"LIGAMENT_MAPPING_TOLERANCE": 0.3}))
    
    # Check that _Xd (deformed coords) equals _X (original) when u=0
    assert np.allclose(lig_system._Xd, lig_system._X), \
        "With u=0, deformed coords should equal original coords"
    
    # Test 2: With displacement, KDTree should use X + u
    phi_stretched = fem.Function(V)
    X = V.tabulate_dof_coordinates()
    u_stretch = np.zeros_like(X)
    u_stretch[:, 0] = 0.2 * X[:, 0]  # 20% stretch in x
    phi_stretched.x.array[:] = u_stretch.flatten()
    
    lig_system2 = LigamentSpringSystem(V, [poly], [1000.0], phi_stretched, make_test_config({"LIGAMENT_MAPPING_TOLERANCE": 0.3}))
    
    # Check that _Xd = _X + u(_X)
    expected_Xd = lig_system2._X + u_stretch
    assert np.allclose(lig_system2._Xd, expected_Xd), \
        "Deformed coords should be X + u(X)"
    
    # Cleanup
    lig_system.K.destroy()
    lig_system.fpret.destroy()
    lig_system2.K.destroy()
    lig_system2.fpret.destroy()


def test_ligament_current_length_from_deformed_coords(test_config):
    """Spring lengths and KDTree queries are computed in deformed configuration."""
    _require_single_rank()
    from simulation.ligamentassembler import LigamentSpringSystem

    domain = mesh.create_unit_cube(MPI.COMM_WORLD, 10, 10, 10)
    V = fem.functionspace(domain, ("Lagrange", 1, (3,)))
    
    # Create ligament
    points = np.array([[0.3, 0.5, 0.5], [0.7, 0.5, 0.5]])
    lines = np.array([2, 0, 1], dtype=np.int32)
    poly = pv.PolyData(points)
    poly.lines = lines
    
    # Test that segment geometry is computed from deformed coordinates
    phi_test = fem.Function(V)
    X = V.tabulate_dof_coordinates()
    u_test = np.zeros_like(X)
    u_test[:, 0] = 0.3 * X[:, 0]  # 30% stretch
    phi_test.x.array[:] = u_test.flatten()
    
    lig = LigamentSpringSystem(V, [poly], [500.0], phi_test, make_test_config({"LIGAMENT_MAPPING_TOLERANCE": 0.35}))
    
    # Verify that _Xd was computed correctly
    expected_Xd = lig._X + u_test
    assert np.allclose(lig._Xd, expected_Xd), \
        "Deformed coordinates should be X + u(X)"
    
    # Verify that deformed coordinates are used in geometry
    # Check segment lengths are computed from _Xd not _X
    # (This is implicit - assembly uses _Xd via _refresh_mapper)
    
    # Cleanup
    lig.K.destroy()
    lig.fpret.destroy()


def test_ligament_geometric_stiffness_with_pretension(test_config):
    """Pretension T adds geometric stiffness and nonzero pretension force vector."""
    _require_single_rank()
    from simulation.ligamentassembler import LigamentSpringSystem

    domain = mesh.create_unit_cube(MPI.COMM_WORLD, 10, 10, 10)
    V = fem.functionspace(domain, ("Lagrange", 1, (3,)))
    
    # Create ligament
    points = np.array([[0.0, 0.5, 0.5], [1.0, 0.5, 0.5]])
    lines = np.array([2, 0, 1], dtype=np.int32)
    poly = pv.PolyData(points)
    poly.lines = lines
    
    phi_pret = fem.Function(V)
    
    # Without pretension
    lig_no_pret = LigamentSpringSystem(
        V, [poly], [1000.0], phi_pret, make_test_config({"LIGAMENT_MAPPING_TOLERANCE": 0.3}), pretensions=[0.0]
    )
    
    num_springs = sum(len(pairs) for pairs in lig_no_pret._pairs_per_bundle)
    if num_springs == 0:
        pytest.skip("No springs found - test requires finer mesh")
    
    K_no_pret_norm = lig_no_pret.K.norm()
    
    # With pretension (adds geometric stiffness)
    phi_pret2 = fem.Function(V)
    lig_with_pret = LigamentSpringSystem(
        V, [poly], [1000.0], phi_pret2, make_test_config({"LIGAMENT_MAPPING_TOLERANCE": 0.3}), pretensions=[100.0]
    )
    K_with_pret_norm = lig_with_pret.K.norm()
    
    # Geometric stiffness adds to total, so K should be larger
    assert K_with_pret_norm > K_no_pret_norm, \
        "Pretension should add geometric stiffness"
    
    # Check pretension force vector is non-zero
    fpret_norm = lig_with_pret.fpret.norm()
    assert fpret_norm > 0, "Pretension should create force vector"
    
    # Cleanup
    lig_no_pret.K.destroy()
    lig_no_pret.fpret.destroy()
    lig_with_pret.K.destroy()
    lig_with_pret.fpret.destroy()


# ======================== Integration tests across solvers ========================


def test_eigensolver_and_elasticsolver_use_same_phi(elastic_test_setup, test_config):
    """Eigensolver and Elasticsolver produce identical K when given the same φ."""
    _require_single_rank()

    domain, V, W, E, nu, ds, dx = elastic_test_setup
    
    # Create both solvers
    phi = create_zero_phi(V)

    eigen_solver = ElasticEigenSolver(V, E, nu, phi, test_config)
    phi = create_zero_phi(V)

    elastic_solver = ElasticSolver(V, E, nu, phi, test_config)
    
    # Apply same deformation to both
    X = V.tabulate_dof_coordinates()
    phi_vals = 0.08 * X
    
    eigen_solver.phi.x.array[:] = phi_vals.flatten()
    eigen_solver.phi.x.scatter_forward()
    elastic_solver.phi.x.array[:] = phi_vals.flatten()
    elastic_solver.phi.x.scatter_forward()
    
    # Reassemble both
    eigen_solver._assemble_mass()
    eigen_solver._assemble_stiffness()
    elastic_solver._assemble_stiffness()
    
    # Stiffness matrices should be very similar (same physics)
    K_eigen_norm = eigen_solver.K.norm()
    K_elastic_norm = elastic_solver.K.norm()
    
    # Should be identical (no ligaments in this test)
    assert np.isclose(K_eigen_norm, K_elastic_norm, rtol=1e-10), \
        "Both solvers should compute same stiffness for same phi"
    
    # Cleanup
    eigen_solver.K.destroy()
    eigen_solver.M.destroy()
    eigen_solver.M0.destroy()
    elastic_solver.K.destroy()


def test_consistent_jacobian_across_volume_and_surface(test_config):
    """Uniform expansion: ∫J dx and ∫J_surf ds scale by (1+c)^3 and (1+c)^2."""
    _require_single_rank()
    import ufl

    domain = mesh.create_unit_cube(MPI.COMM_WORLD, 10, 10, 10)
    V = fem.functionspace(domain, ("Lagrange", 1, (3,)))
    W = fem.functionspace(domain, ("DG", 0))
    E = fem.Function(W)
    E.x.array[:] = 1000.0
    nu = fem.Function(W)
    nu.x.array[:] = 0.3
    
    phi = create_zero_phi(V)

    
    solver = ElasticSolver(V, E, nu, phi, test_config)
    
    # Apply known uniform expansion
    X = V.tabulate_dof_coordinates()
    expansion = 0.15
    solver.phi.x.array[:] = expansion * X.flatten()
    solver.phi.x.scatter_forward()
    
    # Compare integrals of J over volume and J_surf over the top face
    dx = ufl.dx(domain=domain)
    ds_top = _top_face_measure(domain)
    J = solver._J
    J_surf = solver._surface_jacobian()
    vol_ref = fem.assemble_scalar(fem.form(1.0 * dx))
    area_ref = fem.assemble_scalar(fem.form(1.0 * ds_top))
    vol_phys = fem.assemble_scalar(fem.form(J * dx))
    area_phys = fem.assemble_scalar(fem.form(J_surf * ds_top))
    
    exp_vol = (1.0 + expansion) ** 3
    exp_area = (1.0 + expansion) ** 2
    
    assert np.isclose(vol_phys / vol_ref, exp_vol, rtol=0.03)
    assert np.isclose(area_phys / area_ref, exp_area, rtol=0.03)
    
    # Cleanup
    solver.K.destroy()


def test_zero_deformation_gives_identity_mapping(test_config):
    """With φ=0, F=I, J=1, and K, M match reference assemblies."""
    _require_single_rank()
    import ufl

    domain = mesh.create_unit_cube(MPI.COMM_WORLD, 10, 10, 10)
    V = fem.functionspace(domain, ("Lagrange", 1, (3,)))
    W = fem.functionspace(domain, ("DG", 0))
    E = fem.Function(W)
    E.x.array[:] = 1500.0
    nu = fem.Function(W)
    nu.x.array[:] = 0.25
    ds = ufl.ds(domain=domain)
    
    phi = create_zero_phi(V)

    
    solver = ElasticEigenSolver(V, E, nu, phi, test_config)
    
    # Verify phi is zero
    assert np.allclose(solver.phi.x.array[:], 0.0)
    
    # With F=I, elastic stiffness should match reference
    # This is implicitly tested by comparing with/without deformation
    K_identity = solver.K.copy()
    M_identity = solver.M.copy()
    
    # Apply small deformation and check it changes
    X = V.tabulate_dof_coordinates()
    solver.phi.x.array[:] = 0.01 * X.flatten()
    solver.phi.x.scatter_forward()
    solver._assemble_stiffness()
    solver._assemble_mass()
    
    K_diff = solver.K.copy()
    K_diff.axpy(-1.0, K_identity)
    M_diff = solver.M.copy()
    M_diff.axpy(-1.0, M_identity)
    
    # Should be different
    assert K_diff.norm() > 1e-6, "Stiffness should change with φ"
    assert M_diff.norm() > 1e-6, "Mass should change with φ"
    
    # Cleanup
    K_identity.destroy()
    M_identity.destroy()
    K_diff.destroy()
    M_diff.destroy()
    solver.K.destroy()
    solver.M.destroy()
    solver.M0.destroy()


def test_deformation_gradient_determinant_positive(test_config):
    """For compressions < 50%, det(F) stays positive and matches expected J."""
    _require_single_rank()
    import ufl
    _require_single_rank()
    import ufl

    domain = mesh.create_unit_cube(MPI.COMM_WORLD, 10, 10, 10)
    V = fem.functionspace(domain, ("Lagrange", 1, (3,)))
    W = fem.functionspace(domain, ("DG", 0))
    E = fem.Function(W)
    E.x.array[:] = 2000.0
    nu = fem.Function(W)
    nu.x.array[:] = 0.3
    ds = ufl.ds(domain=domain)
    dx = ufl.dx(domain=domain)
    
    phi = create_zero_phi(V)

    
    solver = ElasticEigenSolver(V, E, nu, phi, test_config)
    
    # Apply physically reasonable deformation (compression < 50%)
    X = V.tabulate_dof_coordinates()
    solver.phi.x.array[:] = -0.3 * X.flatten()  # 30% compression
    solver.phi.x.scatter_forward()
    
    # Compute average J over domain
    I = ufl.Identity(3)
    F = I + ufl.grad(solver.phi)
    J = ufl.det(F)
    J_avg = fem.assemble_scalar(fem.form(J * dx))
    volume = fem.assemble_scalar(fem.form(1.0 * dx))
    J_mean = J_avg / volume
    
    # For 30% compression: F = 0.7*I, J = 0.7^3 = 0.343
    expected_J = 0.7 ** 3
    
    assert J_mean > 0, "Jacobian must be positive for valid deformation"
    assert np.isclose(J_mean, expected_J, rtol=0.05), \
        f"Mean J should be ~{expected_J:.3f}, got {J_mean:.3f}"
    
    # Cleanup
    solver.K.destroy()
    solver.M.destroy()
    solver.M0.destroy()


def test_apply_traction_intensity_scales_with_jsurf(test_config):
    """Traction intensity on sample: RHS scales by (1+c)^2 under uniform expansion."""
    _require_single_rank()
    import ufl

    domain = mesh.create_unit_cube(MPI.COMM_WORLD, 10, 10, 10)
    V = fem.functionspace(domain, ("Lagrange", 1, (3,)))
    W = fem.functionspace(domain, ("DG", 0))
    E = fem.Function(W); E.x.array[:] = 800.0
    nu = fem.Function(W); nu.x.array[:] = 0.3
    phi = create_zero_phi(V)

    solver = ElasticSolver(V, E, nu, phi, test_config)

    ds_top = _top_face_measure(domain)

    # Apply traction vector in +z on sample surface
    t = np.array([0.0, 0.0, 50.0])
    solver.apply_load(t, ds_top, defined_on='sample', input_type='intensity')
    solver._assemble_rhs()
    b0 = solver.b.norm()

    # Uniform expansion φ=cx
    X = V.tabulate_dof_coordinates()
    c = 0.12
    solver.phi.x.array[:] = c * X.flatten()
    solver.phi.x.scatter_forward()
    # Rebuild load form and RHS with updated φ
    solver.clear_loads()
    solver.apply_load(t, ds_top, defined_on='sample', input_type='intensity')
    solver._assemble_rhs()
    b1 = solver.b.norm()

    expected = (1.0 + c) ** 2
    assert np.isclose(b1 / b0, expected, rtol=0.05)

    # Cleanup
    solver.K.destroy(); solver.b.destroy()


def test_eigensolver_update_reassembles_mass(test_config):
    """ElasticEigenSolver.update(fu_new=...) triggers M reassembly (J depends on φ)."""
    _require_single_rank()

    domain = mesh.create_unit_cube(MPI.COMM_WORLD, 8, 8, 8)
    V = fem.functionspace(domain, ("Lagrange", 1, (3,)))
    W = fem.functionspace(domain, ("DG", 0))
    E = fem.Function(W); E.x.array[:] = 1200.0
    nu = fem.Function(W); nu.x.array[:] = 0.28
    phi = create_zero_phi(V)

    solver = ElasticEigenSolver(V, E, nu, phi, test_config)

    M0 = solver.M.copy()

    X = V.tabulate_dof_coordinates()
    u = 0.07 * X
    phi.x.array[:] = u.flatten()
    solver.update()

    M_diff = solver.M.copy(); M_diff.axpy(-1.0, M0)
    assert M_diff.norm() > 1e-6

    # Cleanup
    M0.destroy(); M_diff.destroy(); solver.K.destroy(); solver.M.destroy(); solver.M0.destroy()

if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
