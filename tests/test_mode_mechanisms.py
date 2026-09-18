"""
Tests for mode mechanism kinematic metrics and beam-like decomposition.

Validates:
1. Invariance under u -> -u (sign reflection).
2. Subspace rotation invariance Q -> Q * O for degenerate/clustered pairs (e.g. modes 9-10).
3. Exact 100% Frobenius energy closure for Level A and Level B RegionalBeamModel.
4. Canonical deformation patterns (pure axial, bending, transverse shear, St. Venant twist).
5. Exact factor-of-2 shear tensor fraction definition.
6. Rejection/handling of invalid Jacobian weights (J <= 0).
"""

import numpy as np
import pytest

from analysis.mode_mechanism_metrics import (
    RegionalBeamModel,
    compute_level_a_profile,
    compute_subspace_level_a_profile,
    transform_strain_tensor,
)


def test_transform_strain_tensor():
    """Verify coordinate transformation R^T * eps * R."""
    # Diagonal strain
    eps = np.array([[[1.0, 0.0, 0.0],
                     [0.0, 2.0, 0.0],
                     [0.0, 0.0, 3.0]]])
    # 90 deg rotation around z: x -> y, y -> -x
    R = np.array([[0.0, -1.0, 0.0],
                  [1.0,  0.0, 0.0],
                  [0.0,  0.0, 1.0]])
    
    eps_rot = transform_strain_tensor(eps, R)
    assert eps_rot.shape == (1, 3, 3)
    # R^T eps R:
    # [[0, 1, 0], [-1, 0, 0], [0, 0, 1]] * diag(1,2,3) * [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    # = [[0, 2, 0], [-1, 0, 0], [0, 0, 3]] * [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    # = [[2, 0, 0], [0, 1, 0], [0, 0, 3]]
    expected = np.array([[[2.0, 0.0, 0.0],
                          [0.0, 1.0, 0.0],
                          [0.0, 0.0, 3.0]]])
    np.testing.assert_allclose(eps_rot, expected, atol=1e-12)


def test_level_a_fractions_closure_and_factor_of_two():
    """Verify sum of 6 components equals 1.0, and shears include the factor of 2."""
    n_cells = 100
    rng = np.random.default_rng(42)
    # Generate symmetric strain tensors
    A = rng.normal(size=(n_cells, 3, 3))
    eps = 0.5 * (A + np.swapaxes(A, 1, 2))
    volumes = rng.uniform(0.5, 2.0, size=n_cells)
    
    profile = compute_level_a_profile(eps, volumes)
    
    sum_comps = (
        profile["f_ML"] + profile["f_AP"] + profile["f_CC"] +
        profile["f_ML_AP"] + profile["f_ML_CC"] + profile["f_AP_CC"]
    )
    assert np.isclose(sum_comps, 1.0, atol=1e-14)
    assert np.isclose(profile["f_volumetric"] + profile["f_deviatoric"], 1.0, atol=1e-14)
    
    # Manual check of Frobenius norm
    manual_D = np.sum(volumes[:, None, None] * eps * eps)
    assert np.isclose(profile["D_total"], manual_D, atol=1e-12)
    
    # Check that shear fractions have factor of 2:
    # For a pure shear tensor in component 0,1:
    pure_shear = np.zeros((1, 3, 3))
    pure_shear[0, 0, 1] = pure_shear[0, 1, 0] = 0.5
    vol = np.array([1.0])
    ps_profile = compute_level_a_profile(pure_shear, vol)
    # eps : eps = 2 * (0.5)^2 = 0.5
    # 2 * eps_01^2 / (eps:eps) = 2 * 0.25 / 0.5 = 1.0
    assert np.isclose(ps_profile["f_ML_AP"], 1.0, atol=1e-14)
    assert np.isclose(ps_profile["f_ML"], 0.0, atol=1e-14)
    assert np.isclose(ps_profile["f_AP"], 0.0, atol=1e-14)


def test_invariance_under_sign_flip():
    """Verify profile is strictly identical under u -> -u (eps -> -eps)."""
    n_cells = 50
    rng = np.random.default_rng(123)
    A = rng.normal(size=(n_cells, 3, 3))
    eps = 0.5 * (A + np.swapaxes(A, 1, 2))
    volumes = rng.uniform(1.0, 3.0, size=n_cells)
    
    p1 = compute_level_a_profile(eps, volumes)
    p2 = compute_level_a_profile(-eps, volumes)
    
    for k in ["f_ML", "f_AP", "f_CC", "f_ML_AP", "f_ML_CC", "f_AP_CC", "f_volumetric", "f_deviatoric", "D_total"]:
        assert np.isclose(p1[k], p2[k], atol=1e-15)


def test_subspace_rotation_invariance():
    """Verify compute_subspace_level_a_profile is invariant under arbitrary orthogonal mixing Q -> Q * O."""
    n_cells = 100
    rng = np.random.default_rng(999)
    volumes = rng.uniform(0.8, 1.5, size=n_cells)
    
    # Generate two independent strain fields eps1, eps2
    A1 = rng.normal(size=(n_cells, 3, 3))
    A2 = rng.normal(size=(n_cells, 3, 3))
    eps1 = 0.5 * (A1 + np.swapaxes(A1, 1, 2))
    eps2 = 0.5 * (A2 + np.swapaxes(A2, 1, 2))
    
    subspace_profile_orig = compute_subspace_level_a_profile([eps1, eps2], volumes)
    
    # Apply a 2D rotation O(theta)
    for theta in [np.pi / 6, np.pi / 4, np.pi / 2, 2.345]:
        c, s = np.cos(theta), np.sin(theta)
        eps1_rot = c * eps1 - s * eps2
        eps2_rot = s * eps1 + c * eps2
        
        subspace_profile_rot = compute_subspace_level_a_profile([eps1_rot, eps2_rot], volumes)
        
        for k in ["f_ML", "f_AP", "f_CC", "f_ML_AP", "f_ML_CC", "f_AP_CC", "D_total"]:
            assert np.isclose(subspace_profile_orig[k], subspace_profile_rot[k], atol=1e-12)


def test_invalid_jacobian_handling():
    """Verify that negative or zero cell volumes are rejected by default."""
    eps = np.zeros((3, 3, 3))
    eps[:, 0, 0] = 1.0
    vols_with_inversion = np.array([1.0, -0.01, 1.0])

    with pytest.raises(ValueError, match="Invalid cell volumes"):
        compute_level_a_profile(eps, vols_with_inversion, allow_invalid_jacobians=False)

    # When explicitly allowed, inverted cells are filtered
    profile = compute_level_a_profile(eps, vols_with_inversion, allow_invalid_jacobians=True)
    assert profile["total_weight"] == 2.0


def test_regional_beam_model_pure_axial():
    """Verify pure axial strain decomposition."""
    # Synthetic bar discretized into cells
    ny, nz = 10, 10
    y_vals = np.linspace(-5.0, 5.0, ny)
    z_vals = np.linspace(-5.0, 5.0, nz)
    yy, zz = np.meshgrid(y_vals, z_vals, indexing="ij")
    yy = yy.ravel()
    zz = zz.ravel()
    n_cells = len(yy)
    coords = np.zeros((n_cells, 3))
    coords[:, 0] = 0.0  # x (axial)
    coords[:, 1] = yy   # y
    coords[:, 2] = zz   # z
    volumes = np.ones(n_cells)
    
    # Local axis is X (axis 0)
    # Pure axial strain: eps_xx = 0.02, all other components 0
    eps = np.zeros((n_cells, 3, 3))
    eps[:, 0, 0] = 0.02
    
    model = RegionalBeamModel("test_bar", centroids=coords, local_x_axis=np.array([1.0, 0.0, 0.0]))
    result = model.decompose(eps, volumes)
    
    assert np.isclose(result["f_axial"], 1.0, atol=1e-10)
    assert np.isclose(result["f_bending"], 0.0, atol=1e-10)
    assert np.isclose(result["f_shear"], 0.0, atol=1e-10)
    assert np.isclose(result["f_twist"], 0.0, atol=1e-10)
    assert np.isclose(result["f_residual"], 0.0, atol=1e-10)
    assert np.isclose(result["f_axial"] + result["f_bending"] + result["f_shear"] + result["f_twist"] + result["f_residual"], 1.0, atol=1e-12)


def test_regional_beam_model_pure_bending():
    """Verify pure bending strain decomposition."""
    ny, nz = 11, 11
    y_vals = np.linspace(-4.0, 4.0, ny)
    z_vals = np.linspace(-3.0, 3.0, nz)
    yy, zz = np.meshgrid(y_vals, z_vals, indexing="ij")
    yy = yy.ravel()
    zz = zz.ravel()
    n_cells = len(yy)
    coords = np.zeros((n_cells, 3))
    coords[:, 1] = yy
    coords[:, 2] = zz
    volumes = np.ones(n_cells)
    
    # Linear bending: eps_xx(y, z) = b_y * y + b_z * z
    eps = np.zeros((n_cells, 3, 3))
    eps[:, 0, 0] = 0.005 * yy - 0.002 * zz
    
    model = RegionalBeamModel("test_beam", centroids=coords, local_x_axis=np.array([1.0, 0.0, 0.0]))
    result = model.decompose(eps, volumes)
    
    assert np.isclose(result["f_axial"], 0.0, atol=1e-10)
    assert np.isclose(result["f_bending"], 1.0, atol=1e-10)
    assert np.isclose(result["f_shear"], 0.0, atol=1e-10)
    assert np.isclose(result["f_twist"], 0.0, atol=1e-10)
    assert np.isclose(result["f_residual"], 0.0, atol=1e-10)
    assert np.isclose(result["f_axial"] + result["f_bending"] + result["f_shear"] + result["f_twist"] + result["f_residual"], 1.0, atol=1e-12)


def test_regional_beam_model_pure_transverse_shear():
    """Verify pure uniform transverse shear decomposition."""
    ny, nz = 9, 9
    y_vals = np.linspace(-2.0, 2.0, ny)
    z_vals = np.linspace(-2.0, 2.0, nz)
    yy, zz = np.meshgrid(y_vals, z_vals, indexing="ij")
    yy = yy.ravel()
    zz = zz.ravel()
    n_cells = len(yy)
    coords = np.zeros((n_cells, 3))
    coords[:, 1] = yy
    coords[:, 2] = zz
    volumes = np.ones(n_cells)
    
    # eps_xy = 0.01, eps_xz = 0.03
    eps = np.zeros((n_cells, 3, 3))
    eps[:, 0, 1] = eps[:, 1, 0] = 0.01
    eps[:, 0, 2] = eps[:, 2, 0] = 0.03
    
    model = RegionalBeamModel("test_shear", centroids=coords, local_x_axis=np.array([1.0, 0.0, 0.0]))
    result = model.decompose(eps, volumes)
    
    assert np.isclose(result["f_axial"], 0.0, atol=1e-10)
    assert np.isclose(result["f_bending"], 0.0, atol=1e-10)
    assert np.isclose(result["f_shear"], 1.0, atol=1e-10)
    assert np.isclose(result["f_twist"], 0.0, atol=1e-10)
    assert np.isclose(result["f_residual"], 0.0, atol=1e-10)
    assert np.isclose(result["f_axial"] + result["f_bending"] + result["f_shear"] + result["f_twist"] + result["f_residual"], 1.0, atol=1e-12)


def test_regional_beam_model_pure_twist():
    """Verify St. Venant circulatory twist decomposition."""
    ny, nz = 15, 15
    y_vals = np.linspace(-3.0, 3.0, ny)
    z_vals = np.linspace(-3.0, 3.0, nz)
    yy, zz = np.meshgrid(y_vals, z_vals, indexing="ij")
    yy = yy.ravel()
    zz = zz.ravel()
    n_cells = len(yy)
    coords = np.zeros((n_cells, 3))
    coords[:, 1] = yy
    coords[:, 2] = zz
    volumes = np.ones(n_cells)
    
    # Pure twist: eps_xy = -1/2 * t * z, eps_xz = 1/2 * t * y
    t_rate = 0.004
    eps = np.zeros((n_cells, 3, 3))
    eps[:, 0, 1] = eps[:, 1, 0] = -0.5 * t_rate * zz
    eps[:, 0, 2] = eps[:, 2, 0] = 0.5 * t_rate * yy
    
    model = RegionalBeamModel("test_twist", centroids=coords, local_x_axis=np.array([1.0, 0.0, 0.0]))
    result = model.decompose(eps, volumes)
    
    assert np.isclose(result["f_axial"], 0.0, atol=1e-10)
    assert np.isclose(result["f_bending"], 0.0, atol=1e-10)
    assert np.isclose(result["f_shear"], 0.0, atol=1e-10)
    assert np.isclose(result["f_twist"], 1.0, atol=1e-10)
    assert np.isclose(result["f_residual"], 0.0, atol=1e-10)
    assert np.isclose(result["f_axial"] + result["f_bending"] + result["f_shear"] + result["f_twist"] + result["f_residual"], 1.0, atol=1e-12)


def test_regional_beam_model_mixed_orthogonality():
    """Verify that a random mixed strain field sums to 1.0 and is invariant under u -> -u."""
    ny, nz = 7, 7
    y_vals = np.linspace(-2.0, 2.0, ny)
    z_vals = np.linspace(-2.0, 2.0, nz)
    yy, zz = np.meshgrid(y_vals, z_vals, indexing="ij")
    yy = yy.ravel()
    zz = zz.ravel()
    n_cells = len(yy)
    coords = np.zeros((n_cells, 3))
    coords[:, 1] = yy
    coords[:, 2] = zz
    rng = np.random.default_rng(777)
    volumes = rng.uniform(0.5, 1.5, size=n_cells)
    
    # Random full 3x3 symmetric strain
    A = rng.normal(scale=0.01, size=(n_cells, 3, 3))
    eps = 0.5 * (A + np.swapaxes(A, 1, 2))
    
    # Random local axis
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    
    model = RegionalBeamModel("test_mixed", centroids=coords, local_x_axis=axis)
    res_pos = model.decompose(eps, volumes)
    res_neg = model.decompose(-eps, volumes)
    
    # Check closure
    sum_pos = res_pos["f_axial"] + res_pos["f_bending"] + res_pos["f_shear"] + res_pos["f_twist"] + res_pos["f_residual"]
    assert np.isclose(sum_pos, 1.0, atol=1e-12)
    
    # Check invariance under sign flip
    assert np.isclose(res_pos["f_axial"], res_neg["f_axial"], atol=1e-14)
    assert np.isclose(res_pos["f_bending"], res_neg["f_bending"], atol=1e-14)
    assert np.isclose(res_pos["f_shear"], res_neg["f_shear"], atol=1e-14)
    assert np.isclose(res_pos["f_twist"], res_neg["f_twist"], atol=1e-14)
    assert np.isclose(res_pos["f_residual"], res_neg["f_residual"], atol=1e-14)


def test_ligament_kinematics_orthogonality_and_rigid_invariance():
    """Verify ligament kinematics: axial/transverse orthogonality and zero strain under rigid motion."""
    # Two nodes forming a ligament spring
    p0 = np.array([10.0, 20.0, 30.0])
    p1 = np.array([15.0, 25.0, 40.0])
    r = p1 - p0
    L = np.linalg.norm(r)
    e_dir = r / L

    # 1. Rigid translation: u0 = u1 = [5, -3, 2]
    u_rigid = np.array([5.0, -3.0, 2.0])
    delta_u_rigid = u_rigid - u_rigid
    delta_L_rigid = np.dot(delta_u_rigid, e_dir)
    assert np.isclose(delta_L_rigid, 0.0, atol=1e-15)

    # 2. Arbitrary displacement field
    delta_u = np.array([1.2, -0.7, 2.4])
    delta_L = np.dot(delta_u, e_dir)
    eps_axial = delta_L / L

    delta_u_trans = delta_u - delta_L * e_dir
    # Orthogonality: delta_u_trans . e_dir == 0
    assert np.isclose(np.dot(delta_u_trans, e_dir), 0.0, atol=1e-15)

    # Pythagorean theorem: ||delta_u||^2 == delta_L^2 + ||delta_u_trans||^2
    assert np.isclose(np.linalg.norm(delta_u)**2, delta_L**2 + np.linalg.norm(delta_u_trans)**2, atol=1e-14)

