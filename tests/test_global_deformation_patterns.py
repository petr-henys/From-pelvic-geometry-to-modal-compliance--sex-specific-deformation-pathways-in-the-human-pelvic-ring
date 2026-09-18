import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pytest
from analysis.global_deformation_patterns import GlobalDeformationModel, PATTERNS


def test_rigid_motion_is_removed_completely():
    rng = np.random.default_rng(12)
    x = rng.normal(size=(300, 3))
    w = rng.uniform(0.5, 2.0, len(x))
    model = GlobalDeformationModel(x, w)

    # Pure translation and rotation
    rigid = np.array([2.0, 3.0, 1.0]) + np.cross([0.3, 0.5, 0.6], x)
    res_rigid = model.fit(rigid)
    assert not res_rigid['is_valid']

    # Arbitrary nonrigid field
    u = np.column_stack([x[:, 0] * x[:, 1], -0.5 * x[:, 0]**2, np.zeros(len(x))])
    p = model.fit(u)
    q = model.fit(-2.5 * u + rigid)
    assert p['is_valid'] and q['is_valid']
    for key in [*['f_' + k for k in PATTERNS], 'f_residual']:
        np.testing.assert_allclose(p[key], q[key], atol=1e-10)


def test_pure_synthetic_deformations_on_symmetric_domain():
    """On a symmetric domain, pure canonical deformation families are mutually orthogonal."""
    x_1d = np.linspace(-1, 1, 15)
    X, Y, Z = np.meshgrid(x_1d, x_1d, x_1d, indexing='ij')
    points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    weights = np.ones(len(points))
    model = GlobalDeformationModel(points, weights)

    # 1. Pure axial stretch
    u_axial = np.column_stack([points[:, 0], np.zeros(len(points)), np.zeros(len(points))])
    res_ax = model.fit(u_axial)
    assert res_ax['is_valid']
    np.testing.assert_allclose(res_ax['f_axial'], 1.0, atol=1e-10)
    np.testing.assert_allclose(res_ax['f_residual'], 0.0, atol=1e-10)

    # 2. Pure bending
    u_bend = np.column_stack([-points[:, 0]*points[:, 1], 0.5*points[:, 0]**2, np.zeros(len(points))])
    res_bend = model.fit(u_bend)
    assert res_bend['is_valid']
    np.testing.assert_allclose(res_bend['f_residual'], 0.0, atol=1e-10)
    assert res_bend['f_bending'] > 0.75
    np.testing.assert_allclose(res_bend['f_axial'], 0.0, atol=1e-10)
    np.testing.assert_allclose(res_bend['f_torsion'], 0.0, atol=1e-10)

    # 3. Pure shear
    u_shear = np.column_stack([points[:, 1], points[:, 0], np.zeros(len(points))])
    res_shear = model.fit(u_shear)
    assert res_shear['is_valid']
    np.testing.assert_allclose(res_shear['f_shear'], 1.0, atol=1e-10)
    np.testing.assert_allclose(res_shear['f_residual'], 0.0, atol=1e-10)

    # 4. Pure torsion
    u_torsion = np.column_stack([-points[:, 0]*points[:, 2], np.zeros(len(points)), points[:, 0]*points[:, 0]*0])
    # Torsion around Y: u_x = y*z, u_z = -x*y
    u_torsion = np.column_stack([points[:, 1]*points[:, 2], np.zeros(len(points)), -points[:, 0]*points[:, 1]])
    res_tor = model.fit(u_torsion)
    assert res_tor['is_valid']
    np.testing.assert_allclose(res_tor['f_torsion'], 1.0, atol=1e-10)
    np.testing.assert_allclose(res_tor['f_residual'], 0.0, atol=1e-10)


def test_synthetic_deformations_identified_backwards_on_irregular_mesh():
    """On irregular geometries, pure fields have zero residual and dominant Shapley weight."""
    rng = np.random.default_rng(99)
    x = rng.normal(size=(400, 3))
    w = rng.uniform(0.2, 3.0, len(x))
    model = GlobalDeformationModel(x, w)
    c = model.center

    # Pure axial
    u_ax = np.column_stack([x[:, 0] - c[0], 0.5*(x[:, 1] - c[1]), -(x[:, 2] - c[2])])
    res_ax = model.fit(u_ax)
    assert res_ax['f_residual'] < 1e-10
    assert res_ax['f_axial'] > 0.85

    # Pure bending
    u_b = np.column_stack([-(x[:, 0]-c[0])*(x[:, 1]-c[1]), 0.5*(x[:, 0]-c[0])**2, np.zeros(len(x))])
    res_b = model.fit(u_b)
    assert res_b['f_residual'] < 1e-10
    assert res_b['f_bending'] > 0.70

    # Pure shear
    u_sh = np.column_stack([x[:, 1]-c[1], x[:, 0]-c[0], np.zeros(len(x))])
    res_sh = model.fit(u_sh)
    assert res_sh['f_residual'] < 1e-10
    assert res_sh['f_shear'] > 0.80

    # Pure torsion
    u_tor = np.column_stack([-(x[:, 1]-c[1])*(x[:, 2]-c[2]), (x[:, 0]-c[0])*(x[:, 2]-c[2]), np.zeros(len(x))])
    res_tor = model.fit(u_tor)
    assert res_tor['f_residual'] < 1e-10
    assert res_tor['f_torsion'] > 0.90


def test_shapley_closure_and_scale_invariance():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(350, 3))
    u = rng.normal(size=x.shape)
    w = rng.uniform(0.1, 2.0, len(x))
    model = GlobalDeformationModel(x, w)

    fit_orig = model.fit(u)
    fit_scaled = model.fit(-3.7 * u)

    # 100% exact closure
    total_share = sum(fit_orig['f_' + k] for k in (*PATTERNS, 'residual'))
    np.testing.assert_allclose(total_share, 1.0, atol=1e-10)

    # Non-negativity
    for k in (*PATTERNS, 'residual'):
        assert fit_orig['f_' + k] >= -1e-12
        np.testing.assert_allclose(fit_orig['f_' + k], fit_scaled['f_' + k], atol=1e-10)
