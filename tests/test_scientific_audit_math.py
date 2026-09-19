"""Rigorous mathematical tests for scientific audit verifications.

Verifies:
1. Polynomial completeness: 30D quadratic vector space, 6D rigid Lie algebra, 24D nonrigid quotient space.
2. Cross-gradient shear field w = (yz, xz, xy) is irrotational and divergence-free.
3. Intrinsic algebraic overlap between bending and shear on symmetric cube [-1, 1]^3 is -22/45 != 0.
4. Basix degree-4 tetrahedral quadrature rule integrates degree-4 polynomials exactly with positive weights.
5. Basis invariance of single_functional_coupling under arbitrary orthogonal rotations Q -> Q R.
6. Basis invariance of Grassmann distance d_G(Q1 R1, Q2 R2) = d_G(Q1, Q2).
7. Analytical avoided crossing / near-degeneracy formula in 2DOF model.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import basix

from analysis.global_deformation_patterns import GlobalDeformationModel, PATTERNS
from analysis.spectral_metrics import (
    single_functional_coupling,
    canonical_pair_couplings,
    coupling_per_1mm_max,
    orthonormalize_l2,
    compute_principal_angles,
    grassmann_distance,
)


def test_polynomial_space_dimensions_and_completeness():
    """Verify 30D quadratic polynomial space, 6D rigid, 24D nonrigid quotient."""
    rng = np.random.default_rng(42)
    # Random 3D point cloud of 500 points (to have full rank)
    points = rng.uniform(-1.0, 1.0, size=(500, 3))
    weights = np.ones(len(points))

    model = GlobalDeformationModel(points, weights)

    # Check that rigid body basis has rank 6
    assert model.rigid_q.shape[1] == 6

    # Check that each family has the expected rank
    ranks = [len(ids) for ids in model.group_col_ids]
    # axial: 6, bending: 6, shear: 10, torsion: 2
    assert ranks == [6, 6, 10, 2]
    assert sum(ranks) == 24
    assert model.gram.shape == (24, 24)

    # Check Gram condition number is finite and well-conditioned
    assert np.isfinite(model.condition)
    assert model.condition < 100.0


def test_irrotational_triaxial_shear_properties():
    """Verify that w = (yz, xz, xy) = grad(xyz) is curl-free and divergence-free."""
    # Symbolic / numerical gradient verification
    # For u(x,y,z) = (y*z, x*z, x*y):
    # div(u) = d(yz)/dx + d(xz)/dy + d(xy)/dz = 0 + 0 + 0 = 0
    # curl(u)_x = d(xy)/dy - d(xz)/dz = x - x = 0
    # curl(u)_y = d(yz)/dz - d(xy)/dx = y - y = 0
    # curl(u)_z = d(xz)/dx - d(yz)/dy = z - z = 0
    # Cauchy strain tensor:
    # eps_xx = 0, eps_yy = 0, eps_zz = 0 (zero normal strain!)
    # eps_xy = (z + z)/2 = z
    # eps_yz = (x + x)/2 = x
    # eps_zx = (y + y)/2 = y
    # Hence w is a pure shear field whose shear components vary linearly along the transverse axes.
    rng = np.random.default_rng(123)
    coords = rng.uniform(-2, 2, size=(50, 3))
    x, y, z = coords.T

    # Numerical finite differences for curl and div
    h = 1e-6
    def u_w(c):
        return np.column_stack([c[:, 1] * c[:, 2], c[:, 0] * c[:, 2], c[:, 0] * c[:, 1]])

    du_dx = (u_w(coords + [h, 0, 0]) - u_w(coords - [h, 0, 0])) / (2 * h)
    du_dy = (u_w(coords + [0, h, 0]) - u_w(coords - [0, h, 0])) / (2 * h)
    du_dz = (u_w(coords + [0, 0, h]) - u_w(coords - [0, 0, h])) / (2 * h)

    div = du_dx[:, 0] + du_dy[:, 1] + du_dz[:, 2]
    curl_x = du_dy[:, 2] - du_dz[:, 1]
    curl_y = du_dz[:, 0] - du_dx[:, 2]
    curl_z = du_dx[:, 1] - du_dy[:, 0]

    np.testing.assert_allclose(div, 0.0, atol=1e-8)
    np.testing.assert_allclose(curl_x, 0.0, atol=1e-8)
    np.testing.assert_allclose(curl_y, 0.0, atol=1e-8)
    np.testing.assert_allclose(curl_z, 0.0, atol=1e-8)


def test_intrinsic_bending_shear_overlap_on_symmetric_cube():
    """Verify analytical inner product <bending, shear> = -22/45 on [-1, 1]^3."""
    # Let Omega = [-1, 1]^3, volume V = 8.
    # Radius of gyration r_g:
    # int_{-1}^1 int_{-1}^1 int_{-1}^1 (x^2 + y^2 + z^2) dx dy dz = 3 * (2/3 * 2 * 2) = 8.
    # r_g^2 = (1/V) * 8 = 1 => r_g = 1.
    # Scaled coordinates: \tilde x = x.
    # Bending mode b = (-x*y, 0.5*x^2, 0).
    # Shear mode s = (x*y, 0.5*x^2, 0).
    # Nonrigid projection:
    # Rigid translations and rotations are odd or even; b and s have zero mean (orthogonal to translations).
    # Rotations:
    # r_x = (0, -z, y), r_y = (z, 0, -x), r_z = (-y, x, 0).
    # <b, r_z> = int (x*y^2 + 0.5*x^3) = 0 by symmetry.
    # Thus b and s are already orthogonal to all 6 rigid modes on [-1, 1]^3!
    # Norm of b:
    # |b|^2 = x^2 y^2 + 0.25 x^4.
    # int_{-1}^1 x^2 dx = 2/3, int y^2 dy = 2/3, int dz = 2 => int x^2 y^2 = 8/9.
    # int_{-1}^1 0.25 x^4 dx dy dz = 0.25 * (2/5) * 2 * 2 = 2/5.
    # int_Omega |b|^2 = 8/9 + 2/5 = 58/45.
    # Normalized by V=8: ||b||_V^2 = (58/45)/8 = 29/180.
    # Norm of s:
    # |s|^2 = x^2 y^2 + 0.25 x^4 => identical: ||s||_V^2 = 29/180.
    # Inner product <b, s>:
    # b . s = (-x*y)(x*y) + (0.5 x^2)(0.5 x^2) + 0 = -x^2 y^2 + 0.25 x^4.
    # int_Omega (b . s) = -8/9 + 2/5 = -22/45.
    # In normalized V: <b, s>_V = (-22/45)/8 = -11/180.
    # Cosine of angle: (-22/45) / (58/45) = -22/58 = -11/29 approx -0.3793.
    # This is non-zero, proving that bending and shear overlap intrinsically even on a symmetric cube.

    # Exact numerical verification via Gauss-Legendre quadrature:
    pts, wts = np.polynomial.legendre.leggauss(5)
    grid = np.meshgrid(pts, pts, pts, indexing='ij')
    w_grid = np.meshgrid(wts, wts, wts, indexing='ij')
    x, y, z = [g.ravel() for g in grid]
    w = (w_grid[0] * w_grid[1] * w_grid[2]).ravel()

    b = np.column_stack([-x * y, 0.5 * x**2, np.zeros_like(x)])
    s = np.column_stack([x * y, 0.5 * x**2, np.zeros_like(x)])

    integral_bs = np.sum(np.sum(b * s, axis=1) * w)
    expected_integral = -22.0 / 45.0  # -0.488888...
    np.testing.assert_allclose(integral_bs, expected_integral, atol=1e-14)


def test_degree4_tetrahedral_quadrature_exactness():
    """Verify Basix degree-4 rule has positive weights and integrates degree-4 exactly."""
    pts, wts = basix.make_quadrature(basix.CellType.tetrahedron, 4)

    # 1. Check positive weights
    assert len(wts) == 14
    assert np.all(wts > 0.0)
    # Unit tetrahedron volume: 1/6
    np.testing.assert_allclose(np.sum(wts), 1.0 / 6.0, atol=1e-15)

    # 2. Test exact integration of monomial test cases up to degree 4 on reference tet
    # Reference tet: x >= 0, y >= 0, z >= 0, x+y+z <= 1.
    # Analytical integral of x^p y^q z^r is p! q! r! / (p+q+r+3)!
    for p in range(5):
        for q in range(5 - p):
            for r in range(5 - p - q):
                deg = p + q + r
                if deg > 4:
                    continue
                import math
                exact = (math.factorial(p) * math.factorial(q) * math.factorial(r) /
                         math.factorial(p + q + r + 3))
                approx = np.sum(wts * (pts[:, 0]**p * pts[:, 1]**q * pts[:, 2]**r))
                np.testing.assert_allclose(approx, exact, atol=1e-14, err_msg=f"Failed on x^{p} y^{q} z^{r}")


def test_single_functional_coupling_basis_invariance():
    """Verify that single_functional_coupling is strictly invariant under Q -> Q R."""
    rng = np.random.default_rng(2026)
    n_nodes = 100
    n_dofs = n_nodes * 3
    k = 2  # 2D subspace (e.g. modes 9-10)

    # Generate random orthonormal Q
    A = rng.normal(size=(n_dofs, k))
    Q, _ = np.linalg.qr(A)

    # Generate random measurement covector b
    b = rng.normal(size=n_dofs)

    # Baseline coupling
    kappa_orig, psi_orig, sigma_orig = single_functional_coupling(Q, b)

    # Random orthogonal rotation in O(2)
    angle = rng.uniform(0, 2 * np.pi)
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle),  np.cos(angle)]])

    Q_rot = Q @ R

    # Rotated coupling
    kappa_rot, psi_rot, sigma_rot = single_functional_coupling(Q_rot, b)

    # Assert invariant to machine precision
    np.testing.assert_allclose(kappa_rot, kappa_orig, atol=1e-13)
    np.testing.assert_allclose(sigma_rot, sigma_orig, atol=1e-13)
    np.testing.assert_allclose(psi_rot, psi_orig, atol=1e-13)


def test_grassmann_distance_basis_invariance():
    """Verify that Grassmann distance d_G(Q1 R1, Q2 R2) = d_G(Q1, Q2)."""
    rng = np.random.default_rng(2027)
    n_dofs = 90
    k = 2

    Q1, _ = np.linalg.qr(rng.normal(size=(n_dofs, k)))
    Q2, _ = np.linalg.qr(rng.normal(size=(n_dofs, k)))

    theta_orig = compute_principal_angles(Q1, Q2)
    dG_orig = grassmann_distance(theta_orig)

    # Random orthogonal rotations
    R1 = Rotation.from_euler('z', 45, degrees=True).as_matrix()[:2, :2]
    R2 = Rotation.from_euler('z', -73, degrees=True).as_matrix()[:2, :2]

    Q1_rot = Q1 @ R1
    Q2_rot = Q2 @ R2

    theta_rot = compute_principal_angles(Q1_rot, Q2_rot)
    dG_rot = grassmann_distance(theta_rot)

    np.testing.assert_allclose(dG_rot, dG_orig, atol=1e-13)


def test_analytical_2dof_avoided_crossing():
    """Verify avoided crossing eigenvalues lambda_pm = lambda_0 pm sqrt(alpha^2 mu^2 + delta^2)."""
    lambda_0 = 10.0
    alpha = 2.5
    delta = 0.4

    mu_vals = np.linspace(-2.0, 2.0, 50)
    for mu in mu_vals:
        H = np.array([[lambda_0 + alpha * mu, delta],
                      [delta, lambda_0 - alpha * mu]])
        evals = np.linalg.eigvalsh(H)

        expected_lo = lambda_0 - np.sqrt((alpha * mu)**2 + delta**2)
        expected_hi = lambda_0 + np.sqrt((alpha * mu)**2 + delta**2)

        np.testing.assert_allclose(evals, [expected_lo, expected_hi], atol=1e-14)

    # Minimal gap at mu = 0 is exactly 2 * delta
    H0 = np.array([[lambda_0, delta], [delta, lambda_0]])
    evals0 = np.linalg.eigvalsh(H0)
    gap0 = evals0[1] - evals0[0]
    np.testing.assert_allclose(gap0, 2 * abs(delta), atol=1e-14)


def test_single_functional_extremum_identity():
    """Verify kappa_ell(Q) = max_{||alpha||=1} b_ell^T Q alpha = ||Q^T b_ell||_2."""
    rng = np.random.default_rng(42)
    n_dof = 120
    k = 4
    A = rng.normal(size=(n_dof, k))
    Q, _ = np.linalg.qr(A)
    b = rng.normal(size=n_dof)

    # Analytical capacity and optimal field
    c = Q.T @ b
    kappa_exact = np.linalg.norm(c)
    alpha_star = c / kappa_exact
    psi_star = Q @ alpha_star

    # Extremum identity: b^T psi_star == kappa_exact
    assert np.isclose(b @ psi_star, kappa_exact, rtol=1e-14)

    # Numerical verification against random sphere samples
    rand_alphas = rng.normal(size=(k, 10000))
    rand_alphas /= np.linalg.norm(rand_alphas, axis=0, keepdims=True)
    responses = b @ (Q @ rand_alphas)
    assert np.all(responses <= kappa_exact + 1e-14)
    assert np.isclose(np.max(responses), kappa_exact, rtol=1e-2)

    # Any small perturbation from alpha_star on the sphere strictly decreases the response
    for _ in range(50):
        pert = rng.normal(size=k) * 0.1
        pert_alpha = alpha_star + pert
        pert_alpha /= np.linalg.norm(pert_alpha)
        assert b @ (Q @ pert_alpha) < kappa_exact


def test_joint_svd_mathematical_identities():
    """Verify B = [b1, b2], C = Q^T B = U Sigma V^T, (B V[:, k])^T (Q U[:, k]) = sigma_k."""
    rng = np.random.default_rng(101)
    n_dof = 150
    k = 3
    A = rng.normal(size=(n_dof, k))
    Q, _ = np.linalg.qr(A)
    b1 = rng.normal(size=n_dof)
    b2 = rng.normal(size=n_dof)
    B = np.column_stack([b1, b2])

    C = Q.T @ B
    U, sigmas, Vt = np.linalg.svd(C, full_matrices=False)
    V = Vt.T

    # Modes in displacement space and matching functionals in covector space
    psi1 = Q @ U[:, 0]
    psi2 = Q @ U[:, 1]
    b_tilde1 = B @ V[:, 0]
    b_tilde2 = B @ V[:, 1]

    # Exact singular values
    np.testing.assert_allclose(b_tilde1 @ psi1, sigmas[0], atol=1e-13)
    np.testing.assert_allclose(b_tilde2 @ psi2, sigmas[1], atol=1e-13)

    # Cross orthogonality
    np.testing.assert_allclose(b_tilde1 @ psi2, 0.0, atol=1e-13)
    np.testing.assert_allclose(b_tilde2 @ psi1, 0.0, atol=1e-13)
    np.testing.assert_allclose(psi1 @ psi2, 0.0, atol=1e-13)


def test_linear_elasticity_incremental_load_scaling():
    """Verify that incremental static solve scales linearly with external load."""
    rng = np.random.default_rng(77)
    n = 30
    # Symmetric positive definite stiffness matrix
    A = rng.normal(size=(n, n))
    K = A.T @ A + 2.0 * np.eye(n)

    f_ext = rng.normal(size=n) * 400.0
    f_pret = rng.normal(size=n) * 20.0

    # Total and incremental solves
    u_pret = np.linalg.solve(K, f_pret)
    delta_u = np.linalg.solve(K, f_ext)
    u_total = np.linalg.solve(K, f_ext + f_pret)

    np.testing.assert_allclose(u_total, u_pret + delta_u, atol=1e-12)

    # Scaling factor c
    c = 2.5
    u_total_c = np.linalg.solve(K, c * f_ext + f_pret)
    delta_u_c = u_total_c - u_pret

    # Incremental response scales exactly linearly: delta_u(c) == c * delta_u
    np.testing.assert_allclose(delta_u_c, c * delta_u, atol=1e-12)


def test_exact_max_nodal_capacity_properties():
    """Verify mathematical properties of exact max-nodal anatomical capacity."""
    from analysis.spectral_metrics import single_functional_capacity_max_nodal, single_functional_coupling

    rng = np.random.default_rng(2028)
    n_nodes = 50
    n_dofs = n_nodes * 3

    # Test r=1
    A1 = rng.normal(size=(n_dofs, 1))
    Q1, _ = np.linalg.qr(A1)
    b1 = rng.normal(size=n_dofs)
    cap1, psi1, _ = single_functional_capacity_max_nodal(Q1, b1)
    max_u1 = np.sqrt(np.sum(psi1.reshape(-1, 3) ** 2, axis=1)).max()
    np.testing.assert_allclose(max_u1, 1.0, atol=1e-12)

    # Test r=2
    A2 = rng.normal(size=(n_dofs, 2))
    Q2, _ = np.linalg.qr(A2)
    b2 = rng.normal(size=n_dofs)
    cap2, psi2, sigma2 = single_functional_capacity_max_nodal(Q2, b2)
    cap2_old, _, _ = single_functional_coupling(Q2, b2, exact_max_nodal=False)

    # Constraint satisfaction
    max_u2 = np.sqrt(np.sum(psi2.reshape(-1, 3) ** 2, axis=1)).max()
    np.testing.assert_allclose(max_u2, 1.0, atol=1e-6)

    # True max-nodal must be >= old post-hoc sphere normalized value
    assert cap2 >= cap2_old - 1e-12

    # Rotation invariance under O(2)
    angle = rng.uniform(0, 2 * np.pi)
    R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    cap2_rot, _, _ = single_functional_capacity_max_nodal(Q2 @ R, b2)
    np.testing.assert_allclose(cap2_rot, cap2, atol=1e-5)

    # Sign invariance under b -> -b
    cap2_neg, _, _ = single_functional_capacity_max_nodal(Q2, -b2)
    np.testing.assert_allclose(cap2_neg, cap2, atol=1e-12)

    # Test r=4
    A4 = rng.normal(size=(n_dofs, 4))
    Q4, _ = np.linalg.qr(A4)
    b4 = rng.normal(size=n_dofs)
    cap4, psi4, _ = single_functional_capacity_max_nodal(Q4, b4)
    cap4_old, _, _ = single_functional_coupling(Q4, b4, exact_max_nodal=False)
    max_u4 = np.sqrt(np.sum(psi4.reshape(-1, 3) ** 2, axis=1)).max()
    np.testing.assert_allclose(max_u4, 1.0, atol=1e-6)
    assert cap4 >= cap4_old - 1e-6


def test_incremental_load_routing_scale_invariance():
    """Verify that incremental modal energy fractions are strictly scale-invariant."""
    rng = np.random.default_rng(2029)
    n = 20
    A = rng.normal(size=(n, n))
    K = A.T @ A + 3.0 * np.eye(n)
    M = np.eye(n)

    # Solve generalized eigenproblem K phi = lambda M phi
    evals, evecs = np.linalg.eigh(K)

    f_ext = rng.normal(size=n) * 400.0
    u_inc_1 = np.linalg.solve(K, f_ext)

    # Modal coordinates q_i = phi_i^T M u_inc
    q_1 = evecs.T @ M @ u_inc_1
    E_1 = 0.5 * evals * (q_1 ** 2)
    fracs_1 = E_1 / np.sum(E_1)

    # Scaled load: c * f_ext
    c = 3.7
    u_inc_c = np.linalg.solve(K, c * f_ext)
    q_c = evecs.T @ M @ u_inc_c
    E_c = 0.5 * evals * (q_c ** 2)
    fracs_c = E_c / np.sum(E_c)

    # Fractions must be identical to machine precision
    np.testing.assert_allclose(fracs_c, fracs_1, atol=1e-14)

