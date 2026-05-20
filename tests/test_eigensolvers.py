#!/usr/bin/env python3
"""Eigenvalue solver tests: convergence, sensitivity, and accuracy."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from mpi4py import MPI
from dolfinx import mesh, fem
import ufl
import pyvista as pv
from petsc4py import PETSc

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from conftest import _require_single_rank, create_zero_phi, make_test_config
from simulation.eigensolver import ElasticEigenSolver
from simulation.ligamentassembler import LigamentSpringSystem

pytestmark = pytest.mark.simulation


# ======================== Fixtures ========================


@pytest.fixture(scope="session")
def elasticity_setup():
    _require_single_rank()
    from conftest import create_unit_cube
    domain = create_unit_cube()
    gdim = domain.geometry.dim
    V = fem.functionspace(domain, ("Lagrange", 1, (gdim,)))
    W = fem.functionspace(domain, ("DG", 0))

    E = fem.Function(W)
    E.x.array[:] = 1200.0
    nu = fem.Function(W)
    nu.x.array[:] = 0.3

    ds = ufl.ds(domain=domain)
    return domain, V, W, E, nu, ds


def _polyline_from_indices(X, indices):
    pts = X[np.asarray(indices, dtype=int)].astype(float)
    poly = pv.PolyData(pts)
    lines = []
    for a, b in zip(range(len(indices) - 1), range(1, len(indices))):
        lines.extend([2, a, b])
    poly.lines = np.asarray(lines, dtype=np.int32)
    return poly


# ======================== Mass matrix tests ========================


def test_mass_matrix_matches_integral(elasticity_setup, test_config):
    """Verify mass matrix M via integral identity: v^T M v = ∫ ||u||^2 dx."""
    _require_single_rank()
    domain, V, _, E, nu, ds = elasticity_setup
    phi = create_zero_phi(V)

    solver = ElasticEigenSolver(V, E, nu, phi, config=test_config)
    M = solver.M0

    rng = np.random.default_rng(3)
    dof = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
    vec = rng.normal(size=(dof,))

    petsc_vec = PETSc.Vec().createSeq(dof)
    petsc_vec.setArray(vec)
    prod = petsc_vec.dot(M * petsc_vec)

    u = fem.Function(V)
    u.x.array[:] = vec
    u.x.scatter_forward()
    dx = ufl.Measure("dx", domain=domain)
    integral = fem.assemble_scalar(fem.form(ufl.inner(u, u) * dx))

    assert np.isclose(prod, integral, rtol=1e-10, atol=1e-12)

    petsc_vec.destroy()
    solver.K.destroy()
    solver.M.destroy()


def test_mass_norms_matches_matrix_product(elasticity_setup, test_config):
    """Verify mass matrix symmetry and norm properties."""
    _require_single_rank()
    _, V, _, E, nu, ds = elasticity_setup
    phi = create_zero_phi(V)

    solver = ElasticEigenSolver(V, E, nu, phi, config=test_config)

    rng = np.random.default_rng(7)
    vec = rng.normal(size=(1, V.dofmap.index_map.size_local, V.mesh.geometry.dim))
    norms = solver._mass_norms(vec)

    petsc_vec = PETSc.Vec().createSeq(vec.size)
    petsc_vec.setArray(vec.reshape(-1))
    work = solver.M * petsc_vec
    expected = petsc_vec.dot(work)

    assert np.isclose(norms[0], expected)

    petsc_vec.destroy()
    work.destroy()
    solver.K.destroy()
    solver.M.destroy()


def test_mass_norms_scale_quadratic(elasticity_setup, test_config):
    _require_single_rank()
    _, V, _, E, nu, ds = elasticity_setup
    phi = create_zero_phi(V)

    solver = ElasticEigenSolver(V, E, nu, phi, config=test_config)

    rng = np.random.default_rng(11)
    base = rng.normal(size=(V.dofmap.index_map.size_local, V.mesh.geometry.dim))
    modes = np.stack([base, 2.5 * base])
    norms = solver._mass_norms(modes)

    assert norms[1] == pytest.approx((2.5 ** 2) * norms[0], rel=1e-9, abs=1e-12)

    solver.K.destroy()
    solver.M.destroy()


# ======================== Energy tests ========================


def test_mode_region_energies_partition_sum(elasticity_setup, test_config):
    _require_single_rank()
    domain, V, _, E, nu, ds = elasticity_setup
    phi = create_zero_phi(V)

    solver = ElasticEigenSolver(V, E, nu, phi, config=test_config)

    tdim = domain.topology.dim
    domain.topology.create_entities(tdim)
    num_cells = domain.topology.index_map(tdim).size_local
    cell_indices = np.arange(num_cells, dtype=np.int32)
    midpoints = mesh.compute_midpoints(domain, tdim, cell_indices)
    regions = np.where(midpoints[:, 0] >= 0.5, 2, 1).astype(np.int32)
    tags = mesh.meshtags(domain, tdim, cell_indices, regions)

    rng = np.random.default_rng(3)
    mode = rng.normal(size=(1, V.dofmap.index_map.size_local, V.mesh.geometry.dim))

    energies = solver.mode_region_energies(mode, tags, [1, 2])[0]

    u_fn = fem.Function(V)
    u_fn.x.array[:] = mode.reshape(-1)
    u_fn.x.scatter_forward()

    gdim = V.mesh.geometry.dim
    F = ufl.Identity(gdim) + ufl.grad(solver.phi)
    F_inv = ufl.inv(F)
    J = ufl.det(F)
    mu = solver.E / (2.0 * (1.0 + solver.nu))
    lmbda = solver.E * solver.nu / ((1.0 + solver.nu) * (1.0 - 2.0 * solver.nu))

    def eps(w):
        return ufl.sym(ufl.grad(w) * F_inv)

    def sigma(w):
        return lmbda * ufl.tr(eps(w)) * ufl.Identity(gdim) + 2.0 * mu * eps(w)

    energy_density = ufl.inner(sigma(u_fn), eps(u_fn)) * J
    dx = ufl.Measure("dx", domain=domain)
    total_energy = fem.assemble_scalar(fem.form(energy_density * dx))

    assert np.isclose(energies.sum(), total_energy)

    solver.K.destroy()
    solver.M.destroy()


def test_mode_weighted_energies_constant_weight(elasticity_setup, test_config):
    """Weighted energy with constant weight equals weight * total energy."""
    _require_single_rank()
    domain, V, _, E, nu, ds = elasticity_setup
    phi = create_zero_phi(V)

    solver = ElasticEigenSolver(V, E, nu, phi, config=test_config)

    rng = np.random.default_rng(5)
    mode = rng.normal(size=(1, V.dofmap.index_map.size_local, V.mesh.geometry.dim))

    u_fn = fem.Function(V)
    u_fn.x.array[:] = mode.reshape(-1)
    u_fn.x.scatter_forward()
    gdim = V.mesh.geometry.dim
    F = ufl.Identity(gdim) + ufl.grad(solver.phi)
    F_inv = ufl.inv(F)
    J = ufl.det(F)
    mu = solver.E / (2.0 * (1.0 + solver.nu))
    lmbda = solver.E * solver.nu / ((1.0 + solver.nu) * (1.0 - 2.0 * solver.nu))

    def eps(w):
        return ufl.sym(ufl.grad(w) * F_inv)

    def sigma(w):
        return lmbda * ufl.tr(eps(w)) * ufl.Identity(gdim) + 2.0 * mu * eps(w)

    energy_density = ufl.inner(sigma(u_fn), eps(u_fn)) * J
    total_energy = fem.assemble_scalar(fem.form(energy_density * ufl.dx(domain=domain)))

    c = 3.7
    w = np.full_like(E.x.array, fill_value=c, dtype=float)
    weighted = solver.mode_weighted_energies(mode, w)

    assert weighted.shape == (1,)
    assert np.isfinite(weighted[0])
    assert weighted[0] == pytest.approx(c * total_energy, rel=1e-9, abs=1e-12)

    solver.K.destroy()
    solver.M.destroy()


def test_mode_weighted_energies_invalid_weight_shape_raises(elasticity_setup, test_config):
    """Passing cell_weights with wrong shape should raise ValueError."""
    _require_single_rank()
    _, V, W, E, nu, ds = elasticity_setup
    phi = create_zero_phi(V)
    solver = ElasticEigenSolver(V, E, nu, phi, config=test_config)

    rng = np.random.default_rng(5)
    mode = rng.normal(size=(1, V.dofmap.index_map.size_local, V.mesh.geometry.dim))

    bad_weights = np.ones(E.x.array.shape[0] - 1, dtype=float)
    with pytest.raises(ValueError):
        solver.mode_weighted_energies(mode, bad_weights)

    solver.K.destroy()
    solver.M.destroy()


# ======================== Sensitivity tests ========================


def test_eigenvalue_scale_sensitivity_matches_fd(elasticity_setup, test_config):
    """For uniform E scaling, dλ/dα = (1/α) * (∫ energy) / (φᵀ M φ) ≈ FD."""
    _require_single_rank()
    _, V, _, E, nu, ds = elasticity_setup

    phi = create_zero_phi(V)
    solver = ElasticEigenSolver(V, E, nu, phi, config=test_config)
    solver.fixed_dirichlet(gamma=1e3, ds=ds)

    lam, vecs = solver.solve(nev=6, target=1e-3)
    mnorm = solver._mass_norms(vecs)[0]

    alpha = float(E.x.array[0])
    w_alpha = np.full_like(E.x.array, fill_value=(1.0 / alpha), dtype=float)
    weighted = solver.mode_weighted_energies(vecs[:1], w_alpha)[0]
    dlam_dalpha_pred = weighted / mnorm

    def eigenvalue_with_scale(scale: float) -> float:
        E_new = fem.Function(E.function_space)
        E_new.x.array[:] = alpha * scale
        E_new.x.scatter_forward()
        phi_fd = create_zero_phi(V)
        sol = ElasticEigenSolver(V, E_new, nu, phi_fd, config=test_config)
        sol.fixed_dirichlet(gamma=1e3, ds=ds)
        lam_s, _ = sol.solve(nev=6, target=1e-3)
        val = lam_s[0]
        sol.K.destroy()
        sol.M.destroy()
        return float(val)

    eps = 1e-6
    fd = (eigenvalue_with_scale(1.0 + eps) - eigenvalue_with_scale(1.0 - eps)) / (2.0 * alpha * eps)

    assert dlam_dalpha_pred == pytest.approx(fd, rel=5e-3, abs=1e-8)

    solver.K.destroy()
    solver.M.destroy()


def test_eigenvalue_sensitivity_matches_fd(elasticity_setup, test_config):
    _require_single_rank()
    _, V, _, E, nu, ds = elasticity_setup

    X = V.tabulate_dof_coordinates()
    bundle = _polyline_from_indices(X, [0, 1])

    stiffness0 = 15.0
    phi = create_zero_phi(V)

    lig = LigamentSpringSystem(
        V, [bundle], [stiffness0], phi, make_test_config({"LIGAMENT_MAPPING_TOLERANCE": 1e-6})
    )
    solver = ElasticEigenSolver(V, E, nu, phi, config=test_config, ligament_system=lig)
    solver.fixed_dirichlet(gamma=1e3, ds=ds)

    nev = 4
    lam, vecs = solver.solve(nev=nev, target=1e-3)
    analytic = solver.eigenvalue_sensitivities(vecs)[0, 0]

    def eigenvalue(stiffness_value: float) -> float:
        phi = create_zero_phi(V)
        lig_fd = LigamentSpringSystem(
            V, [bundle], [stiffness_value], phi, make_test_config({"LIGAMENT_MAPPING_TOLERANCE": 1e-6})
        )
        solver_fd = ElasticEigenSolver(V, E, nu, phi, test_config, ligament_system=lig_fd)
        solver_fd.fixed_dirichlet(gamma=1e3, ds=ds)
        lam_fd, _ = solver_fd.solve(nev=nev, target=1e-3)
        val = lam_fd[0]
        solver_fd.K.destroy()
        solver_fd.M.destroy()
        lig_fd.K.destroy()
        return val

    delta = stiffness0 * 1e-6
    fd = (eigenvalue(stiffness0 + delta) - eigenvalue(stiffness0 - delta)) / (2.0 * delta)

    assert np.isclose(analytic, fd, rtol=1e-3, atol=1e-8)

    solver.K.destroy()
    solver.M.destroy()
    lig.K.destroy()


def test_eigenvalue_pretension_sensitivity_matches_fd(elasticity_setup, test_config):
    """Verify eigenvalue sensitivity to ligament pretension via finite differences."""
    _require_single_rank()
    _, V, _, E, nu, ds = elasticity_setup

    X = V.tabulate_dof_coordinates()
    bundle = _polyline_from_indices(X, [0, 1])

    stiffness0 = 15.0
    T0 = 5.0
    phi = create_zero_phi(V)
    lig = LigamentSpringSystem(
        V,
        [bundle],
        [stiffness0],
        phi,
        make_test_config({"LIGAMENT_MAPPING_TOLERANCE": 1e-6}),
        pretensions=[T0],
    )
    solver = ElasticEigenSolver(V, E, nu, phi, config=test_config, ligament_system=lig)
    solver.fixed_dirichlet(gamma=1e3, ds=ds)

    nev = 4
    lam, vecs = solver.solve(nev=nev, target=1e-3)
    analytic = solver.eigenvalue_pretension_sensitivities(vecs)[0, 0]

    def eigenvalue_at_pretension(T_value: float) -> float:
        phi_fd = create_zero_phi(V)
        lig_fd = LigamentSpringSystem(
            V,
            [bundle],
            [stiffness0],
            phi_fd,
            make_test_config({"LIGAMENT_MAPPING_TOLERANCE": 1e-6}),
            pretensions=[T_value],
        )
        solver_fd = ElasticEigenSolver(V, E, nu, phi_fd, config=test_config, ligament_system=lig_fd)
        solver_fd.fixed_dirichlet(gamma=1e3, ds=ds)
        lam_fd, _ = solver_fd.solve(nev=nev, target=1e-3)
        val = lam_fd[0]
        solver_fd.K.destroy()
        solver_fd.M.destroy()
        lig_fd.K.destroy()
        return val

    delta = T0 * 1e-6
    fd = (eigenvalue_at_pretension(T0 + delta) - eigenvalue_at_pretension(T0 - delta)) / (2.0 * delta)

    assert np.isclose(analytic, fd, rtol=1e-3, atol=1e-8)

    solver.K.destroy()
    solver.M.destroy()
    lig.K.destroy()


def test_eigenvalue_pretension_sensitivity_zero_with_no_ligaments(elasticity_setup, test_config):
    """Verify pretension sensitivity returns empty array when no ligaments present."""
    _require_single_rank()
    _, V, _, E, nu, ds = elasticity_setup
    phi = create_zero_phi(V)

    solver = ElasticEigenSolver(V, E, nu, phi, config=test_config, ligament_system=None)
    solver.fixed_dirichlet(gamma=1e3, ds=ds)

    nev = 4
    lam, vecs = solver.solve(nev=nev, target=1e-3)
    pretension_sens = solver.eigenvalue_pretension_sensitivities(vecs)

    num_modes = len(lam)
    assert pretension_sens.shape == (num_modes, 0)

    solver.K.destroy()
    solver.M.destroy()


def test_eigenvalue_pretension_sensitivity_multiple_bundles(elasticity_setup, test_config):
    """Verify pretension sensitivity shape with multiple ligament bundles."""
    _require_single_rank()
    _, V, _, E, nu, ds = elasticity_setup
    X = V.tabulate_dof_coordinates()

    bundle1 = _polyline_from_indices(X, [0, 1])
    bundle2 = _polyline_from_indices(X, [2, 3])
    bundle3 = _polyline_from_indices(X, [4, 5])

    stiffnesses = [10.0, 15.0, 20.0]
    T = [2.0, 3.0, 4.0]

    phi = create_zero_phi(V)
    lig = LigamentSpringSystem(
        V,
        [bundle1, bundle2, bundle3],
        stiffnesses,
        phi,
        make_test_config({"LIGAMENT_MAPPING_TOLERANCE": 1e-6}),
        pretensions=T,
    )
    solver = ElasticEigenSolver(V, E, nu, phi, config=test_config, ligament_system=lig)
    solver.fixed_dirichlet(gamma=1e3, ds=ds)

    nev = 4
    lam, vecs = solver.solve(nev=nev, target=1e-3)

    stiff_sens = solver.eigenvalue_sensitivities(vecs)
    pretension_sens = solver.eigenvalue_pretension_sensitivities(vecs)

    num_modes = len(lam)
    num_bundles = 3

    assert stiff_sens.shape == (num_modes, num_bundles)
    assert pretension_sens.shape == (num_modes, num_bundles)

    assert np.all(np.isfinite(stiff_sens))
    assert np.all(np.isfinite(pretension_sens))

    solver.K.destroy()
    solver.M.destroy()
    lig.K.destroy()


# ======================== SLEPc convergence tests ========================


def test_solve_filters_rigid_modes(elasticity_setup, test_config):
    _require_single_rank()
    _, V, _, E, nu, ds = elasticity_setup
    phi = create_zero_phi(V)

    solver = ElasticEigenSolver(V, E, nu, phi, config=test_config)

    lam, vecs = solver.solve(nev=8, target=1e-3)

    assert np.all(lam > solver.rigid_mode_tolerance)
    diffs = np.diff(lam)
    assert np.all(diffs >= -1e-8)

    solver.K.destroy()
    solver.M.destroy()


def test_eigenvalue_residual_accuracy(elasticity_setup, test_config):
    """Test that eigenvalue residuals ||Kx - λMx|| satisfy tolerance."""
    _require_single_rank()
    _, V, _, E, nu, ds = elasticity_setup

    phi = create_zero_phi(V)
    solver = ElasticEigenSolver(V, E, nu, phi, config=test_config)
    solver.fixed_dirichlet(gamma=1e6)

    lam, vecs = solver.solve(nev=6, target=1e-6)

    residuals = []
    vec_petsc = solver.K.createVecRight()
    work = solver.K.createVecRight()

    for i, (eigenval, eigenvec) in enumerate(zip(lam, vecs)):
        vec_petsc.setArray(eigenvec.flatten())
        solver.K.mult(vec_petsc, work)

        mx_vec = solver.K.createVecRight()
        solver.M.mult(vec_petsc, mx_vec)
        work.axpy(-eigenval, mx_vec)

        r_norm = work.norm()
        mx_vec.destroy()

        solver.M.mult(vec_petsc, work)
        lambda_mx_norm = abs(eigenval) * work.norm()

        rel_residual = r_norm / lambda_mx_norm if lambda_mx_norm > 0 else r_norm
        residuals.append(rel_residual)

    vec_petsc.destroy()
    work.destroy()

    assert len(residuals) > 0, "No eigenvalues converged"
    max_residual = max(residuals)
    assert max_residual < 1e-6, f"Excessive residual: {max_residual:.2e}"

    solver.K.destroy()
    solver.M.destroy()


def test_shift_invert_target_proximity(elasticity_setup, test_config):
    """Test that eigenvalues are found near the specified target."""
    _require_single_rank()
    _, V, _, E, nu, ds = elasticity_setup

    phi = create_zero_phi(V)
    solver = ElasticEigenSolver(V, E, nu, phi, config=test_config)
    solver.fixed_dirichlet(gamma=1e6)

    target = 100.0
    lam, vecs = solver.solve(nev=8, target=target)

    assert len(lam) >= 6, f"Only {len(lam)} eigenvalues converged, expected ≥6"

    min_eig = np.min(lam)
    max_eig = np.max(lam)

    assert min_eig > 0, f"Non-positive eigenvalue: {min_eig}"
    assert min_eig >= target * 0.01, f"Smallest eigenvalue {min_eig:.2e} too far below target {target:.2e}"
    assert max_eig <= target * 1000, f"Largest eigenvalue {max_eig:.2e} too far above target {target:.2e}"

    solver.K.destroy()
    solver.M.destroy()


def test_eigenvalue_orthogonality(elasticity_setup, test_config):
    """Test mass-orthogonality: x_i^T M x_j ≈ δ_ij."""
    _require_single_rank()
    _, V, _, E, nu, ds = elasticity_setup

    phi = create_zero_phi(V)
    solver = ElasticEigenSolver(V, E, nu, phi, config=test_config)
    solver.fixed_dirichlet(gamma=1e6)

    lam, vecs = solver.solve(nev=6, target=100.0)

    assert len(lam) >= 4, f"Only {len(lam)} eigenvalues, need at least 4 for orthogonality test"

    num_modes = len(lam)
    gram_matrix = np.zeros((num_modes, num_modes))

    vec_i = solver.M.createVecRight()
    vec_j = solver.M.createVecRight()
    work = solver.M.createVecRight()

    for i in range(num_modes):
        vec_i.setArray(vecs[i].flatten())

        for j in range(num_modes):
            vec_j.setArray(vecs[j].flatten())
            solver.M.mult(vec_j, work)
            gram_matrix[i, j] = vec_i.dot(work)

    vec_i.destroy()
    vec_j.destroy()
    work.destroy()

    diag = np.sqrt(np.diag(gram_matrix))
    normalized_gram = gram_matrix / np.outer(diag, diag)

    identity = np.eye(num_modes)
    error = np.linalg.norm(normalized_gram - identity, ord="fro")

    assert error < 0.1, f"Poor M-orthogonality: ||G - I||_F = {error:.2e}"

    if num_modes > 1:
        off_diag = normalized_gram - np.diag(np.diag(normalized_gram))
        max_off_diag = np.max(np.abs(off_diag))
        assert max_off_diag < 0.05, f"Large off-diagonal element: {max_off_diag:.2e}"

    solver.K.destroy()
    solver.M.destroy()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
