"""FD-verified test for the cartilage/symphysis dλ/dE sensitivity formula.

This mirrors the per-region modulus sensitivity computed by
``simulation_core._compute_cartilage_sensitivities``:

    dλ_i / dE_R  ≈  (∫_R energy_density(u_i) dx) / E_R / (u_iᵀ M u_i)

The test builds a cube split into two cell regions, holds bone-region E
fixed and perturbs cartilage-region E, and compares the analytical
sensitivity against a central finite difference.

Regression purpose: the previous implementation rescaled
``mode_region_energies`` by ``E_func``, which silently broke this
identity for non-uniform ``E``. Keeping this test in the suite catches
any reintroduction of that bug.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import ufl
from mpi4py import MPI
from dolfinx import fem, mesh as dmesh

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from conftest import _require_single_rank, create_zero_phi, make_test_config
from simulation.eigensolver import ElasticEigenSolver

pytestmark = pytest.mark.simulation


def _two_region_cube_with_tags(n: int = 10):
    """Return (domain, V, W, E, nu, ds, cell_tags, region_ids).

    Cells with centroid x < 0.5 are tagged region 1 ("bone"),
    others region 2 ("cartilage").
    """
    domain = dmesh.create_unit_cube(MPI.COMM_WORLD, n, n, n)
    tdim = domain.topology.dim
    domain.topology.create_connectivity(tdim, 0)
    num_cells = domain.topology.index_map(tdim).size_local
    cell_indices = np.arange(num_cells, dtype=np.int32)

    # Compute cell centroids via geometry midpoints
    midpoints = dmesh.compute_midpoints(domain, tdim, cell_indices)
    region_values = np.where(midpoints[:, 0] < 0.5, 1, 2).astype(np.int32)
    cell_tags = dmesh.meshtags(domain, tdim, cell_indices, region_values)

    gdim = domain.geometry.dim
    V = fem.functionspace(domain, ("Lagrange", 1, (gdim,)))
    W = fem.functionspace(domain, ("DG", 0))

    # Set per-region E using cell_tags
    E_bone = 1500.0
    E_cart = 200.0
    E = fem.Function(W)
    E.x.array[cell_tags.find(1)] = E_bone
    E.x.array[cell_tags.find(2)] = E_cart
    nu = fem.Function(W)
    nu.x.array[:] = 0.3

    ds = ufl.ds(domain=domain)
    return domain, V, W, E, nu, ds, cell_tags, [1, 2], (E_bone, E_cart)


def _analytical_cartilage_sens(
    solver: ElasticEigenSolver,
    cell_tags,
    region_id: int,
    region_E: float,
    eigenvectors: np.ndarray,
) -> np.ndarray:
    """Replicate the formula used by SimulationCore._compute_cartilage_sensitivities."""
    energies = solver.mode_region_energies(eigenvectors, cell_tags, [region_id])[:, 0]
    mass_norms = solver._mass_norms(eigenvectors)
    out = np.zeros_like(energies)
    valid = mass_norms > 1e-12
    out[valid] = (energies[valid] / region_E) / mass_norms[valid]
    return out


def test_cartilage_modulus_sensitivity_matches_fd():
    """dλ/dE_cart from per-region energy formula matches central FD."""
    _require_single_rank()
    cfg = make_test_config({})

    domain, V, W, E, nu, ds, cell_tags, region_ids, (E_bone, E_cart0) = (
        _two_region_cube_with_tags(n=10)
    )
    cart_id = 2

    phi = create_zero_phi(V)
    solver = ElasticEigenSolver(V, E, nu, phi, config=cfg)
    solver.fixed_dirichlet(gamma=1e3, ds=ds)

    nev = 4
    lam0, vecs0 = solver.solve(nev=nev, target=1e-3)
    assert len(lam0) >= 1, "No eigenvalues converged"

    analytic = _analytical_cartilage_sens(solver, cell_tags, cart_id, E_cart0, vecs0)

    def eigenvalue_at_E_cart(E_cart_val: float) -> float:
        E_pert = fem.Function(W)
        E_pert.x.array[cell_tags.find(1)] = E_bone
        E_pert.x.array[cell_tags.find(cart_id)] = E_cart_val
        nu_pert = fem.Function(W)
        nu_pert.x.array[:] = 0.3
        phi_pert = create_zero_phi(V)
        sol = ElasticEigenSolver(V, E_pert, nu_pert, phi_pert, config=cfg)
        sol.fixed_dirichlet(gamma=1e3, ds=ds)
        lam, _ = sol.solve(nev=nev, target=1e-3)
        val = float(lam[0])
        sol.K.destroy()
        sol.M.destroy()
        sol.M0.destroy()
        return val

    delta = E_cart0 * 1e-5
    fd = (eigenvalue_at_E_cart(E_cart0 + delta) - eigenvalue_at_E_cart(E_cart0 - delta)) / (2.0 * delta)

    assert np.isclose(analytic[0], fd, rtol=5e-3, atol=1e-9), (
        f"Cartilage sensitivity mismatch: analytic={analytic[0]:.6e}, FD={fd:.6e}"
    )

    solver.K.destroy()
    solver.M.destroy()
    solver.M0.destroy()


def test_cartilage_sensitivity_zero_when_no_region():
    """Empty region list returns zero-column sensitivity array."""
    _require_single_rank()
    cfg = make_test_config({})

    domain, V, W, E, nu, ds, cell_tags, _, _ = _two_region_cube_with_tags(n=6)
    phi = create_zero_phi(V)
    solver = ElasticEigenSolver(V, E, nu, phi, config=cfg)
    solver.fixed_dirichlet(gamma=1e3, ds=ds)
    _, vecs = solver.solve(nev=2, target=1e-3)

    out = solver.mode_region_energies(vecs, cell_tags, [])
    assert out.shape == (vecs.shape[0], 0)

    solver.K.destroy()
    solver.M.destroy()
    solver.M0.destroy()


def test_cartilage_energy_partition_sums_to_total():
    """Region energies must sum to the total elastic strain energy on the mode."""
    _require_single_rank()
    cfg = make_test_config({})

    domain, V, W, E, nu, ds, cell_tags, region_ids, _ = _two_region_cube_with_tags(n=8)
    phi = create_zero_phi(V)
    solver = ElasticEigenSolver(V, E, nu, phi, config=cfg)
    solver.fixed_dirichlet(gamma=1e3, ds=ds)

    _, vecs = solver.solve(nev=3, target=1e-3)
    per_region = solver.mode_region_energies(vecs, cell_tags, region_ids)

    # Total energy via mode_weighted_energies with weight=1.
    total = solver.mode_weighted_energies(vecs, np.ones_like(E.x.array))
    assert np.allclose(per_region.sum(axis=1), total, rtol=1e-8, atol=1e-10)

    solver.K.destroy()
    solver.M.destroy()
    solver.M0.destroy()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
