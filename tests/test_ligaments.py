from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from mpi4py import MPI
from dolfinx import mesh, fem
from petsc4py import PETSc
import pyvista as pv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from conftest import _require_single_rank, test_config, make_test_config

from simulation.ligamentassembler import LigamentSpringSystem

pytestmark = pytest.mark.simulation


@pytest.fixture(scope="session")
def V_X_bs():
    _require_single_rank()
    from conftest import create_unit_cube
    domain = create_unit_cube()
    gdim = domain.geometry.dim
    V = fem.functionspace(domain, ("Lagrange", 1, (gdim,)))
    X = V.tabulate_dof_coordinates()
    bs = int(V.dofmap.index_map_bs)
    assert bs == gdim
    return V, X, bs

def create_phi_function(V, displacement_values=None):
    """Create a phi fem.Function with given displacement values or zeros."""
    phi = fem.Function(V)
    if displacement_values is not None:
        phi.x.array[:] = displacement_values.reshape(-1)
    else:
        phi.x.array[:] = 0.0
    return phi

def polyline_from_indices(X, indices):
    pts = X[np.array(indices, dtype=int)].astype(float)
    poly = pv.PolyData(pts)
    lines = []
    for a, b in zip(range(len(indices) - 1), range(1, len(indices))):
        lines.extend([2, a, b])
    poly.lines = np.array(lines, dtype=np.int32)
    return poly

def get_block(K, i, j, bs):
    rows = list(range(i * bs, (i + 1) * bs))
    cols = list(range(j * bs, (j + 1) * bs))
    return K.getValues(rows, cols)

def block_row_sum(K, ib, bs):
    # Sum K[ib, j] over all block columns j seen in the sparsity
    cols = set()
    for r in range(ib * bs, (ib + 1) * bs):
        col_idx, _ = K.getRow(r)
        cols.update((c // bs) for c in col_idx)
    S = np.zeros((bs, bs), dtype=float)
    for jb in cols:
        S += get_block(K, ib, jb, bs)
    return S

def vectorize_blocks(vec_blocks):
    # vec_blocks: array (nb, bs) -> flattened scalar vector
    return vec_blocks.reshape(-1)


def _central_difference(func, x0, dx):
    return (func(x0 + dx) - func(x0 - dx)) / (2.0 * dx)

def test_type_and_blocksize(V_X_bs, test_config):
    from simulation.ligamentassembler import LigamentSpringSystem
    V, X, bs = V_X_bs
    poly = polyline_from_indices(X, [0, 1, 2])
    phi = create_phi_function(V)
    sysK = LigamentSpringSystem(V, [poly], [1.0], phi, make_test_config({"LIGAMENT_MAPPING_TOLERANCE": 1e-6}))
    K = sysK.K
    assert "aij" in K.getType().lower()
    assert K.getBlockSize() == bs

def test_block_row_equilibrium(V_X_bs, test_config):
    from simulation.ligamentassembler import LigamentSpringSystem
    V, X, bs = V_X_bs
    polylines = [polyline_from_indices(X, [0, 1]), polyline_from_indices(X, [3, 4]), polyline_from_indices(X, [2, 5])]
    stiffnesses = [5.0, 7.5, 3.3]
    phi = create_phi_function(V)
    sysK = LigamentSpringSystem(V, polylines, stiffnesses, phi, make_test_config({"LIGAMENT_MAPPING_TOLERANCE": 1e-6}))
    K = sysK.K
    nb = X.shape[0]
    for ib in range(min(nb, 10)):  # sample a few
        S = block_row_sum(K, ib, bs)
        assert np.allclose(S, 0.0, atol=1e-12)

def test_translational_nullspace(V_X_bs, test_config):
    from simulation.ligamentassembler import LigamentSpringSystem
    from petsc4py import PETSc
    V, X, bs = V_X_bs
    poly = polyline_from_indices(X, [0, 1, 5, 6])
    stiffness = 9.0
    phi = create_phi_function(V)
    sysK = LigamentSpringSystem(V, [poly], [stiffness], phi, make_test_config({"LIGAMENT_MAPPING_TOLERANCE": 1e-6}))
    K = sysK.K
    t = np.array([0.11, -0.22, 0.33], dtype=float)
    nb = X.shape[0]
    U_blocks = np.tile(t, (nb, 1))
    U = PETSc.Vec().createSeq(nb * bs)
    U.setArray(vectorize_blocks(U_blocks))
    y = K * U
    assert np.allclose(y.getArray(), 0.0, atol=1e-12)

def test_rotational_nullspace(V_X_bs, test_config):
    from simulation.ligamentassembler import LigamentSpringSystem
    from petsc4py import PETSc
    V, X, bs = V_X_bs
    poly = polyline_from_indices(X, [0, 1, 5, 6, 7])
    stiffness = 4.2
    phi = create_phi_function(V)
    sysK = LigamentSpringSystem(V, [poly], [stiffness], phi, make_test_config({"LIGAMENT_MAPPING_TOLERANCE": 1e-6}))
    K = sysK.K
    # infinitesimal rotation u_i = omega x x_i
    omega = np.array([0.3, -0.2, 0.1], dtype=float)
    U_blocks = np.cross(np.tile(omega, (X.shape[0], 1)), X)
    from petsc4py import PETSc
    U = PETSc.Vec().createSeq(U_blocks.size)
    U.setArray(vectorize_blocks(U_blocks))
    y = K * U
    assert np.allclose(y.getArray(), 0.0, atol=1e-11)

def test_zero_stiffness_bundle_ignored(V_X_bs, test_config):
    from simulation.ligamentassembler import LigamentSpringSystem
    V, X, bs = V_X_bs
    i0, i1, i2 = 0, 1, 2
    poly1 = polyline_from_indices(X, [i0, i1])
    poly2 = polyline_from_indices(X, [i1, i2])
    phi = create_phi_function(V)
    sysK = LigamentSpringSystem(V, [poly1, poly2], [0.0, 5.0], phi, make_test_config({"LIGAMENT_MAPPING_TOLERANCE": 1e-6}))
    W12 = get_block(sysK.K, i1, i2, bs)
    Z01 = get_block(sysK.K, i0, i1, bs)
    assert np.allclose(W12, get_block(sysK.K, i2, i1, bs), atol=1e-12)  # symmetry sanity
    assert np.allclose(Z01, 0.0, atol=1e-14)  # zero stiffness means no contribution


def test_construction_is_deterministic(V_X_bs, test_config):
    """Two independent constructions with identical inputs must yield identical K."""
    _require_single_rank()

    V, X, bs = V_X_bs
    poly = polyline_from_indices(X, [0, 1])

    phi_a = create_phi_function(V)
    system_a = LigamentSpringSystem(V, [poly], [6.0], phi_a, make_test_config({"LIGAMENT_MAPPING_TOLERANCE": 1e-8}))

    phi_b = create_phi_function(V)
    system_b = LigamentSpringSystem(V, [poly], [6.0], phi_b, make_test_config({"LIGAMENT_MAPPING_TOLERANCE": 1e-8}))

    for i in range(2):
        for j in range(2):
            block_a = get_block(system_a.K, i, j, bs)
            block_b = get_block(system_b.K, i, j, bs)
            assert np.allclose(block_a, block_b, atol=1e-14)

    # Sanity: K is non-trivial (at least one nonzero diagonal block on the bundle).
    diag00 = get_block(system_a.K, 0, 0, bs)
    assert np.linalg.norm(diag00) > 0.0

def test_reversed_duplicates_in_one_bundle_dedup_note(V_X_bs, test_config):
    """Reversed duplicate segments must collapse to a single undirected pair."""
    from simulation.ligamentassembler import LigamentSpringSystem

    V, X, bs = V_X_bs
    a, b = 0, 1
    pts = X[[a, b]].astype(float)
    poly = pv.PolyData(pts)
    # Add both directions: 0-1 and 1-0
    poly.lines = np.array([2, 0, 1, 2, 1, 0], dtype=np.int32)

    stiffness = 2.0
    phi = create_phi_function(V)
    sysK = LigamentSpringSystem(V, [poly], [stiffness], phi, make_test_config({"LIGAMENT_MAPPING_TOLERANCE": 1e-6}))
    W = get_block(sysK.K, a, a, bs)
    direction = X[b] - X[a]
    direction = direction.astype(float) / np.linalg.norm(direction)
    assert np.allclose(W, stiffness * np.outer(direction, direction), atol=1e-12)


def test_update_with_none_displacement_resets_geometry(V_X_bs, test_config):
    _require_single_rank()

    V, X, bs = V_X_bs
    poly = polyline_from_indices(X, [0, 1])

    phi = create_phi_function(V)
    system = LigamentSpringSystem(
        V,
        [poly],
        [5.5],
        phi,
        make_test_config({"LIGAMENT_MAPPING_TOLERANCE": 1e-8}),
    )

    baseline = get_block(system.K, 0, 1, bs)

    # Apply shear displacement
    shear_disp = np.zeros_like(X)
    shear_disp[:, 2] = 0.15 * X[:, 0]
    phi_shear = create_phi_function(V, shear_disp)
    system._phi = phi_shear
    system.update()
    sheared = get_block(system.K, 0, 1, bs)
    assert not np.allclose(sheared, baseline, atol=1e-12)

    # Reset to zero displacement
    phi_zero = create_phi_function(V)
    system._phi = phi_zero
    system.update()
    restored = get_block(system.K, 0, 1, bs)
    assert np.allclose(restored, baseline, atol=1e-12)


def test_mode_bundle_quadratic_forms_matches_fd(V_X_bs, test_config):
    _require_single_rank()

    V, X, bs = V_X_bs
    poly = polyline_from_indices(X, [0, 1, 2])
    stiffness0 = 7.5
    phi = create_phi_function(V)
    system = LigamentSpringSystem(V, [poly], [stiffness0], phi, make_test_config({"LIGAMENT_MAPPING_TOLERANCE": 1e-6}))

    nb = X.shape[0]
    mode_blocks = np.zeros((nb, bs), dtype=float)
    mode_blocks[0] = [0.05, -0.02, 0.01]
    mode_blocks[1] = [-0.03, 0.04, -0.02]
    mode_blocks[2] = [0.01, 0.02, -0.05]

    mode_vec = PETSc.Vec().createSeq(nb * bs)
    mode_vec.setArray(vectorize_blocks(mode_blocks))

    # Ensure baseline state and analytic sensitivity
    system.update(stiffnesses=[stiffness0])
    analytic = system.mode_bundle_quadratic_forms(mode_blocks[np.newaxis, ...])[0, 0]

    def energy(stiffness_value: float) -> float:
        system.update(stiffnesses=[stiffness_value])
        tmp = system.K * mode_vec
        val = mode_vec.dot(tmp)
        tmp.destroy()
        return val

    delta = 1e-6 * stiffness0
    fd = _central_difference(energy, stiffness0, delta)
    system.update(stiffnesses=[stiffness0])  # restore baseline for cleanliness

    assert np.isclose(analytic, fd, rtol=1e-5, atol=1e-11)

    mode_vec.destroy()


def test_mode_bundle_geometric_forms_match_fd(V_X_bs, test_config):
    _require_single_rank()

    V, X, bs = V_X_bs
    poly = polyline_from_indices(X, [0, 1])
    stiffness0 = 9.0
    T0 = 4.0

    phi = create_phi_function(V)
    system = LigamentSpringSystem(
        V,
        [poly],
        [stiffness0],
        phi,
        make_test_config({"LIGAMENT_MAPPING_TOLERANCE": 1e-8}),
        pretensions=[T0],
    )

    nb = X.shape[0]
    mode_blocks = np.zeros((nb, bs), dtype=float)
    mode_blocks[0] = [-0.02, 0.01, 0.03]
    mode_blocks[1] = [0.015, -0.025, -0.005]

    mode_vec = PETSc.Vec().createSeq(nb * bs)
    mode_vec.setArray(vectorize_blocks(mode_blocks))

    analytic = system.mode_bundle_geometric_quadratic_forms(mode_blocks[np.newaxis, ...])[0, 0]

    def energy(pretension: float) -> float:
        system.update(pretensions=[pretension])
        tmp = system.K * mode_vec
        val = mode_vec.dot(tmp)
        tmp.destroy()
        return val

    delta = 1e-6 * max(1.0, abs(T0))
    fd = _central_difference(energy, T0, delta)
    system.update(pretensions=[T0])

    assert np.isclose(analytic, fd, rtol=1e-5, atol=1e-10)

    mode_vec.destroy()


def test_mode_bundle_geometric_forms_translation_invariant(V_X_bs, test_config):
    """Geometric quadratic forms must be invariant under rigid translation of φ
    (segment direction & length unchanged ⇒ K_geo unchanged)."""
    _require_single_rank()

    V, X, bs = V_X_bs
    poly = polyline_from_indices(X, [0, 1])
    stiffness0 = 6.0
    T0 = 3.5

    phi = create_phi_function(V)
    system = LigamentSpringSystem(
        V,
        [poly],
        [stiffness0],
        phi,
        make_test_config({"LIGAMENT_MAPPING_TOLERANCE": 1e-6}),
        pretensions=[T0],
    )

    rng = np.random.default_rng(42)
    mode_blocks = rng.normal(size=(X.shape[0], bs)) * 0.01

    baseline = system.mode_bundle_geometric_quadratic_forms(mode_blocks[np.newaxis, ...])

    # Apply pure translation via update() — same pattern as
    # test_translation_update_preserves_energy.
    translation = np.tile(np.array([0.12, -0.05, 0.08]), (X.shape[0], 1))
    phi_trans = create_phi_function(V, translation)
    system._phi = phi_trans
    system.update()
    translated = system.mode_bundle_geometric_quadratic_forms(mode_blocks[np.newaxis, ...])

    assert np.allclose(translated, baseline, atol=1e-10, rtol=1e-10)


def test_translation_update_preserves_energy(V_X_bs, test_config):
    _require_single_rank()

    V, X, bs = V_X_bs
    poly = polyline_from_indices(X, [0, 1, 2])
    stiffness0 = 6.5
    phi = create_phi_function(V)
    system = LigamentSpringSystem(V, [poly], [stiffness0], phi, make_test_config({"LIGAMENT_MAPPING_TOLERANCE": 1e-6}))

    rng = np.random.default_rng(123)
    mode = rng.normal(size=(X.shape[0], bs))
    vec = system.K.createVecRight()
    vec.setArray(mode.reshape(-1))
    baseline = vec.dot(system.K * vec)

    # Apply translation displacement
    translation = np.tile(np.array([0.12, -0.05, 0.08]), (X.shape[0], 1))
    phi_translate = create_phi_function(V, translation)
    system._phi = phi_translate
    system.update()
    translated_energy = vec.dot(system.K * vec)

    assert translated_energy == pytest.approx(baseline, rel=1e-6, abs=1e-9)

    vec.destroy()


def test_ligament_matrix_positive_semidefinite(V_X_bs, test_config):
    _require_single_rank()

    V, X, bs = V_X_bs
    polylines = [polyline_from_indices(X, [0, 3]), polyline_from_indices(X, [2, 5])]
    stiffnesses = [4.0, 9.0]
    phi = create_phi_function(V)
    system = LigamentSpringSystem(V, polylines, stiffnesses, phi, make_test_config({"LIGAMENT_MAPPING_TOLERANCE": 1e-6}))

    rng = np.random.default_rng(77)
    for _ in range(5):
        mode = rng.normal(size=(X.shape[0], bs))
        vec = system.K.createVecRight()
        vec.setArray(mode.reshape(-1))
        energy = vec.dot(system.K * vec)
        assert energy >= -1e-9
        vec.destroy()


def test_segments_filtered_by_tolerance(V_X_bs, test_config):
    _require_single_rank()
    from simulation.ligamentassembler import LigamentSpringSystem

    V, X, bs = V_X_bs
    far_pts = np.array([[50.0, 50.0, 50.0], [51.0, 50.5, 49.8]], dtype=float)
    poly = pv.PolyData(far_pts)
    poly.lines = np.array([2, 0, 1], dtype=np.int32)

    phi = create_phi_function(V)
    system = LigamentSpringSystem(V, [poly], [12.0], phi, make_test_config({"LIGAMENT_MAPPING_TOLERANCE": 1e-3}))

    random_mode = np.random.default_rng(5).normal(size=(1, X.shape[0], bs))
    sensitivity = system.mode_bundle_quadratic_forms(random_mode)[0, 0]
    assert sensitivity == pytest.approx(0.0, abs=1e-12)


def test_zero_length_segment_ignored(V_X_bs, test_config):
    """Segments with identical endpoints should contribute no stiffness."""
    _require_single_rank()
    from simulation.ligamentassembler import LigamentSpringSystem
    V, X, bs = V_X_bs
    import pyvista as pv
    pts = np.array([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]], dtype=float)
    poly = pv.PolyData(pts)
    poly.lines = np.array([2, 0, 1], dtype=np.int32)
    phi = create_phi_function(V)
    system = LigamentSpringSystem(V, [poly], [10.0], phi, make_test_config({"LIGAMENT_MAPPING_TOLERANCE": 1e-6}))
    # Random mode energy should be ~0 due to ignored segment
    rng = np.random.default_rng(0)
    mode = rng.normal(size=(1, X.shape[0], bs))
    energy = system.mode_bundle_quadratic_forms(mode)[0, 0]
    assert energy == pytest.approx(0.0, abs=1e-12)


def test_update_displacement_recomputes_stiffness(V_X_bs, test_config):
    _require_single_rank()

    V, X, bs = V_X_bs
    i0, i1 = 0, 1
    poly = polyline_from_indices(X, [i0, i1])
    stiffness = 3.5

    phi = create_phi_function(V)
    system = LigamentSpringSystem(V, [poly], [stiffness], phi, make_test_config({"LIGAMENT_MAPPING_TOLERANCE": 1e-8}))

    initial_block = get_block(system.K, i0, i1, bs)

    # Create shear displacement
    shear_disp = np.zeros_like(X)
    shear_disp[:, 1] = 0.2 * X[:, 0]
    phi_shear = create_phi_function(V, shear_disp)
    system._phi = phi_shear
    system.update()

    Xd = X + shear_disp
    direction = Xd[i1] - Xd[i0]
    length = np.linalg.norm(direction)
    direction /= length
    expected_W = (stiffness / 1.0) * np.outer(direction, direction)

    updated_block = get_block(system.K, i0, i1, bs)

    assert not np.allclose(updated_block, initial_block, atol=1e-12)
    assert np.allclose(updated_block, -expected_W, atol=1e-12)


def test_update_stiffness_scales_matrix(V_X_bs, test_config):
    _require_single_rank()

    V, X, bs = V_X_bs
    poly = polyline_from_indices(X, [0, 1, 2])
    stiffness = 8.0
    phi = create_phi_function(V)
    system = LigamentSpringSystem(V, [poly], [stiffness], phi, make_test_config({"LIGAMENT_MAPPING_TOLERANCE": 1e-6}))

    block_before = get_block(system.K, 0, 1, bs)
    scale = 1.7
    system.update(stiffnesses=[stiffness * scale])
    block_after = get_block(system.K, 0, 1, bs)

    assert np.allclose(block_after, scale * block_before, atol=1e-12)


def test_pretension_adds_geometric_stiffness_energy_delta(V_X_bs, test_config):
    _require_single_rank()

    V, X, bs = V_X_bs
    i0, i1 = 0, 1
    poly = polyline_from_indices(X, [i0, i1])
    stiffness = 10.0
    T = 5.0  # N pretension

    phi = create_phi_function(V)
    system = LigamentSpringSystem(V, [poly], [stiffness], phi, make_test_config({"LIGAMENT_MAPPING_TOLERANCE": 1e-8}), pretensions=[0.0])

    # Build a mode with displacement difference orthogonal to the segment direction
    e_dir = X[i1] - X[i0]
    L = np.linalg.norm(e_dir)
    assert L > 0
    e_dir = e_dir / L
    # pick any vector not parallel to e_dir and project to orthogonal plane
    v = np.array([0.3, -0.4, 0.2])
    v = v - np.dot(v, e_dir) * e_dir
    if np.linalg.norm(v) < 1e-12:
        v = np.array([e_dir[1], -e_dir[0], 0.0])  # fallback orthogonal
        v = v - np.dot(v, e_dir) * e_dir
    v = v / np.linalg.norm(v) * 0.1  # small amplitude

    nb = X.shape[0]
    U_blocks = np.zeros((nb, bs), dtype=float)
    U_blocks[i0] = -0.5 * v
    U_blocks[i1] = 0.5 * v
    vec = PETSc.Vec().createSeq(nb * bs)
    vec.setArray(vectorize_blocks(U_blocks))

    energy0 = vec.dot(system.K * vec)
    system.update(pretensions=[T])
    energyT = vec.dot(system.K * vec)

    expected_delta = (T / L) * np.linalg.norm(v) ** 2
    assert energyT - energy0 == pytest.approx(expected_delta, rel=1e-6, abs=1e-9)

    vec.destroy()


def test_sensitivity_pretension_no_gate_equals_baseline(V_X_bs, test_config):
    _require_single_rank()

    V, X, bs = V_X_bs
    i0, i1 = 0, 1
    poly = polyline_from_indices(X, [i0, i1])
    stiffness = 5.0
    # pretension T>0 -> s0 < 0, gate inactive
    T_pos = 3.0
    phi = create_phi_function(V)
    system = LigamentSpringSystem(V, [poly], [stiffness], phi, make_test_config({"LIGAMENT_MAPPING_TOLERANCE": 1e-8}), pretensions=[T_pos])

    e_dir = X[i1] - X[i0]
    e_dir = e_dir / np.linalg.norm(e_dir)
    dL = 0.01
    nb = X.shape[0]
    mode = np.zeros((nb, bs), dtype=float)
    mode[i1] = dL * e_dir
    sens_pret = system.mode_bundle_quadratic_forms(mode[np.newaxis, ...])[0, 0]

    # Baseline system (no pretension) should give identical sensitivity for the same mode
    system.update(pretensions=[0.0])
    sens_base = system.mode_bundle_quadratic_forms(mode[np.newaxis, ...])[0, 0]
    assert sens_pret == pytest.approx(sens_base, rel=1e-12, abs=1e-12)

if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
