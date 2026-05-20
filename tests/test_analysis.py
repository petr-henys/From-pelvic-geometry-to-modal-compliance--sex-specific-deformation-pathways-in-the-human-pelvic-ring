from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from mpi4py import MPI
from petsc4py import PETSc

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from conftest import _require_single_rank, MAC_TOL, EPS_TOL
from mode_pairing import mac_mass, pair_modes

pytestmark = pytest.mark.analysis


def _identity_mass(n: int, comm) -> PETSc.Mat:
    M = PETSc.Mat().createAIJ([n, n], comm=comm)
    M.setUp()
    diag = PETSc.Vec().createMPI(n, comm=comm)
    diag.set(1.0)
    M.setDiagonal(diag, addv=PETSc.InsertMode.INSERT_VALUES)
    diag.destroy()
    M.assemble()
    return M


def test_mac_mass_identity():
    _require_single_rank()
    vec = np.array([1.0, 2.0, -1.0], dtype=float)
    M = _identity_mass(len(vec), MPI.COMM_WORLD)

    value = mac_mass(vec, vec, M)
    assert value == pytest.approx(1.0, abs=MAC_TOL)

    M.destroy()


def test_mac_mass_zero_norm_returns_eps():
    _require_single_rank()
    zero = np.zeros(3)
    v = np.array([1.0, 0.0, 0.0])
    M = _identity_mass(3, MPI.COMM_WORLD)

    value = mac_mass(zero, v, M)
    assert value == pytest.approx(MAC_TOL, rel=1e-6, abs=EPS_TOL)

    M.destroy()


def test_pair_modes_recovers_permutation():
    _require_single_rank()
    gdim = 1
    nodes = 3
    ref_vecs = np.array([
        [[1.0], [0.0], [0.0]],
        [[0.0], [1.0], [0.0]],
    ])
    samp_vecs = np.array([
        [[0.0], [1.0], [0.0]],
        [[1.0], [0.0], [0.0]],
    ])
    ref_vals = np.array([1.0, 2.0])
    samp_vals = np.array([2.0, 1.0])

    M = _identity_mass(nodes * gdim, MPI.COMM_WORLD)
    perm = pair_modes(ref_vecs, ref_vals, samp_vecs, samp_vals, M, mac_cut=0.5)

    assert perm.tolist() == [1, 0]
    M.destroy()


def test_pair_modes_marks_low_mac_unpaired():
    _require_single_rank()
    M = _identity_mass(3, MPI.COMM_WORLD)

    ref_vecs = np.array([[[1.0], [0.0], [0.0]]])
    samp_vecs = np.array([[[0.0], [1.0], [0.0]]])
    ref_vals = np.array([1.0])
    samp_vals = np.array([1.0])

    perm = pair_modes(ref_vecs, ref_vals, samp_vecs, samp_vals, M, mac_cut=0.9)
    assert perm.tolist() == [-1]
    M.destroy()


def test_pair_modes_handles_extra_sample_modes():
    _require_single_rank()
    M = _identity_mass(3, MPI.COMM_WORLD)

    ref_vecs = np.array([[[1.0], [0.0], [0.0]]])
    samp_vecs = np.array([
        [[1.0], [0.0], [0.0]],
        [[0.0], [1.0], [0.0]],
    ])
    ref_vals = np.array([1.0])
    samp_vals = np.array([1.0, 2.0])

    perm = pair_modes(ref_vecs, ref_vals, samp_vecs, samp_vals, M, mac_cut=0.2)
    assert perm.tolist() == [0]

    M.destroy()


# ======================== Anatomy analyser tests ========================

from conftest import (
    ABS_TOL,
    ANGLE_TOL,
    DISPLACEMENT_TOL,
    MAPPING_ERR_TOL,
    SHEAR_TOL,
    NEG_TOL,
    DIST_MAX_TOL,
    REL_TOL,
)
from anatomy_analyser import sij_relative_angles
import anatomy_analyser as aa

def random_proper_rotation(rng):
    # random orthonormal via QR + det=+1
    A = rng.standard_normal((3,3))
    Q, _ = np.linalg.qr(A)
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1.0
    return Q


@pytest.fixture
def rng():
    return np.random.default_rng(42)


# Require real 'ants' from the configured conda environment; no stubs/monkeypatching.


def test_kabsch_recovers_rigid(rng):
    P = rng.standard_normal((300, 3))
    R_true = random_proper_rotation(rng)
    t_true = rng.standard_normal(3) * 5.0
    Q = (R_true @ P.T).T + t_true
    R_fit, t_fit = aa._rigid_from_kabsch(P, Q)
    # Orthogonality + det + mapping error
    assert np.allclose(R_fit @ R_fit.T, np.eye(3), atol=ABS_TOL)
    assert np.isclose(np.linalg.det(R_fit), 1.0, atol=ABS_TOL)
    err = np.mean(np.linalg.norm((R_fit @ P.T).T + t_fit - Q, axis=1))
    assert err < MAPPING_ERR_TOL


def test_displacement_world_equals_sac_formula(rng):
    R_sac = random_proper_rotation(rng)
    R_ili = random_proper_rotation(rng)
    t_sac = rng.standard_normal(3)
    t_ili = rng.standard_normal(3)
    p = rng.standard_normal(3)

    R_rel, t_rel = aa._compose_relative(R_ili, t_ili, R_sac, t_sac)

    d_world_A = (R_ili - R_sac) @ p + (t_ili - t_sac)
    d_sac = (R_rel - np.eye(3)) @ p + t_rel
    d_world_B = R_sac @ d_sac

    assert np.allclose(d_world_A, d_world_B, atol=DISPLACEMENT_TOL)


def test_chunked_min_dists_nonnegative(rng):
    A = rng.standard_normal((100, 3))
    B = A + rng.normal(scale=1e-12, size=A.shape)
    d, idx = aa._chunked_min_dists(A, B)
    assert np.all(d >= -NEG_TOL)
    assert float(np.max(d)) < DIST_MAX_TOL


def test_kabsch_returns_proper_rotation(rng):
    P = rng.standard_normal((50,3))
    Q = rng.standard_normal((50,3))
    R = aa._kabsch(P, Q)
    assert np.allclose(R.T @ R, np.eye(3), atol=ABS_TOL)
    assert np.isclose(np.linalg.det(R), 1.0, atol=ABS_TOL)


class DummyPart:
    def __init__(self, points: np.ndarray):
        pts = np.asarray(points, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError("Points must be of shape (N, 3)")
        self.points = pts

    @property
    def n_points(self) -> int:
        return self.points.shape[0]

class DummyTemplate:
    def __init__(self, left: np.ndarray, right: np.ndarray, sacrum: np.ndarray):
        self._left = DummyPart(left)
        self._right = DummyPart(right)
        self._sacrum = DummyPart(sacrum)
        empty = DummyPart(np.empty((0, 3)))
        self._others = (empty, empty)
        self.points = np.vstack([self._left.points, self._right.points, self._sacrum.points])

    def split_bodies(self):
        return self._left, self._right, self._sacrum, *self._others


def _block_points(center: tuple[float, float, float], half_sizes=(0.2, 0.15, 0.1)) -> np.ndarray:
    cx, cy, cz = center
    hx, hy, hz = half_sizes
    grid = np.stack(
        np.meshgrid(
            np.linspace(-hx, hx, 4),
            np.linspace(-hy, hy, 4),
            np.linspace(-hz, hz, 4),
            indexing="ij",
        ),
        axis=-1,
    ).reshape(-1, 3)
    return grid + np.array([cx, cy, cz])


@pytest.fixture
def reference_template():
    left = _block_points(center=(-1.2, 0.0, 0.0))
    right = _block_points(center=(1.2, 0.0, 0.0))
    sacrum = _block_points(center=(0.0, 0.0, 0.0), half_sizes=(0.3, 0.2, 0.2))
    return DummyTemplate(left, right, sacrum)


def _rotate_x(points: np.ndarray, angle_deg: float, origin: np.ndarray) -> np.ndarray:
    theta = np.deg2rad(angle_deg)
    R = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(theta), -np.sin(theta)],
            [0.0, np.sin(theta), np.cos(theta)],
        ],
        dtype=float,
    )
    shifted = points - origin
    return (R @ shifted.T).T + origin


def test_sij_relative_angles_identity(reference_template):
    template_def = DummyTemplate(
        reference_template._left.points.copy(),
        reference_template._right.points.copy(),
        reference_template._sacrum.points.copy(),
    )

    result = sij_relative_angles(reference_template, template_def, sij_gap_mm=100.0, trim_frac=0.0)

    for side in ("SIJ_left", "SIJ_right"):
        angles = result[side]["angles_deg"]
        disps = result[side]["displacement_mm"]
        assert np.allclose([angles["alpha_x"], angles["beta_y"], angles["gamma_z"]], 0.0, atol=ANGLE_TOL)
        assert np.allclose([disps["dx_ML"], disps["dy_AP"], disps["dz_CC"]], 0.0, atol=DISPLACEMENT_TOL)


def test_sij_relative_angles_detects_left_rotation(reference_template):
    rotation_deg = 15.0
    left_center = reference_template._left.points.mean(axis=0)
    rotated_left = _rotate_x(reference_template._left.points, rotation_deg, left_center)
    template_def = DummyTemplate(rotated_left, reference_template._right.points.copy(), reference_template._sacrum.points.copy())

    result = sij_relative_angles(reference_template, template_def, sij_gap_mm=100.0, trim_frac=0.0)

    left_angles = result["SIJ_left"]["angles_deg"]
    right_angles = result["SIJ_right"]["angles_deg"]

    assert left_angles["alpha_x"] == pytest.approx(rotation_deg, abs=0.5)
    assert right_angles["alpha_x"] == pytest.approx(0.0, abs=0.5)
    # Obliquity and axial angles remain near zero for pure x-rotation
    assert left_angles["beta_y"] == pytest.approx(0.0, abs=0.5)
    assert left_angles["gamma_z"] == pytest.approx(0.0, abs=0.5)


def test_sij_relative_angles_pure_translation(reference_template):
    translation = np.array([0.0, 0.6, -0.25])
    template_def = DummyTemplate(
        reference_template._left.points + translation,
        reference_template._right.points.copy(),
        reference_template._sacrum.points.copy(),
    )

    result = sij_relative_angles(reference_template, template_def, sij_gap_mm=100.0, trim_frac=0.0)

    left_disp = result["SIJ_left"]["displacement_mm"]
    right_disp = result["SIJ_right"]["displacement_mm"]

    left_vec = np.array([left_disp["dx_ML"], left_disp["dy_AP"], left_disp["dz_CC"]])
    assert np.linalg.norm(left_vec) == pytest.approx(
        np.linalg.norm(translation), rel=REL_TOL, abs=DISPLACEMENT_TOL
    )

    assert np.allclose(
        [right_disp["dx_ML"], right_disp["dy_AP"], right_disp["dz_CC"]],
        0.0,
        atol=DISPLACEMENT_TOL,
    )


def test_sij_relative_angles_mismatched_points_raises(reference_template):
    fewer_left = reference_template._left.points[:-5]
    template_def = DummyTemplate(fewer_left, reference_template._right.points.copy(), reference_template._sacrum.points.copy())

    with pytest.raises(ValueError, match="Point counts must match"):
        sij_relative_angles(reference_template, template_def)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
