"""Geometric and I/O utilities for pelvic morphology analysis.

Provides coordinate transformations (ANTs warping), geometry primitives
(symmetry plane detection, angles, projections), and mesh manipulation
(PyVista operations, morphing, validation).

No DOLFINx dependencies - pure NumPy/PyVista for portability.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import pyvista as pv

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def warp(points: np.ndarray, warp_path: str | Path, affine_path: str | Path) -> np.ndarray:
    """Apply ANTs forward transforms (nonlinear warp + affine) to 3D points.

    Parameters
    ----------
    points : np.ndarray
        Coordinates array (n_points, 3) in XYZ order
    warp_path : str | Path
        Path to ANTs nonlinear displacement field (.nii.gz or .mat)
    affine_path : str | Path
        Path to ANTs affine matrix file (.mat)

    Returns
    -------
    np.ndarray
        Transformed coordinates (n_points, 3)
        
    Notes
    -----
    Uses ants.apply_transforms_to_points with forward transforms (no inversion).
    Requires ants and pandas packages.
    """
    import ants  # type: ignore
    import pandas as pd  # type: ignore

    df_points = pd.DataFrame(np.asarray(points, dtype=float), columns=["x", "y", "z"])
    transformed = ants.apply_transforms_to_points(
        dim=3,
        points=df_points,
        transformlist=[str(warp_path), str(affine_path)],
        whichtoinvert=[False, False],
    )
    return np.asarray(transformed.values, dtype=float)


def load_json_points(json_path: str | Path) -> np.ndarray:
    """Load 3D point coordinates from 3D Slicer markup JSON.
    
    Extracts controlPoints[*].position from first markup in file.
    
    Returns
    -------
    np.ndarray
        Point array (n_points, 3) in XYZ order
    """

    with open(json_path, "r", encoding="utf8") as handle:
        data = json.load(handle)

    control_points = data["markups"][0]["controlPoints"]
    coords = [[p["position"][i] for p in control_points] for i in range(3)]
    return np.asarray(coords, dtype=float).T


# ---------------------------------------------------------------------------
# Geometry primitives
# ---------------------------------------------------------------------------

def angle_between_vectors(vec1: Sequence[float], vec2: Sequence[float]) -> float:
    """Compute angle between two 3D vectors in degrees.
    
    Uses arccos(v1·v2 / |v1||v2|). Raises ValueError if either vector is zero.
    """

    v1 = np.asarray(vec1, dtype=float)
    v2 = np.asarray(vec2, dtype=float)
    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)
    if v1_norm == 0.0 or v2_norm == 0.0:
        raise ValueError("Input vectors must be non-zero.")

    cos_angle = float(np.dot(v1, v2) / (v1_norm * v2_norm))
    cos_angle = float(np.clip(cos_angle, -1.0, 1.0))
    angle_rad = float(np.arccos(cos_angle))
    return float(np.degrees(angle_rad))


def project_points_onto_plane(points: np.ndarray, plane: pv.PolyData) -> np.ndarray:
    """Project points onto a PyVista plane-like PolyData object."""

    plane_normal = np.asarray(plane.active_normals.mean(0), dtype=float)
    plane_origin = np.asarray(plane.center, dtype=float)
    return project_points_onto_plane_manual(points, plane_origin, plane_normal)


def project_points_onto_plane_manual(
    points: np.ndarray,
    plane_origin: Sequence[float],
    plane_normal: Sequence[float],
) -> np.ndarray:
    """Project points onto a plane defined by origin and normal."""

    pts = np.asarray(points, dtype=float)
    normal = np.asarray(plane_normal, dtype=float)
    origin = np.asarray(plane_origin, dtype=float)

    norm = np.linalg.norm(normal)
    if norm == 0.0:
        raise ValueError("Plane normal must be non-zero.")
    n_hat = normal / norm

    vecs = pts - origin
    dist = vecs @ n_hat
    proj = dist[:, np.newaxis] * n_hat
    return pts - proj


def create_polar_system(normal: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    """Create an orthonormal basis inside a plane given its normal."""

    z_axis = np.asarray(normal, dtype=float)
    if np.linalg.norm(z_axis) == 0.0:
        raise ValueError("Normal vector must be non-zero.")
    z_axis /= np.linalg.norm(z_axis)

    reference = np.array([1.0, 0.0, 0.0], dtype=float)
    if np.allclose(z_axis, reference):
        reference = np.array([0.0, 1.0, 0.0], dtype=float)

    x_axis = np.cross(z_axis, reference)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    return x_axis, y_axis


def cartesian_to_polar(
    point: Sequence[float],
    center: Sequence[float],
    x_axis: Sequence[float],
    y_axis: Sequence[float],
) -> Tuple[float, float]:
    """Convert Cartesian coordinates to polar coordinates in a plane."""

    point_centered = np.asarray(point, dtype=float) - np.asarray(center, dtype=float)
    x_hat = np.asarray(x_axis, dtype=float)
    y_hat = np.asarray(y_axis, dtype=float)

    projection = np.dot(point_centered, x_hat) * x_hat + np.dot(point_centered, y_hat) * y_hat
    radius = float(np.linalg.norm(projection))
    theta = float(np.arctan2(np.dot(projection, y_hat), np.dot(projection, x_hat)))
    return radius, float(np.degrees(theta))


def fit_plane(points: np.ndarray) -> np.ndarray:
    """Fit a plane to a cloud of points via least squares and return its normal."""

    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("Points must have shape (N, 3).")

    A = np.c_[pts[:, 0], pts[:, 1], np.ones(pts.shape[0])]
    C, *_ = np.linalg.lstsq(A, pts[:, 2], rcond=None)
    normal = np.array([C[0], C[1], -1.0], dtype=float)
    return normal / np.linalg.norm(normal)


def find_symmetry_plane(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute a symmetry plane from a point cloud using PCA."""

    pts = np.asarray(points, dtype=float)
    if pts.size == 0:
        raise ValueError("Cannot compute symmetry plane of an empty point set.")

    origin = pts.mean(axis=0)
    centered = pts - origin
    cov = np.cov(centered, rowvar=False) + np.eye(3) * 1e-12
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    normal = eigenvectors[:, int(np.argmax(eigenvalues))]
    if normal[0] < 0:
        normal = -normal
    return normal, origin


def split_points_by_plane(
    points: np.ndarray,
    normal: Sequence[float],
    origin: Sequence[float],
    *,
    tolerance: float = 1e-9,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split indices of points by their signed distance to a plane."""

    pts = np.asarray(points, dtype=float)
    if pts.size == 0:
        logger.warning("split_points_by_plane received an empty point array")
        return (np.array([], dtype=int),) * 3

    normal = np.asarray(normal, dtype=float)
    origin = np.asarray(origin, dtype=float)
    norm = np.linalg.norm(normal)
    if norm == 0.0:
        raise ValueError("Normal vector must be non-zero.")

    signed_dist = (pts - origin) @ (normal / norm)
    idx_pos = np.where(signed_dist > tolerance)[0]
    idx_neg = np.where(signed_dist < -tolerance)[0]
    idx_zero = np.where(np.abs(signed_dist) <= tolerance)[0]

    logger.debug(
        "Split points: side1=%d side2=%d on_plane=%d",
        idx_pos.size,
        idx_neg.size,
        idx_zero.size,
    )
    return idx_pos, idx_neg, idx_zero


# ---------------------------------------------------------------------------
# Mechanics helpers
# ---------------------------------------------------------------------------

def compute_moment(
    eigenvalue: float,
    eigenvector: np.ndarray,
    node_positions: np.ndarray,
    reference_point: Sequence[float],
) -> np.ndarray:
    """Compute the resultant moment of an eigenmode about a reference point."""

    if eigenvector.shape != node_positions.shape:
        raise ValueError(
            f"Eigenvector shape {eigenvector.shape} does not match node positions {node_positions.shape}"
        )

    forces = eigenvalue * eigenvector
    r = node_positions - np.asarray(reference_point, dtype=float)
    return np.sum(np.cross(r, forces), axis=0)


def compute_force(eigenvalue: float, eigenvector: np.ndarray) -> np.ndarray:
    """Compute the resultant force of an eigenmode."""

    return np.sum(eigenvalue * eigenvector, axis=0)


def ALE(mesh, u) -> None:  # type: ignore[no-untyped-def]
    """Apply an ALE update to a mesh using a displacement function."""

    gdim = mesh.geometry.dim
    expected = mesh.geometry.x.shape[0] * gdim
    if u.x.array.size != expected:
        raise ValueError(
            f"Displacement array size {u.x.array.size} does not match expected {expected}"
        )

    deformation = u.x.array.reshape((-1, gdim))
    mesh.geometry.x[:, :gdim] += deformation


def ensure_nonnegative(field: np.ndarray, min_value: float) -> np.ndarray:
    """Clamp array values so they are not below ``min_value``."""

    clipped = np.array(field, copy=True)
    clipped[clipped <= min_value] = min_value
    return clipped
