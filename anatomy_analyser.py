
"""Sacroiliac joint (SIJ) kinematic analysis from deformed pelvis geometry.

Computes relative rotation (Euler angles) and translation of ilium with respect
to sacrum in pelvis-aligned coordinate frame.

Algorithm:
1. Establish pelvic reference frame (x_ML, y_AP, z_CC) from symmetry plane + PCA
2. Remove global affine deformation via sacrum reference fit (robust pinv)
3. Define SIJ contact regions (ROI) within gap tolerance of opposing bone surfaces
4. Rigid registration (Kabsch) on each side's ROI (sacrum, ilium) with outlier trimming
5. Compute relative transformation T_rel = T_sac^{-1} · T_ilium
6. Extract Euler XYZ angles (deg) and contact point displacement (mm) in pelvic frame

Output units match input geometry (typically mm for clinical data).
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pyvista as pv

sys.path.insert(0, str(Path(__file__).resolve().parent))

logger = logging.getLogger(__name__)

# Constants
EPS = 1e-12
EUL_TOL = 1e-6
CHUNK_SIZE = 2000

def _normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, float).reshape(-1)
    n = np.linalg.norm(v)
    if n == 0:
        raise ValueError("Zero-length vector.")
    return v / n

def _pca_axis_in_plane(points_xyz: np.ndarray, x_hat: np.ndarray, origin: np.ndarray) -> np.ndarray:
    """Find dominant direction in plane perpendicular to x_hat via PCA after projection."""
    P = np.ascontiguousarray(points_xyz, float) - origin
    P_proj = P - np.outer(P @ x_hat, x_hat)
    X = P_proj - P_proj.mean(axis=0)
    if X.shape[0] < 3:
        z_like = np.array([0.0, 0.0, 1.0], float)
    else:
        _, _, Vt = np.linalg.svd(X, full_matrices=False)
        z_like = Vt[0]
    z_like -= (z_like @ x_hat) * x_hat
    return _normalize(z_like)

def _build_pelvis_frame(points_all: np.ndarray, ml_normal: np.ndarray, origin: np.ndarray) -> np.ndarray:
    """Build right-handed pelvic frame P_frame with columns [x_ML, y_AP, z_CC]."""
    x_hat = _normalize(ml_normal)
    z_like = _pca_axis_in_plane(points_all, x_hat, origin)
    z_hat = _normalize(z_like - (z_like @ x_hat) * x_hat)
    y_hat = _normalize(np.cross(z_hat, x_hat))
    z_hat = _normalize(np.cross(x_hat, y_hat))
    return np.column_stack([x_hat, y_hat, z_hat])

def _ensure_lr(P_frame: np.ndarray, left_pts: np.ndarray, right_pts: np.ndarray, origin: np.ndarray) -> np.ndarray:
    """Ensure x_ML direction: left -> right (preserves right-handedness)."""
    P = P_frame.copy()
    x = P[:, 0]
    l = left_pts.mean(axis=0) - origin
    r = right_pts.mean(axis=0) - origin
    if (l @ x) > (r @ x):
        P[:, 0] *= -1.0
        P[:, 1] *= -1.0
    return P

def _kabsch(P_xyz: np.ndarray, Q_xyz: np.ndarray) -> np.ndarray:
    """Kabsch: return R such that Q ≈ R @ P. Correct composition is R = V @ U^T when C = P^T Q."""
    P = np.ascontiguousarray(P_xyz, float)
    Q = np.ascontiguousarray(Q_xyz, float)
    if P.shape != Q.shape or P.shape[1] != 3:
        raise ValueError("P and Q must be Nx3 with identical order")
    Pc = P - P.mean(axis=0)
    Qc = Q - Q.mean(axis=0)
    C = Pc.T @ Qc
    U, _, Vt = np.linalg.svd(C)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1.0
        R = Vt.T @ U.T
    return R

def _rigid_from_kabsch(P_xyz: np.ndarray, Q_xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (R, t) such that Q ≈ R @ P + t via Kabsch rotation and center difference."""
    R = _kabsch(P_xyz, Q_xyz)
    t = Q_xyz.mean(axis=0) - R @ P_xyz.mean(axis=0)
    return R, t

def _rigid_from_kabsch_trimmed(P_xyz: np.ndarray, Q_xyz: np.ndarray, trim_frac: float = 0.1, iters: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """Trimmed Kabsch: suppress outliers via iterative trimming and refitting."""
    if not (0.0 <= trim_frac < 0.5):
        raise ValueError("trim_frac must be in [0, 0.5)")
    P = np.ascontiguousarray(P_xyz, float)
    Q = np.ascontiguousarray(Q_xyz, float)
    n = P.shape[0]
    keep = np.ones(n, dtype=bool)
    for _ in range(max(1, iters)):
        R, t = _rigid_from_kabsch(P[keep], Q[keep])
        res = np.linalg.norm((R @ P.T).T + t - Q, axis=1)
        if trim_frac <= EPS:
            break
        k = int(np.floor((1.0 - trim_frac) * n))
        if k < 3:
            break
        idx = np.argpartition(res, kth=k-1)[:k]
        keep[:] = False
        keep[idx] = True
    R, t = _rigid_from_kabsch(P[keep], Q[keep])
    return R, t

def _compose_relative(R_a: np.ndarray, t_a: np.ndarray,
                      R_b: np.ndarray, t_b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """T_rel = T_b^{-1} · T_a → 'a in b-frame'."""
    R_rel = R_b.T @ R_a
    t_rel = R_b.T @ (t_a - t_b)
    return R_rel, t_rel

def _euler_xyz_in_frame(R: np.ndarray, P_frame: np.ndarray) -> Tuple[float, float, float]:
    """Euler XYZ angles (deg) in pelvic frame P_frame with singularity handling."""
    Rloc = P_frame.T @ R @ P_frame
    r20 = float(np.clip(Rloc[2, 0], -1.0, 1.0))
    r21 = float(np.clip(Rloc[2, 1], -1.0, 1.0))
    r22 = float(np.clip(Rloc[2, 2], -1.0, 1.0))
    r10 = float(np.clip(Rloc[1, 0], -1.0, 1.0))
    r00 = float(np.clip(Rloc[0, 0], -1.0, 1.0))
    if np.isclose(abs(r20), 1.0, atol=EUL_TOL):
        beta = -np.pi/2 if r20 > 0 else np.pi/2
        a = float(np.clip(-Rloc[0, 1], -1.0, 1.0))
        b = float(np.clip(Rloc[1, 1], -1.0, 1.0))
        alpha = np.arctan2(a, b)
        gamma = 0.0
    else:
        alpha = np.arctan2(r21, r22)
        beta = np.arcsin(-r20)
        gamma = np.arctan2(r10, r00)
    return (float(np.degrees(alpha)), float(np.degrees(beta)), float(np.degrees(gamma)))

def _vec_in_frame(v: np.ndarray, P_frame: np.ndarray) -> np.ndarray:
    """Express vector in pelvic frame."""
    return P_frame.T @ np.asarray(v, float).reshape(3)

# ----- affine removal on sacrum -----

def _affine_fit(P_xyz: np.ndarray, Q_xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Find affine map (A, b) such that Q ≈ A @ P + b via least squares. Returns A(3x3), b(3,)."""
    P = np.ascontiguousarray(P_xyz, float)
    Q = np.ascontiguousarray(Q_xyz, float)
    if P.shape != Q.shape or P.shape[1] != 3:
        raise ValueError("P and Q must be Nx3 with identical order")
    N = P.shape[0]
    X = np.hstack([P, np.ones((N, 1))])
    M, *_ = np.linalg.lstsq(X, Q, rcond=None)
    A = M[:3, :].T
    b = M[3, :]
    return A, b

def _apply_inv_affine(A: np.ndarray, b: np.ndarray, Q_xyz: np.ndarray) -> np.ndarray:
    """Q' = A^{-1} @ (Q - b) — transform def→ref space (remove global affinity via robust pinv)."""
    Qi = np.ascontiguousarray(Q_xyz, float)
    Ainv = np.linalg.pinv(A)
    return (Ainv @ (Qi - b).T).T

# ----- vzdálenosti / ROI -----

def _chunked_min_dists(A: np.ndarray, B: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """For each point in A, find nearest point in B. Returns (distance, argmin_idx)."""
    m, n = A.shape[0], B.shape[0]
    if m == 0 or n == 0:
        return np.zeros(m, float), np.full(m, -1, int)
    min_d2 = np.full(m, np.inf)
    arg = np.full(m, -1, int)
    B2 = np.sum(B * B, axis=1)
    for i in range(0, m, CHUNK_SIZE):
        Ai = A[i:i+CHUNK_SIZE]
        Ai2 = np.sum(Ai * Ai, axis=1, keepdims=True)
        d2 = Ai2 + B2[None, :] - 2.0 * (Ai @ B.T)
        jmin = np.argmin(d2, axis=1)
        min_d2[i:i+CHUNK_SIZE] = d2[np.arange(d2.shape[0]), jmin]
        arg[i:i+CHUNK_SIZE] = jmin
    return np.sqrt(np.maximum(min_d2, 0.0)), arg

def _sided_mask(pts: np.ndarray, P_frame: np.ndarray, origin: np.ndarray, side: str) -> np.ndarray:
    """Boolean mask by side: left (x<0), right (x>0) in pelvic frame."""
    x_hat = P_frame[:, 0]
    side_sign = -1.0 if side.lower().startswith('l') else 1.0
    rel = (pts - origin) @ x_hat
    return (rel * side_sign) > 0

def _sij_roi_masks(sac_pts: np.ndarray, ili_pts: np.ndarray,
                   P_frame: np.ndarray, origin: np.ndarray,
                   side: str, gap_mm: float, min_pts: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return boolean ROI masks for sacrum and ilium on given side."""
    sac_side_mask = _sided_mask(sac_pts, P_frame, origin, side)
    ili_side_mask = _sided_mask(ili_pts, P_frame, origin, side)
    sac_side = sac_pts[sac_side_mask]
    ili_side = ili_pts[ili_side_mask]

    if sac_side.shape[0] == 0:
        sac_side = sac_pts
        sac_side_mask = np.ones(sac_pts.shape[0], bool)
    if ili_side.shape[0] == 0:
        ili_side = ili_pts
        ili_side_mask = np.ones(ili_pts.shape[0], bool)

    # Sacrum (side) -> Ilium (side)
    d_sac, _ = _chunked_min_dists(sac_side, ili_side)
    sac_roi_side_mask = d_sac <= gap_mm
    sac_roi_idx_global = np.where(sac_side_mask)[0][sac_roi_side_mask]
    if sac_roi_idx_global.size < min_pts and sac_side.shape[0] > 0:
        k = min(min_pts, sac_side.shape[0])
        idx_local = np.argpartition(d_sac, kth=k-1)[:k]
        sac_roi_idx_global = np.where(sac_side_mask)[0][idx_local]

    sac_roi_mask = np.zeros(sac_pts.shape[0], dtype=bool)
    if sac_roi_idx_global.size > 0:
        sac_roi_mask[sac_roi_idx_global] = True

    # Ilium (side) -> Sacrum (side)
    d_ili, _ = _chunked_min_dists(ili_side, sac_side)
    ili_roi_side_mask = d_ili <= gap_mm
    ili_roi_idx_global = np.where(ili_side_mask)[0][ili_roi_side_mask]
    if ili_roi_idx_global.size < min_pts and ili_side.shape[0] > 0:
        k = min(min_pts, ili_side.shape[0])
        idx_local = np.argpartition(d_ili, kth=k-1)[:k]
        ili_roi_idx_global = np.where(ili_side_mask)[0][idx_local]

    ili_roi_mask = np.zeros(ili_pts.shape[0], dtype=bool)
    if ili_roi_idx_global.size > 0:
        ili_roi_mask[ili_roi_idx_global] = True

    return sac_roi_mask, ili_roi_mask

# ---------- main ----------

def _check_pairwise_counts(left0, left1, right0, right1, sac0, sac1):
    """Validate point counts match between reference and deformed states."""
    errs = []
    if left0.n_points != left1.n_points:
        errs.append(f"left ilium: {left0.n_points} (ref) vs {left1.n_points} (def)")
    if right0.n_points != right1.n_points:
        errs.append(f"right ilium: {right0.n_points} (ref) vs {right1.n_points} (def)")
    if sac0.n_points != sac1.n_points:
        errs.append(f"sacrum: {sac0.n_points} (ref) vs {sac1.n_points} (def)")
    if errs:
        raise ValueError("Point counts must match pairwise:\n  - " + "\n  - ".join(errs))

def sij_relative_angles(template_ref, template_def, sij_gap_mm: float = 5.0, trim_frac: float = 0.1) -> dict:
    """Compute SIJ kinematics: rotation (Euler XYZ deg) and translation (mm) in pelvic frame.
    
    Parameters:
        sij_gap_mm: Distance threshold for ROI inclusion (mm)
        trim_frac: Outlier fraction for trimmed rigid fit (0..0.5)
    
    Returns:
        Dictionary with SIJ_left and SIJ_right, each containing angles_deg and displacement_mm
    """
    from simulation.utils import find_symmetry_plane

    logger.debug("SIJ relative angles: gap=%.2f mm, trim=%.2f", sij_gap_mm, trim_frac)

    # Pelvic frame from reference
    ml_normal, origin = find_symmetry_plane(template_ref.points)
    origin = np.asarray(origin, float).reshape(3)
    P_frame = _build_pelvis_frame(template_ref.points, ml_normal, origin)

    # Split bodies (expects 1:1 node correspondence)
    left0, right0, sac0, _, _ = template_ref.split_bodies()
    left1, right1, sac1, _, _ = template_def.split_bodies()
    _check_pairwise_counts(left0, left1, right0, right1, sac0, sac1)
    P_frame = _ensure_lr(P_frame, left0.points, right0.points, origin)

    # Remove global affine via sacrum fit, apply inverse to deformed
    A_sac, b_sac = _affine_fit(sac0.points, sac1.points)
    sac1_aff = _apply_inv_affine(A_sac, b_sac, sac1.points)
    left1_aff = _apply_inv_affine(A_sac, b_sac, left1.points)
    right1_aff = _apply_inv_affine(A_sac, b_sac, right1.points)

    # Define ROI in reference (left/right)
    sacL_mask, iliL_mask = _sij_roi_masks(sac0.points, left0.points, P_frame, origin, 'left', sij_gap_mm, min_pts=80)
    sacR_mask, iliR_mask = _sij_roi_masks(sac0.points, right0.points, P_frame, origin, 'right', sij_gap_mm, min_pts=80)

    if sacL_mask.sum() == 0 or iliL_mask.sum() == 0:
        raise ValueError("Empty left SIJ ROI")
    if sacR_mask.sum() == 0 or iliR_mask.sum() == 0:
        raise ValueError("Empty right SIJ ROI")

    # Rigid fits on ROI only (trimmed)
    R_sac_L, t_sac_L = _rigid_from_kabsch_trimmed(sac0.points[sacL_mask], sac1_aff[sacL_mask], trim_frac=trim_frac)
    R_ili_L, t_ili_L = _rigid_from_kabsch_trimmed(left0.points[iliL_mask], left1_aff[iliL_mask], trim_frac=trim_frac)

    R_sac_R, t_sac_R = _rigid_from_kabsch_trimmed(sac0.points[sacR_mask], sac1_aff[sacR_mask], trim_frac=trim_frac)
    R_ili_R, t_ili_R = _rigid_from_kabsch_trimmed(right0.points[iliR_mask], right1_aff[iliR_mask], trim_frac=trim_frac)

    # Relative rigid (ilium in sacrum frame)
    R_rel_L, t_rel_L = _compose_relative(R_ili_L, t_ili_L, R_sac_L, t_sac_L)
    R_rel_R, t_rel_R = _compose_relative(R_ili_R, t_ili_R, R_sac_R, t_sac_R)

    # SIJ contact point = sacral ROI centroid
    p_L = sac0.points[sacL_mask].mean(axis=0)
    p_R = sac0.points[sacR_mask].mean(axis=0)

    # Displacement in world frame
    dL_world = (R_ili_L - R_sac_L) @ p_L + (t_ili_L - t_sac_L)
    dR_world = (R_ili_R - R_sac_R) @ p_R + (t_ili_R - t_sac_R)

    # Angles + displacement in pelvic frame
    eul_L = _euler_xyz_in_frame(R_rel_L, P_frame)
    eul_R = _euler_xyz_in_frame(R_rel_R, P_frame)
    dpel_L = _vec_in_frame(dL_world, P_frame)
    dpel_R = _vec_in_frame(dR_world, P_frame)

    return {
        "SIJ_left": {
            "angles_deg": {"alpha_x": eul_L[0], "beta_y": eul_L[1], "gamma_z": eul_L[2]},
            "displacement_mm": {"dx_ML": float(dpel_L[0]), "dy_AP": float(dpel_L[1]), "dz_CC": float(dpel_L[2])},
        },
        "SIJ_right": {
            "angles_deg": {"alpha_x": eul_R[0], "beta_y": eul_R[1], "gamma_z": eul_R[2]},
            "displacement_mm": {"dx_ML": float(dpel_R[0]), "dy_AP": float(dpel_R[1]), "dz_CC": float(dpel_R[2])},
        },
    }

if __name__ == '__main__':
    template = pv.read('data/pelvic.vtk')
    deformed = pv.read('data/pelvic.vtk')
    out = sij_relative_angles(template, deformed, sij_gap_mm=5.0, trim_frac=0.1)
    import pprint
    pprint.pprint(out)
