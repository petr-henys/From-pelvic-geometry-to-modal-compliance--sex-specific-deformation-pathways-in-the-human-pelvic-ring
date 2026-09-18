"""Kinematic deformation mechanism metrics for pelvic eigenmodes.

Implements:
1. Level A: Exact local tensor profiles (normal ML/AP/CC, shear ML-AP/ML-CC/AP-CC,
   volumetric/deviatoric splits, principal strains).
2. Level B: Regional orthogonal mechanism decomposition (axial stretching/compression,
   bending, uniform transverse shear, circulatory twist, and residual).
3. Subspace-level invariant aggregation across near-degenerate clusters.
"""
from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# Level A: Exact Local Tensor Profile
# =============================================================================

def transform_strain_tensor(
    strain: np.ndarray,
    R: np.ndarray,
) -> np.ndarray:
    """Rotate strain tensors into an orthonormal anatomical frame.

    Parameters
    ----------
    strain : ndarray of shape (..., 3, 3)
        Symmetric strain tensors.
    R : ndarray of shape (3, 3)
        Orthonormal rotation matrix whose columns are anatomical unit vectors
        [e_ML, e_AP, e_CC].

    Returns
    -------
    strain_prime : ndarray of shape (..., 3, 3)
        Rotated strain tensors eps' = R^T eps R.
    """
    R = np.asarray(R, dtype=float)
    return np.einsum("ki,...kl,lj->...ij", R, strain, R)


def compute_level_a_profile(
    strain: np.ndarray,
    weights: np.ndarray,
    admissible_mask: np.ndarray | None = None,
    tissue_mask: np.ndarray | None = None,
    eps_threshold: float = 1e-15,
    allow_invalid_jacobians: bool = True,
) -> dict[str, float]:
    """Compute exact Level A volume-weighted strain tensor fractions.

    The total quadratic strain norm is:
        D = sum_c w_c (eps_c : eps_c)
          = sum_c w_c [ (eps_11)^2 + (eps_22)^2 + (eps_33)^2
                        + 2*(eps_12)^2 + 2*(eps_13)^2 + 2*(eps_23)^2 ]

    Parameters
    ----------
    strain : ndarray of shape (num_cells, 3, 3)
        Strain tensors in the desired coordinate system.
    weights : ndarray of shape (num_cells,)
        Cell integration weights w_c = J_c * V_ref,c.
    admissible_mask : ndarray of shape (num_cells,), optional
        Boolean mask of geometrically admissible cells (e.g. J > 0).
    tissue_mask : ndarray of shape (num_cells,), optional
        Boolean mask selecting a specific tissue domain (e.g. bone, SIJ).
    eps_threshold : float
        Threshold for D below which the profile is considered degenerate.
    allow_invalid_jacobians : bool, default True
        If False, raises ValueError if any w <= 0. If True, filters out w <= 0.

    Returns
    -------
    profile : dict
        Dictionary containing quadratic norm D, 6 component fractions
        (summing to 1.0), volumetric/deviatoric fractions, and cell stats.
    """
    strain = np.asarray(strain, dtype=float)
    weights = np.asarray(weights, dtype=float)

    if not allow_invalid_jacobians and np.any(weights <= 0.0):
        n_bad = int(np.sum(weights <= 0.0))
        raise ValueError(f"Invalid cell volumes / Jacobians encountered: {n_bad} cells have w <= 0.")

    mask = weights > 0.0
    if admissible_mask is not None:
        mask &= np.asarray(admissible_mask, dtype=bool)
    if tissue_mask is not None:
        mask &= np.asarray(tissue_mask, dtype=bool)

    if not np.any(mask):
        return {
            "D_total": 0.0,
            "f_ML": np.nan, "f_AP": np.nan, "f_CC": np.nan,
            "f_ML_AP": np.nan, "f_ML_CC": np.nan, "f_AP_CC": np.nan,
            "f_volumetric": np.nan, "f_deviatoric": np.nan,
            "n_cells": 0, "total_weight": 0.0, "is_valid": False,
        }

    w = weights[mask]
    eps = strain[mask]

    e11 = eps[:, 0, 0]
    e22 = eps[:, 1, 1]
    e33 = eps[:, 2, 2]
    e12 = eps[:, 0, 1]
    e13 = eps[:, 0, 2]
    e23 = eps[:, 1, 2]

    # Quadratic energy/norm components
    D_11 = float(np.sum(w * (e11 ** 2)))
    D_22 = float(np.sum(w * (e22 ** 2)))
    D_33 = float(np.sum(w * (e33 ** 2)))
    D_12 = float(np.sum(2.0 * w * (e12 ** 2)))
    D_13 = float(np.sum(2.0 * w * (e13 ** 2)))
    D_23 = float(np.sum(2.0 * w * (e23 ** 2)))

    D_total = D_11 + D_22 + D_33 + D_12 + D_13 + D_23

    if D_total <= eps_threshold:
        return {
            "D_total": D_total,
            "f_ML": 0.0, "f_AP": 0.0, "f_CC": 0.0,
            "f_ML_AP": 0.0, "f_ML_CC": 0.0, "f_AP_CC": 0.0,
            "f_volumetric": 0.0, "f_deviatoric": 0.0,
            "n_cells": int(np.sum(mask)), "total_weight": float(np.sum(w)),
            "is_valid": False,
        }

    # Volumetric and deviatoric decomposition
    # tr(eps) = e11 + e22 + e33
    # eps_vol = (1/3) * tr(eps) * I
    # ||eps_vol||^2 = (1/3) * tr(eps)^2
    tr_eps = e11 + e22 + e33
    D_vol = float(np.sum(w * (tr_eps ** 2) / 3.0))
    D_dev = D_total - D_vol

    return {
        "D_total": D_total,
        "f_ML": D_11 / D_total,
        "f_AP": D_22 / D_total,
        "f_CC": D_33 / D_total,
        "f_ML_AP": D_12 / D_total,
        "f_ML_CC": D_13 / D_total,
        "f_AP_CC": D_23 / D_total,
        "f_volumetric": D_vol / D_total,
        "f_deviatoric": D_dev / D_total,
        "D_11": D_11,
        "D_22": D_22,
        "D_33": D_33,
        "D_12": D_12,
        "D_13": D_13,
        "D_23": D_23,
        "n_cells": int(np.sum(mask)),
        "total_weight": float(np.sum(w)),
        "is_valid": True,
    }


def compute_subspace_level_a_profile(
    modal_strains: Sequence[np.ndarray],
    weights: np.ndarray,
    admissible_mask: np.ndarray | None = None,
    tissue_mask: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute basis-invariant Level A fractions for a clustered subspace.

    Given modal strain tensors [eps_1, ..., eps_r] corresponding to an
    orthonormal basis of the subspace, the aggregate quadratic norm and
    fractions are summed across the modes:
        bar_D_comp = sum_{j=1}^r D_comp(eps_j)
        bar_D_total = sum_{j=1}^r D_total(eps_j)
        f_comp(Q) = bar_D_comp / bar_D_total

    Because D_comp is quadratic in the displacement field, any orthogonal
    internal basis rotation Q' = Q * O leaves bar_D_comp and f_comp invariant.
    """
    total_D = 0.0
    sum_D_11 = 0.0
    sum_D_22 = 0.0
    sum_D_33 = 0.0
    sum_D_12 = 0.0
    sum_D_13 = 0.0
    sum_D_23 = 0.0

    for strain in modal_strains:
        prof = compute_level_a_profile(strain, weights, admissible_mask, tissue_mask)
        if not prof["is_valid"]:
            continue
        total_D += prof["D_total"]
        sum_D_11 += prof["D_11"]
        sum_D_22 += prof["D_22"]
        sum_D_33 += prof["D_33"]
        sum_D_12 += prof["D_12"]
        sum_D_13 += prof["D_13"]
        sum_D_23 += prof["D_23"]

    if total_D <= 1e-15:
        return {
            "D_total": 0.0,
            "f_ML": np.nan, "f_AP": np.nan, "f_CC": np.nan,
            "f_ML_AP": np.nan, "f_ML_CC": np.nan, "f_AP_CC": np.nan,
            "is_valid": False,
        }

    return {
        "D_total": total_D,
        "f_ML": sum_D_11 / total_D,
        "f_AP": sum_D_22 / total_D,
        "f_CC": sum_D_33 / total_D,
        "f_ML_AP": sum_D_12 / total_D,
        "f_ML_CC": sum_D_13 / total_D,
        "f_AP_CC": sum_D_23 / total_D,
        "is_valid": True,
    }


# =============================================================================
# Level B: Regional Mechanism Decomposition
# =============================================================================

class RegionalBeamModel:
    """Orthogonal beam/branch mechanism model for an anatomical pelvic region.

    Decomposes local strain fields into:
    1. Axial extension/compression: eps_ss ~ a
    2. Bending: eps_ss ~ b_y * y + b_z * z
    3. Uniform transverse shear: (eps_sy, eps_sz) ~ (g_y, g_z)
    4. Circulatory twist / torsion: (eps_sy, eps_sz) ~ t * (-z, y)^T
    5. Residual strain: eps_res

    Ensures exact Frobenius orthogonality under volume weighting, so that
    fractions strictly sum to 100%.
    """

    def __init__(
        self,
        name: str = "region",
        cell_indices: np.ndarray | None = None,
        coords_ref: np.ndarray | None = None,
        cells: np.ndarray | None = None,
        centroids: np.ndarray | None = None,
        e_long: np.ndarray | None = None,
        e_trans1: np.ndarray | None = None,
        e_trans2: np.ndarray | None = None,
        local_x_axis: np.ndarray | None = None,
    ):
        self.name = name
        
        # Resolve axial direction (e_long or local_x_axis)
        if e_long is None and local_x_axis is not None:
            e_long = local_x_axis
        elif e_long is None and local_x_axis is None:
            e_long = np.array([1.0, 0.0, 0.0])
        self.e_long = np.asarray(e_long, dtype=float)
        self.e_long /= np.linalg.norm(self.e_long)

        # Resolve transverse directions
        if e_trans1 is None:
            v = np.array([0.0, 1.0, 0.0]) if abs(self.e_long[1]) < 0.9 else np.array([0.0, 0.0, 1.0])
            e_trans1 = np.cross(self.e_long, v)
            e_trans1 /= np.linalg.norm(e_trans1)
        else:
            e_trans1 = np.asarray(e_trans1, dtype=float)
            e_trans1 /= np.linalg.norm(e_trans1)
            
        if e_trans2 is None:
            e_trans2 = np.cross(self.e_long, e_trans1)
            e_trans2 /= np.linalg.norm(e_trans2)
        else:
            e_trans2 = np.asarray(e_trans2, dtype=float)
            e_trans2 /= np.linalg.norm(e_trans2)

        self.e_trans1 = e_trans1
        self.e_trans2 = e_trans2

        # Orthonormal transformation matrix R = [e_trans1, e_trans2, e_long]
        # so local coords are (y, z, s)
        self.R = np.column_stack([self.e_trans1, self.e_trans2, self.e_long])

        # Centroids and cell indices
        if centroids is not None:
            self.centroids_ref = np.asarray(centroids, dtype=float)
            if cell_indices is None:
                self.cell_indices = np.arange(len(self.centroids_ref))
            else:
                self.cell_indices = np.asarray(cell_indices, dtype=int)
        elif cell_indices is not None and coords_ref is not None and cells is not None:
            self.cell_indices = np.asarray(cell_indices, dtype=int)
            cell_nodes = cells[self.cell_indices]
            self.centroids_ref = np.mean(coords_ref[cell_nodes], axis=1)
        else:
            self.cell_indices = np.array([], dtype=int)
            self.centroids_ref = np.zeros((0, 3))

    def decompose(
        self,
        strain: np.ndarray,
        weights: np.ndarray | None = None,
        coords_phys: np.ndarray | None = None,
        cells: np.ndarray | None = None,
    ) -> dict[str, float]:
        """Perform orthogonal mechanism decomposition on this region.

        Parameters
        ----------
        strain : ndarray of shape (num_cells, 3, 3) or (len(cell_indices), 3, 3)
            Global strain tensors.
        weights : ndarray of shape (num_cells,) or (len(cell_indices),), optional
            Integration weights w_c = J_c * V_ref,c. Default is equal weighting (1.0).
        coords_phys : ndarray of shape (num_nodes, 3), optional
            Physical coordinates to compute deformed centroids.
        cells : ndarray of shape (num_cells, 4), optional
            Cell connectivity array.
        """
        strain = np.asarray(strain, dtype=float)
        n_reg = len(self.cell_indices)

        if strain.shape[0] == n_reg:
            eps_sub = strain
            if weights is not None:
                w = np.asarray(weights, dtype=float)
            else:
                w = np.ones(n_reg)
        else:
            eps_sub = strain[self.cell_indices]
            if weights is not None:
                w = np.asarray(weights, dtype=float)[self.cell_indices]
            else:
                w = np.ones(n_reg)

        W_tot = float(np.sum(w))
        if W_tot <= 1e-15 or len(w) == 0:
            return {
                "is_valid": False, "D_total": 0.0,
                "f_axial": 0.0, "f_bending": 0.0, "f_shear": 0.0, "f_twist": 0.0, "f_residual": 0.0,
                "D_axial": 0.0, "D_bending": 0.0, "D_shear": 0.0, "D_twist": 0.0, "D_residual": 0.0,
            }

        # Transform strain to local (y, z, s) frame
        eps_local = np.einsum("ki,ckl,lj->cij", self.R, eps_sub, self.R)

        # Cell centroids in local frame
        if coords_phys is not None and cells is not None:
            cell_nodes = cells[self.cell_indices]
            centroids = np.mean(coords_phys[cell_nodes], axis=1)
        else:
            centroids = self.centroids_ref

        # Project centroids onto local (y, z) transverse axes
        y_raw = np.dot(centroids, self.e_trans1)
        z_raw = np.dot(centroids, self.e_trans2)

        # Centering with volume weights to ensure orthogonality: sum w * y_c = 0
        y_bar = float(np.sum(w * y_raw) / W_tot)
        z_bar = float(np.sum(w * z_raw) / W_tot)
        y = y_raw - y_bar
        z = z_raw - z_bar

        # Total regional strain norm D_reg = sum w (eps : eps)
        # Components:
        # eps_local has indices: 0=y, 1=z, 2=s
        e_yy = eps_local[:, 0, 0]
        e_zz = eps_local[:, 1, 1]
        e_ss = eps_local[:, 2, 2]
        e_yz = eps_local[:, 0, 1]
        e_sy = eps_local[:, 2, 0]
        e_sz = eps_local[:, 2, 1]

        D_total = float(np.sum(w * (
            e_yy**2 + e_zz**2 + e_ss**2 + 2*e_yz**2 + 2*e_sy**2 + 2*e_sz**2
        )))

        if D_total <= 1e-15:
            return {"is_valid": False, "D_total": 0.0}

        # 1. Axial stretching / compression: eps_ss ~ a
        a = float(np.sum(w * e_ss) / W_tot)
        D_axial = float(W_tot * (a ** 2))

        # 2. Bending: eps_ss - a ~ b_y * y + b_z * z
        # Inertia tensor [I_yy, I_yz; I_yz, I_zz]
        I_yy = float(np.sum(w * (y ** 2)))
        I_zz = float(np.sum(w * (z ** 2)))
        I_yz = float(np.sum(w * y * z))

        rhs_y = float(np.sum(w * (e_ss - a) * y))
        rhs_z = float(np.sum(w * (e_ss - a) * z))

        A_bend = np.array([[I_yy, I_yz], [I_yz, I_zz]])
        cond_bend = float(np.linalg.cond(A_bend)) if np.linalg.det(A_bend) > 1e-12 else 1e12

        if cond_bend < 1e6:
            b = np.linalg.solve(A_bend, np.array([rhs_y, rhs_z]))
            b_y, b_z = float(b[0]), float(b[1])
            eps_bend_ss = b_y * y + b_z * z
            D_bend = float(np.sum(w * (eps_bend_ss ** 2)))
        else:
            b_y, b_z = 0.0, 0.0
            D_bend = 0.0

        # 3. Uniform transverse shear: (eps_sy, eps_sz) ~ (g_y, g_z)
        # Factor 2 because of tensor norm 2*w*(eps_sy^2 + eps_sz^2)
        g_y = float(np.sum(w * e_sy) / W_tot)
        g_z = float(np.sum(w * e_sz) / W_tot)
        D_shear = float(2.0 * W_tot * (g_y**2 + g_z**2))

        # 4. Circulatory twist / torsion: (eps_sy, eps_sz) ~ t * (-z, y)
        # Centered twist pattern h_T = (-z, y) has sum w * h_T = 0,
        # so it is orthogonal to constant transverse shear (g_y, g_z)!
        denom_twist = float(np.sum(w * (y**2 + z**2)))
        if denom_twist > 1e-12:
            num_twist = float(np.sum(w * (e_sy * (-z) + e_sz * y)))
            t = num_twist / denom_twist
            D_twist = float(2.0 * (t ** 2) * denom_twist)
        else:
            t = 0.0
            D_twist = 0.0

        # 5. Residual: exactly D_total - (D_axial + D_bend + D_shear + D_twist)
        D_fitted = D_axial + D_bend + D_shear + D_twist
        D_res = max(0.0, D_total - D_fitted)

        return {
            "D_total": D_total,
            "f_axial": D_axial / D_total,
            "f_bending": D_bend / D_total,
            "f_shear": D_shear / D_total,
            "f_twist": D_twist / D_total,
            "f_residual": D_res / D_total,
            "D_axial": D_axial,
            "D_bending": D_bend,
            "D_shear": D_shear,
            "D_twist": D_twist,
            "D_residual": D_res,
            "a_axial": a,
            "b_y": b_y,
            "b_z": b_z,
            "g_y": g_y,
            "g_z": g_z,
            "t_twist": t,
            "cond_bend": cond_bend,
            "W_tot": W_tot,
            "is_valid": True,
        }
