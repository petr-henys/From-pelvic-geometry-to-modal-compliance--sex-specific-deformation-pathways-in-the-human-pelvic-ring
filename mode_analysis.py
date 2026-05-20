# -*- coding: utf-8 -*-
"""Modal decomposition and energy partitioning for static deformations.

Projects static displacement fields onto eigenvector basis using mass-weighted
inner products. Computes modal participation factors, MAC scores, and strain
energy distribution across modes.

Key functions:
- compute_mac_u_vs_modes: MAC correlation between displacement and each mode
- project_to_modes: Modal amplitudes via M-orthogonal projection
- reconstruct_from_modes: Rebuild displacement from modal amplitudes
- modal_strain_energy: Energy contribution per mode

Supports mass matrix (PETSc) for consistent metrics or Euclidean fallback.
Input shapes: eigenvectors (modes, nodes, dims) or (dof, modes)
               displacement (nodes, dims) or (dof,)
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple, Optional
import numpy as np
import pandas as pd
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))


from petsc4py import PETSc


logger = logging.getLogger(__name__)


def _as_Phi_and_u(eigvecs: np.ndarray, u_stat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Normalize eigenvector and displacement arrays to consistent DOF layout.
    
    Converts to:
    - Phi: (n_dof, n_modes) with modes as columns
    - u: (n_dof,) flattened displacement
    
    Accepts:
    - eigvecs: (n_modes, n_nodes, n_comp) or (n_dof, n_modes)
    - u_stat: (n_nodes, n_comp) or (n_dof,)
    """
    if eigvecs.ndim == 3:
        m, n_nodes, n_comp = eigvecs.shape
        Phi = eigvecs.reshape(m, n_nodes * n_comp).T  # (n_dof, m)
    elif eigvecs.ndim == 2:
        Phi = eigvecs
        m = Phi.shape[1]
    else:
        raise ValueError(f"eigvecs must be shape (m, n_nodes, n_comp) or (n_dof, m), got {eigvecs.shape}")

    if u_stat.ndim == 2:
        u = u_stat.reshape(-1)
    elif u_stat.ndim == 1:
        u = u_stat
    else:
        raise ValueError(f"u_stat must be shape (n_nodes, n_comp) or (n_dof,), got {u_stat.shape}")

    if Phi.shape[0] != u.size:
        raise ValueError(f"Inconsistent DOF: Phi has {Phi.shape[0]} rows, u has {u.size} elements")

    return Phi, u


def _dot_M(x: np.ndarray, y: np.ndarray, M: Optional['PETSc.Mat']) -> complex:
    """Compute mass-weighted inner product x^H M y.
    
    Uses PETSc matrix-vector product if M provided, else Euclidean x^H y.
    Zero-copy array wrapping for efficiency.
    """
    if M is None:
        return np.vdot(x, y)
    

    vx = PETSc.Vec().createWithArray(x, comm=M.comm)
    vy = PETSc.Vec().createWithArray(y, comm=M.comm)
    return vx.dot(M * vy)


def compute_mac_u_vs_modes(
    eigvecs: np.ndarray, 
    u_stat: np.ndarray, 
    *, 
    mass_matrix: Optional['PETSc.Mat'] = None
) -> np.ndarray:
    """Compute MAC scores between displacement field and each eigenmode.
    
    MAC(u, φ_i) = |u^H M φ_i|^2 / [(u^H M u)(φ_i^H M φ_i)]
    
    Parameters
    ----------
    eigvecs : np.ndarray
        Eigenvectors (n_modes, n_nodes, n_comp) or (n_dof, n_modes)
    u_stat : np.ndarray
        Displacement field (n_nodes, n_comp) or (n_dof,)
    mass_matrix : PETSc.Mat | None
        Mass matrix for inner product (None = Euclidean)
        
    Returns
    -------
    np.ndarray
        MAC values (n_modes,) in range [0, 1]
    """
    Phi, u = _as_Phi_and_u(eigvecs, u_stat)
    
    # u^H M u
    u_norm2 = np.real(_dot_M(u, u, mass_matrix))
    if u_norm2 <= 0:
        logger.warning("Zero norm displacement field, returning zero MAC scores")
        return np.zeros(Phi.shape[1], dtype=float)

    # φ_i^H M u for all i
    dots = np.empty(Phi.shape[1], dtype=complex)
    phi_norm2 = np.empty(Phi.shape[1], dtype=float)
    
    for i in range(Phi.shape[1]):
        col = Phi[:, i]
        dots[i] = _dot_M(col, u, mass_matrix)
        phi_norm2[i] = float(np.real(_dot_M(col, col, mass_matrix)))

    mac = np.zeros_like(dots, dtype=float)
    denom = phi_norm2 * u_norm2
    mask = denom > 0
    mac[mask] = (np.abs(dots[mask]) ** 2) / denom[mask]
    
    return np.clip(mac, 0.0, 1.0)


def rank_modes_nonorthonormal(
    eigvecs: np.ndarray,
    u_stat: np.ndarray,
    lambdas: np.ndarray,
    lambda_tol: Optional[float] = None,
    rcond: Optional[float] = None,
    *,
    mass_matrix: Optional['PETSc.Mat'] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, np.ndarray, float, np.ndarray, np.ndarray]:
    """
    Robust projection without orthonormality assumption in metric M.

    Projection: q = argmin ||Φ q - u||_M, i.e., (Φ^H M Φ) q = Φ^H M u
    Mode energy: U_i = 0.5 * λ_i * q_i^2 * (φ_i^H M φ_i)
    Fraction: η_i = U_i / sum(U_i), computed only for λ_i > lambda_tol

    Parameters
    ----------
    eigvecs : np.ndarray
        Eigenvectors (n_modes, n_nodes, n_comp) or (n_dof, n_modes)
    u_stat : np.ndarray
        Displacement field (n_nodes, n_comp) or (n_dof,)
    lambdas : np.ndarray
        Eigenvalues (n_modes,)
    lambda_tol : float, optional
        Tolerance for filtering rigid/degenerate modes
    rcond : float, optional
        Condition number cutoff for least squares
    mass_matrix : PETSc.Mat, optional
        Mass matrix for inner products

    Returns
    -------
    idx_sorted : np.ndarray
        Mode indices sorted by energy contribution (0-based)
    fracs_sorted : np.ndarray
        Energy fractions for sorted modes
    q_sorted : np.ndarray
        Participation factors for sorted modes
    U_sorted : np.ndarray
        Strain energies for sorted modes
    U_tot : float
        Total strain energy
    u_rec : np.ndarray
        Reconstructed displacement (same shape as u_stat input)
    rel_err : float
        Relative reconstruction error in M-norm
    norms : np.ndarray
        Mode norms in M-metric (φ_i^H M φ_i)
    q_full : np.ndarray
        Full participation factors for all modes
    """
    Phi, u = _as_Phi_and_u(eigvecs, u_stat)

    lambdas = np.asarray(lambdas, dtype=float).reshape(-1)
    m = Phi.shape[1]
    if lambdas.size != m:
        raise ValueError(f"Number of eigenvalues ({lambdas.size}) doesn't match modes ({m})")

    if lambda_tol is None:
        lambda_tol = 1e-12 * (np.max(np.abs(lambdas)) if lambdas.size else 1.0)

    # Mask for usable modes (exclude rigid/degenerate)
    mask = np.abs(lambdas) > lambda_tol
    if not np.any(mask):
        logger.warning("No modes above lambda_tol, returning empty results")
        u_rec = np.zeros_like(u).reshape(u_stat.shape)
        return (np.array([], dtype=int), np.array([]), np.array([]), np.array([]),
                0.0, u_rec, 0.0, np.array([]), np.zeros_like(lambdas))

    # Project into subspace (mass-weighted)
    Phi_use = Phi[:, mask]
    m_sub = Phi_use.shape[1]
    
    # Build Gram matrix G = Φ^H M Φ and RHS b = Φ^H M u
    G = np.empty((m_sub, m_sub), dtype=float)
    b = np.empty(m_sub, dtype=float)
    
    for i in range(m_sub):
        vi = Phi_use[:, i]
        b[i] = float(np.real(_dot_M(vi, u, mass_matrix)))
        for j in range(i, m_sub):
            vj = Phi_use[:, j]
            val = float(np.real(_dot_M(vi, vj, mass_matrix)))
            G[i, j] = val
            G[j, i] = val
    
    # Solve G q = b
    q_use = np.linalg.solve(G, b)
    
    q_full = np.zeros_like(lambdas, dtype=float)
    q_full[mask] = np.real_if_close(q_use)

    # Mode norms in M-metric: φ_i^H M φ_i
    norms = np.empty(m, dtype=float)
    for i in range(m):
        norms[i] = float(np.real(_dot_M(Phi[:, i], Phi[:, i], mass_matrix)))

    # Mode energies (ensure non-negative numerically)
    U_i = 0.5 * lambdas * (q_full ** 2) * norms
    U_i = np.clip(U_i, 0.0, None)

    # Sort by energy within mask
    idx_mask = np.nonzero(mask)[0]
    order = np.argsort(U_i[mask])[::-1]
    idx_sorted = idx_mask[order]

    U_sorted = U_i[idx_sorted]
    U_tot = float(U_sorted.sum())
    fracs_sorted = (U_sorted / U_tot) if U_tot > 0 else np.zeros_like(U_sorted)
    q_sorted = q_full[idx_sorted]

    # Reconstruction and error
    u_rec = (Phi @ q_full).reshape(u_stat.shape)
    err = (u_rec.reshape(-1) - u.reshape(-1))
    
    denom = np.sqrt(max(0.0, float(np.real(_dot_M(u.reshape(-1), u.reshape(-1), mass_matrix)))))
    num = np.sqrt(max(0.0, float(np.real(_dot_M(err, err, mass_matrix)))))
    rel_err = (num / denom) if denom > 0 else 0.0

    return idx_sorted, fracs_sorted, q_sorted, U_sorted, U_tot, u_rec, rel_err, norms, q_full


def export_mac_and_energy_to_xlsx(
    eigvecs: np.ndarray,
    u_stat: np.ndarray,
    lambdas: np.ndarray,
    xlsx_path: str,
    lambda_tol: Optional[float] = None,
    rcond: Optional[float] = None,
    top: Optional[int] = None,
    *,
    mass_matrix: Optional['PETSc.Mat'] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """
    Compute MAC(u, φ_i) and energy ranking, export to Excel with three sheets.
    
    Parameters
    ----------
    eigvecs : np.ndarray
        Eigenvectors
    u_stat : np.ndarray
        Displacement field
    lambdas : np.ndarray
        Eigenvalues
    xlsx_path : str
        Output Excel file path
    lambda_tol : float, optional
        Tolerance for filtering modes
    rcond : float, optional
        Condition number for least squares
    top : int, optional
        Number of top modes to export (None = all)
    mass_matrix : PETSc.Mat, optional
        Mass matrix
        
    Returns
    -------
    df_en : pd.DataFrame
        Energy ranking dataframe
    df_mac : pd.DataFrame
        MAC scores dataframe
    xlsx_path : str
        Output path (echoed back)
    """
    mac = compute_mac_u_vs_modes(eigvecs, u_stat, mass_matrix=mass_matrix)

    (idx_sorted, fracs_sorted, q_sorted, U_sorted, U_tot,
     u_rec, rel_err, norms, q_full) = rank_modes_nonorthonormal(
        eigvecs, u_stat, lambdas, lambda_tol=lambda_tol, rcond=rcond, mass_matrix=mass_matrix
    )

    lambdas = np.asarray(lambdas, float).reshape(-1)
    Phi, _ = _as_Phi_and_u(eigvecs, u_stat)
    
    mode_norm2 = np.empty(Phi.shape[1], dtype=float)
    for i in range(Phi.shape[1]):
        mode_norm2[i] = float(np.real(_dot_M(Phi[:, i], Phi[:, i], mass_matrix)))

    # MAC dataframe (1-based mode indexing for user)
    df_mac = pd.DataFrame({
        "mode": np.arange(1, lambdas.size + 1, dtype=int),
        "MAC_u_vs_mode": mac,
        "lambda": lambdas,
        "mode_norm2": mode_norm2,
        "q_from_LS": q_full,
        "U_i_from_LS": 0.5 * lambdas * (q_full**2) * mode_norm2
    }).sort_values("MAC_u_vs_mode", ascending=False, kind="mergesort").reset_index(drop=True)

    # Energy ranking dataframe (1-based mode indexing)
    df_en = pd.DataFrame({
        "rank": np.arange(1, len(idx_sorted) + 1, dtype=int),
        "mode": idx_sorted + 1,  # Convert 0-based to 1-based
        "frac": fracs_sorted,
        "U_i": U_sorted,
        "q_i": q_sorted,
        "lambda": lambdas[idx_sorted],
        "mode_norm2": mode_norm2[idx_sorted]
    })
    
    if top is not None and top > 0:
        df_en = df_en.iloc[:top].copy()

    # Write to Excel
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as xlw:
        pd.DataFrame({
            "U_total": [U_tot], 
            "rel_reconstruction_error_norm": [rel_err]
        }).to_excel(xlw, sheet_name="Info", index=False)
        df_mac.to_excel(xlw, sheet_name="MAC", index=False)
        df_en.to_excel(xlw, sheet_name="StrainEnergy", index=False)

    logger.info(f"Exported MAC and energy analysis to {xlsx_path}")
    return df_en, df_mac, xlsx_path
