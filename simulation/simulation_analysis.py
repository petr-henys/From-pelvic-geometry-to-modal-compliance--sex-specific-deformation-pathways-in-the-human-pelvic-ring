"""Shared analysis utilities for eigenstiffness simulation."""
from __future__ import annotations

import numpy as np
from dolfinx import fem

from mode_analysis import compute_mac_u_vs_modes, rank_modes_nonorthonormal
from utils.log_utils import fmt_matrix_with_columns


def format_sensitivity_debug(matrix: np.ndarray, labels: list[str]) -> str:
    """Format sensitivity matrix for debug output."""
    return fmt_matrix_with_columns(matrix, labels, digits=3, sci=True, max_width=160)


def function_to_array(function: fem.Function) -> np.ndarray:
    """Convert DOLFINx function to numpy array."""
    gdim = function.function_space.mesh.geometry.dim
    return function.x.array.reshape((-1, gdim))


def compute_mode_contributions(
    eigvecs: np.ndarray,
    eigvals: np.ndarray,
    displacement: fem.Function,
    *,
    mass_matrix=None,
) -> dict[str, np.ndarray | float]:
    """Compute MAC, energy, and modal amplitudes for displacement field."""
    disp_array = function_to_array(displacement)

    modes = np.nan_to_num(eigvecs, nan=0.0)
    lambdas = np.nan_to_num(eigvals, nan=0.0)
    mac = compute_mac_u_vs_modes(modes, disp_array, mass_matrix=mass_matrix)

    (
        _idx_sorted,
        _fracs_sorted,
        _q_sorted,
        _U_sorted,
        _U_tot,
        _u_rec,
        rel_err,
        norms,
        q_full,
    ) = rank_modes_nonorthonormal(modes, disp_array, lambdas, mass_matrix=mass_matrix)

    norms = np.nan_to_num(norms, nan=0.0)
    q_full = np.nan_to_num(q_full, nan=0.0)
    mac = np.nan_to_num(mac, nan=0.0).astype(float, copy=False)

    mode_energy = 0.5 * lambdas * (q_full**2) * norms
    mode_energy = np.nan_to_num(mode_energy, nan=0.0)
    energy_total = float(np.sum(mode_energy))

    if energy_total > 0.0:
        energy_frac = mode_energy / energy_total
    else:
        energy_frac = np.zeros_like(mode_energy)

    return {
        "mac": mac,
        "energy": mode_energy,
        "energy_fraction": energy_frac,
        "modal_amplitude": q_full,
        "reconstruction_error": float(rel_err),
        "energy_total": energy_total,
    }


def sij_angles_to_array(stats: dict) -> np.ndarray:
    """Extract SIJ angles from stats dict to numpy array."""
    return np.array(
        [
            [
                stats["SIJ_left"]["angles_deg"]["alpha_x"],
                stats["SIJ_left"]["angles_deg"]["beta_y"],
                stats["SIJ_left"]["angles_deg"]["gamma_z"],
            ],
            [
                stats["SIJ_right"]["angles_deg"]["alpha_x"],
                stats["SIJ_right"]["angles_deg"]["beta_y"],
                stats["SIJ_right"]["angles_deg"]["gamma_z"],
            ],
        ],
        dtype=float,
    )


def sij_trans_to_array(stats: dict) -> np.ndarray:
    """Extract SIJ translations from stats dict to numpy array."""
    return np.array(
        [
            [
                stats["SIJ_left"]["displacement_mm"]["dx_ML"],
                stats["SIJ_left"]["displacement_mm"]["dy_AP"],
                stats["SIJ_left"]["displacement_mm"]["dz_CC"],
            ],
            [
                stats["SIJ_right"]["displacement_mm"]["dx_ML"],
                stats["SIJ_right"]["displacement_mm"]["dy_AP"],
                stats["SIJ_right"]["displacement_mm"]["dz_CC"],
            ],
        ],
        dtype=float,
    )


def log_sij_stats(label: str, angles: np.ndarray, translations: np.ndarray) -> str:
    """Format SIJ stats as string for logging."""
    lines = [
        f"[{label}] SIJ NUT/AP/CC (L,R): "
        f"({angles[0, 0]:.2f}, {angles[0, 1]:.2f}, {angles[0, 2]:.2f}) | "
        f"({angles[1, 0]:.2f}, {angles[1, 1]:.2f}, {angles[1, 2]:.2f})",
        f"[{label}] SIJ Tx/Ty/Tz (L,R): "
        f"({translations[0, 0]:.2f}, {translations[0, 1]:.2f}, {translations[0, 2]:.2f}) | "
        f"({translations[1, 0]:.2f}, {translations[1, 1]:.2f}, {translations[1, 2]:.2f})"
    ]
    return "\n".join(lines)
