"""Mass-weighted modal pairing using MAC and Hungarian algorithm.

Implements mode correlation via Modal Assurance Criterion (MAC) with mass matrix
inner product. Uses linear sum assignment (Hungarian algorithm) to find optimal
one-to-one pairing between reference and sample eigenmodes.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import logging
import time
from typing import Iterable

import numpy as np
from petsc4py import PETSc
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger(__name__)


def _dot_mass(x: np.ndarray, y: np.ndarray, mass_matrix: PETSc.Mat) -> complex:
    """Compute mass-weighted inner product x^H M y.
    
    Uses zero-copy PETSc vector wrapping for efficiency.
    PETSc objects are explicitly destroyed to prevent memory leaks.
    """

    vec_x = PETSc.Vec().createWithArray(x, comm=mass_matrix.comm)
    vec_y = PETSc.Vec().createWithArray(y, comm=mass_matrix.comm)
    tmp = mass_matrix * vec_y
    result = vec_x.dot(tmp)
    tmp.destroy()
    vec_y.destroy()
    vec_x.destroy()
    return result


def mac_mass(
    x: np.ndarray,
    y: np.ndarray,
    mass_matrix: PETSc.Mat,
    *,
    eps: float = 1e-12,
) -> float:
    """Compute mass-weighted Modal Assurance Criterion between mode vectors.
    
    MAC = |x^H M y|^2 / [(x^H M x)(y^H M y)]
    
    Returns value in [0, 1] where 1 = perfect correlation, 0 = orthogonal.
    """

    numerator = abs(_dot_mass(x, y, mass_matrix)) ** 2
    denom = _dot_mass(x, x, mass_matrix).real * _dot_mass(y, y, mass_matrix).real
    return float(eps if denom < eps else numerator / denom)


def _log_mac_progress(current: int, total: int) -> None:
    """Log MAC matrix assembly progress at 25% intervals (throttled)."""

    if total <= 10:
        return
    quarter = max(1, total // 4)
    if (current + 1) % quarter == 0:
        logger.debug("MAC computation progress: %d/%d reference modes", current + 1, total)


def _validate_lengths(vectors: Iterable[np.ndarray], values: np.ndarray, label: str) -> None:
    """Validate eigenvector and eigenvalue array lengths match."""
    if len(vectors) != len(values):
        raise ValueError(f"{label} vectors ({len(vectors)}) and values ({len(values)}) length mismatch")


def pair_modes(
    ref_vecs: np.ndarray,
    ref_vals: np.ndarray,
    samp_vecs: np.ndarray,
    samp_vals: np.ndarray,
    mass_matrix: PETSc.Mat,
    *,
    mac_cut: float = 0.05,
) -> np.ndarray:
    """Pair sample eigenmodes to reference via mass-weighted MAC and Hungarian algorithm.

    Computes MAC matrix between all mode pairs, then solves optimal assignment
    problem to maximize total correlation. Pairs below mac_cut threshold are rejected.

    Parameters
    ----------
    ref_vecs, samp_vecs : np.ndarray
        Eigenvector arrays (n_modes, ...) with spatial DOFs flattened for comparison
    ref_vals, samp_vals : np.ndarray
        Eigenvalue arrays (used only for diagnostic logging)
    mass_matrix : PETSc.Mat
        Mass matrix defining inner product (typically includes Jacobian)
    mac_cut : float
        Minimum MAC threshold for accepting pairings (default 0.05)

    Returns
    -------
    np.ndarray
        Permutation array (ref_modes,) where perm[i] = j means ref_mode[i] pairs
        with samp_mode[j], or -1 if no acceptable match exists
        
    Notes
    -----
    Uses scipy.optimize.linear_sum_assignment (Hungarian algorithm) to solve
    maximum weight bipartite matching on -MAC values.
    """

    _validate_lengths(ref_vecs, ref_vals, "Reference")
    _validate_lengths(samp_vecs, samp_vals, "Sample")
    if len(ref_vecs) == 0 or len(samp_vecs) == 0:
        raise ValueError("Cannot pair empty mode sets")

    logger.info("Pair modes ref=%d sample=%d cut=%.3f", len(ref_vecs), len(samp_vecs), mac_cut)

    ref_flat = ref_vecs.reshape(len(ref_vecs), -1)
    samp_flat = samp_vecs.reshape(len(samp_vecs), -1)
    if ref_flat.shape[1] != samp_flat.shape[1]:
        raise ValueError(
            f"DOF mismatch: reference {ref_flat.shape[1]} vs sample {samp_flat.shape[1]}"
        )

    logger.debug("Flattened mode shapes: ref=%s sample=%s", ref_flat.shape, samp_flat.shape)

    logger.info("MAC matrix %d x %d", len(ref_flat), len(samp_flat))
    t_mac_start = time.perf_counter()
    mac = np.zeros((len(ref_flat), len(samp_flat)), dtype=float)
    for i, ref_mode in enumerate(ref_flat):
        for j, samp_mode in enumerate(samp_flat):
            mac[i, j] = mac_mass(ref_mode, samp_mode, mass_matrix)
        _log_mac_progress(i, len(ref_flat))
    t_mac_end = time.perf_counter()
    logger.info("MAC matrix computed in %.3f seconds", t_mac_end - t_mac_start)

    max_mac = float(np.max(mac))
    mean_mac = float(np.mean(mac))
    above_threshold = int(np.sum(mac >= mac_cut))
    logger.info(
        "MAC stats max=%.4f mean=%.4f kept=%d (%.1f%%)",
        max_mac,
        mean_mac,
        above_threshold,
        100 * above_threshold / mac.size,
    )

    logger.info("Solving optimal assignment problem (Hungarian algorithm)...")
    t_h_start = time.perf_counter()
    row_ind, col_ind = linear_sum_assignment(-mac)
    t_h_end = time.perf_counter()
    logger.debug("Hungarian algorithm completed in %.4f seconds", t_h_end - t_h_start)

    perm = np.full(len(ref_flat), -1, dtype=np.int32)
    pair_scores: list[float] = []
    for i_ref, j_samp in zip(row_ind, col_ind):
        score = mac[i_ref, j_samp]
        if score >= mac_cut:
            perm[i_ref] = j_samp
            pair_scores.append(score)
            if score > 0.9:
                logger.debug(
                    "Excellent pairing: ref_mode[%d] <-> samp_mode[%d] (MAC=%.4f)",
                    i_ref,
                    j_samp,
                    score,
                )
            elif score < 0.2:
                logger.debug(
                    "Poor pairing: ref_mode[%d] <-> samp_mode[%d] (MAC=%.4f)",
                    i_ref,
                    j_samp,
                    score,
                )
        else:
            logger.debug(
                "Failed pairing: ref_mode[%d] <-> samp_mode[%d] (MAC=%.4f < %.3f)",
                i_ref,
                j_samp,
                score,
                mac_cut,
            )

    paired = int(np.count_nonzero(perm >= 0))
    logger.info("Paired %d/%d (%.1f%%)", paired, len(ref_flat), 100 * paired / len(ref_flat))

    if pair_scores:
        logger.info(
            "Pairing quality - avg: %.4f, range: [%.4f, %.4f]",
            float(np.mean(pair_scores)),
            float(np.min(pair_scores)),
            float(np.max(pair_scores)),
        )
    else:
        logger.warning("No modes met the MAC threshold %.3f", mac_cut)

    if paired < len(ref_flat):
        logger.warning("%d modes could not be paired (MAC < %.3f)", len(ref_flat) - paired, mac_cut)
        unpaired = np.where(perm < 0)[0]
        if unpaired.size:
            eigs = ref_vals[unpaired]
            logger.debug(
                "Unpaired reference eigenvalue range: [%.4e, %.4e]",
                float(np.min(eigs)),
                float(np.max(eigs)),
            )

    if np.any(~np.isfinite(mac)):
        raise ValueError("Invalid MAC matrix computed (contains NaN or Inf)")

    if logger.isEnabledFor(logging.DEBUG):
        for i in range(min(5, len(ref_flat))):
            best_idx = int(np.argmax(mac[i, :]))
            logger.debug(
                "ref_mode[%d]: best_MAC=%.4f with samp_mode[%d]",
                i,
                float(np.max(mac[i, :])),
                best_idx,
            )
        if len(ref_flat) > 5:
            logger.debug("... and %d more reference modes", len(ref_flat) - 5)

    return perm
