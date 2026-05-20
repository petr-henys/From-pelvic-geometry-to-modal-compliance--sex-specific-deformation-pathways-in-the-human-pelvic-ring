"""Tail-risk metric helpers (pure NumPy).

Shared computations used by tail-risk analysis/visualisation scripts:
element-level scalar fields (SED, principal strains, von-Mises) and
volume-weighted tail/dispersion statistics.
"""
from __future__ import annotations

import warnings

import numpy as np


def element_scalars(
    strain_3x3: np.ndarray,
    stress_3x3: np.ndarray,
    *,
    symmetrize: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute SED, principal strains, and von-Mises stress from DG0 tensors.

    Parameters
    ----------
    strain_3x3, stress_3x3
        Arrays shaped ``(num_cells, 3, 3)``.
    symmetrize
        If True, symmetrizes strain/stress before computing invariants. This is a
        numerical guard; in the underlying FE pipeline both are symmetric by
        construction.

    Returns
    -------
    (sed, eps1, eps3, vms)
        Arrays shaped ``(num_cells,)``.
    """
    if symmetrize:
        strain_3x3 = 0.5 * (strain_3x3 + np.swapaxes(strain_3x3, -2, -1))
        stress_3x3 = 0.5 * (stress_3x3 + np.swapaxes(stress_3x3, -2, -1))

    # SED = 0.5 ε:σ
    sed = 0.5 * np.einsum("eij,eij->e", strain_3x3, stress_3x3)

    # Principal strains from symmetric tensor
    eigs = np.linalg.eigvalsh(strain_3x3)  # ascending → (:, 3)
    eps1 = eigs[:, 2]  # max (tensile)
    eps3 = eigs[:, 0]  # min (compressive)

    # Von-Mises stress: σ_vm = √(3/2 s:s), s = σ − ⅓ tr(σ) I
    tr_sig = np.trace(stress_3x3, axis1=1, axis2=2)
    dev = stress_3x3 - (tr_sig / 3.0)[:, None, None] * np.eye(3)[None, :, :]
    vms = np.sqrt(np.maximum(1.5 * np.einsum("eij,eij->e", dev, dev), 0.0))

    return sed, eps1, eps3, vms


def sed_only(
    strain_3x3: np.ndarray,
    stress_3x3: np.ndarray,
    *,
    symmetrize: bool = True,
) -> np.ndarray:
    """Fast path: compute SED only (no eigvalsh)."""
    if symmetrize:
        strain_3x3 = 0.5 * (strain_3x3 + np.swapaxes(strain_3x3, -2, -1))
        stress_3x3 = 0.5 * (stress_3x3 + np.swapaxes(stress_3x3, -2, -1))
    return 0.5 * np.einsum("eij,eij->e", strain_3x3, stress_3x3)


def weighted_percentile_robust(
    values: np.ndarray,
    weights: np.ndarray,
    q: float,
) -> float:
    """NaN/Inf-safe weighted percentile via sorted cumulative weight.

    Parameters
    ----------
    values : 1-D array
    weights : 1-D array (same length, positive)
    q : percentile in [0, 100]
    """
    ok = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if ok.sum() < 1:
        return np.nan
    v, w = values[ok], weights[ok]
    idx = np.argsort(v)
    sv, sw = v[idx], w[idx]
    cw = np.cumsum(sw)
    cw /= cw[-1]
    return float(np.interp(q / 100.0, cw, sv))


def compute_tail_metrics(
    sed: np.ndarray,
    eps1: np.ndarray,
    eps3: np.ndarray,
    volumes: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float]:
    """NaN-robust volume-weighted tail metrics for masked elements.

    Does **not** compute exceedance (requires τ).
    """
    nan_keys = [
        "P99_eps1", "P99_abs_eps3", "P99_SED", "P95_SED",
        "top1pct_mean_SED", "entropy_SED", "gini_SED",
        "median_SED", "IQR_SED", "MAD_SED",
    ]

    s, e1, e3, v = sed[mask], eps1[mask], eps3[mask], volumes[mask]
    ok = np.isfinite(s) & np.isfinite(e1) & np.isfinite(e3) & np.isfinite(v) & (v > 0)
    n_ok = int(ok.sum())
    if n_ok < 10:
        warnings.warn(
            f"Too few valid elements ({n_ok}) — returning NaN metrics.",
            stacklevel=2,
        )
        return {k: np.nan for k in nan_keys}

    s, e1, e3, v = s[ok], e1[ok], e3[ok], v[ok]
    total_vol = float(v.sum())

    # ── Weighted percentiles ──
    P99_eps1 = weighted_percentile_robust(e1, v, 99.0)
    P99_abs_eps3 = weighted_percentile_robust(np.abs(e3), v, 99.0)
    P99_SED = weighted_percentile_robust(s, v, 99.0)
    P95_SED = weighted_percentile_robust(s, v, 95.0)

    # ── Top-1 % volume mean SED ──
    desc = np.argsort(s)[::-1]
    cum_v = np.cumsum(v[desc])
    thresh = 0.01 * total_vol
    # Select the minimal prefix with cumulative volume >= 1 % (or at least 1 element).
    k = int(np.searchsorted(cum_v, thresh, side="left"))
    k = max(0, min(k, len(desc) - 1))
    top_sel = np.zeros_like(desc, dtype=bool)
    top_sel[: k + 1] = True
    top1pct_mean_SED = float(np.average(s[desc][top_sel], weights=v[desc][top_sel]))

    # ── Normalised Shannon entropy of SED·V distribution ──
    wv = np.maximum(s * v, 0.0)
    wv_sum = float(wv.sum())
    if wv_sum > 0:
        p = wv / wv_sum
        p_pos = p[p > 0]
        if p_pos.size > 1:
            entropy_norm = float(-np.sum(p_pos * np.log(p_pos)) / np.log(p_pos.size))
        else:
            entropy_norm = 0.0
    else:
        entropy_norm = 0.0

    # ── Gini coefficient ──
    wv_sorted = np.sort(np.maximum(s * v, 0.0))
    n = int(wv_sorted.size)
    if n > 0 and float(wv_sorted.sum()) > 0:
        cum = np.cumsum(wv_sorted)
        gini = float(1.0 - 2.0 * cum.sum() / (n * cum[-1]) + 1.0 / n)
    else:
        gini = 0.0

    # ── Dispersion (weighted median, IQR, MAD) ──
    median_SED = weighted_percentile_robust(s, v, 50.0)
    IQR_SED = weighted_percentile_robust(s, v, 75.0) - weighted_percentile_robust(s, v, 25.0)
    MAD_SED = weighted_percentile_robust(np.abs(s - median_SED), v, 50.0)

    return {
        "P99_eps1": P99_eps1,
        "P99_abs_eps3": P99_abs_eps3,
        "P99_SED": P99_SED,
        "P95_SED": P95_SED,
        "top1pct_mean_SED": top1pct_mean_SED,
        "entropy_SED": entropy_norm,
        "gini_SED": gini,
        "median_SED": median_SED,
        "IQR_SED": IQR_SED,
        "MAD_SED": MAD_SED,
    }


def exceedance_volfrac(
    sed: np.ndarray,
    volumes: np.ndarray,
    mask: np.ndarray,
    tau: float,
) -> float:
    """Volume fraction of elements where SED > τ within a domain mask."""
    s, v = sed[mask], volumes[mask]
    ok = np.isfinite(s) & np.isfinite(v) & (v > 0)
    if ok.sum() < 1 or tau <= 0:
        return np.nan
    s, v = s[ok], v[ok]
    total_vol = float(v.sum())
    if total_vol <= 0:
        return np.nan
    return float(v[s > tau].sum() / total_vol)

