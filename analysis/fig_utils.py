"""Statistical and data-loading helpers for killer figure pipeline.

Provides Cliff's delta, matched-pair analysis, and data merging routines
reused across all five figures.  No plotting lives here.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

# ---------------------------------------------------------------------------
# Project-root on sys.path so `analysis.*` and `utils.*` are importable
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

log = logging.getLogger(__name__)

from analysis.spectral_config import DATA_DIR_FULL
from analysis.spectral_data import _open_zarr_array

# ── Constants ─────────────────────────────────────────────────
# Zarr loadcase axis order (validated in simulation/simulation_dataset.py)
LOADCASE_INDEX: dict[str, int] = {
    "SP2leg": 0,
    "SP1leg": 1,
    "LAB_phase1": 2,
    "LAB_phase2": 3,
    "LAB_phase3": 4,
}

# 0-based mode indices for the 9–10 block
MODE9_IDX = 8
MODE10_IDX = 9


# ===================================================================
# Data loaders
# ===================================================================
def load_tail_csv(input_dir: Path) -> pd.DataFrame:
    """Load per-loadcase tail-risk CSV."""
    p = input_dir / "tail_risk" / "metrics_tail.csv"
    if not p.exists():
        p = input_dir / "metrics_tail.csv"
    if not p.exists():
        raise FileNotFoundError(f"metrics_tail.csv not found in {input_dir}")
    return pd.read_csv(p)


def load_tail_aggregated_csv(input_dir: Path) -> pd.DataFrame:
    """Load aggregated (standing / labour) tail-risk CSV."""
    p = input_dir / "tail_risk" / "metrics_tail_aggregated.csv"
    if not p.exists():
        p = input_dir / "metrics_tail_aggregated.csv"
    if not p.exists():
        raise FileNotFoundError(
            f"metrics_tail_aggregated.csv not found in {input_dir}"
        )
    return pd.read_csv(p)


def load_energy_fractions(data_dir: Path) -> np.ndarray:
    """Load mode_energy_fraction.zarr → (N_subj, N_modes, N_loadcases).

    Reuses the validated ``_open_zarr_array`` from analysis.spectral_data.
    """
    return _open_zarr_array(data_dir / "mode_energy_fraction.zarr")


def load_modal_amplitudes(data_dir: Path) -> np.ndarray:
    """Load mode_modal_amplitude.zarr → (N_subj, N_modes, N_loadcases).

    Reuses the validated ``_open_zarr_array`` from analysis.spectral_data.
    """
    return _open_zarr_array(data_dir / "mode_modal_amplitude.zarr")


def build_routing_df(
    labour_case: str,
    data_dir: Path,
) -> pd.DataFrame:
    """Per-subject 9-10 routing coefficients for a given loadcase.

    Returns DataFrame with columns:
        subject_idx, E9, E10, a9, a10, theta, ratio
    where
        theta = atan2(|a10|, |a9|)     (sign-invariant mixing angle, radians in [0, π/2])
        ratio = E9 / (E9 + E10)        (energy partition)
    """
    lc_idx = LOADCASE_INDEX.get(labour_case)
    if lc_idx is None:
        raise ValueError(
            f"Unknown loadcase '{labour_case}'. "
            f"Valid: {list(LOADCASE_INDEX)}"
        )

    frac = load_energy_fractions(data_dir)  # (N, 15, 5)
    amp = load_modal_amplitudes(data_dir)    # (N, 15, 5)

    e9 = frac[:, MODE9_IDX, lc_idx]
    e10 = frac[:, MODE10_IDX, lc_idx]
    a9 = amp[:, MODE9_IDX, lc_idx]
    a10 = amp[:, MODE10_IDX, lc_idx]

    # NOTE: modal-amplitude sign is not identifiable unless eigenvector signs are
    # aligned across subjects; use a sign-invariant definition for publication figures.
    theta = np.arctan2(np.abs(a10), np.abs(a9))
    denom = e9 + e10
    ratio = np.where(denom > 0, e9 / denom, np.nan)

    return pd.DataFrame({
        "subject_idx": np.arange(len(e9)),
        "E9": e9,
        "E10": e10,
        "a9": a9,
        "a10": a10,
        "theta": theta,
        "ratio": ratio,
    })


def build_routing_df_labour_avg(
    data_dir: Path,
) -> pd.DataFrame:
    """Like build_routing_df but averaged over 3 labour phases."""
    frac = load_energy_fractions(data_dir)
    amp = load_modal_amplitudes(data_dir)

    lab_idx = [LOADCASE_INDEX["LAB_phase1"],
               LOADCASE_INDEX["LAB_phase2"],
               LOADCASE_INDEX["LAB_phase3"]]

    e9 = frac[:, MODE9_IDX, lab_idx].mean(axis=1)
    e10 = frac[:, MODE10_IDX, lab_idx].mean(axis=1)
    a9 = amp[:, MODE9_IDX, lab_idx].mean(axis=1)
    a10 = amp[:, MODE10_IDX, lab_idx].mean(axis=1)

    theta = np.arctan2(np.abs(a10), np.abs(a9))
    denom = e9 + e10
    ratio = np.where(denom > 0, e9 / denom, np.nan)

    return pd.DataFrame({
        "subject_idx": np.arange(len(e9)),
        "E9": e9,
        "E10": e10,
        "a9": a9,
        "a10": a10,
        "theta": theta,
        "ratio": ratio,
    })


# ===================================================================
# Statistical helpers
# ===================================================================
def cliffs_delta(x: np.ndarray, y: np.ndarray) -> tuple[float, str]:
    """Cliff's delta effect size with qualitative label.

    Returns (delta, magnitude) where magnitude ∈ {negligible, small, medium, large}.
    Thresholds following Romano et al. (2006).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) == 0 or len(y) == 0:
        return np.nan, "undefined"
    n_x, n_y = len(x), len(y)
    # Dominance count without O(n_x*n_y) allocations:
    # more  = #pairs (x_i > y_j)
    # less  = #pairs (x_i < y_j)
    y_sorted = np.sort(y)
    more = np.searchsorted(y_sorted, x, side="left").sum()
    less = (n_y - np.searchsorted(y_sorted, x, side="right")).sum()
    delta = float((more - less) / (n_x * n_y))
    ad = abs(delta)
    if ad < 0.147:
        mag = "negligible"
    elif ad < 0.33:
        mag = "small"
    elif ad < 0.474:
        mag = "medium"
    else:
        mag = "large"
    return delta, mag


def cliffs_delta_ci(
    x: np.ndarray,
    y: np.ndarray,
    alpha: float = 0.05,
    n_boot: int = 2000,
    rng_seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap percentile CI for Cliff's delta.

    Returns (delta, ci_lo, ci_hi).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) == 0 or len(y) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(rng_seed)
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        bx = rng.choice(x, size=len(x), replace=True)
        by = rng.choice(y, size=len(y), replace=True)
        deltas[i] = cliffs_delta(bx, by)[0]
    delta0 = cliffs_delta(x, y)[0]
    lo = np.nanpercentile(deltas, 100 * alpha / 2)
    hi = np.nanpercentile(deltas, 100 * (1 - alpha / 2))
    return delta0, lo, hi


def mann_whitney_test(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[float, float]:
    """Two-sided Mann–Whitney U test. Returns (U, p_value)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    if len(x) < 3 or len(y) < 3:
        return np.nan, np.nan
    u, p = sp_stats.mannwhitneyu(x, y, alternative="two-sided")
    return u, p


def spearman_test(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[float, float]:
    """Spearman rank correlation. Returns (rho, p)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 5:
        return np.nan, np.nan
    return sp_stats.spearmanr(x[mask], y[mask])


def nearest_neighbor_match(
    df: pd.DataFrame,
    treat_col: str,
    covariates: list[str],
    *,
    caliper: float = 0.5,
    rng_seed: int = 42,
) -> pd.DataFrame:
    """1:1 nearest-neighbor matching on standardised covariates.

    Returns matched subset (treated + matched controls) with a `_pair_id` column.
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import NearestNeighbors

    treat = df[treat_col].astype(int)
    cov = df[covariates].copy()
    # Drop rows with NaN in covariates
    valid = cov.notna().all(axis=1)
    df_v = df.loc[valid].copy()
    treat_v = treat.loc[valid]
    cov_v = cov.loc[valid]

    scaler = StandardScaler()
    X = scaler.fit_transform(cov_v)

    idx_t = np.where(treat_v == 1)[0]
    idx_c = np.where(treat_v == 0)[0]
    if len(idx_t) == 0 or len(idx_c) == 0:
        log.warning("No treated or control subjects for matching.")
        return df_v.iloc[:0]

    nn = NearestNeighbors(n_neighbors=1).fit(X[idx_c])
    dists, inds = nn.kneighbors(X[idx_t])

    rng = np.random.default_rng(rng_seed)
    matched_c_set: set[int] = set()
    matched_pairs: list[tuple[int, int]] = []
    # Greedy match with caliper
    order = rng.permutation(len(idx_t))
    for o in order:
        ci = int(inds[o, 0])
        d = dists[o, 0]
        real_ci = idx_c[ci]
        if d <= caliper and real_ci not in matched_c_set:
            matched_c_set.add(real_ci)
            matched_pairs.append((idx_t[o], real_ci))

    if len(matched_pairs) == 0:
        log.warning("Matching produced 0 pairs (caliper=%.2f).", caliper)
        return df_v.iloc[:0]

    keep = [i for pair in matched_pairs for i in pair]
    out = df_v.iloc[keep].copy()
    out["_pair_id"] = np.repeat(np.arange(len(matched_pairs), dtype=int), 2)
    return out


def ols_robust(
    df: pd.DataFrame,
    formula: str,
    cov_type: str = "HC3",
) -> object:
    """OLS with heteroskedastic-robust standard errors.

    Returns statsmodels RegressionResultsWrapper.
    """
    import statsmodels.formula.api as smf

    model = smf.ols(formula, data=df).fit(cov_type=cov_type)
    return model


def lowess_smooth(
    x: np.ndarray,
    y: np.ndarray,
    frac: float = 0.4,
) -> tuple[np.ndarray, np.ndarray]:
    """LOWESS smooth. Returns sorted (x_smooth, y_smooth)."""
    import statsmodels.nonparametric.smoothers_lowess as lo

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 10:
        return np.array([]), np.array([])
    result = lo.lowess(y[mask], x[mask], frac=frac, return_sorted=True)
    return result[:, 0], result[:, 1]


# ===================================================================
# Manifest / summary writers
# ===================================================================
def write_manifest(
    fig_dir: Path,
    entries: list[dict] | dict,
    *,
    meta: dict | None = None,
) -> Path:
    """Write figure_manifest.json.

    Backwards-compatible:
    - If called with a list and no `meta`, writes the legacy list.
    - If called with a list and `meta`, writes `{"meta": ..., "figures": [...]}`.
    - If called with a dict, writes it as-is (assumed already structured).
    """
    p = fig_dir / "figure_manifest.json"
    with open(p, "w") as f:
        if isinstance(entries, dict):
            payload = entries
        elif meta is None:
            payload = entries
        else:
            payload = {"meta": meta, "figures": entries}
        json.dump(payload, f, indent=2)
    return p


def append_stats_row(
    rows: list[dict],
    *,
    figure: str,
    description: str,
    N: int,
    effect_size: float | str,
    effect_label: str,
    p_value: float | str,
    method: str,
    notes: str = "",
    domain: str = "",
) -> None:
    """Append one row to the stats accumulator."""
    rows.append({
        "domain": domain,
        "figure": figure,
        "description": description,
        "N": N,
        "effect_size": effect_size,
        "effect_label": effect_label,
        "p_value": p_value,
        "method": method,
        "notes": notes,
    })


def write_stats_summary(fig_dir: Path, rows: list[dict]) -> Path:
    """Write stats_summary.csv."""
    p = fig_dir / "stats_summary.csv"
    df = pd.DataFrame(rows)
    if "p_value" in df.columns and len(df) > 0:
        p_raw = pd.to_numeric(df["p_value"], errors="coerce")
        mask = p_raw.notna()
        if int(mask.sum()) > 0:
            from statsmodels.stats.multitest import multipletests

            df.loc[mask, "p_fdr_bh"] = multipletests(
                p_raw.loc[mask].to_numpy(dtype=float),
                method="fdr_bh",
            )[1]
    df.to_csv(p, index=False)
    return p



