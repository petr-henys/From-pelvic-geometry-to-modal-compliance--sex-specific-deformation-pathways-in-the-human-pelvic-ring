#!/usr/bin/env python3
"""Generate publication-ready figures for 9-10 block near-degeneracy story.

Produces core PNG figures + combined Fig1/Fig2 dashboard +
matched-subset replot + stats_summary.csv + figure_manifest.json.
Reuses validated code from analysis/ and utils/.

Usage
-----
    python analysis/make_killer_figures.py \\
        --input_dir analysis_outputs \\
        --fig_dir analysis_outputs/figures \\
        --routing_data_dir results/ref_S1P_fixed_new2/data \\
        --routing_case LAB_phase1 \\
        --domain bone \\
        --dpi 300

    python analysis/make_killer_figures.py --dry_run
"""
from __future__ import annotations

import argparse
import subprocess
import logging
import sys
from importlib.metadata import version as pkg_version
from pathlib import Path

# ── Project root on path ──────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

# ── Reused project utilities ──────────────────────────────────
from utils.plot_utils import (
    ANNOT_SIZE,
    FULL_WIDTH,
    HALF_WIDTH,
    ROW_H,
    SMALL_ANNOT_SIZE,
    setup_plot_style,
    MALE_COLOR,
    FEMALE_COLOR,
    SCATTER_ALPHA,
    FILL_ALPHA,
    panel_label,
)
from analysis.fig_utils import (
    load_tail_aggregated_csv,
    build_routing_df,
    build_routing_df_labour_avg,
    cliffs_delta,
    cliffs_delta_ci,
    mann_whitney_test,
    spearman_test,
    nearest_neighbor_match,
    ols_robust,
    lowess_smooth,
    write_manifest,
    write_stats_summary,
    append_stats_row,
    LOADCASE_INDEX,
)

log = logging.getLogger(__name__)

# ── Marker styles ─────────────────────────────────────────────
SWAP0_MARKER = "o"
SWAP1_MARKER = "^"
SEX_COLORS = {"M": MALE_COLOR, "F": FEMALE_COLOR}

REGRESSION_ERRORS = (ValueError, KeyError, np.linalg.LinAlgError)


def _optional_package_version(package_name: str) -> str | None:
    """Return installed package version or None."""
    from importlib.metadata import PackageNotFoundError as _PNF
    try:
        return pkg_version(package_name)
    except _PNF:
        return None


# ===================================================================
# CLI
# ===================================================================
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    default_routing_dir_new = _PROJECT_ROOT / "results" / "ref_S1P_fixed_new2" / "data"
    default_routing_dir_fixed = _PROJECT_ROOT / "results" / "ref_S1P_fixed" / "data"
    default_routing_dir = (
        default_routing_dir_new if default_routing_dir_new.exists() else default_routing_dir_fixed
    )

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input_dir", type=Path, default=_PROJECT_ROOT / "analysis_outputs",
                   help="Directory containing tail_risk/ sub-folder with CSVs")
    p.add_argument("--fig_dir", type=Path, default=_PROJECT_ROOT / "analysis_outputs" / "figures",
                   help="Output directory for PNGs")
    p.add_argument(
        "--routing_data_dir",
        type=Path,
        default=default_routing_dir,
        help="Zarr data dir for routing coefficients (expects mode_energy_fraction.zarr and mode_modal_amplitude.zarr)",
    )
    p.add_argument(
        "--routing_case",
        dest="routing_case",
        default="LAB_phase1",
        help="Loadcase used for routing coefficients (e.g. LAB_phase1). "
             "NOTE: tail metrics use an aggregated parturition category (standing vs LAB), not a sequential load-case staging.",
    )
    # Backwards-compatible alias (deprecated)
    p.add_argument("--labour_case", dest="routing_case", help=argparse.SUPPRESS)
    p.add_argument("--domain", default="bone",
                   help="Tissue domain (default: bone)")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--dry_run", action="store_true",
                   help="Print module provenance and exit")
    return p.parse_args(argv)


# ===================================================================
# Helpers
# ===================================================================
def _savefig(fig: plt.Figure, fig_dir: Path, name: str, dpi: int) -> Path:
    p = fig_dir / f"{name}.pdf"
    fig.savefig(p, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", p)
    return p


def _require_columns(df: pd.DataFrame, cols: list[str], *, context: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{context}: missing required columns {missing}")


def _coerce_numeric(s: pd.Series, *, name: str) -> np.ndarray:
    out = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(out).any():
        log.warning("Column %s has no finite numeric values after coercion.", name)
    return out


def _clean_complete_cases(df: pd.DataFrame) -> pd.DataFrame:
    return df.replace([np.inf, -np.inf], np.nan).dropna()


def _prepare_ols_df(
    reg_df: pd.DataFrame,
    *,
    min_n: int,
    context: str,
) -> pd.DataFrame | None:
    reg_df = _clean_complete_cases(reg_df)
    n = len(reg_df)
    if n < min_n:
        log.warning("%s: skipping OLS, too few complete cases (n=%d).", context, n)
        return None
    return reg_df


def _fit_ols_model(
    reg_df: pd.DataFrame,
    formula: str,
    *,
    context: str,
):
    try:
        return ols_robust(reg_df, formula)
    except REGRESSION_ERRORS as exc:
        log.warning("%s: regression failed: %s", context, exc)
        return None


def _mask_valid_binary(x: np.ndarray) -> np.ndarray:
    return np.isfinite(x) & np.isin(x, [0.0, 1.0])


def _rank_biserial_from_diffs(d: np.ndarray) -> float:
    """Rank-biserial correlation for paired differences (Wilcoxon-style).

    Returns in [-1, 1]. If all diffs are 0 or non-finite, returns NaN.
    """
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d)]
    d = d[d != 0.0]
    n = int(d.size)
    if n == 0:
        return np.nan
    r = sp_stats.rankdata(np.abs(d))
    w_pos = float(r[d > 0].sum())
    w_neg = float(r[d < 0].sum())
    tot = w_pos + w_neg
    if tot <= 0:
        return np.nan
    return (w_pos - w_neg) / tot


def _sex_scatter(ax, x, y, sex, swap=None, alpha=SCATTER_ALPHA, s=16):
    """Scatter coloured by sex, optionally shaped by swap."""
    for sx, col in SEX_COLORS.items():
        mask_sex = sex == sx
        if swap is not None:
            for sv, mk in [(0, SWAP0_MARKER), (1, SWAP1_MARKER)]:
                sel = mask_sex & (swap == sv)
                if sel.sum() == 0:
                    continue
                ax.scatter(x[sel], y[sel], c=col, marker=mk, s=s,
                           alpha=alpha, edgecolors="none",
                           label=f"{sx}, swap={sv}")
        else:
            ax.scatter(x[mask_sex], y[mask_sex], c=col, marker="o", s=s,
                       alpha=alpha, edgecolors="none", label=sx)


def _box_swap(ax, vals_0, vals_1, positions=(0, 1)):
    """Draw side-by-side box plots for swap=0 vs swap=1."""
    bp = ax.boxplot(
        [vals_0[np.isfinite(vals_0)], vals_1[np.isfinite(vals_1)]],
        positions=positions,
        widths=0.45,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="k", linewidth=1.2),
    )
    for patch, col in zip(bp["boxes"], ["0.82", "0.65"]):
        patch.set_facecolor(col)
        patch.set_edgecolor("0.3")
        patch.set_linewidth(0.6)
    ax.set_xticks(positions)
    ax.set_xticklabels(["swap=0", "swap=1"])


def _annotate_mwu(ax, x0, x1, *, prefix="", y_frac=0.95):
    """Add MWU annotation at top of axes."""
    u, p = mann_whitney_test(x0, x1)
    d, mag = cliffs_delta(x0, x1)
    txt = f"{prefix}MWU p={p:.2e}, Δ={d:+.2f} ({mag})"
    ax.text(0.5, y_frac, txt, transform=ax.transAxes, ha="center",
            fontsize=SMALL_ANNOT_SIZE, style="italic")
    return u, p, d, mag


def _build_labour_specificity_base(df_agg: pd.DataFrame) -> pd.DataFrame:
    """Return per-subject LAB-vs-standing deltas used by Fig2-style plots."""
    _require_columns(
        df_agg,
        ["subject_idx", "category", "P99_SED", "exceedance_volfrac_SED", "swap_9_10", "sex", "age"],
        context="Labour-specificity base input",
    )

    stand = df_agg[df_agg["category"] == "standing"].set_index("subject_idx")
    labour = df_agg[df_agg["category"] == "labour"].set_index("subject_idx")
    common = stand.index.intersection(labour.index)

    base = pd.DataFrame(
        {
            "subject_idx": common.to_numpy(),
            "swap": labour.loc[common, "swap_9_10"].to_numpy(),
            "sex": labour.loc[common, "sex"].astype(str).to_numpy(),
            "age": labour.loc[common, "age"].to_numpy(),
            "P99_lab": labour.loc[common, "P99_SED"].to_numpy(),
            "P99_stand": stand.loc[common, "P99_SED"].to_numpy(),
            "ex_lab": labour.loc[common, "exceedance_volfrac_SED"].to_numpy(),
            "ex_stand": stand.loc[common, "exceedance_volfrac_SED"].to_numpy(),
        }
    )

    p99_lab = _coerce_numeric(base["P99_lab"], name="P99_lab")
    p99_st = _coerce_numeric(base["P99_stand"], name="P99_stand")
    delta_log = np.full(len(base), np.nan, dtype=float)
    ok = np.isfinite(p99_lab) & np.isfinite(p99_st) & (p99_lab > 0) & (p99_st > 0)
    delta_log[ok] = np.log10(p99_lab[ok]) - np.log10(p99_st[ok])
    base["delta_logP99"] = delta_log
    base["delta_exceed"] = _coerce_numeric(base["ex_lab"], name="ex_lab") - _coerce_numeric(
        base["ex_stand"], name="ex_stand"
    )
    return base


# ===================================================================
# FIGURE 1 — Swap reduces LAB upper-tail extremes
# ===================================================================
def fig1_swap_tail(df: pd.DataFrame, fig_dir: Path, dpi: int,
                   stats_rows: list[dict], *, domain: str) -> dict:
    """FIG 1: box/violin + scatter of tail metrics by swap_9_10."""
    _require_columns(
        df,
        ["swap_9_10", "sex", "P99_SED", "top1pct_mean_SED"],
        context="Fig1 input",
    )
    metrics = ["P99_SED", "top1pct_mean_SED"]
    metric_pretty = {
        "P99_SED": "P99 SED",
        "top1pct_mean_SED": "Top-1% mean SED",
    }
    paths = []
    for metric in metrics:
        fig, ax = plt.subplots(figsize=(HALF_WIDTH, HALF_WIDTH * 0.85))
        raw = _coerce_numeric(df[metric], name=metric)
        swap_raw = _coerce_numeric(df["swap_9_10"], name="swap_9_10")
        sex = df["sex"].astype(str).to_numpy()

        mask = np.isfinite(raw) & (raw > 0) & _mask_valid_binary(swap_raw)
        n_drop = int((~mask).sum())
        if n_drop:
            log.warning("Fig1 %s: dropped %d/%d rows (non-finite/non-positive metric or invalid swap).",
                        metric, n_drop, len(df))

        log_vals = np.log10(raw[mask])
        swap = swap_raw[mask].astype(int, copy=False)
        sex = sex[mask]

        v0 = log_vals[swap == 0]
        v1 = log_vals[swap == 1]

        # Box
        _box_swap(ax, v0, v1)

        # Jittered scatter coloured by sex
        rng = np.random.default_rng(0)
        for sv, pos in [(0, 0), (1, 1)]:
            sel = swap == sv
            for sx, col in SEX_COLORS.items():
                idx = sel & (sex == sx)
                n_pts = idx.sum()
                if n_pts == 0:
                    continue
                ax.scatter(
                    pos + rng.uniform(-0.15, 0.15, size=n_pts),
                    log_vals[idx],
                    c=col, s=12, alpha=0.45, edgecolors="none",
                    label=sx if sv == 0 else None, zorder=3,
                )

        pretty = metric_pretty.get(metric, metric)
        ax.set_ylabel(f"log₁₀({pretty})")
        ax.set_title(f"{pretty} ({domain}, LAB)", fontweight="bold")

        # Overall MWU
        if len(v0) >= 3 and len(v1) >= 3:
            _, p, d, mag = _annotate_mwu(ax, v0, v1, prefix="All: ", y_frac=0.97)
            append_stats_row(
                stats_rows,
                figure="Fig1",
                description=f"All: {metric} swap0 vs swap1",
                N=int(len(v0) + len(v1)),
                effect_size=d,
                effect_label=f"Cliff_d ({mag})",
                p_value=p,
                method="Mann-Whitney U",
                domain=domain,
            )
            d_full, d_lo, d_hi = cliffs_delta_ci(v0, v1)
            append_stats_row(
                stats_rows,
                figure="Fig1",
                description=f"All: {metric} Cliff CI",
                N=int(len(v0) + len(v1)),
                effect_size=f"{d_full:.3f} [{d_lo:.3f}, {d_hi:.3f}]",
                effect_label="Cliff_d_CI",
                p_value="",
                method="Bootstrap 2000",
                domain=domain,
            )

        # Within-male
        m_mask = sex == "M"
        v0m = log_vals[m_mask & (swap == 0)]
        v1m = log_vals[m_mask & (swap == 1)]
        if len(v0m) >= 3 and len(v1m) >= 3:
            _, p_m, d_m, mag_m = _annotate_mwu(ax, v0m, v1m, prefix="Males: ", y_frac=0.91)
            append_stats_row(
                stats_rows,
                figure="Fig1",
                description=f"Males: {metric} swap0 vs swap1",
                N=int(len(v0m) + len(v1m)),
                effect_size=d_m,
                effect_label=f"Cliff_d ({mag_m})",
                p_value=p_m,
                method="Mann-Whitney U (within males)",
                domain=domain,
            )

        # Within-female (if N>=10 per group)
        f_mask = sex == "F"
        v0f = log_vals[f_mask & (swap == 0)]
        v1f = log_vals[f_mask & (swap == 1)]
        if len(v0f) >= 10 and len(v1f) >= 10:
            _, p_f, d_f, mag_f = _annotate_mwu(ax, v0f, v1f, prefix="Females: ", y_frac=0.85)
            append_stats_row(
                stats_rows,
                figure="Fig1",
                description=f"Females: {metric} swap0 vs swap1",
                N=int(len(v0f) + len(v1f)),
                effect_size=d_f,
                effect_label=f"Cliff_d ({mag_f})",
                p_value=p_f,
                method="Mann-Whitney U (within females)",
                domain=domain,
            )

        # Sample sizes per swap group (aggregate only; avoids sex×swap breakdown).
        ax.text(
            0.02,
            0.02,
            f"n0={len(v0)}, n1={len(v1)}",
            transform=ax.transAxes,
            fontsize=SMALL_ANNOT_SIZE,
            color="0.25",
        )

        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc="upper right", fontsize=SMALL_ANNOT_SIZE, markerscale=1.2)

        tag = metric.replace("_SED", "").lower()
        p_out = _savefig(fig, fig_dir, f"fig1_swap_tail_{domain}_{tag}", dpi)
        paths.append(p_out)

    return {
        "files": [str(p) for p in paths],
        "purpose": f"Swap in 9-10 block vs LAB upper-tail extremes in {domain} (box + scatter, two metrics)."
    }


# ===================================================================
# FIGURE 1+2 DASHBOARD — Combined swap + specificity
# ===================================================================
def fig12_swap_specificity_dashboard(
    df_lab: pd.DataFrame,
    df_agg: pd.DataFrame,
    fig_dir: Path,
    dpi: int,
    stats_rows: list[dict],
    *,
    domain: str,
) -> dict:
    """Combined Fig1/Fig2 dashboard with compact titles and larger panels."""
    _require_columns(
        df_lab,
        ["swap_9_10", "sex", "P99_SED", "top1pct_mean_SED"],
        context="Fig1+2 dashboard input",
    )
    base = _build_labour_specificity_base(df_agg)

    fig, axs = plt.subplots(2, 2, figsize=(FULL_WIDTH, 4.0 * ROW_H))
    axes = axs.ravel()
    rng = np.random.default_rng(42)

    panel_specs: list[dict[str, object]] = []

    for metric, ylabel, panel in [
        ("P99_SED", r"log$_{10}$(P99 SED)", "A. P99 tail (LAB)"),
        ("top1pct_mean_SED", r"log$_{10}$(Top-1% mean SED)", "B. Top-1% tail (LAB)"),
    ]:
        raw = _coerce_numeric(df_lab[metric], name=metric)
        swap_raw = _coerce_numeric(df_lab["swap_9_10"], name="swap_9_10")
        sex = df_lab["sex"].astype(str).to_numpy()
        mask = np.isfinite(raw) & (raw > 0) & _mask_valid_binary(swap_raw)
        panel_specs.append(
            {
                "vals": np.log10(raw[mask]),
                "swap": swap_raw[mask].astype(int, copy=False),
                "sex": sex[mask],
                "ylabel": ylabel,
                "panel": panel,
                "tag": metric,
                "zero_line": False,
                "figure_tag": "Fig12",
            }
        )

    for metric, ylabel, panel in [
        ("delta_logP99", r"$\Delta$ log$_{10}$(P99) [LAB-SP]", "C. LAB specificity (P99)"),
        ("delta_exceed", r"$\Delta$ exceedance [LAB-SP]", "D. LAB specificity (exceedance)"),
    ]:
        vals_raw = _coerce_numeric(base[metric], name=metric)
        swap_raw = _coerce_numeric(base["swap"], name="swap_9_10")
        sex = base["sex"].astype(str).to_numpy()
        mask = np.isfinite(vals_raw) & _mask_valid_binary(swap_raw)
        panel_specs.append(
            {
                "vals": vals_raw[mask],
                "swap": swap_raw[mask].astype(int, copy=False),
                "sex": sex[mask],
                "ylabel": ylabel,
                "panel": panel,
                "tag": metric,
                "zero_line": True,
                "figure_tag": "Fig12",
            }
        )

    for i, (ax, spec) in enumerate(zip(axes, panel_specs)):
        vals = np.asarray(spec["vals"], dtype=float)
        swap = np.asarray(spec["swap"], dtype=int)
        sex = np.asarray(spec["sex"], dtype=str)

        v0 = vals[swap == 0]
        v1 = vals[swap == 1]
        _box_swap(ax, v0, v1)

        for sv, pos in [(0, 0), (1, 1)]:
            sel = swap == sv
            for sx, col in SEX_COLORS.items():
                idx = sel & (sex == sx)
                n_pts = int(idx.sum())
                if n_pts == 0:
                    continue
                ax.scatter(
                    pos + rng.uniform(-0.14, 0.14, size=n_pts),
                    vals[idx],
                    c=col,
                    s=11,
                    alpha=0.45,
                    edgecolors="none",
                    label=sx if (sv == 0 and i == 0) else None,
                    zorder=3,
                )

        if bool(spec["zero_line"]):
            ax.axhline(0.0, color="0.5", ls="--", lw=0.6, zorder=0)

        ax.set_ylabel(str(spec["ylabel"]))
        panel_label(ax, str(spec["panel"]))

        if len(v0) >= 3 and len(v1) >= 3:
            _, p, d, mag = _annotate_mwu(ax, v0, v1, prefix="All: ", y_frac=0.97)
            append_stats_row(
                stats_rows,
                figure=str(spec["figure_tag"]),
                description=f"All: {spec['tag']} swap0 vs swap1",
                N=int(len(v0) + len(v1)),
                effect_size=d,
                effect_label=f"Cliff_d ({mag})",
                p_value=p,
                method="Mann-Whitney U",
                domain=domain,
            )

        ax.text(
            0.02,
            0.02,
            f"n0={len(v0)}, n1={len(v1)}",
            transform=ax.transAxes,
            fontsize=SMALL_ANNOT_SIZE,
            color="0.25",
        )

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(handles, labels, loc="upper right", fontsize=SMALL_ANNOT_SIZE, title="Sex")

    p_out = _savefig(fig, fig_dir, f"fig12_swap_specificity_dashboard_{domain}", dpi)
    return {
        "files": [str(p_out)],
        "purpose": f"Combined dashboard for swap-vs-tail (Fig1) and LAB-specificity deltas (Fig2) in {domain}.",
    }


# ===================================================================
# FIGURE 3 — Mechanism bridge: within-subspace routing
# ===================================================================
def fig3_mechanism_bridge(df_agg: pd.DataFrame, routing: pd.DataFrame,
                          fig_dir: Path, dpi: int,
                          stats_rows: list[dict], *, domain: str) -> dict:
    """FIG 3: routing variable (theta / ratio) vs tail (association + adjusted models)."""
    # Merge
    lab = df_agg[df_agg["category"] == "labour"].copy()
    lab = lab.merge(routing, on="subject_idx", how="inner")

    _require_columns(
        lab,
        ["P99_SED", "swap_9_10", "sex", "age", "gap_in_9_10", "dG_9_10", "theta", "ratio"],
        context="Fig3 merged input",
    )
    p99 = _coerce_numeric(lab["P99_SED"], name="P99_SED")
    swap_raw = _coerce_numeric(lab["swap_9_10"], name="swap_9_10")
    sex = lab["sex"].astype(str).to_numpy()
    age = _coerce_numeric(lab["age"], name="age")
    gap = _coerce_numeric(lab["gap_in_9_10"], name="gap_in_9_10")
    dG = _coerce_numeric(lab["dG_9_10"], name="dG_9_10")

    base_mask = np.isfinite(p99) & (p99 > 0) & _mask_valid_binary(swap_raw) & np.isfinite(age) & np.isfinite(gap) & np.isfinite(dG)
    if int((~base_mask).sum()):
        log.warning("Fig3: dropped %d/%d rows (non-finite/non-positive P99_SED, invalid swap, or missing covariates).",
                    int((~base_mask).sum()), len(lab))

    lab = lab.loc[base_mask].copy()
    log_p99 = np.log10(_coerce_numeric(lab["P99_SED"], name="P99_SED"))
    swap = _coerce_numeric(lab["swap_9_10"], name="swap_9_10").astype(int, copy=False)
    sex = lab["sex"].astype(str).to_numpy()

    paths = []
    for xvar, xlabel in [
        ("theta", "Mixing angle θ = atan2(|a₁₀|, |a₉|)  [rad]"),
        ("ratio", "Energy partition  E₉ / (E₉ + E₁₀)"),
    ]:
        xvals_raw = _coerce_numeric(lab[xvar], name=xvar)
        mask = np.isfinite(xvals_raw) & np.isfinite(log_p99)
        xvals = xvals_raw[mask]
        yvals = log_p99[mask]
        sex_m = sex[mask]
        swap_m = swap[mask]
        age_m = _coerce_numeric(lab["age"], name="age")[mask]
        gap_m = _coerce_numeric(lab["gap_in_9_10"], name="gap_in_9_10")[mask]
        dG_m = _coerce_numeric(lab["dG_9_10"], name="dG_9_10")[mask]

        fig, ax = plt.subplots(figsize=(HALF_WIDTH, HALF_WIDTH * 0.85))

        _sex_scatter(ax, xvals, yvals, sex_m, swap=swap_m, s=14)

        # LOWESS trend
        xs, ys = lowess_smooth(xvals, yvals, frac=0.45)
        if len(xs) > 0:
            ax.plot(xs, ys, color="k", lw=1.4, ls="-", zorder=5, label="LOWESS")

        # Spearman
        rho, p_rho = spearman_test(xvals, yvals)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(f"log₁₀(P99 SED)  [{domain}, LAB]")
        ax.set_title("Routing ↔ tail (association)", fontweight="bold")
        ax.text(0.02, 0.97, f"Spearman ρ={rho:.3f}, p={p_rho:.2e}",
                transform=ax.transAxes, fontsize=SMALL_ANNOT_SIZE, va="top")
        append_stats_row(stats_rows, figure="Fig3",
                         description=f"All: {xvar} vs log10(P99_SED)",
                         N=int(np.isfinite(xvals).sum()), effect_size=rho, effect_label="Spearman_rho",
                         p_value=p_rho, method="Spearman", domain=domain)

        # Within-male
        m_mask = sex_m == "M"
        rho_m, p_m = spearman_test(xvals[m_mask], yvals[m_mask])
        ax.text(0.02, 0.91, f"Males ρ={rho_m:.3f}, p={p_m:.2e}",
                transform=ax.transAxes, fontsize=SMALL_ANNOT_SIZE, va="top", color=MALE_COLOR)
        append_stats_row(stats_rows, figure="Fig3",
                         description=f"Males: {xvar} vs log10(P99_SED)",
                         N=int(np.isfinite(xvals[m_mask]).sum()), effect_size=rho_m,
                         effect_label="Spearman_rho", p_value=p_m,
                         method="Spearman (within males)", domain=domain)

        f_mask = sex_m == "F"
        rho_f, p_f = spearman_test(xvals[f_mask], yvals[f_mask])
        ax.text(0.02, 0.85, f"Females ρ={rho_f:.3f}, p={p_f:.2e}",
                transform=ax.transAxes, fontsize=SMALL_ANNOT_SIZE, va="top", color=FEMALE_COLOR)
        append_stats_row(
            stats_rows,
            figure="Fig3",
            description=f"Females: {xvar} vs log10(P99_SED)",
            N=int(np.isfinite(xvals[f_mask]).sum()),
            effect_size=rho_f,
            effect_label="Spearman_rho",
            p_value=p_f,
            method="Spearman (within females)",
            domain=domain,
        )

        # Adjusted models (robust SE): include operating-point + demographics
        reg_df = pd.DataFrame({
            "log_P99": yvals,
            "x": xvals,
            "swap": swap_m,
            "sex_F": (sex_m == "F").astype(int),
            "age": age_m,
            "gap": gap_m,
            "dG": dG_m,
        })
        reg_df = _prepare_ols_df(reg_df, min_n=20, context=f"Fig3 {xvar}")
        if reg_df is not None:
            mod_base = _fit_ols_model(
                reg_df, "log_P99 ~ x + swap + sex_F + age + gap + dG",
                context=f"Fig3 {xvar} base",
            )
            mod_int = _fit_ols_model(
                reg_df, "log_P99 ~ x * swap + sex_F + age + gap + dG",
                context=f"Fig3 {xvar} interaction",
            )
            if mod_base is not None and mod_int is not None:
                ax.text(
                    0.02,
                    0.02,
                    f"OLS x coef={mod_base.params['x']:+.3f}, p={mod_base.pvalues['x']:.2e}\n"
                    f"x×swap p={mod_int.pvalues.get('x:swap', np.nan):.2e}",
                    transform=ax.transAxes,
                    fontsize=SMALL_ANNOT_SIZE,
                    va="bottom",
                )
                append_stats_row(
                    stats_rows,
                    figure="Fig3",
                    description=f"OLS log10(P99_SED)~{xvar}+swap+sex+age+gap+dG: coef_x",
                    N=int(mod_base.nobs),
                    effect_size=float(mod_base.params["x"]),
                    effect_label="OLS_coef_x",
                    p_value=float(mod_base.pvalues["x"]),
                    method="OLS HC3",
                    domain=domain,
                )
                append_stats_row(
                    stats_rows,
                    figure="Fig3",
                    description=f"OLS interaction log10(P99_SED)~{xvar}*swap+sex+age+gap+dG: coef_x:swap",
                    N=int(mod_int.nobs),
                    effect_size=float(mod_int.params.get("x:swap", np.nan)),
                    effect_label="OLS_coef_x:swap",
                    p_value=float(mod_int.pvalues.get("x:swap", np.nan)),
                    method="OLS HC3 (interaction)",
                    domain=domain,
                )

        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="lower right", fontsize=SMALL_ANNOT_SIZE, ncol=2, markerscale=1.0)

        p_out = _savefig(fig, fig_dir, f"fig3_mechanism_{domain}_{xvar}", dpi)
        paths.append(p_out)

    return {
        "files": [str(p) for p in paths],
        "purpose": f"Routing variables (theta/ratio in 9-10 block) vs tail cost in {domain}; includes unadjusted correlations and adjusted OLS."
    }


# ===================================================================
# FIGURE 4 — Operating point: gap/dG vs tail
# ===================================================================
def fig4_operating_point(df_agg: pd.DataFrame, fig_dir: Path, dpi: int,
                         stats_rows: list[dict], *, domain: str) -> dict:
    """FIG 4: gap/dG vs log10(P99_SED) with swap overlay."""
    lab = df_agg[df_agg["category"] == "labour"].copy()
    _require_columns(lab, ["P99_SED", "swap_9_10", "sex", "gap_in_9_10", "dG_9_10"], context="Fig4 input")
    p99 = _coerce_numeric(lab["P99_SED"], name="P99_SED")
    swap_raw = _coerce_numeric(lab["swap_9_10"], name="swap_9_10")
    sex = lab["sex"].astype(str).to_numpy()
    mask = np.isfinite(p99) & (p99 > 0) & _mask_valid_binary(swap_raw)
    if int((~mask).sum()):
        log.warning("Fig4: dropped %d/%d rows (non-finite/non-positive P99_SED or invalid swap).",
                    int((~mask).sum()), len(lab))
    lab = lab.loc[mask].copy()
    log_p99 = np.log10(_coerce_numeric(lab["P99_SED"], name="P99_SED"))
    swap = _coerce_numeric(lab["swap_9_10"], name="swap_9_10").astype(int, copy=False)
    sex = lab["sex"].astype(str).to_numpy()

    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, 2 * ROW_H), layout="constrained")
    paths = []

    for ax, xvar, xlabel in zip(
        axes,
        ["gap_in_9_10", "dG_9_10"],
        ["gap_in(9-10)", "Grassmann distance d_G(9-10)"],
    ):
        xvals = lab[xvar].values.astype(float)
        _sex_scatter(ax, xvals, log_p99, sex, swap=swap, s=14)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(f"log₁₀(P99 SED)  [{domain}, LAB]")

        # Spearman overall
        rho, p_rho = spearman_test(xvals, log_p99)
        ax.text(0.02, 0.97, f"All: ρ={rho:.3f}, p={p_rho:.2e}",
                transform=ax.transAxes, fontsize=SMALL_ANNOT_SIZE, va="top")
        append_stats_row(stats_rows, figure="Fig4",
                         description=f"All: {xvar} vs log10(P99_SED)",
                         N=int(np.isfinite(xvals).sum()), effect_size=rho, effect_label="Spearman_rho",
                         p_value=p_rho, method="Spearman", domain=domain)

        # Within-male
        mm = sex == "M"
        rho_m, p_m = spearman_test(xvals[mm], log_p99[mm])
        ax.text(0.02, 0.91, f"M: ρ={rho_m:.3f}, p={p_m:.2e}",
                transform=ax.transAxes, fontsize=SMALL_ANNOT_SIZE, va="top", color=MALE_COLOR)
        append_stats_row(stats_rows, figure="Fig4",
                         description=f"Males: {xvar} vs log10(P99_SED)",
                         N=int(np.isfinite(xvals[mm]).sum()), effect_size=rho_m,
                         effect_label="Spearman_rho", p_value=p_m,
                         method="Spearman (males)", domain=domain)

        # Within-female
        fm = sex == "F"
        rho_f, p_f = spearman_test(xvals[fm], log_p99[fm])
        ax.text(0.02, 0.85, f"F: ρ={rho_f:.3f}, p={p_f:.2e}",
                transform=ax.transAxes, fontsize=SMALL_ANNOT_SIZE, va="top", color=FEMALE_COLOR)
        append_stats_row(stats_rows, figure="Fig4",
                         description=f"Females: {xvar} vs log10(P99_SED)",
                         N=int(np.isfinite(xvals[fm]).sum()), effect_size=rho_f,
                         effect_label="Spearman_rho", p_value=p_f,
                         method="Spearman (females)", domain=domain)

    axes[0].set_title("(a) Gap vs tail", fontweight="bold")
    axes[1].set_title("(b) Grassmann distance vs tail", fontweight="bold")
    if axes[1].get_legend_handles_labels()[0]:
        axes[1].legend(loc="lower right", fontsize=SMALL_ANNOT_SIZE, ncol=2, markerscale=1.0)

    p_out = _savefig(fig, fig_dir, f"fig4_operating_point_{domain}", dpi)
    paths.append(p_out)

    return {
        "files": [str(p) for p in paths],
        "purpose": f"Operating-point (gap_in, d_G) correlates with tail in {domain}, swap overlaid."
    }


# ===================================================================
# FIGURE 5 — Sex effect + swap as internal valve
# ===================================================================
def fig5_sex_valve(df_agg: pd.DataFrame, fig_dir: Path, dpi: int,
                   stats_rows: list[dict], *, domain: str) -> dict:
    """FIG 5: sex distribution + regression controlling for swap."""
    lab = df_agg[df_agg["category"] == "labour"].copy()
    _require_columns(
        lab,
        ["P99_SED", "swap_9_10", "gap_in_9_10", "dG_9_10", "age", "sex"],
        context="Fig5 input",
    )
    p99 = _coerce_numeric(lab["P99_SED"], name="P99_SED")
    mask = np.isfinite(p99) & (p99 > 0)
    if int((~mask).sum()):
        log.warning("Fig5: dropped %d/%d rows (non-finite/non-positive P99_SED).",
                    int((~mask).sum()), len(lab))
    lab = lab.loc[mask].copy()
    log_p99 = np.log10(_coerce_numeric(lab["P99_SED"], name="P99_SED"))
    sex = lab["sex"].astype(str).to_numpy()

    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, 2 * ROW_H),
                             gridspec_kw={"width_ratios": [1.3, 1]},
                             layout="constrained")
    ax_hist, ax_reg = axes

    # ── Panel (a): overlapping histograms ─────────────────────
    bins = np.linspace(np.nanmin(log_p99) - 0.1, np.nanmax(log_p99) + 0.1, 35)
    for sx, col in SEX_COLORS.items():
        ax_hist.hist(log_p99[sex == sx], bins=bins, color=col, alpha=FILL_ALPHA,
                     label=sx, edgecolor="white", linewidth=0.3)
    ax_hist.set_xlabel(f"log₁₀(P99 SED)  [{domain}, LAB]")
    ax_hist.set_ylabel("Count")
    panel_label(ax_hist, "(a) Sex distribution of tail cost")
    ax_hist.legend(fontsize=SMALL_ANNOT_SIZE)

    # MWU sex test
    xm = log_p99[sex == "M"]
    xf = log_p99[sex == "F"]
    if len(xm) >= 3 and len(xf) >= 3:
        _, pm = mann_whitney_test(xm, xf)
        dm, magm = cliffs_delta(xm, xf)
        ax_hist.text(0.02, 0.95, f"MWU p={pm:.2e}, Δ={dm:+.2f} ({magm})",
                     transform=ax_hist.transAxes, fontsize=SMALL_ANNOT_SIZE, va="top")
        append_stats_row(stats_rows, figure="Fig5",
                         description="Sex diff log10(P99_SED) M vs F",
                         N=int(len(xm) + len(xf)), effect_size=dm, effect_label=f"Cliff_d ({magm})",
                         p_value=pm, method="Mann-Whitney U", domain=domain)

    # ── Panel (b): regression coefficients ────────────────────
    reg_df = lab[["P99_SED", "swap_9_10", "gap_in_9_10", "dG_9_10", "age", "sex"]].copy()
    reg_df["P99_SED"] = _coerce_numeric(reg_df["P99_SED"], name="P99_SED")
    reg_df = reg_df[reg_df["P99_SED"] > 0].copy()
    reg_df = _clean_complete_cases(reg_df)
    reg_df["log_P99"] = np.log10(reg_df["P99_SED"].values.astype(float))
    reg_df["sex_F"] = (reg_df["sex"] == "F").astype(int)

    # Standardize continuous predictors to make coefficients comparable
    for c in ["gap_in_9_10", "dG_9_10", "age"]:
        reg_df[c] = pd.to_numeric(reg_df[c], errors="coerce")
    reg_df = reg_df.dropna(subset=["swap_9_10", "gap_in_9_10", "dG_9_10", "age", "sex_F", "log_P99"])
    for c in ["gap_in_9_10", "dG_9_10", "age"]:
        mu = float(reg_df[c].mean())
        sd = float(reg_df[c].std(ddof=0))
        reg_df[c] = (reg_df[c] - mu) / (sd if sd > 0 else 1.0)

    coefs_data = []
    reg_df = _prepare_ols_df(reg_df, min_n=20, context="Fig5")
    if reg_df is not None:
        mod_full = _fit_ols_model(
            reg_df,
            "log_P99 ~ sex_F + swap_9_10 + gap_in_9_10 + dG_9_10 + age",
            context="Fig5",
        )
        if mod_full is not None:
            for var in ["sex_F", "swap_9_10", "gap_in_9_10", "dG_9_10", "age"]:
                coefs_data.append({
                    "variable": var,
                    "coef": mod_full.params[var],
                    "se": mod_full.bse[var],
                    "p": mod_full.pvalues[var],
                })
                append_stats_row(stats_rows, figure="Fig5",
                                 description=f"OLS coef: {var}",
                                 N=int(mod_full.nobs), effect_size=mod_full.params[var],
                                 effect_label="OLS_coef",
                                 p_value=mod_full.pvalues[var],
                                 method="OLS HC3",
                                 notes=f"SE={mod_full.bse[var]:.4f} (gap/dG/age standardized)",
                                 domain=domain)

    if coefs_data:
        cdf = pd.DataFrame(coefs_data)
        y_pos = np.arange(len(cdf))
        ax_reg.barh(y_pos, cdf["coef"], xerr=1.96 * cdf["se"],
                    color="0.55", edgecolor="0.3", linewidth=0.5, height=0.6,
                    capsize=2)
        ax_reg.set_yticks(y_pos)
        ax_reg.set_yticklabels(cdf["variable"], fontsize=SMALL_ANNOT_SIZE)
        ax_reg.axvline(0, color="k", lw=0.5, ls="--")
        ax_reg.set_xlabel("OLS coefficient (HC3 SE)")
        panel_label(ax_reg, "(b) Adjusted coefficients")
        for i, row in cdf.iterrows():
            stars = "***" if row["p"] < 0.001 else ("**" if row["p"] < 0.01 else ("*" if row["p"] < 0.05 else ""))
            ax_reg.text(row["coef"] + 1.96 * row["se"] + 0.01, i,
                        f"p={row['p']:.2e}{stars}", fontsize=SMALL_ANNOT_SIZE, va="center")

    p_out = _savefig(fig, fig_dir, f"fig5_sex_valve_{domain}", dpi)

    return {
        "files": [str(p_out)],
        "purpose": f"Sex effect on tail in {domain} with adjusted regression controlling for swap, gap, dG, age."
    }


# ===================================================================
# MATCHED SUBSET — replot Fig 1 for propensity-matched males
# ===================================================================
def fig1_matched(df_agg: pd.DataFrame, fig_dir: Path, dpi: int,
                 stats_rows: list[dict], *, domain: str) -> dict:
    """Replot Fig 1 for nearest-neighbor matched males."""
    lab = df_agg[df_agg["category"] == "labour"].copy()
    males = lab[lab["sex"] == "M"].copy().reset_index(drop=True)

    covariates = ["gap_in_9_10", "dG_9_10", "age"]
    # Check all covariates exist
    for c in covariates:
        if c not in males.columns:
            log.warning("Covariate %s not in data; skipping matching.", c)
            return {"files": [], "purpose": "Skipped — missing covariate."}

    matched = nearest_neighbor_match(males, "swap_9_10", covariates, caliper=1.0)
    if len(matched) < 10 or "_pair_id" not in matched.columns:
        log.warning("Matched N=%d too small; skipping.", len(matched))
        return {"files": [], "purpose": "Skipped — too few matched pairs."}

    p99 = pd.to_numeric(matched["P99_SED"], errors="coerce").to_numpy(float)
    swap_raw = pd.to_numeric(matched["swap_9_10"], errors="coerce").to_numpy(float)
    ok = np.isfinite(p99) & (p99 > 0) & _mask_valid_binary(swap_raw)
    matched = matched.loc[ok].copy()
    matched["log_P99"] = np.log10(pd.to_numeric(matched["P99_SED"], errors="coerce").to_numpy(float))

    fig, ax = plt.subplots(figsize=(HALF_WIDTH, HALF_WIDTH * 0.85))
    pairs: list[tuple[float, float]] = []
    for pid, sub in matched.groupby("_pair_id"):
        if sub["swap_9_10"].nunique() != 2:
            continue
        y0 = float(sub.loc[sub["swap_9_10"] == 0, "log_P99"].iloc[0])
        y1 = float(sub.loc[sub["swap_9_10"] == 1, "log_P99"].iloc[0])
        if np.isfinite(y0) and np.isfinite(y1):
            pairs.append((y0, y1))
    if len(pairs) < 5:
        log.warning("Matched pairs after cleaning too small (n_pairs=%d); skipping.", len(pairs))
        plt.close(fig)
        return {"files": [], "purpose": "Skipped — too few clean matched pairs."}

    pairs_arr = np.asarray(pairs, dtype=float)
    v0 = pairs_arr[:, 0]
    v1 = pairs_arr[:, 1]
    _box_swap(ax, v0, v1)

    rng = np.random.default_rng(7)
    for y0, y1 in pairs_arr:
        j0 = float(rng.uniform(-0.08, 0.08))
        j1 = float(rng.uniform(-0.08, 0.08))
        ax.plot([0 + j0, 1 + j1], [y0, y1], color="0.7", lw=0.6, alpha=0.6, zorder=2)
        ax.scatter([0 + j0, 1 + j1], [y0, y1], c=MALE_COLOR, s=12, alpha=0.7,
                   edgecolors="none", zorder=3)

    ax.set_ylabel("log₁₀(P99 SED)")
    ax.set_title(f"Matched males ({domain}): paired swap effect", fontweight="bold")

    diff = v1 - v0
    nonzero_diff = diff[np.isfinite(diff) & (diff != 0.0)]
    if len(nonzero_diff) >= 3:
        _, p = sp_stats.wilcoxon(nonzero_diff, zero_method="wilcox", alternative="two-sided")
    else:
        p = np.nan
    rbc = _rank_biserial_from_diffs(diff)
    rng = np.random.default_rng(42)
    boots = rng.choice(diff, size=(2000, len(diff)), replace=True)
    med_boot = np.median(boots, axis=1)
    lo, hi = np.percentile(med_boot, [2.5, 97.5])

    ax.text(
        0.5,
        0.94,
        f"Wilcoxon p={p:.2e} | median Δ={np.median(diff):+.3f} [{lo:+.3f}, {hi:+.3f}] | RBC={rbc:+.2f}",
        transform=ax.transAxes,
        ha="center",
        fontsize=SMALL_ANNOT_SIZE,
        style="italic",
    )
    ax.text(0.5, 0.85, f"N matched={len(diff)} pairs", transform=ax.transAxes,
            ha="center", fontsize=SMALL_ANNOT_SIZE)

    append_stats_row(
        stats_rows,
        figure="Fig1_matched",
        description="Matched males: paired swap effect on log10(P99_SED)",
        N=int(len(diff)),
        effect_size=float(rbc) if np.isfinite(rbc) else np.nan,
        effect_label="Rank_biserial_corr",
        p_value=float(p) if np.isfinite(p) else np.nan,
        method="Wilcoxon signed-rank (paired)",
        notes=f"median_diff={np.median(diff):+.4f} CI[{lo:+.4f},{hi:+.4f}] (log10 units)",
        domain=domain,
    )

    p_out = _savefig(fig, fig_dir, f"fig1_matched_males_{domain}", dpi)

    return {
        "files": [str(p_out)],
        "purpose": f"Propensity-matched (gap, dG, age) male subset in {domain} with paired inference (Wilcoxon)."
    }


# ===================================================================
# LaTeX table export
# ===================================================================
def _write_swap_latex_table(
    stats_rows: list[dict], domain: str, table_dir: Path,
) -> None:
    """Write a LaTeX table of swap-test stats for the manuscript."""
    table_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(stats_rows)
    display_figure = "Fig12" if (df["figure"] == "Fig12").any() else "Fig1"
    display_order = [
        "All: P99_SED swap0 vs swap1",
        "All: top1pct_mean_SED swap0 vs swap1",
    ]
    display_mask = (
        (df["figure"] == display_figure)
        & df["description"].isin(display_order)
        & df["method"].str.contains("Mann-Whitney", na=False)
    )
    sub = df.loc[
        display_mask,
        ["figure", "description", "N", "effect_size", "effect_label", "p_value"],
    ].copy()
    if sub.empty:
        log.warning("No swap stats rows for LaTeX table.")
        return

    family_figures = [display_figure, "Fig3", "Fig4", "Fig5", "Fig1_matched"]
    family_mask = df["figure"].isin(family_figures)
    p_family = pd.to_numeric(df.loc[family_mask, "p_value"], errors="coerce")
    p_family_mask = p_family.notna()
    q_family = pd.Series(np.nan, index=df.index, dtype=float)
    if int(p_family_mask.sum()) > 0:
        from statsmodels.stats.multitest import multipletests

        q_family.loc[p_family.loc[p_family_mask].index] = multipletests(
            p_family.loc[p_family_mask].to_numpy(dtype=float), method="fdr_bh",
        )[1]
    sub["q_FDR"] = q_family.loc[sub.index].to_numpy()

    order_map = {desc: i for i, desc in enumerate(display_order)}
    sub["sort_key"] = sub["description"].map(order_map)
    sub = sub.sort_values(["sort_key", "description"]).drop(columns=["sort_key"])

    # Sanitize cell values for LaTeX (escape underscores, etc.)
    _tex_esc = str.maketrans({"_": r"\_", "&": r"\&", "%": r"\%", "#": r"\#"})
    for col in ("description", "effect_label"):
        sub[col] = sub[col].astype(str).map(lambda s: s.translate(_tex_esc))
    # Pretty metric names
    sub["description"] = (
        sub["description"]
        .str.replace(r"P99\_SED", r"$P_{99}$ SED", regex=False)
        .str.replace(r"top1pct\_mean\_SED", r"Top-1\% mean SED", regex=False)
        .str.replace(r"delta\_logP99", r"$\Delta\log P_{99}$", regex=False)
        .str.replace(r"delta\_exceed", r"$\Delta f_{\mathrm{exc}}$", regex=False)
    )
    sub["effect_label"] = sub["effect_label"].str.replace(r"Cliff\_d", r"Cliff $\delta$", regex=False)
    # Better column names for LaTeX
    sub = sub.rename(columns={
        "description": "Comparison",
        "effect_size": r"Cliff $\delta$",
        "effect_label": "Magnitude",
        "p_value": "$p$",
        "q_FDR": "$q$ (BH-FDR)",
    })
    sub = sub.drop(columns=["figure"], errors="ignore")
    tex_path = table_dir / f"tail_risk_swap_{domain}.tex"
    sub.to_latex(tex_path, index=False, escape=False, float_format="%.4f")
    log.info("Wrote LaTeX table: %s", tex_path)


# ===================================================================
# MAIN
# ===================================================================
def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    args = parse_args(argv)

    if args.dry_run:
        print("Dry run — imports OK.")
        return

    # ── Setup ────────────────────────────────────────────────
    setup_plot_style(dpi=args.dpi)
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ────────────────────────────────────────────
    log.info("Loading CSVs from %s", args.input_dir)
    df_agg = load_tail_aggregated_csv(args.input_dir)

    # Filter to requested domain
    domain = args.domain
    df_agg = df_agg[df_agg["group"] == domain].copy()

    # For Fig 1 we use the aggregated LAB category (more robust)
    df_agg_lab = df_agg[df_agg["category"] == "labour"].copy()

    # Load routing data
    routing_case = args.routing_case
    routing_dir = args.routing_data_dir
    log.info("Loading routing coefficients from Zarr: %s (case=%s)", routing_dir, routing_case)
    if routing_case in LOADCASE_INDEX:
        routing = build_routing_df(routing_case, data_dir=routing_dir)
    elif routing_case.lower() in {"labour_avg", "labour-avg", "avg"}:
        routing = build_routing_df_labour_avg(data_dir=routing_dir)
    else:
        raise ValueError(
            f"Unknown --routing_case '{routing_case}'. "
            f"Use one of {list(LOADCASE_INDEX)} or 'labour_avg'."
        )

    # Routing / CSV cohort sanity check
    n_subj_csv = int(df_agg["subject_idx"].nunique())
    n_subj_routing = int(routing["subject_idx"].nunique())
    if n_subj_csv != n_subj_routing:
        raise ValueError(
            "Subject count mismatch between CSV and routing Zarr: "
            f"CSV n={n_subj_csv}, routing n={n_subj_routing}. "
            "Ensure both refer to the same cohort ordering."
        )

    # ── Stats accumulator ────────────────────────────────────
    stats_rows: list[dict] = []
    manifest: list[dict] = []

    # ── Generate figures ─────────────────────────────────────
    log.info("=== FIG 1: Swap reduces LAB upper-tail extremes ===")
    m1 = fig1_swap_tail(df_agg_lab, args.fig_dir, args.dpi, stats_rows, domain=domain)
    manifest.append(m1)

    log.info("=== FIG 1+2 DASHBOARD: swap + LAB-specificity ===")
    m12 = fig12_swap_specificity_dashboard(
        df_agg_lab,
        df_agg,
        args.fig_dir,
        args.dpi,
        stats_rows,
        domain=domain,
    )
    manifest.append(m12)

    log.info("=== FIG 3: Mechanism bridge ===")
    m3 = fig3_mechanism_bridge(df_agg, routing, args.fig_dir, args.dpi, stats_rows, domain=domain)
    manifest.append(m3)

    log.info("=== FIG 4: Operating point ===")
    m4 = fig4_operating_point(df_agg, args.fig_dir, args.dpi, stats_rows, domain=domain)
    manifest.append(m4)

    log.info("=== FIG 5: Sex effect + valve ===")
    m5 = fig5_sex_valve(df_agg, args.fig_dir, args.dpi, stats_rows, domain=domain)
    manifest.append(m5)

    log.info("=== FIG 1 MATCHED: propensity-matched males ===")
    m6 = fig1_matched(df_agg, args.fig_dir, args.dpi, stats_rows, domain=domain)
    manifest.append(m6)

    # ── Write outputs ────────────────────────────────────────
    meta: dict[str, object] = {
        "script": str(Path(__file__).resolve()),
        "input_dir": str(args.input_dir),
        "fig_dir": str(args.fig_dir),
        "domain": domain,
        "routing_data_dir": str(routing_dir),
        "routing_case": routing_case,
        "dpi": int(args.dpi),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "matplotlib": plt.matplotlib.__version__,
        "scipy": _optional_package_version("scipy"),
        "statsmodels": _optional_package_version("statsmodels"),
        "sklearn": _optional_package_version("scikit-learn"),
    }
    try:
        meta["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(_PROJECT_ROOT),
            text=True,
        ).strip()
    except (FileNotFoundError, OSError, subprocess.CalledProcessError):
        meta["git_commit"] = None

    write_manifest(args.fig_dir, manifest, meta=meta)
    write_stats_summary(args.fig_dir, stats_rows)

    # ── LaTeX swap-test table for manuscript ─────────────────
    _write_swap_latex_table(stats_rows, domain, args.fig_dir.parent / "tables")

    log.info("Done. Figures in %s", args.fig_dir)


if __name__ == "__main__":
    main()
