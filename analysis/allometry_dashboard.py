#!/usr/bin/env python3
"""Allometry / body-size scaling dashboard.

Synthesises pre-computed results into a single publication-ready figure that
quantifies how isotropic pelvic scale (cbrt of volume ratio to template)
influences:

  A – Eigenvalue scaling exponents β_scale per mode with 95 % CI
  B – Variance decomposition (LMG R²) showing scale share per mode
  C – Mode-swap probability vs scale (logistic fit + sex stratification)
  D – Spectral gap_in(9–10) vs scale scatter with LOWESS + sex dimorphism
  E – SIJ micromotion metrics vs scale (rotation, translation)
  F – Energy-fraction sensitivity (pp change per IQR of log-scale, 5 loads)

All data are read from existing CSV / Excel / Zarr artefacts — nothing is
re-computed.  Output: analysis_outputs/figures/allometry_dashboard.pdf and
analysis_outputs/tables/allometry_summary.csv/.tex.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy import stats

from utils.plot_utils import (
    setup_plot_style,
    FULL_WIDTH,
    ROW_H,
    ROW_H_SMALL,
    ANNOT_SIZE,
    SMALL_ANNOT_SIZE,
    SCALE_EFFECT_COLOR,
    SEX_EFFECT_COLOR,
    SHAPE_EFFECT_COLOR,
    MATERIAL_EFFECT_COLOR,
    MALE_COLOR,
    FEMALE_COLOR,
    NEUTRAL_COLOR,
    SWAP_LINE_COLOR,
    SCATTER_ALPHA,
    LINE_WIDTH,
    BAR_EDGE_LW,
    BAR_EDGE_COLOR,
    panel_label,
    format_mode_label_short,
)

# ── Paths ─────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "results" / "ref_S1P_fixed_new2" / "data"
ANALYSIS_DIR = REPO_ROOT / "results" / "ref_S1P_fixed_new2" / "analysis"
OUT_DIR = REPO_ROOT / "analysis_outputs"
FIG_DIR = OUT_DIR / "figures"
TAB_DIR = OUT_DIR / "tables"

NUM_MODES = 15
ALPHA = 0.05
LOAD_CASES = ["SP2leg", "SP1leg", "LAB_phase1", "LAB_phase2", "LAB_phase3"]
LOAD_LABELS = ["SP 2-leg", "SP 1-leg", "LAB\u2081", "LAB\u2082", "LAB\u2083"]


# =====================================================================
# Data loading (all pre-computed)
# =====================================================================

def load_eigen_results() -> pd.DataFrame:
    """Load eigen_results.csv from the analysis directory."""
    return pd.read_csv(ANALYSIS_DIR / "eigen_results.csv").sort_values("mode").reset_index(drop=True)


def load_fractions_results() -> pd.DataFrame:
    """Load fractions_results.csv from the analysis directory."""
    return pd.read_csv(ANALYSIS_DIR / "fractions_results.csv")


def load_metadata() -> pd.DataFrame:
    """Merge allometry + demography into one per-subject table."""
    allo = pd.read_excel(DATA_DIR / "allometry.xlsx")
    demo = pd.read_excel(DATA_DIR / "demography.xlsx")
    demo["sex"] = demo["sex"].astype(str).str.strip().str.upper()
    return pd.merge(allo, demo, on="patient_id", how="inner")


def load_permutations() -> pd.DataFrame:
    """Load mode permutation table from eigen_data.xlsx."""
    return pd.read_excel(DATA_DIR / "eigen_data.xlsx", sheet_name="permutations")


def load_eigenvalues() -> pd.DataFrame:
    """Load eigenvalue modes table (278 × 15) from eigen_data.xlsx."""
    return pd.read_excel(DATA_DIR / "eigen_data.xlsx", sheet_name="modes")


def load_sij_subject_level() -> pd.DataFrame:
    """Load SIJ micromotion subject-level data."""
    path = OUT_DIR / "sij_micromotion" / "sij_micromotion_subject_level.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


# =====================================================================
# Derived quantities
# =====================================================================

def compute_swap_9_10(perm: pd.DataFrame) -> np.ndarray:
    """Binary indicator: does subject swap modes 9 and 10?

    Permutation columns are 0-based: perm_9 holds the reference-mode index
    that ended up at rank position 9 for each subject.  A swap means the
    reference mode at position 9 is not 8 (0-based mode 9) AND vice-versa.
    """
    p9 = perm["perm_9"].to_numpy()
    p10 = perm["perm_10"].to_numpy()
    # Identity: perm_9==8, perm_10==9 (0-based).  Swap: perm_9==9, perm_10==8.
    return ((p9 == 9) & (p10 == 8)).astype(float)


def compute_gap_in(eigenvalues: pd.DataFrame, pair: tuple[int, int] = (9, 10)) -> np.ndarray:
    """Relative spectral gap between pair of adjacent modes (1-based).

    Uses the canonical formula gap_in(i) = (λ_{i+1} - λ_i) / λ_i
    consistent with spectral_metrics.compute_gap_in.
    """
    a, b = pair
    la = eigenvalues[f"eig_{a}"].to_numpy(float)
    lb = eigenvalues[f"eig_{b}"].to_numpy(float)
    return (lb - la) / la


# =====================================================================
# Panel A — Scaling exponents β_scale per mode
# =====================================================================

def plot_panel_A(ax: plt.Axes, res: pd.DataFrame) -> None:
    """Bar chart of β_scale (log-log scale exponent) with 95 % CI."""
    x = np.arange(NUM_MODES)
    y = res["beta_scale"].to_numpy()
    lo = res["beta_scale_CI_lo"].to_numpy()
    hi = res["beta_scale_CI_hi"].to_numpy()
    yerr = np.vstack([y - lo, hi - y])

    bars = ax.bar(x, y, color=SCALE_EFFECT_COLOR, yerr=yerr, capsize=2,
                  edgecolor=BAR_EDGE_COLOR, linewidth=BAR_EDGE_LW)
    ax.axhline(0, color="0.4", lw=0.6, ls="--")

    # Classical elastic scaling reference line λ ~ L^(-2)
    ax.axhline(-2.0, color="0.6", lw=0.5, ls=":", label="elastic λ∝L⁻²")

    # Significance markers
    for i in range(NUM_MODES):
        sig = res["beta_scale_signif"].iloc[i]
        if sig:
            ax.text(i, hi[i] + 0.08, "★", ha="center", va="bottom",
                    fontsize=SMALL_ANNOT_SIZE, color=SCALE_EFFECT_COLOR)

    ax.set_xticks(x)
    ax.set_xticklabels([format_mode_label_short(i) for i in range(NUM_MODES)],
                       fontsize=ANNOT_SIZE)
    ax.set_ylabel(r"$\beta_{\rm scale}$ (log–log)")
    ax.legend(fontsize=SMALL_ANNOT_SIZE, loc="lower right")
    panel_label(ax, "A  Scale exponent")


# =====================================================================
# Panel B — R² variance decomposition (stacked bar)
# =====================================================================

def plot_panel_B(ax: plt.Axes, res: pd.DataFrame) -> None:
    """Stacked-bar R² decomposition: scale, sex, shape, material."""
    x = np.arange(NUM_MODES)
    w = 0.7

    r2_scale = res["R2pct_scale"].to_numpy()
    r2_sex = res["R2pct_sex"].to_numpy()
    r2_shape = res["R2pct_shape"].to_numpy()
    r2_material = res["R2pct_material"].to_numpy()

    bottom = np.zeros(NUM_MODES)
    for vals, color, label in [
        (r2_scale, SCALE_EFFECT_COLOR, "Scale"),
        (r2_sex, SEX_EFFECT_COLOR, "Sex"),
        (r2_shape, SHAPE_EFFECT_COLOR, "Shape"),
        (r2_material, MATERIAL_EFFECT_COLOR, "Material"),
    ]:
        ax.bar(x, vals, w, bottom=bottom, color=color, edgecolor="white",
               linewidth=0.3, label=label)
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels([format_mode_label_short(i) for i in range(NUM_MODES)],
                       fontsize=ANNOT_SIZE)
    ax.set_ylabel("Explained variance (%)")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=SMALL_ANNOT_SIZE, ncol=4, loc="upper right")
    panel_label(ax, "B  Variance split")


# =====================================================================
# Panel C — Swap probability vs scale (logistic + sex stratification)
# =====================================================================

def plot_panel_C(ax: plt.Axes, meta: pd.DataFrame, perm: pd.DataFrame) -> None:
    """Swap_9_10 probability vs isotropic scale, sex-stratified."""
    swap = compute_swap_9_10(perm)
    scale = meta["scale"].to_numpy(float)
    sex = meta["sex"].to_numpy()

    # Logistic regression: swap ~ scale
    from scipy.special import expit
    mask = np.isfinite(scale) & np.isfinite(swap)
    X = scale[mask]
    Y = swap[mask].astype(float)
    if Y.sum() > 5 and (1 - Y).sum() > 5:
        slope, intercept, _, _, _ = stats.linregress(X, Y)
        x_fit = np.linspace(X.min(), X.max(), 200)
        # Simple logistic via statsmodels if available, else linear approx
        try:
            import statsmodels.api as sm
            logit_model = sm.Logit(Y, sm.add_constant(X)).fit(disp=0)
            y_fit = logit_model.predict(sm.add_constant(x_fit))
        except Exception:
            y_fit = np.clip(intercept + slope * x_fit, 0, 1)
        ax.plot(x_fit, y_fit, color=SWAP_LINE_COLOR, lw=1.2, zorder=3)

    # Sex-stratified scatter
    for sex_val, color, label, marker in [
        ("M", MALE_COLOR, "Male", "o"), ("F", FEMALE_COLOR, "Female", "s")
    ]:
        m = sex == sex_val
        jitter = np.random.default_rng(42).uniform(-0.03, 0.03, size=m.sum())
        ax.scatter(scale[m], swap[m] + jitter, c=color, s=12,
                   alpha=SCATTER_ALPHA, marker=marker, label=label, zorder=2,
                   linewidths=0)

    # Swap rate by sex annotation
    m_rate = swap[sex == "M"].mean()
    f_rate = swap[sex == "F"].mean()
    ax.text(0.02, 0.92, f"Swap rate: M={m_rate:.1%}, F={f_rate:.1%}",
            transform=ax.transAxes, fontsize=SMALL_ANNOT_SIZE, va="top")

    ax.set_xlabel("Isotropic scale")
    ax.set_ylabel("P(swap 9–10)")
    ax.set_ylim(-0.12, 1.12)
    ax.legend(fontsize=SMALL_ANNOT_SIZE, loc="center right")
    panel_label(ax, "C  Swap vs size")


# =====================================================================
# Panel D — Spectral gap(9–10) vs scale
# =====================================================================

def plot_panel_D(ax: plt.Axes, meta: pd.DataFrame, eigenvalues: pd.DataFrame) -> None:
    """Relative spectral gap_in(9–10) vs isotropic scale, sex-coloured."""
    scale = meta["scale"].to_numpy(float)
    sex = meta["sex"].to_numpy()
    gap = compute_gap_in(eigenvalues, (9, 10))

    for sex_val, color, label, marker in [
        ("M", MALE_COLOR, "Male", "o"), ("F", FEMALE_COLOR, "Female", "s")
    ]:
        m = sex == sex_val
        ax.scatter(scale[m], gap[m], c=color, s=12, alpha=SCATTER_ALPHA,
                   marker=marker, label=label, linewidths=0)

    # LOWESS trend
    try:
        import statsmodels.api as sm
        lowess = sm.nonparametric.lowess(gap, scale, frac=0.4)
        ax.plot(lowess[:, 0], lowess[:, 1], color="0.2", lw=1.0, ls="-",
                label="LOWESS", zorder=4)
    except Exception:
        pass

    rho, p = stats.spearmanr(scale, gap, nan_policy="omit")
    ax.text(0.02, 0.92, f"ρ={rho:.3f}, p={p:.1e}",
            transform=ax.transAxes, fontsize=SMALL_ANNOT_SIZE, va="top")

    ax.set_xlabel("Isotropic scale")
    ax.set_ylabel("gap$_{\\rm in}$(9–10)")
    ax.legend(fontsize=SMALL_ANNOT_SIZE, loc="upper right")
    panel_label(ax, "D  Gap vs size")


# =====================================================================
# Panel E — SIJ micromotion vs scale
# =====================================================================

def plot_panel_E(ax: plt.Axes, sij: pd.DataFrame, meta: pd.DataFrame) -> None:
    """SIJ rotation and translation magnitude vs scale at LAB."""
    if sij.empty:
        ax.text(0.5, 0.5, "SIJ data not available", ha="center", va="center",
                transform=ax.transAxes)
        panel_label(ax, "E  SIJ micromotion vs body size")
        return

    lab = sij[sij["load_case"].isin(["LAB_phase1", "LAB_phase2", "LAB_phase3"])].copy()
    if lab.empty:
        lab = sij[sij["load_case"] == sij["load_case"].iloc[0]].copy()

    # Merge with metadata for scale
    lab = lab.merge(
        meta[["patient_id", "scale"]].reset_index().rename(columns={"index": "subject_idx"}),
        left_on="subject_idx", right_on="subject_idx", how="left"
    )

    if "scale" not in lab.columns:
        # Fallback: merge by index
        lab["scale"] = meta["scale"].values[lab["subject_idx"].values]

    scale = lab["scale"].to_numpy(float)
    rot = lab["rot_mag_deg"].to_numpy(float)
    trans = lab["trans_mag_mm"].to_numpy(float)
    sex = lab["sex"].to_numpy()

    # Rotation
    for sex_val, color, marker in [("M", MALE_COLOR, "o"), ("F", FEMALE_COLOR, "s")]:
        m = sex == sex_val
        ax.scatter(scale[m], rot[m], c=color, s=10, alpha=SCATTER_ALPHA,
                   marker=marker, linewidths=0)

    rho_rot, p_rot = stats.spearmanr(scale, rot, nan_policy="omit")
    rho_trans, p_trans = stats.spearmanr(scale, trans, nan_policy="omit")
    ax.text(0.02, 0.92, f"Rot: ρ={rho_rot:.3f}, p={p_rot:.2e}\n"
                         f"Trans: ρ={rho_trans:.3f}, p={p_trans:.2e}",
            transform=ax.transAxes, fontsize=SMALL_ANNOT_SIZE, va="top")

    # Add LOWESS
    try:
        import statsmodels.api as sm
        lowess = sm.nonparametric.lowess(rot, scale, frac=0.4)
        ax.plot(lowess[:, 0], lowess[:, 1], color=MALE_COLOR, lw=1.0, ls="-",
                alpha=0.8, zorder=4)
    except Exception:
        pass

    ax.set_xlabel("Isotropic scale")
    ax.set_ylabel("SIJ rotation (deg)")
    panel_label(ax, "E  SIJ vs size")


# =====================================================================
# Panel F — Energy fraction scale sensitivity per load case
# =====================================================================

def plot_panel_F(ax: plt.Axes, frac: pd.DataFrame) -> None:
    """Grouped bars: pp_scale_IQR per mode for each load case."""
    n_loads = len(LOAD_CASES)
    n_modes = NUM_MODES
    x = np.arange(n_modes)
    total_w = 0.75
    bar_w = total_w / n_loads

    cmap = plt.cm.viridis
    colors = [cmap(i / (n_loads - 1)) for i in range(n_loads)]

    for li, load in enumerate(LOAD_CASES):
        sub = frac[frac["load_case"] == load].sort_values("mode").reset_index(drop=True)
        if sub.empty:
            continue
        y = sub["pp_scale_IQR"].to_numpy()[:n_modes]
        offset = (li - n_loads / 2 + 0.5) * bar_w
        ax.bar(x[:len(y)] + offset, y, bar_w * 0.9, color=colors[li],
               edgecolor="white", linewidth=0.2, label=LOAD_LABELS[li])

    ax.axhline(0, color="0.4", lw=0.6, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels([format_mode_label_short(i) for i in range(n_modes)],
                       fontsize=ANNOT_SIZE)
    ax.set_ylabel("Scale IQR (pp)")
    ax.legend(fontsize=SMALL_ANNOT_SIZE, ncol=n_loads, loc="lower right")
    panel_label(ax, "F  Energy vs size")


# =====================================================================
# Summary table export
# =====================================================================

def build_summary_table(res: pd.DataFrame, frac: pd.DataFrame) -> pd.DataFrame:
    """Build a concise allometry summary table for manuscript."""
    # Robust p-values in the upstream artefact can be written as 0.0 for
    # highly significant modes; recompute from t-statistics to avoid
    # reporting 0.0e+00 in the manuscript table.
    from statsmodels.stats.multitest import multipletests

    t_vals = res["beta_scale_t"].to_numpy(dtype=float)
    df_resid = res["df_resid"].to_numpy(dtype=float)
    p_raw = np.array(
        [2.0 * stats.t.sf(abs(t), df=int(df)) for t, df in zip(t_vals, df_resid, strict=True)],
        dtype=float,
    )
    p_fdr = multipletests(p_raw, method="fdr_bh")[1]

    rows = []
    for i, r in res.iterrows():
        mode = int(r["mode"])
        rows.append({
            "Mode": mode,
            "$\\beta_{\\rm scale}$": f"{r['beta_scale']:.2f}",
            "95\\% CI": f"[{r['beta_scale_CI_lo']:.2f}, {r['beta_scale_CI_hi']:.2f}]",
            "$p_{\\rm FDR}$": f"{p_fdr[i]:.1e}" if p_fdr[i] < 0.001 else f"{p_fdr[i]:.3f}",
            "$\\Delta\\lambda$ IQR (\\%)": f"{r['pct_scale_IQR']:.1f}",
            "R$^2$ scale (\\%)": f"{r['R2pct_scale']:.1f}",
            "R$^2$ shape (\\%)": f"{r['R2pct_shape']:.1f}",
            "R$^2$ material (\\%)": f"{r['R2pct_material']:.1f}",
            "R$^2$ sex (\\%)": f"{r['R2pct_sex']:.1f}",
        })
    return pd.DataFrame(rows)


def export_latex_table(summary: pd.DataFrame, path: Path) -> None:
    """Export summary table as LaTeX (manual formatting, no jinja2)."""
    cols = summary.columns.tolist()
    col_fmt = "c" * len(cols)
    header = " & ".join(cols) + " \\\\\n"

    rows_str = ""
    for _, row in summary.iterrows():
        vals = " & ".join(str(row[c]) for c in cols)
        rows_str += vals + " \\\\\n"

    latex = (
        "\\begin{table}[!ht]\n"
        "\\centering\n"
        "\\caption{Allometric scaling exponents and variance decomposition per eigenvalue mode. "
        "$\\beta_{\\rm scale}$ is the IV-2SLS log--log coefficient of eigenvalue on isotropic scale, "
        "with HC-robust 95\\,\\% CIs and FDR-corrected $p$-values. "
        "$\\Delta\\lambda$ per IQR gives the percent eigenvalue change for an interquartile "
        "range shift in log-scale. R$^2$ columns show the LMG (Shapley) variance share for "
        "each predictor group.}\n"
        "\\label{tab:allometry-summary}\n"
        "\\small\n"
        "\\resizebox{\\linewidth}{!}{%\n"
        f"\\begin{{tabular}}{{{col_fmt}}}\n"
        "\\toprule\n"
        f"{header}"
        "\\midrule\n"
        f"{rows_str}"
        "\\bottomrule\n"
        "\\end{tabular}%\n"
        "}\n"
        "\\end{table}\n"
    )
    path.write_text(latex)


# =====================================================================
# Main assembly
# =====================================================================

def main() -> None:
    setup_plot_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)

    # Load all pre-computed data
    res = load_eigen_results()
    frac = load_fractions_results()
    meta = load_metadata()
    perm = load_permutations()
    eigen = load_eigenvalues()
    sij = load_sij_subject_level()

    # ── Figure: 3 × 2 mosaic ──────────────────────────────────
    fig = plt.figure(figsize=(FULL_WIDTH, 2 * ROW_H + 0.45), constrained_layout=False)
    mosaic = [
        ["A", "B", "C"],
        ["D", "E", "F"],
    ]
    axes = fig.subplot_mosaic(mosaic, gridspec_kw={"wspace": 0.20, "hspace": 0.28})

    plot_panel_A(axes["A"], res)
    plot_panel_B(axes["B"], res)
    plot_panel_C(axes["C"], meta, perm)
    plot_panel_D(axes["D"], meta, eigen)
    plot_panel_E(axes["E"], sij, meta)
    plot_panel_F(axes["F"], frac)

    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.08, top=0.97)
    fig.set_constrained_layout(False)
    fig.set_layout_engine("none")
    fig_path = FIG_DIR / "allometry_dashboard"
    fig.savefig(fig_path.with_suffix(".pdf"), bbox_inches=None)
    fig.set_constrained_layout(False)
    fig.set_layout_engine("none")
    fig.savefig(fig_path.with_suffix(".png"), dpi=300, bbox_inches=None)
    plt.close(fig)
    print(f"Figure saved: {fig_path}.pdf / .png")

    # ── Summary table ──────────────────────────────────────────
    summary = build_summary_table(res, frac)
    csv_path = TAB_DIR / "allometry_summary.csv"
    summary.to_csv(csv_path, index=False)
    print(f"CSV saved: {csv_path}")

    tex_path = TAB_DIR / "allometry_summary.tex"
    export_latex_table(summary, tex_path)
    print(f"LaTeX table saved: {tex_path}")


if __name__ == "__main__":
    main()
