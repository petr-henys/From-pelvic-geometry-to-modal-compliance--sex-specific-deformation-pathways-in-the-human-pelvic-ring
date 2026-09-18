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

from analysis.publication_style import (
    apply_publication_style,
    panel_label as pub_panel_label,
    style_axis,
    MALE_COLOR,
    FEMALE_COLOR,
    NEUTRAL_COLOR,
    format_pvalue,
    save_publication_figure,
    COL2_WIDTH,
    ROW_H,
)
from utils.plot_utils import (
    SCALE_EFFECT_COLOR,
    SEX_EFFECT_COLOR,
    SHAPE_EFFECT_COLOR,
    MATERIAL_EFFECT_COLOR,
    SWAP_LINE_COLOR,
    BAR_EDGE_LW,
    BAR_EDGE_COLOR,
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
    """Relative spectral gap between pair of adjacent solver ranks (1-based).

    Uses the canonical formula gap_in(i) = (λ_{i+1} - λ_i) / λ_i
    where λ are sorted in ascending solver order,
    consistent with spectral_metrics.compute_gap_in.
    """
    cols = [c for c in eigenvalues.columns if c.startswith("eig_")]
    vals = eigenvalues[cols].to_numpy(float)
    sorted_vals = np.sort(vals, axis=1)
    a, b = pair
    la = sorted_vals[:, a - 1]
    lb = sorted_vals[:, b - 1]
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
    ax.axhline(0, color="#6b7280", lw=0.6, ls="--")

    # Classical elastic scaling reference line λ ~ L^(-2)
    ax.axhline(-2.0, color="#9ca3af", lw=0.6, ls=":", label=r"Elastic scaling $\lambda \propto L^{-2}$")

    # Significance markers
    for i in range(NUM_MODES):
        sig = res["beta_scale_signif"].iloc[i]
        if sig:
            ax.text(i, hi[i] + 0.08, "*", ha="center", va="bottom",
                    fontsize=7.5, color=SCALE_EFFECT_COLOR, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([str(i + 1) for i in range(NUM_MODES)], fontsize=7.0)
    ax.set_xlabel("Mode rank", fontsize=7.5)
    ax.set_ylabel(r"$\beta_{\rm scale}$ (log–log)", fontsize=7.5)
    ax.legend(fontsize=6.5, loc="lower right", frameon=False)
    pub_panel_label(ax, "A")
    ax.set_title(r"Eigenvalue scaling $\beta_{\rm scale}$", pad=5, fontsize=7.8, fontweight="medium")
    style_axis(ax, y_grid=True)


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
    ax.set_xticklabels([str(i + 1) for i in range(NUM_MODES)], fontsize=7.0)
    ax.set_xlabel("Mode rank", fontsize=7.5)
    ax.set_ylabel("Explained variance (%)", fontsize=7.5)
    ax.set_ylim(0, 122)
    ax.legend(fontsize=6.5, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.0), frameon=False)
    pub_panel_label(ax, "B")
    ax.set_title("Variance decomposition", pad=5, fontsize=7.8, fontweight="medium")
    style_axis(ax, y_grid=True)


# =====================================================================
# Panel C — Swap probability vs scale (logistic + sex stratification)
# =====================================================================

def plot_panel_C(ax: plt.Axes, meta: pd.DataFrame, perm: pd.DataFrame) -> None:
    """Swap_9_10 probability vs isotropic scale, sex-stratified."""
    swap = compute_swap_9_10(perm)
    scale = meta["scale"].to_numpy(float)
    sex = meta["sex"].to_numpy()

    # Logistic regression: swap ~ scale
    mask = np.isfinite(scale) & np.isfinite(swap)
    X = scale[mask]
    Y = swap[mask].astype(float)
    if Y.sum() > 5 and (1 - Y).sum() > 5:
        slope, intercept, _, _, _ = stats.linregress(X, Y)
        x_fit = np.linspace(X.min(), X.max(), 200)
        try:
            import statsmodels.api as sm
            logit_model = sm.Logit(Y, sm.add_constant(X)).fit(disp=0)
            y_fit = logit_model.predict(sm.add_constant(x_fit))
        except Exception:
            y_fit = np.clip(intercept + slope * x_fit, 0, 1)
        ax.plot(x_fit, y_fit, color="#374151", lw=1.4, zorder=3, label="Logistic fit")

    # Sex-stratified scatter
    for sex_val, color, label, marker in [
        ("M", MALE_COLOR, "Male", "o"), ("F", FEMALE_COLOR, "Female", "s")
    ]:
        m = sex == sex_val
        jitter = np.random.default_rng(42).uniform(-0.025, 0.025, size=m.sum())
        ax.scatter(scale[m], swap[m] + jitter, c=color, s=14,
                   alpha=0.55, marker=marker, label=label, zorder=2,
                   linewidths=0)

    # Swap rate by sex annotation (placed in central open space)
    m_rate = swap[sex == "M"].mean()
    f_rate = swap[sex == "F"].mean()
    ax.text(
        0.04, 0.70,
        f"Swap rate:\nMale: {m_rate:.1%}\nFemale: {f_rate:.1%}",
        transform=ax.transAxes, fontsize=6.8, va="center",
        bbox=dict(boxstyle="round,pad=0.3", fc="#ffffff", ec="#e5e7eb", alpha=0.9),
    )

    ax.set_xlabel("Isotropic scale $s$", fontsize=7.5)
    ax.set_ylabel(r"$P(\mathrm{swap}_{9–10})$", fontsize=7.5)
    ax.set_ylim(-0.10, 1.10)
    ax.legend(fontsize=6.5, loc="center right", frameon=False)
    pub_panel_label(ax, "C")
    ax.set_title("Mode 9–10 swap probability", pad=5, fontsize=7.8, fontweight="medium")
    style_axis(ax, y_grid=True)


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
        ax.scatter(scale[m], gap[m], c=color, s=14, alpha=0.55,
                   marker=marker, label=label, linewidths=0)

    # LOWESS trend
    try:
        import statsmodels.api as sm
        lowess = sm.nonparametric.lowess(gap, scale, frac=0.4)
        ax.plot(lowess[:, 0], lowess[:, 1], color="#1f2937", lw=1.2, ls="-",
                label="LOWESS", zorder=4)
    except Exception:
        pass

    rho, p = stats.spearmanr(scale, gap, nan_policy="omit")
    ax.text(0.04, 0.93, f"$\\rho = {rho:.2f}$, {format_pvalue(p)}",
            transform=ax.transAxes, fontsize=6.8, va="top")

    ax.set_xlabel("Isotropic scale $s$", fontsize=7.5)
    ax.set_ylabel(r"$\mathrm{gap}_{\rm in}(9–10)$", fontsize=7.5)
    ax.set_ylim(-0.02, 0.46)
    ax.legend(fontsize=6.5, loc="upper right", frameon=False)
    pub_panel_label(ax, "D")
    ax.set_title(r"Spectral gap $\mathrm{gap}_{\rm in}(9–10)$", pad=5, fontsize=7.8, fontweight="medium")
    style_axis(ax, y_grid=True)


# =====================================================================
# Panel E — SIJ micromotion vs scale
# =====================================================================

def plot_panel_E(ax: plt.Axes, sij: pd.DataFrame, meta: pd.DataFrame) -> None:
    """SIJ rotation and translation magnitude vs scale at LAB."""
    if sij.empty:
        ax.text(0.5, 0.5, "SIJ data not available", ha="center", va="center",
                transform=ax.transAxes)
        pub_panel_label(ax, "E")
        ax.set_title("SIJ micromotion vs body size", pad=5, fontsize=7.8, fontweight="medium")
        style_axis(ax)
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
        lab["scale"] = meta["scale"].values[lab["subject_idx"].values]

    scale = lab["scale"].to_numpy(float)
    rot = lab["rot_mag_deg"].to_numpy(float)
    trans = lab["trans_mag_mm"].to_numpy(float)
    sex = lab["sex"].to_numpy()

    # Rotation
    for sex_val, color, marker in [("M", MALE_COLOR, "o"), ("F", FEMALE_COLOR, "s")]:
        m = sex == sex_val
        ax.scatter(scale[m], rot[m], c=color, s=12, alpha=0.55,
                   marker=marker, linewidths=0)

    rho_rot, p_rot = stats.spearmanr(scale, rot, nan_policy="omit")
    rho_trans, p_trans = stats.spearmanr(scale, trans, nan_policy="omit")
    ax.text(0.04, 0.95, f"Rot: $\\rho = {rho_rot:.2f}$, {format_pvalue(p_rot)}\n"
                         f"Trans: $\\rho = {rho_trans:.2f}$, {format_pvalue(p_trans)}",
            transform=ax.transAxes, fontsize=6.8, va="top")

    # Add LOWESS
    try:
        import statsmodels.api as sm
        lowess = sm.nonparametric.lowess(rot, scale, frac=0.4)
        ax.plot(lowess[:, 0], lowess[:, 1], color="#1f2937", lw=1.2, ls="-",
                alpha=0.85, zorder=4)
    except Exception:
        pass

    ax.set_xlabel("Isotropic scale $s$", fontsize=7.5)
    ax.set_ylabel("SIJ rotation (°)", fontsize=7.5)
    ax.set_ylim(-0.2, 5.8)
    pub_panel_label(ax, "E")
    ax.set_title("SIJ micromotion vs scale", pad=5, fontsize=7.8, fontweight="medium")
    style_axis(ax, y_grid=True)


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

    ax.axhline(0, color="#6b7280", lw=0.6, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels([str(i + 1) for i in range(n_modes)], fontsize=7.0)
    ax.set_xlabel("Mode rank", fontsize=7.5)
    ax.set_ylabel(r"Scale sensitivity $\Delta$ (pp / IQR)", fontsize=7.5)
    ax.legend(fontsize=6.5, ncol=3, loc="upper right", frameon=False)
    pub_panel_label(ax, "F")
    ax.set_title("Modal energy sensitivity to scale", pad=5, fontsize=7.8, fontweight="medium")
    style_axis(ax, y_grid=True)


# =====================================================================
# Summary table export
# =====================================================================

def build_summary_table(res: pd.DataFrame, frac: pd.DataFrame) -> pd.DataFrame:
    """Build a concise allometry summary table for manuscript."""
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
    apply_publication_style()
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
    fig = plt.figure(figsize=(COL2_WIDTH * 1.05, ROW_H * 2.3), layout="constrained")
    mosaic = [
        ["A", "B", "C"],
        ["D", "E", "F"],
    ]
    axes = fig.subplot_mosaic(mosaic)

    plot_panel_A(axes["A"], res)
    plot_panel_B(axes["B"], res)
    plot_panel_C(axes["C"], meta, perm)
    plot_panel_D(axes["D"], meta, eigen)
    plot_panel_E(axes["E"], sij, meta)
    plot_panel_F(axes["F"], frac)

    fig_path = FIG_DIR / "allometry_dashboard"
    save_publication_figure(fig, fig_path)
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
