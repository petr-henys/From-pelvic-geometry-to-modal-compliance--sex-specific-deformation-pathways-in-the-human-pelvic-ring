"""Plotting functions for spectral panel analysis.

Each function produces one figure (or a set of per-cluster figures) and
optionally saves it.  No computation beyond what is needed for layout.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from analysis.publication_style import (
    apply_publication_style,
    panel_label as pub_panel_label,
    style_axis,
    style_distribution,
    format_pvalue,
    save_publication_figure,
    MALE_COLOR,
    FEMALE_COLOR,
    NEUTRAL_COLOR,
)
from analysis.spectral_config import (
    COL2_WIDTH,
    EPS_DEG,
    FIG_DPI,
    FIG_DIR,
    FIG_FMT,
    FIG_FORMATS,
    N_MODES,
    ROW_H,
    TAB_DIR,
)
from utils.plot_utils import (
    AGE_COLOR,
    ANNOT_SIZE,
    BAR_EDGE_COLOR,
    BAR_EDGE_LW,
    BOX_LW,
    DEFAULT_FONT_SIZE,
    FILL_ALPHA,
    MATERIAL_EFFECT_COLOR,
    MEDIAN_LW,
    SCATTER_ALPHA,
    SHAPE_EFFECT_COLOR,
    SMALL_ANNOT_SIZE,
    SUPTITLE_SIZE,
    panel_label,
    set_suptitle,
)


# ============================================================
# Internal helpers
# ============================================================
def _save_fig(fig: plt.Figure, name: str) -> Path:
    """Save figure in all configured formats and close."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    primary = FIG_DIR / f"{name}.pdf"
    save_publication_figure(fig, FIG_DIR / name)
    plt.close(fig)
    return primary


def _sanitize_tex(text: str) -> str:
    """Replace Unicode characters that pdflatex cannot handle.

    Underscores inside math mode ($...$) are left intact; outside math
    they are escaped to ``\\_``.
    """
    text = (
        text
        .replace("\u21c4", "$\\rightleftarrows$")
        .replace("\u2013", "--")
        .replace("\u2192", "$\\to$")
        .replace("\u00b1", "$\\pm$")
        .replace("%", "\\%")
    )
    # Escape underscores only outside math mode
    import re
    parts = re.split(r"(\$[^$]*\$)", text)
    for i, part in enumerate(parts):
        if not part.startswith("$"):
            parts[i] = part.replace("_", "\\_")
    return "".join(parts)


def _save_table(
    df: pd.DataFrame,
    name: str,
    *,
    tex_columns: list[str] | None = None,
    tex_col_labels: dict[str, str] | None = None,
) -> None:
    """Save table as CSV (full) and LaTeX (optionally slimmed).

    Parameters
    ----------
    df : Full DataFrame.
    name : Output base name.
    tex_columns : Subset of columns to include in .tex. If None, all columns.
    tex_col_labels : Column rename map for .tex (applied after subsetting).
    """
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(TAB_DIR / f"{name}.csv", index=False)

    df_tex = df[tex_columns].copy() if tex_columns else df.copy()
    if tex_col_labels:
        df_tex = df_tex.rename(columns=tex_col_labels)

    tex_path = TAB_DIR / f"{name}.tex"
    # Use a general-format float printer to avoid reporting small p-values as 0.0000.
    df_tex.to_latex(tex_path, index=False, float_format="%.4g", na_rep="--")
    # Sanitize Unicode in the generated tex file
    raw = tex_path.read_text(encoding="utf-8")
    tex_path.write_text(_sanitize_tex(raw), encoding="utf-8")


def _format_metric_title(metric: str) -> str:
    if metric.startswith("gap_in"):
        core = metric.replace("gap_in", "").strip(" ()").replace("-", "–").replace("\u2013", "–")
        return f"gap$_{{\\rm in}}$({core})"
    if metric.startswith("d_G"):
        core = metric.replace("d_G", "").strip(" ()").replace("modes_", "").replace("_", "–")
        return f"$d_G$({core})"
    if metric == "swap_count_per_subject":
        return "Swapped modes / subject"
    if "permutation" in metric:
        return "Pairing exchange rate"
    return metric


def _sex_boxplot(
    ax: plt.Axes,
    vals_m: np.ndarray,
    vals_f: np.ndarray,
    ylabel: str,
    title: str = "",
    show_p: bool = True,
) -> None:
    """Styled sex-stratified distribution plot with Mann-Whitney U annotation."""
    vals_m = np.asarray(vals_m, dtype=float)
    vals_f = np.asarray(vals_f, dtype=float)
    vals_m = vals_m[np.isfinite(vals_m)]
    vals_f = vals_f[np.isfinite(vals_f)]
    if len(vals_m) == 0 or len(vals_f) == 0:
        return

    style_distribution(
        ax,
        [vals_m, vals_f],
        positions=[0, 1],
        labels=["Male", "Female"],
        color=[MALE_COLOR, FEMALE_COLOR],
        pt_size=9.0,
        pt_alpha=0.35,
        width=0.45,
    )
    ax.set_ylabel(ylabel, fontsize=7.5)
    if title:
        ax.set_title(_format_metric_title(title), fontsize=8.0, fontweight="medium", pad=5)
    style_axis(ax, y_grid=True)

    if show_p and len(vals_m) > 1 and len(vals_f) > 1:
        u_stat, p_val = stats.mannwhitneyu(vals_m, vals_f, alternative="two-sided")
        n_m, n_f = len(vals_m), len(vals_f)
        r_eff = 1.0 - 2.0 * u_stat / (n_m * n_f)
        p_txt = format_pvalue(p_val, prefix="p = ")
        eff_txt = f"$r_{{rb}} = {r_eff:+.2f}$"
        ax.text(
            0.5, 0.95,
            f"{p_txt}, {eff_txt}",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=6.8,
            color="#111827",
            fontweight="medium",
        )
        curr_min, curr_max = ax.get_ylim()
        span = curr_max - curr_min
        ax.set_ylim(curr_min, curr_max + 0.16 * span)


# ============================================================
# 1. Sex & age analysis  (3×2 panel)
# ============================================================
def plot_sex_age_analysis(
    *,
    age: np.ndarray,
    mask_m: np.ndarray,
    mask_f: np.ndarray,
    sex: np.ndarray,
    beta_sex: np.ndarray,
    beta_age: np.ndarray,
    ci_sex: np.ndarray,
    ci_age: np.ndarray,
    pval_sex: np.ndarray,
    pval_age: np.ndarray,
    swap_rate_m: np.ndarray,
    swap_rate_f: np.ndarray,
    pval_swap_sex: np.ndarray,
    n_swapped_per_subj: np.ndarray,
    rho_swap_age: float,
    p_swap_age: float,
    gap_r_sex: np.ndarray,
    gap_rho_age: np.ndarray,
    gap_p_sex: np.ndarray,
    gap_p_age: np.ndarray,
) -> None:
    """2×3 panel: age histogram, OLS coefficients, swap rates, gap effects."""
    modes_1 = np.arange(1, N_MODES + 1)
    gap_pairs_1 = np.arange(1, N_MODES)
    _annot = ANNOT_SIZE
    w = 0.35

    fig, axes = plt.subplots(
        2, 3, figsize=(COL2_WIDTH, 2 * ROW_H),
        layout="constrained",
    )

    # (a) Age histogram by sex
    ax = axes[0, 0]
    bins = np.arange(
        int(age.min()) - 1,
        int(age.max()) + 2,
        max(1, (int(age.max()) - int(age.min())) // 15),
    )
    ax.hist(age[mask_m], bins=bins, color=MALE_COLOR, alpha=FILL_ALPHA, label="Male")
    ax.hist(age[mask_f], bins=bins, color=FEMALE_COLOR, alpha=FILL_ALPHA, label="Female")
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Count")
    ax.legend(loc="upper right")
    panel_label(ax, "(a) Age distribution")

    # (b) Sex β on log λ
    ax = axes[0, 1]
    ax.errorbar(
        modes_1, beta_sex,
        yerr=[beta_sex - ci_sex[:, 0], ci_sex[:, 1] - beta_sex],
        fmt="o", color=FEMALE_COLOR, ms=4, capsize=3, lw=1.0,
        label="$\\beta_{\\mathrm{sex}}$  (F vs M)",
    )
    sig = pval_sex < 0.05
    if sig.any():
        ax.scatter(
            modes_1[sig], beta_sex[sig],
            marker="o", s=50, facecolors="none", edgecolors="black", lw=0.8,
            label="$p < 0.05$", zorder=5,
        )
    ax.axhline(0, color="0.4", ls="--", lw=0.7)
    ax.set_xlabel("Mode")
    ax.set_ylabel("$\\beta_{\\mathrm{sex}}$  (log scale)")
    ax.set_xticks(modes_1)
    ax.legend(loc="upper right")
    panel_label(ax, "(b) Sex effect on $\\log\\lambda_i$")

    # (c) Age β on log λ
    ax = axes[0, 2]
    ax.errorbar(
        modes_1, beta_age,
        yerr=[beta_age - ci_age[:, 0], ci_age[:, 1] - beta_age],
        fmt="s", color=AGE_COLOR, ms=4, capsize=3, lw=1.0,
        label="$\\beta_{\\mathrm{age}}$",
    )
    sig_age = pval_age < 0.05
    if sig_age.any():
        ax.scatter(
            modes_1[sig_age], beta_age[sig_age],
            marker="s", s=50, facecolors="none", edgecolors="black", lw=0.8,
            label="$p < 0.05$", zorder=5,
        )
    ax.axhline(0, color="0.4", ls="--", lw=0.7)
    ax.set_xlabel("Mode")
    ax.set_ylabel("$\\beta_{\\mathrm{age}}$  (per year)")
    ax.set_xticks(modes_1)
    ax.legend(loc="upper right")
    panel_label(ax, "(c) Age effect on $\\log\\lambda_i$")

    # (d) Per-mode swap rate by sex
    ax = axes[1, 0]
    ax.bar(modes_1 - w / 2, swap_rate_m, w, color=MALE_COLOR, alpha=FILL_ALPHA,
           label="Male", edgecolor=BAR_EDGE_COLOR, lw=BAR_EDGE_LW)
    ax.bar(modes_1 + w / 2, swap_rate_f, w, color=FEMALE_COLOR, alpha=FILL_ALPHA,
           label="Female", edgecolor=BAR_EDGE_COLOR, lw=BAR_EDGE_LW)
    for m in range(N_MODES):
        if pval_swap_sex[m] < 0.05:
            y_top = max(swap_rate_m[m], swap_rate_f[m])
            ax.text(modes_1[m], y_top + 0.01, "*", ha="center", va="bottom",
                    fontsize=_annot, fontweight="bold")
    ax.set_xlabel("Mode")
    ax.set_ylabel("Swap rate")
    ax.set_xticks(modes_1)
    ax.legend(loc="upper right")
    panel_label(ax, "(d) Permutation rate by sex")

    # (e) Swapped-mode count vs age
    ax = axes[1, 1]
    colors_scatter = np.where(sex == "M", MALE_COLOR, FEMALE_COLOR)
    ax.scatter(age, n_swapped_per_subj, c=colors_scatter, s=8, alpha=SCATTER_ALPHA,
               rasterized=True, linewidths=0)
    z = np.polyfit(age, n_swapped_per_subj, 1)
    age_line = np.linspace(age.min(), age.max(), 50)
    ax.plot(age_line, np.polyval(z, age_line), "k-", lw=1.2, alpha=0.7)
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("# swapped modes")
    ax.text(
        0.02, 0.97,
        f"$\\rho_s$ = {rho_swap_age:.3f}  (p = {p_swap_age:.2e})",
        transform=ax.transAxes, ha="left", va="top", fontsize=_annot,
        color="crimson" if p_swap_age < 0.05 else "0.4",
    )
    panel_label(ax, "(e) Permutation burden vs age")

    # (f) Gap sex effect + age effect per pair
    ax = axes[1, 2]
    ax.bar(gap_pairs_1 - w / 2, gap_r_sex, w, color=FEMALE_COLOR, alpha=FILL_ALPHA,
           label="Sex  $r_{rb}$", edgecolor=BAR_EDGE_COLOR, lw=BAR_EDGE_LW)
    ax.bar(gap_pairs_1 + w / 2, gap_rho_age, w, color=AGE_COLOR, alpha=FILL_ALPHA,
           label="Age  $\\rho_s$", edgecolor=BAR_EDGE_COLOR, lw=BAR_EDGE_LW)
    for gi in range(N_MODES - 1):
        if gap_p_sex[gi] < 0.05:
            y_val = gap_r_sex[gi]
            ax.text(gap_pairs_1[gi] - w / 2, y_val + np.sign(y_val) * 0.02,
                    "*", ha="center", va="bottom" if y_val >= 0 else "top",
                    fontsize=_annot, fontweight="bold", color=FEMALE_COLOR)
        if gap_p_age[gi] < 0.05:
            y_val = gap_rho_age[gi]
            ax.text(gap_pairs_1[gi] + w / 2, y_val + np.sign(y_val) * 0.02,
                    "*", ha="center", va="bottom" if y_val >= 0 else "top",
                    fontsize=_annot, fontweight="bold", color=AGE_COLOR)
    ax.axhline(0, color="0.4", ls="--", lw=0.7)
    ax.set_xlabel("Mode pair $(i, i{+}1)$")
    ax.set_ylabel("Effect size")
    ax.set_xticks(gap_pairs_1)
    ax.set_xticklabels([f"{i}\u2013{i+1}" for i in gap_pairs_1])
    ax.legend(loc="upper right")
    panel_label(ax, "(f) Gap sensitivity to sex & age")

    _save_fig(fig, "sex_age_analysis")


# ============================================================
# 2. Sex-stratified summary
# ============================================================
def plot_sex_stratified_summary(
    summary_rows: list[dict],
    gaps_full: np.ndarray,
    grassmann_results: dict[str, np.ndarray],
    n_swaps_m: np.ndarray,
    n_swaps_f: np.ndarray,
    rate_m: float,
    rate_f: float,
    mask_m: np.ndarray,
    mask_f: np.ndarray,
) -> None:
    """Grid of sex-stratified box/bar plots for all summary metrics."""
    apply_publication_style()
    n_metrics = len(summary_rows)
    if n_metrics == 0:
        return
    n_cols = min(n_metrics, 4)
    n_rows_fig = (n_metrics + n_cols - 1) // n_cols

    fig, axes_sum = plt.subplots(
        n_rows_fig, n_cols,
        figsize=(COL2_WIDTH, ROW_H * n_rows_fig * 1.15),
        squeeze=False,
        layout="constrained",
    )
    axes_flat = axes_sum.ravel()

    for i, srow in enumerate(summary_rows):
        ax = axes_flat[i]
        metric = srow["metric"]

        if metric.startswith("gap_in"):
            parts = metric.split("(")[1].rstrip(")").split("\u2013")
            if len(parts) < 2:
                parts = metric.split("(")[1].rstrip(")").split("-")
            gi = int(parts[0]) - 1
            vals_m = gaps_full[mask_m, gi]
            vals_f = gaps_full[mask_f, gi]
            _sex_boxplot(ax, vals_m, vals_f,
                         ylabel=r"$\mathrm{gap}_{\mathrm{in}}$", title=metric)
        elif metric.startswith("d_G"):
            cl_key = metric.split("(")[1].rstrip(")")
            vals_m = grassmann_results[cl_key][mask_m]
            vals_f = grassmann_results[cl_key][mask_f]
            _sex_boxplot(ax, vals_m, vals_f, ylabel="$d_G$", title=metric)
        elif metric == "swap_count_per_subject":
            _sex_boxplot(ax, n_swaps_m, n_swaps_f,
                         ylabel="# swapped modes", title=metric)
        elif "permutation" in metric:
            ax.bar(
                ["Male", "Female"], [rate_m, rate_f],
                color=[MALE_COLOR, FEMALE_COLOR], alpha=0.88,
                edgecolor="none", width=0.5,
            )
            ax.set_ylabel("Swap rate", fontsize=7.5)
            ax.set_title(_format_metric_title(metric), fontsize=8.0, fontweight="medium", pad=5)
            p_txt = format_pvalue(srow['p_value'], prefix="p = ")
            ax.text(
                0.5, 0.95, p_txt,
                transform=ax.transAxes, ha="center", va="top",
                fontsize=6.8,
                color="#111827",
                fontweight="medium",
            )
            ax.set_ylim(0, 1.15)
            style_axis(ax, y_grid=True)

    for j in range(n_metrics, len(axes_flat)):
        axes_flat[j].set_visible(False)

    _save_fig(fig, "sex_stratified_summary")


# ============================================================
# 3. Spectral structure dashboard  (3-row vertical)
# ============================================================

def plot_spectral_structure_dashboard(
    gaps_full: np.ndarray,
    eps_in: float,
    swap_rate: np.ndarray,
    fail_rate: np.ndarray,
    swap_pair_counter: dict[tuple[int, int], int],
    n_subj: int,
    ev_full: np.ndarray,
    ev_shape: np.ndarray | None,
    ev_material: np.ndarray | None,
) -> None:
    """1×3 dashboard: (a) gap boxplot, (b) swap bars, (c) CV decomposition."""
    from analysis.spectral_config import EPS_IN_PERCENTILE

    nonid_rate = swap_rate + fail_rate
    _annot_fs = ANNOT_SIZE
    modes_1b = np.arange(1, N_MODES + 1)

    has_cv = ev_shape is not None and ev_material is not None
    n_cols = 3 if has_cv else 2

    fig, axes_row = plt.subplots(
        1, n_cols,
        figsize=(COL2_WIDTH, ROW_H * 1.6),
        layout="constrained",
    )

    # ── (a) Eigenvalue gap distribution ───────────────────────
    ax_gap = axes_row[0]
    mode_pairs = [f"{i + 1}\u2013{i + 2}" for i in range(N_MODES - 1)]
    bp = ax_gap.boxplot(
        [gaps_full[:, i] for i in range(N_MODES - 1)],
        tick_labels=mode_pairs,
        patch_artist=True, showfliers=False,
        medianprops=dict(color="black", lw=MEDIAN_LW),
        boxprops=dict(linewidth=BOX_LW),
        whiskerprops=dict(linewidth=BOX_LW),
        capprops=dict(linewidth=BOX_LW),
    )
    for patch in bp["boxes"]:
        patch.set_facecolor(NEUTRAL_COLOR)
        patch.set_alpha(FILL_ALPHA)
    ax_gap.axhline(
        eps_in, color="crimson", ls="--", lw=0.9, alpha=0.8,
        label=f"$\\varepsilon_{{\\mathrm{{in}}}}$ = {eps_in:.3f}  (P{EPS_IN_PERCENTILE})",
    )
    ax_gap.axhline(
        EPS_DEG, color="0.5", ls=":", lw=0.8, alpha=0.7,
        label=f"$\\varepsilon_{{\\mathrm{{deg}}}}$ = {EPS_DEG:.1e}",
    )
    ax_gap.set_xlabel("Mode pair")
    ax_gap.set_ylabel("$\\mathrm{gap}_{\\mathrm{in}}(i)$")
    ax_gap.legend(loc="upper right", fontsize=ANNOT_SIZE)
    panel_label(ax_gap, "(a) Eigenvalue gap distribution")
    ax_gap.tick_params(axis="x", labelrotation=45, labelsize=SMALL_ANNOT_SIZE)

    # ── (b) Per-mode swap rate bar chart ──────────────────────
    ax_bar = axes_row[1]
    bars = ax_bar.bar(
        modes_1b, swap_rate,
        color=NEUTRAL_COLOR, alpha=FILL_ALPHA + 0.1,
        edgecolor=BAR_EDGE_COLOR, lw=BAR_EDGE_LW,
        label="Swap (paired)",
    )
    ax_bar.bar(
        modes_1b, fail_rate, bottom=swap_rate,
        color="0.6", alpha=FILL_ALPHA,
        edgecolor=BAR_EDGE_COLOR, lw=BAR_EDGE_LW,
        label="Unpaired",
    )
    for i, (bar, rate) in enumerate(zip(bars, nonid_rate)):
        if rate > 0.1:
            ax_bar.text(
                bar.get_x() + bar.get_width() / 2,
                float(swap_rate[i] + fail_rate[i]) + 0.01,
                f"{rate:.0%}", ha="center", va="bottom", fontsize=_annot_fs,
            )
    ax_bar.set_xlabel("Mode")
    ax_bar.set_ylabel("Rate")
    panel_label(ax_bar, "(b) Permutation swap rates")
    ax_bar.set_xticks(modes_1b)
    ax_bar.set_ylim(0, min(1.0, nonid_rate.max() * 1.25 + 0.05))
    ax_bar.legend(loc="upper left", fontsize=ANNOT_SIZE)

    # ── (c) CV comparison ─────────────────────────────────────
    if has_cv:
        cv_full = np.std(ev_full, axis=0) / np.mean(ev_full, axis=0)
        cv_shape = np.std(ev_shape, axis=0) / np.mean(ev_shape, axis=0)
        cv_mat = np.std(ev_material, axis=0) / np.mean(ev_material, axis=0)
        w = 0.25

        ax_cv = axes_row[2]
        ax_cv.bar(
            modes_1b - w, cv_full, w,
            label="Full", color=NEUTRAL_COLOR,
            alpha=FILL_ALPHA + 0.1,
            edgecolor=BAR_EDGE_COLOR, lw=BAR_EDGE_LW,
        )
        ax_cv.bar(
            modes_1b, cv_shape, w,
            label="Shape only", color=SHAPE_EFFECT_COLOR,
            alpha=FILL_ALPHA + 0.1,
            edgecolor=BAR_EDGE_COLOR, lw=BAR_EDGE_LW,
        )
        ax_cv.bar(
            modes_1b + w, cv_mat, w,
            label="Material only", color=MATERIAL_EFFECT_COLOR,
            alpha=FILL_ALPHA + 0.1,
            edgecolor=BAR_EDGE_COLOR, lw=BAR_EDGE_LW,
        )
        ax_cv.set_xlabel("Mode")
        ax_cv.set_ylabel("CV($\\lambda_i$)")
        ax_cv.set_xticks(modes_1b)
        ax_cv.legend(loc="upper right", fontsize=ANNOT_SIZE)
        panel_label(ax_cv, "(c) Eigenvalue variability decomposition")

    _save_fig(fig, "spectral_structure_dashboard")


# ============================================================
# 4. Subspace analysis dashboard
# ============================================================
def plot_subspace_analysis_dashboard(
    top_clusters: list[tuple[int, int, int]],
    grassmann_results: dict[str, np.ndarray],
    clusters_all: list[list[tuple[int, int]]],
    canon_data: list[dict],
    canon_outlet_data: list[dict],
    mask_m: np.ndarray,
    mask_f: np.ndarray,
    cluster_label_fn,
) -> None:
    """Dashboard: Grassmann (row 0) + inlet canonical (rows 1-2) + outlet canonical (rows 3-4)."""
    n_clust = len(top_clusters)
    n_canon = len(canon_data)
    n_outlet = len(canon_outlet_data)
    if n_clust == 0:
        return
    n_cols = max(n_clust, n_canon, n_outlet)
    n_rows = 1 + (2 if n_canon > 0 else 0) + (2 if n_outlet > 0 else 0)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(COL2_WIDTH, ROW_H * n_rows),
        squeeze=False,
        layout="constrained",
    )

    # ── Row 0: Grassmann distances ────────────────────────────
    for col_i, (cs, ce, _) in enumerate(top_clusters):
        cl_label = cluster_label_fn(cs, ce)
        cl_key = f"modes_{cs + 1}_{ce}"
        ax = axes[0, col_i]
        if cl_key not in grassmann_results:
            ax.set_visible(False)
            continue
        dG = grassmann_results[cl_key]
        member_mask = np.array(
            [(cs, ce) in clist for clist in clusters_all], dtype=bool,
        )
        _sex_boxplot(
            ax,
            dG[mask_m & member_mask], dG[mask_f & member_mask],
            ylabel="$d_G$" if col_i == 0 else "",
            title=cl_label,
        )
        if col_i > 0:
            ax.set_ylabel("")
    for col_i in range(n_clust, n_cols):
        axes[0, col_i].set_visible(False)

    inlet_row_offset = 1
    # ── Rows 1–2: Canonical inlet scores ──────────────────────
    if n_canon > 0:
        for col_i, cd in enumerate(canon_data):
            mm = cd["member_mask"]
            for row_i, (key, ylabel_short) in enumerate([
                ("ap", "AP (mm/mm)"),
                ("ml", "ML (mm/mm)"),
            ]):
                ax = axes[inlet_row_offset + row_i, col_i]
                vals = cd[key]
                vm = vals[mask_m & mm]
                vf = vals[mask_f & mm]
                vm = vm[np.isfinite(vm)]
                vf = vf[np.isfinite(vf)]
                _sex_boxplot(
                    ax, vm, vf,
                    ylabel=ylabel_short if col_i == 0 else "",
                    title=cd["label"] if row_i == 0 else "",
                )
                if col_i > 0:
                    ax.set_ylabel("")
                    ax.set_yticklabels([])
        for col_i in range(n_canon, n_cols):
            for row_i in range(2):
                axes[inlet_row_offset + row_i, col_i].set_visible(False)
        axes[inlet_row_offset, 0].set_ylabel("AP opening\nper 1 mm max")
        axes[inlet_row_offset + 1, 0].set_ylabel("ML opening\nper 1 mm max")

    outlet_row_offset = inlet_row_offset + (2 if n_canon > 0 else 0)
    # ── Rows 3–4: Canonical outlet scores ─────────────────────
    if n_outlet > 0:
        for col_i, cd in enumerate(canon_outlet_data):
            mm = cd["member_mask"]
            for row_i, (key, ylabel_short) in enumerate([
                ("bis", "BIS (mm/mm)"),
                ("bit", "BIT (mm/mm)"),
            ]):
                ax = axes[outlet_row_offset + row_i, col_i]
                vals = cd[key]
                vm = vals[mask_m & mm]
                vf = vals[mask_f & mm]
                vm = vm[np.isfinite(vm)]
                vf = vf[np.isfinite(vf)]
                _sex_boxplot(
                    ax, vm, vf,
                    ylabel=ylabel_short if col_i == 0 else "",
                    title=cd["label"] if row_i == 0 else "",
                )
                if col_i > 0:
                    ax.set_ylabel("")
                    ax.set_yticklabels([])
        for col_i in range(n_outlet, n_cols):
            for row_i in range(2):
                axes[outlet_row_offset + row_i, col_i].set_visible(False)
        axes[outlet_row_offset, 0].set_ylabel("Biischiadic\nper 1 mm max")
        axes[outlet_row_offset + 1, 0].set_ylabel("Bituberous\nper 1 mm max")

    _save_fig(fig, "subspace_analysis_dashboard")
