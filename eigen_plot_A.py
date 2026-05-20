#!/usr/bin/env python3
"""Generate Panel A: Split violin plots showing eigenvalue distributions by sex."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, LogFormatterMathtext
from mpl_toolkits.axes_grid1 import make_axes_locatable

from utils.plot_utils import (
    ANNOT_SIZE,
    FULL_WIDTH,
    ROW_H,
    setup_plot_style,
    MALE_COLOR,
    FEMALE_COLOR,
    PERMUTATION_COLORS,
    PERMUTATION_DISPLAY,
)
from utils.stats_utils import fit_beta

NUM_MODES = 15
GAP_EPS = 1e-16
GAP_AXIS_MAX_FRACTION = 1.0 / 3.0
GAP_BAR_ALPHA = 0.35

DATA_DIR = Path("results/ref_S1P_fixed_new2/data")
OUTPUT_FILE = Path("results/ref_S1P_fixed_new2/analysis/panel_A.png")


def summary_stats_by_mode(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return median and 95% CI bounds per mode."""
    eig_cols = [f"eig_{i+1}" for i in range(NUM_MODES)]
    medians = np.full(NUM_MODES, np.nan)
    ci_low = np.full(NUM_MODES, np.nan)
    ci_high = np.full(NUM_MODES, np.nan)

    for idx, col in enumerate(eig_cols):
        values = pd.to_numeric(df[col], errors="coerce").to_numpy()
        valid = values[np.isfinite(values) & (values > 0)]
        medians[idx] = np.median(valid)
        ci_low[idx] = np.percentile(valid, 2.5)
        ci_high[idx] = np.percentile(valid, 97.5)

    return medians, ci_low, ci_high


def relative_gaps_from_medians(medians: np.ndarray) -> np.ndarray:
    """Compute symmetric relative gaps from consecutive eigenvalue medians."""
    gaps = np.full(NUM_MODES - 1, np.nan)
    for idx in range(NUM_MODES - 1):
        first, second = medians[idx], medians[idx + 1]
        denominator = max(abs(first) + abs(second), GAP_EPS)
        gaps[idx] = 2.0 * (second - first) / denominator
    return gaps


def compute_mode_gaps(eigen_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute median eigenvalues per mode and symmetric relative gaps."""
    medians, _, _ = summary_stats_by_mode(eigen_df)
    gaps = relative_gaps_from_medians(medians)
    x_mid = np.arange(NUM_MODES - 1) + 0.5
    return medians, x_mid, gaps


def gap_ci_from_bounds(ci_low: np.ndarray, ci_high: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert percentile bounds to gap CIs."""
    n_gaps = NUM_MODES - 1
    gap_ci_low = np.full(n_gaps, np.nan)
    gap_ci_high = np.full(n_gaps, np.nan)

    for idx in range(n_gaps):
        low_first, low_second = ci_low[idx], ci_low[idx + 1]
        high_first, high_second = ci_high[idx], ci_high[idx + 1]

        denom_low = max(abs(low_first) + abs(low_second), GAP_EPS)
        denom_high = max(abs(high_first) + abs(high_second), GAP_EPS)
        
        gap_low = 2.0 * (low_second - high_first) / denom_low
        gap_high = 2.0 * (high_second - low_first) / denom_high
        
        gap_ci_low[idx] = gap_low
        gap_ci_high[idx] = gap_high

    return gap_ci_low, gap_ci_high


def compute_sex_gap_components(male_df: pd.DataFrame, female_df: pd.DataFrame) -> tuple[tuple, tuple]:
    """Compute sex-specific gap components with CIs."""
    male_medians, male_ci_low, male_ci_high = summary_stats_by_mode(male_df)
    female_medians, female_ci_low, female_ci_high = summary_stats_by_mode(female_df)
    
    male_gaps = relative_gaps_from_medians(male_medians)
    female_gaps = relative_gaps_from_medians(female_medians)
    
    male_gap_ci_low, male_gap_ci_high = gap_ci_from_bounds(male_ci_low, male_ci_high)
    female_gap_ci_low, female_gap_ci_high = gap_ci_from_bounds(female_ci_low, female_ci_high)
    
    return (male_gaps, male_gap_ci_low, male_gap_ci_high), (female_gaps, female_gap_ci_low, female_gap_ci_high)


def plot_permutation_ribbon(ax: plt.Axes, perm_summary) -> None:
    """Append permutation summary ribbon above Panel A."""
    divider = make_axes_locatable(ax)
    ribbon_ax = divider.append_axes("top", size="12%", pad=0.15, sharex=ax)
    ribbon_ax.set_facecolor("white")

    x_positions = np.arange(NUM_MODES)
    bar_width = 0.34
    offsets = {"M": -0.18, "F": 0.18}

    for sex, offset in offsets.items():
        fractions = perm_summary["fractions"][sex]
        for mode_idx, x_pos in enumerate(x_positions):
            base = 0.0
            for cat_idx, category in enumerate(perm_summary["categories"]):
                height = fractions[mode_idx, cat_idx]
                color = PERMUTATION_COLORS[category]
                ribbon_ax.bar(x_pos + offset, height, width=bar_width, bottom=base,
                            color=color, edgecolor="none", alpha=0.82 if category == "aligned" else 0.92)
                base += height

    ribbon_ax.set_ylim(0.0, 1.0)
    ribbon_ax.set_yticks([0.0, 0.5, 1.0])
    ribbon_ax.set_yticklabels(["0", "0.5", "1"], fontsize=ANNOT_SIZE)
    ribbon_ax.tick_params(axis="x", which="both", labelbottom=False, bottom=False)
    ribbon_ax.set_ylabel("Permutation share", fontsize=ANNOT_SIZE, labelpad=10)
    ribbon_ax.grid(True, which="major", axis="y", linestyle=":", linewidth=0.6, color="0.8", alpha=0.8)
    for spine in ribbon_ax.spines.values():
        spine.set_alpha(0.3)
    ribbon_ax.set_xlim(-0.6, NUM_MODES - 0.4)

    for sex, offset in offsets.items():
        ribbon_ax.text(x_positions[0] + offset, 1.04, sex, fontsize=ANNOT_SIZE, fontweight="bold",
                      ha="center", va="bottom", color="0.3")


def load_permutation_summary() -> dict:
    """Load permutation data and compute summary."""
    perm_df = pd.read_excel(DATA_DIR / "eigen_data.xlsx", sheet_name="permutations")
    demo_df = pd.read_excel(DATA_DIR / "demography.xlsx")
    
    perm_cols = [col for col in perm_df.columns if col.startswith("perm_")]
    perm_cols.sort(key=lambda name: int(name.split("_")[1]))
    perm_cols = perm_cols[:NUM_MODES]
    
    fractions = {}
    perm_matrix = perm_df[perm_cols].apply(pd.to_numeric, errors="coerce").to_numpy()
    sex_series = demo_df["sex"].astype(str).str.upper()
    
    for sex in ["M", "F"]:
        sex_mask = sex_series == sex
        sex_fractions = np.zeros((NUM_MODES, 3))
        sex_perm = perm_matrix[sex_mask]
        
        for mode_idx in range(NUM_MODES):
            col = sex_perm[:, mode_idx]
            valid = np.isfinite(col)
            values = col[valid]
            total = values.size
            
            aligned = np.sum(values == mode_idx)
            unpaired = np.sum(values < 0)
            reassigned = total - aligned - unpaired
            
            sex_fractions[mode_idx] = np.array([aligned, reassigned, unpaired]) / total
        
        fractions[sex] = sex_fractions
    
    return {"categories": ("aligned", "reassigned", "unpaired"), "fractions": fractions}


def main():
    """Generate Panel A figure."""
    setup_plot_style()
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    # Load data
    eigen_df = pd.read_excel(DATA_DIR / "eigen_data.xlsx")
    demography_df = pd.read_excel(DATA_DIR / "demography.xlsx")
    perm_summary = load_permutation_summary()
    
    eig_cols = [f"eig_{i+1}" for i in range(NUM_MODES)]
    sex_labels = demography_df["sex"].astype(str).str.upper().values
    
    eigen_df = eigen_df.assign(sex=sex_labels)
    
    male_df = eigen_df[eigen_df["sex"] == "M"]
    female_df = eigen_df[eigen_df["sex"] == "F"]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 4 * ROW_H), constrained_layout=False)
    fig.subplots_adjust(top=0.88, bottom=0.1, left=0.08, right=0.92)
    
    # Add main title to figure
    fig.suptitle("Eigenvalue Distributions", x=0.08, y=0.965, ha="left")
    
    positions = np.arange(NUM_MODES)
    
    # Plot violins
    for i, col in enumerate(eig_cols):
        for subset, color, side_clip in [
            (male_df[col].values, MALE_COLOR, lambda v, idx=i: np.clip(v, -np.inf, idx)),
            (female_df[col].values, FEMALE_COLOR, lambda v, idx=i: np.clip(v, idx, np.inf)),
        ]:
            parts = ax.violinplot([subset], positions=[i], widths=0.88,
                                 showmeans=False, showmedians=False, showextrema=False)
            for pc in parts["bodies"]:
                pc.set_facecolor(color)
                pc.set_alpha(0.75)
                pc.set_edgecolor("none")
                vertices = pc.get_paths()[0].vertices
                vertices[:, 0] = side_clip(vertices[:, 0])
        
        male_median = np.median(male_df[col].values)
        female_median = np.median(female_df[col].values)
        ax.hlines(male_median, i - 0.4, i - 0.02, colors=MALE_COLOR, linewidth=2.0, zorder=5)
        ax.hlines(female_median, i + 0.02, i + 0.4, colors=FEMALE_COLOR, linewidth=2.0, zorder=5)
    
    ax.set_yscale("log")
    ax.set_xlim(-0.6, NUM_MODES - 0.4)
    ax.set_xticks(positions)
    ax.set_xticklabels([str(i + 1) for i in range(NUM_MODES)])
    ax.set_xlabel("Mode Number", labelpad=8)
    ax.set_ylabel("Eigenvalue (N/mm)", labelpad=8)
    
    ax.yaxis.set_major_locator(LogLocator(base=10))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
    ax.grid(True, which="major", axis="y", linestyle=":", linewidth=0.9, color="0.7", alpha=0.9)
    ax.grid(True, which="minor", axis="y", linestyle=":", linewidth=0.6, color="0.8", alpha=0.8)
    ax.grid(True, which="major", axis="x", linestyle=":", linewidth=0.6, color="0.85", alpha=0.7)
    
    y_lo, y_hi = ax.get_ylim()
    ax.set_ylim(y_lo, y_hi * 1.13)
    
    # Add gap overlay
    _, x_mid, gaps = compute_mode_gaps(eigen_df)
    male_stats, female_stats = compute_sex_gap_components(male_df, female_df)
    male_gaps, male_ci_low, male_ci_high = male_stats
    female_gaps, female_ci_low, female_ci_high = female_stats
    
    ax2 = ax.twinx()
    ax2.set_facecolor("none")
    ax2.grid(False)
    
    total_mask = np.isfinite(gaps)
    if not np.any(total_mask):
        raise ValueError("No finite gap values available for overlay.")
    max_abs_gap = np.nanmax(np.abs(gaps[total_mask]))
    axis_limit = max(max_abs_gap / GAP_AXIS_MAX_FRACTION, 0.1)
    positive_limit = axis_limit
    pad_pos = positive_limit * 0.05
    
    min_gap = np.nanmin(gaps[total_mask])
    negative_extent = abs(min_gap) if min_gap < 0 else 0.0
    negative_limit = negative_extent
    pad_neg = negative_limit * 0.25 if negative_limit > 0 else 0.015
    ax2.set_ylim(-(negative_limit + pad_neg), positive_limit + pad_pos)
    
    # Plot gap bars
    bar_width = 0.6
    ci_cap_width = bar_width * 0.35
    ci_offset = bar_width * 0.18
    eps = 1e-12
    
    for idx, x_pos in enumerate(x_mid):
        total_gap = gaps[idx]
        female_gap = female_gaps[idx]
        male_gap = male_gaps[idx]
        female_weight = abs(female_gap)
        male_weight = abs(male_gap)
        weight_sum = female_weight + male_weight
        
        female_raw = female_weight * np.sign(female_gap)
        male_raw = male_weight * np.sign(male_gap)
        raw_sum = female_raw + male_raw
        if abs(raw_sum) > eps:
            scale_factor = total_gap / raw_sum
            female_component = female_raw * scale_factor
            male_component = male_raw * scale_factor
        elif weight_sum > eps:
            # Opposite-sign components can cancel numerically. In that case,
            # split the total gap by absolute contribution weights.
            female_component = total_gap * (female_weight / weight_sum)
            male_component = total_gap - female_component
        else:
            female_component = 0.0
            male_component = 0.0
        
        bottom_pos = 0.0
        bottom_neg = 0.0
        
        for label, component, gap_value, gap_ci_lo, gap_ci_hi, color in [
            ("female", female_component, female_gap, female_ci_low[idx], female_ci_high[idx], FEMALE_COLOR),
            ("male", male_component, male_gap, male_ci_low[idx], male_ci_high[idx], MALE_COLOR),
        ]:
            if component > 0:
                start = bottom_pos
                bottom_pos += component
            else:
                start = bottom_neg
                bottom_neg += component
            
            ax2.bar(
                x_pos,
                component,
                width=bar_width,
                bottom=start,
                color=color,
                alpha=GAP_BAR_ALPHA,
                edgecolor="none",
            )
            if not np.isfinite(gap_value) or abs(gap_value) <= eps:
                continue
            scale = component / gap_value
            scaled_low = scale * gap_ci_lo
            scaled_high = scale * gap_ci_hi
            ci_low = min(scaled_low, scaled_high)
            ci_high = max(scaled_low, scaled_high)
            ci_low_plot = start + ci_low
            ci_high_plot = start + ci_high
            x_ci = x_pos - ci_offset if label == "female" else x_pos + ci_offset
            ax2.vlines(x_ci, ci_low_plot, ci_high_plot, color=color, linewidth=1.0, alpha=0.9)
            ax2.hlines(ci_low_plot, x_ci - ci_cap_width / 2, x_ci + ci_cap_width / 2,
                      color=color, linewidth=1.0, alpha=0.9)
            ax2.hlines(ci_high_plot, x_ci - ci_cap_width / 2, x_ci + ci_cap_width / 2,
                      color=color, linewidth=1.0, alpha=0.9)
    
    ax2.axhline(0.0, color="0.3", linestyle=":", linewidth=0.8, alpha=0.7)
    ax2.set_ylabel("Relative gap δ (sym.)", labelpad=8)
    ax2.set_zorder(ax.get_zorder() + 1)
    
    # Add permutation ribbon
    plot_permutation_ribbon(ax, perm_summary)
    
    # Legend (merged)
    legend_elements = [
        Patch(facecolor=MALE_COLOR, alpha=0.75, edgecolor="none", label="Male"),
        Patch(facecolor=FEMALE_COLOR, alpha=0.75, edgecolor="none", label="Female"),
        Patch(facecolor=FEMALE_COLOR, alpha=GAP_BAR_ALPHA, edgecolor="none", label="δ contribution (Female)"),
        Patch(facecolor=MALE_COLOR, alpha=GAP_BAR_ALPHA, edgecolor="none", label="δ contribution (Male)"),
        Line2D([0], [0], color=FEMALE_COLOR, linewidth=1.1, alpha=0.9, label="δ 95% CI (Female)"),
        Line2D([0], [0], color=MALE_COLOR, linewidth=1.1, alpha=0.9, label="δ 95% CI (Male)"),
        Patch(facecolor=PERMUTATION_COLORS["aligned"], edgecolor="none", alpha=0.9, label=PERMUTATION_DISPLAY["aligned"]),
        Patch(facecolor=PERMUTATION_COLORS["reassigned"], edgecolor="none", alpha=0.9, label=PERMUTATION_DISPLAY["reassigned"]),
        Patch(facecolor=PERMUTATION_COLORS["unpaired"], edgecolor="none", alpha=0.9, label=PERMUTATION_DISPLAY["unpaired"]),
    ]
    
    ax.legend(handles=legend_elements, loc="upper left", bbox_to_anchor=(0.0, 0.82),
             frameon=True, framealpha=0.95, edgecolor="0.8", ncol=2, columnspacing=1.0, handlelength=1.6)
    
    # Save
    fig.canvas.draw()
    plt.savefig(OUTPUT_FILE, bbox_inches='tight', pad_inches=0.2)
    plt.close()
    print(f"Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
