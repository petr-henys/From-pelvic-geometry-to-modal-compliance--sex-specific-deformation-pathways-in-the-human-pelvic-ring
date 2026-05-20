#!/usr/bin/env python3
"""Fractions Panel C: Coefficients (β scale, γ sex, β shape, β material) with 95% CIs for three load cases.

Reads:
- results/ref_S1P_fixed/analysis/fractions_combined_results.csv

Produces:
- results/ref_S1P_fixed/analysis/fractions_panel_C.png
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.plot_utils import (
    ANNOT_SIZE,
    FULL_WIDTH,
    ROW_H,
    setup_plot_style,
    SCALE_EFFECT_COLOR,
    SEX_EFFECT_COLOR,
    SHAPE_EFFECT_COLOR,
    MATERIAL_EFFECT_COLOR,
)

ANALYSIS_DIR = Path("results/ref_S1P_fixed_new2/analysis")
OUTPUT_FILE = ANALYSIS_DIR / "fractions_panel_C.png"
PLOT_LOADS = ["SP2leg", "SP1leg", "LAB_phase1", "LAB_phase2", "LAB_phase3"]


def draw_bars_with_ci(ax, y, lo, hi, color, ylabel, pvals=None):
    """Draw bar chart with confidence intervals and significance markers."""
    x = np.arange(len(y))
    yerr = np.vstack([y - lo, hi - y])
    ax.bar(x, y, color=color, yerr=yerr, capsize=2, linewidth=0.0)
    ax.set_ylabel(ylabel)
    ax.axhline(0, color="0.3", lw=0.8, ls="--", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([str(i + 1) for i in range(len(y))], fontsize=ANNOT_SIZE)
    
    if pvals is not None:
        ymax = np.nanmax(hi)
        ymin = np.nanmin(lo)
        for i, p in enumerate(pvals):
            if not np.isfinite(p):
                continue
            if p < 0.05:
                ax.text(i, hi[i] + 0.02 * (ymax - ymin), "★", ha="center", va="bottom")


def main() -> None:
    """Generate Panel C figure with four coefficient plots per load case."""
    setup_plot_style()
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    df = pd.read_csv(ANALYSIS_DIR / "fractions_results.csv")

    fig, axes = plt.subplots(len(PLOT_LOADS), 4, figsize=(FULL_WIDTH, ROW_H * len(PLOT_LOADS)), sharex=False)

    for r, load in enumerate(PLOT_LOADS):
        sub = df[df["load_case"] == load].sort_values("mode").reset_index(drop=True)
        
        # γ (sex)
        y = sub["gamma_sex"].to_numpy()
        lo = sub.get("gamma_sex_CI_lo", pd.Series([np.nan] * len(sub))).to_numpy()
        hi = sub.get("gamma_sex_CI_hi", pd.Series([np.nan] * len(sub))).to_numpy()
        draw_bars_with_ci(axes[r, 0], y, lo, hi, SEX_EFFECT_COLOR, "γ (sex)", sub["gamma_sex_p"].to_numpy())
        axes[r, 0].set_title(load, loc="left")
        
        # β (scale)
        yb = sub["beta_scale"].to_numpy()
        lob = sub.get("beta_scale_CI_lo", pd.Series([np.nan] * len(sub))).to_numpy()
        hib = sub.get("beta_scale_CI_hi", pd.Series([np.nan] * len(sub))).to_numpy()
        draw_bars_with_ci(axes[r, 1], yb, lob, hib, SCALE_EFFECT_COLOR, "β (scale)", sub["beta_scale_p"].to_numpy())
        
        # β (shape)
        ys = sub.get("beta_shape", pd.Series([np.nan] * len(sub))).to_numpy()
        los = sub.get("beta_shape_CI_lo", pd.Series([np.nan] * len(sub))).to_numpy()
        his = sub.get("beta_shape_CI_hi", pd.Series([np.nan] * len(sub))).to_numpy()
        draw_bars_with_ci(axes[r, 2], ys, los, his, SHAPE_EFFECT_COLOR, "β (shape)", 
                         sub.get("beta_shape_p", pd.Series([np.nan] * len(sub))).to_numpy())
        
        # β (material)
        ym = sub.get("beta_material", pd.Series([np.nan] * len(sub))).to_numpy()
        lom = sub.get("beta_material_CI_lo", pd.Series([np.nan] * len(sub))).to_numpy()
        him = sub.get("beta_material_CI_hi", pd.Series([np.nan] * len(sub))).to_numpy()
        draw_bars_with_ci(axes[r, 3], ym, lom, him, MATERIAL_EFFECT_COLOR, "β (material)", 
                         sub.get("beta_material_p", pd.Series([np.nan] * len(sub))).to_numpy())

    axes[0, 0].set_title("Coefficients with 95% CI", loc="left")
    for ax in axes[-1, :]:
        ax.set_xlabel("Mode Number")
    
    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
