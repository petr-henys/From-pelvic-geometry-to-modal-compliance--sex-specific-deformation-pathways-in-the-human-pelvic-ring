#!/usr/bin/env python3
"""Fractions Panel B: IQR percent-point effect sizes with 95% CIs for three load cases.

Reads:
- results/ref_S1P_fixed/analysis/fractions_results.csv

Produces:
- results/ref_S1P_fixed/analysis/fractions_panel_B.png
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
OUTPUT_FILE = ANALYSIS_DIR / "fractions_panel_B.png"
PLOT_LOADS = ["SP2leg", "SP1leg", "LAB_phase1", "LAB_phase2", "LAB_phase3"]


def draw_bars(ax, y, lo, hi, color, ylabel):
    """Draw bar chart with confidence intervals."""
    x = np.arange(len(y))
    yerr = np.vstack([y - lo, hi - y])
    ax.bar(x, y, color=color, yerr=yerr, capsize=2, linewidth=0.0)
    ax.set_ylabel(ylabel)
    ax.axhline(0, color="0.3", lw=0.8, ls="--", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([str(i + 1) for i in range(len(y))], fontsize=ANNOT_SIZE)


def main() -> None:
    """Generate Panel B figure with four effect size subplots per load case."""
    setup_plot_style()
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    df = pd.read_csv(ANALYSIS_DIR / "fractions_results.csv")

    fig, axes = plt.subplots(len(PLOT_LOADS), 4, figsize=(FULL_WIDTH, ROW_H * len(PLOT_LOADS)), sharex=False)
    
    for r, load in enumerate(PLOT_LOADS):
        sub = df[df["load_case"] == load].sort_values("mode").reset_index(drop=True)
        
        # Note: For fractions we don't have pct_* columns like eigenvalues,
        # so we use pp_*_IQR which are the IQR-based percent-point changes
        
        # Scale IQR effect
        y_s = sub.get("pp_scale_IQR", pd.Series([np.nan] * len(sub))).to_numpy()
        lo_s = sub.get("pp_scale_IQR_CI_lo", pd.Series([np.nan] * len(sub))).to_numpy()
        hi_s = sub.get("pp_scale_IQR_CI_hi", pd.Series([np.nan] * len(sub))).to_numpy()
        draw_bars(axes[r, 0], y_s, lo_s, hi_s, SCALE_EFFECT_COLOR, "Scale IQR (pp)")
        axes[r, 0].set_title(load, loc="left")
        
        # Sex effect
        y_x = sub["pp_sex"].to_numpy()
        lo_x = sub.get("pp_sex_CI_lo", pd.Series([np.nan] * len(sub))).to_numpy()
        hi_x = sub.get("pp_sex_CI_hi", pd.Series([np.nan] * len(sub))).to_numpy()
        draw_bars(axes[r, 1], y_x, lo_x, hi_x, SEX_EFFECT_COLOR, "Sex (pp)")
        
        # Shape IQR effect
        y_sh = sub.get("pp_shape_IQR", pd.Series([np.nan] * len(sub))).to_numpy()
        lo_sh = sub.get("pp_shape_IQR_CI_lo", pd.Series([np.nan] * len(sub))).to_numpy()
        hi_sh = sub.get("pp_shape_IQR_CI_hi", pd.Series([np.nan] * len(sub))).to_numpy()
        draw_bars(axes[r, 2], y_sh, lo_sh, hi_sh, SHAPE_EFFECT_COLOR, "Shape IQR (pp)")
        
        # Material IQR effect
        y_mt = sub.get("pp_material_IQR", pd.Series([np.nan] * len(sub))).to_numpy()
        lo_mt = sub.get("pp_material_IQR_CI_lo", pd.Series([np.nan] * len(sub))).to_numpy()
        hi_mt = sub.get("pp_material_IQR_CI_hi", pd.Series([np.nan] * len(sub))).to_numpy()
        draw_bars(axes[r, 3], y_mt, lo_mt, hi_mt, MATERIAL_EFFECT_COLOR, "Material IQR (pp)")

    axes[0, 0].set_title("Effect Sizes (percent-point IQR) with 95% CI", loc="left")
    for ax in axes[-1, :]:
        ax.set_xlabel("Mode Number")
    
    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
