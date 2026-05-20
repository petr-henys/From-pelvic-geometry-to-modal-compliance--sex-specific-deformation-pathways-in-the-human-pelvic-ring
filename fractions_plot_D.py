#!/usr/bin/env python3
"""Fractions Panel D: Pie charts showing coefficient magnitudes for each mode across three load cases.

Reads:
- results/ref_S1P_fixed/analysis/fractions_combined_results.csv

Produces:
- results/ref_S1P_fixed/analysis/fractions_panel_D.png
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

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
OUTPUT_FILE = ANALYSIS_DIR / "fractions_panel_D.png"
PLOT_LOADS = ["SP2leg", "SP1leg", "LAB_phase1", "LAB_phase2", "LAB_phase3"]


def main() -> None:
    """Generate Panel D figure with pie charts for each mode and load case."""
    setup_plot_style()
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    df = pd.read_csv(ANALYSIS_DIR / "fractions_results.csv")
    
    # Determine number of modes from first load case
    first_load_df = df[df["load_case"] == PLOT_LOADS[0]]
    num_modes = len(first_load_df)
    
    # Create subplots: 5 rows (load cases) × num_modes columns
    fig, axes = plt.subplots(len(PLOT_LOADS), num_modes, figsize=(FULL_WIDTH, ROW_H * len(PLOT_LOADS)))
    
    for row_idx, load in enumerate(PLOT_LOADS):
        sub = df[df["load_case"] == load].sort_values("mode").reset_index(drop=True)
        
        for mode_idx in range(num_modes):
            ax = axes[row_idx, mode_idx]
            row = sub.iloc[mode_idx]
            
            # Get absolute values of coefficients
            beta_scale = abs(row.get("beta_scale", 0.0))
            gamma_sex = abs(row.get("gamma_sex", 0.0))
            beta_shape = abs(row.get("beta_shape", 0.0))
            beta_material = abs(row.get("beta_material", 0.0))
            
            sizes = [beta_scale, gamma_sex, beta_shape, beta_material]
            colors = [SCALE_EFFECT_COLOR, SEX_EFFECT_COLOR, SHAPE_EFFECT_COLOR, MATERIAL_EFFECT_COLOR]
            
            # Only plot if at least one value is non-zero
            if sum(sizes) > 0:
                ax.pie(
                    sizes,
                    colors=colors,
                    startangle=90,
                    counterclock=False,
                    wedgeprops={"linewidth": 0.5, "edgecolor": "white"},
                )
            
            # Add title only for top row
            if row_idx == 0:
                ax.set_title(f"Mode {mode_idx + 1}", fontsize=ANNOT_SIZE)
            
            # Add load case label only for first column
            if mode_idx == 0:
                ax.text(-1.5, 0, load, ha="right", va="center", rotation=0)
    
    # Add main title
    fig.suptitle("Coefficient Magnitudes (|β|, |γ|) per Mode and Load Case", y=0.99, x=0.01, ha="left")
    
    # Add legend
    legend_elements = [
        Patch(facecolor=SCALE_EFFECT_COLOR, label="Scale"),
        Patch(facecolor=SEX_EFFECT_COLOR, label="Sex"),
        Patch(facecolor=SHAPE_EFFECT_COLOR, label="Shape"),
        Patch(facecolor=MATERIAL_EFFECT_COLOR, label="Material"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
    )
    
    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
