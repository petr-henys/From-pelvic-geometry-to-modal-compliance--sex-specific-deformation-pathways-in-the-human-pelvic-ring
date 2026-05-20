#!/usr/bin/env python3
"""Generate Panel D: Variance Contribution Pie Charts per Mode."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from utils.plot_utils import (
    FULL_WIDTH,
    ROW_H,
    setup_plot_style,
    SCALE_EFFECT_COLOR,
    SEX_EFFECT_COLOR,
    SHAPE_EFFECT_COLOR,
    MATERIAL_EFFECT_COLOR,
)

NUM_MODES = 15

DATA_DIR = Path("results/ref_S1P_fixed_new2/analysis")
OUTPUT_FILE = Path("results/ref_S1P_fixed_new2/analysis/panel_D.png")


def main():
    setup_plot_style()
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    results_df = pd.read_csv(DATA_DIR / "eigen_results.csv")

    fig, axes = plt.subplots(1, NUM_MODES, figsize=(FULL_WIDTH, 2 * ROW_H))
    axes = axes.flatten()

    for mode_idx in range(NUM_MODES):
        row = results_df[results_df["mode"] == mode_idx + 1].iloc[0]
        
        pct_scale = row["R2pct_scale"]
        pct_sex = row["R2pct_sex"]
        pct_shape = row["R2pct_shape"]
        pct_material = row["R2pct_material"]
        
        sizes = [pct_scale, pct_sex, pct_shape, pct_material]
        colors = [SCALE_EFFECT_COLOR, SEX_EFFECT_COLOR, SHAPE_EFFECT_COLOR, MATERIAL_EFFECT_COLOR]
        
        axes[mode_idx].pie(
            sizes,
            colors=colors,
            startangle=90,
            counterclock=False,
            wedgeprops={"linewidth": 0.5, "edgecolor": "white"},
        )
        axes[mode_idx].set_title(f"Mode {mode_idx + 1}")

    fig.suptitle("Variance Contribution per Mode", y=0.98, x=0.01, ha="left")

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
        bbox_to_anchor=(0.5, -0.02),
    )

    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
