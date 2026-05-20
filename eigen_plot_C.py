#!/usr/bin/env python3
"""Generate Panel C: Bar charts with 95% CIs for regression coefficients."""
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
    ROW_H_SMALL,
    setup_plot_style,
    SCALE_EFFECT_COLOR,
    SEX_EFFECT_COLOR,
    SHAPE_EFFECT_COLOR,
    MATERIAL_EFFECT_COLOR,
)

NUM_MODES = 15

DATA_DIR = Path("results/ref_S1P_fixed_new2/analysis")
OUTPUT_FILE = Path("results/ref_S1P_fixed_new2/analysis/panel_C.png")


def main():
    """Generate Panel C figure."""
    setup_plot_style()
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    # Load results
    results_df = pd.read_csv(DATA_DIR / "eigen_results.csv")
    results_df = results_df.sort_values("mode").reset_index(drop=True)
    
    mode_labels = results_df["mode"].astype(str).tolist()
    
    # Create one figure with 4 bar charts + 95% CI
    x = np.arange(NUM_MODES)
    fig, axes = plt.subplots(4, 1, figsize=(FULL_WIDTH, 4 * ROW_H_SMALL), sharex=True)
    axes = axes.flatten()

    # 1) γ (sex)
    gamma_sex = results_df["gamma_sex"].to_numpy()
    gs_lo = results_df.get("gamma_sex_CI_lo", pd.Series([np.nan] * NUM_MODES)).to_numpy()
    gs_hi = results_df.get("gamma_sex_CI_hi", pd.Series([np.nan] * NUM_MODES)).to_numpy()
    y = gamma_sex
    yerr = np.vstack([y - gs_lo, gs_hi - y])
    axes[0].bar(x, y, color=SEX_EFFECT_COLOR, yerr=yerr, capsize=2, linewidth=0.0)
    axes[0].set_ylabel("γ (sex)")
    axes[0].set_title("Regression Coefficients with 95% CI", loc="left")
    # mark significance
    for i, p in enumerate(results_df["gamma_sex_p"].to_numpy()):
        if p < 0.05:
            axes[0].text(i, gs_hi[i] + 0.02 * (np.nanmax(gs_hi) - np.nanmin(gs_lo)), "★", ha="center", va="bottom")

    # 2) β (scale)
    beta_scale = results_df["beta_scale"].to_numpy()
    bs_lo = results_df.get("beta_scale_CI_lo", pd.Series([np.nan] * NUM_MODES)).to_numpy()
    bs_hi = results_df.get("beta_scale_CI_hi", pd.Series([np.nan] * NUM_MODES)).to_numpy()
    y = beta_scale
    yerr = np.vstack([y - bs_lo, bs_hi - y])
    axes[1].bar(x, y, color=SCALE_EFFECT_COLOR, yerr=yerr, capsize=2, linewidth=0.0)
    axes[1].set_ylabel("β (scale)")
    for i, p in enumerate(results_df["beta_scale_p"].to_numpy()):
        if p < 0.05:
            axes[1].text(i, bs_hi[i] + 0.02 * (np.nanmax(bs_hi) - np.nanmin(bs_lo)), "★", ha="center", va="bottom")

    # 3) β (shape)
    beta_shape = results_df["beta_shape"].to_numpy()
    bsh_lo = results_df.get("beta_shape_CI_lo", pd.Series([np.nan] * NUM_MODES)).to_numpy()
    bsh_hi = results_df.get("beta_shape_CI_hi", pd.Series([np.nan] * NUM_MODES)).to_numpy()
    y = beta_shape
    yerr = np.vstack([y - bsh_lo, bsh_hi - y])
    axes[2].bar(x, y, color=SHAPE_EFFECT_COLOR, yerr=yerr, capsize=2, linewidth=0.0)
    axes[2].set_ylabel("β (shape)")
    for i, p in enumerate(results_df["beta_shape_p"].to_numpy()):
        if p < 0.05:
            axes[2].text(i, bsh_hi[i] + 0.02 * (np.nanmax(bsh_hi) - np.nanmin(bsh_lo)), "★", ha="center", va="bottom")

    # 4) β (material)
    beta_material = results_df["beta_material"].to_numpy()
    bm_lo = results_df.get("beta_material_CI_lo", pd.Series([np.nan] * NUM_MODES)).to_numpy()
    bm_hi = results_df.get("beta_material_CI_hi", pd.Series([np.nan] * NUM_MODES)).to_numpy()
    y = beta_material
    yerr = np.vstack([y - bm_lo, bm_hi - y])
    axes[3].bar(x, y, color=MATERIAL_EFFECT_COLOR, yerr=yerr, capsize=2, linewidth=0.0)
    axes[3].set_ylabel("β (material)")
    axes[3].set_xlabel("Mode Number")
    axes[3].set_xticks(x)
    axes[3].set_xticklabels(mode_labels, fontsize=ANNOT_SIZE)
    for i, p in enumerate(results_df["beta_material_p"].to_numpy()):
        if p < 0.05:
            axes[3].text(i, bm_hi[i] + 0.02 * (np.nanmax(bm_hi) - np.nanmin(bm_lo)), "★", ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, bbox_inches="tight")
    plt.close()
    print(f"Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
