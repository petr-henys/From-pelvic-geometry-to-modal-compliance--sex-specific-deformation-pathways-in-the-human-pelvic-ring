#!/usr/bin/env python3
"""Generate Panel B: Bar charts with 95% CIs for percent effects."""
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
OUTPUT_FILE = Path("results/ref_S1P_fixed_new2/analysis/eigen_plot_B.png")


def main():
    """Generate Panel B figure with four bar charts."""
    setup_plot_style()
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Load results
    results_df = pd.read_csv(DATA_DIR / "eigen_results.csv")
    results_df = results_df.sort_values("mode").reset_index(drop=True)

    x = np.arange(NUM_MODES)
    mode_labels = results_df["mode"].astype(str).tolist()

    fig, axes = plt.subplots(4, 1, figsize=(FULL_WIDTH, 4 * ROW_H_SMALL), sharex=True)
    axes = axes.flatten()

    # 1) Scale IQR percent effect with CI
    y = results_df["pct_scale_IQR"].to_numpy()
    s_lo = results_df.get("pct_scale_IQR_CI_lo", pd.Series([np.nan] * NUM_MODES)).to_numpy()
    s_hi = results_df.get("pct_scale_IQR_CI_hi", pd.Series([np.nan] * NUM_MODES)).to_numpy()
    yerr = np.vstack([y - s_lo, s_hi - y])
    axes[0].bar(x, y, color=SCALE_EFFECT_COLOR, yerr=yerr, capsize=2, linewidth=0.0)
    axes[0].axhline(0.0, color="0.3", linewidth=0.8, linestyle="--", alpha=0.5)
    axes[0].set_ylabel("Scale IQR (%)")
    axes[0].set_title("Effect Sizes (percent) with 95% CI", loc="left")

    # 2) Sex percent effect with CI
    y = results_df["pct_sex"].to_numpy()
    sx_lo = results_df.get("pct_sex_CI_lo", pd.Series([np.nan] * NUM_MODES)).to_numpy()
    sx_hi = results_df.get("pct_sex_CI_hi", pd.Series([np.nan] * NUM_MODES)).to_numpy()
    yerr = np.vstack([y - sx_lo, sx_hi - y])
    axes[1].bar(x, y, color=SEX_EFFECT_COLOR, yerr=yerr, capsize=2, linewidth=0.0)
    axes[1].axhline(0.0, color="0.3", linewidth=0.8, linestyle="--", alpha=0.5)
    axes[1].set_ylabel("Sex (%)")

    # 3) Shape percent effect (IQR) with CI
    y = results_df["pct_shape_IQR"].to_numpy()
    sh_lo = results_df.get("pct_shape_IQR_CI_lo", pd.Series([np.nan] * NUM_MODES)).to_numpy()
    sh_hi = results_df.get("pct_shape_IQR_CI_hi", pd.Series([np.nan] * NUM_MODES)).to_numpy()
    yerr = np.vstack([y - sh_lo, sh_hi - y])
    axes[2].bar(x, y, color=SHAPE_EFFECT_COLOR, yerr=yerr, capsize=2, linewidth=0.0)
    axes[2].axhline(0.0, color="0.3", linewidth=0.8, linestyle="--", alpha=0.5)
    axes[2].set_ylabel("Shape IQR (%)")

    # 4) Material percent effect (IQR) with CI
    y = results_df["pct_material_IQR"].to_numpy()
    mt_lo = results_df.get("pct_material_IQR_CI_lo", pd.Series([np.nan] * NUM_MODES)).to_numpy()
    mt_hi = results_df.get("pct_material_IQR_CI_hi", pd.Series([np.nan] * NUM_MODES)).to_numpy()
    yerr = np.vstack([y - mt_lo, mt_hi - y])
    axes[3].bar(x, y, color=MATERIAL_EFFECT_COLOR, yerr=yerr, capsize=2, linewidth=0.0)
    axes[3].axhline(0.0, color="0.3", linewidth=0.8, linestyle="--", alpha=0.5)
    axes[3].set_ylabel("Material IQR (%)")
    axes[3].set_xlabel("Mode Number")
    axes[3].set_xticks(x)
    axes[3].set_xticklabels(mode_labels, fontsize=ANNOT_SIZE)

    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, bbox_inches="tight")
    plt.close()
    print(f"Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
