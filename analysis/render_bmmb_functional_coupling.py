#!/usr/bin/env python3
"""Render the BMMB anatomical-capacity panels from archived subject metrics."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from analysis.publication_style import (
    COL2_WIDTH, BLOCK_COLORS, NEUTRAL_COLOR, apply_publication_style,
    panel_label, style_axis, style_distribution, save_publication_figure,
)

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "analysis_outputs/plos_revision/tables/subject_metrics.csv")
blocks = ["9-10", "12-15", "5-6", "12-13", "14-15"]
config = [
    ("AP", "A", "AP inlet"),
    ("ML", "B", "ML inlet"),
    ("BIS", "C", "Interspinous midpelvis"),
    ("BIT", "D", "Bituberous outlet"),
]
apply_publication_style()
fig, axs = plt.subplots(2, 2, figsize=(COL2_WIDTH, 5.4), layout="constrained")
for ax, (key, letter, title) in zip(axs.flat, config):
    values = [df.loc[(df.block == block) & df.member, key].to_numpy() for block in blocks]
    colors = [BLOCK_COLORS.get(block, NEUTRAL_COLOR) for block in blocks]
    style_distribution(
        ax, values, positions=np.arange(len(blocks)), labels=blocks,
        color=colors, width=0.46, pt_alpha=0.28, pt_size=7, rng_seed=42,
    )
    ax.set_ylabel("Basis-invariant capacity (mm/mm)")
    ax.set_xlabel("Rank block")
    panel_label(ax, letter)
    ax.set_title(title, loc="left", fontsize=8.5, color="#374151", pad=4.0)
    style_axis(ax, spines=("left", "bottom"), y_grid=True)
save_publication_figure(
    fig, ROOT / "manuscripts/bmmb/Fig3", formats=("pdf",), dpi=400, verbose=False,
)
plt.close(fig)
