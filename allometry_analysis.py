#!/usr/bin/env python3
"""Allometry scatter-matrix colored by sex with regression lines.

Simple, explicit script. Assumes required files and columns exist.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils.plot_utils import FEMALE_COLOR, MALE_COLOR, setup_plot_style

PROJECT_ROOT = Path(__file__).resolve().parent
SIMULATION_DIR = PROJECT_ROOT / "results" / "ref_S1P_fixed_new2"
DATA_DIR = SIMULATION_DIR / "data"
ANALYSIS_DIR = SIMULATION_DIR / "analysis"
OUTPUT_PATH = ANALYSIS_DIR / "allometry_pairmatrix.png"


def main() -> None:
    setup_plot_style()
    sns.set_palette([MALE_COLOR, FEMALE_COLOR])

    # Load data (assumes files/columns exist)
    allometry = pd.read_excel(DATA_DIR / "allometry.xlsx")
    demography = pd.read_excel(DATA_DIR / "demography.xlsx")

    # Merge on patient_id; keep sex and age
    demography = demography[["patient_id", "sex", "age"]]
    df = pd.merge(allometry, demography, on="patient_id", how="inner")

    # Select variables: all numeric allometry + age, excluding identifiers and sex
    exclude = {"patient_id", "sex"}
    vars_to_plot = [
        c for c in df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    ]

    # PairGrid with hue by sex
    g = sns.PairGrid(df.dropna(subset=vars_to_plot + ["sex"]), vars=vars_to_plot, hue="sex", height=1.2)
    g.map_upper(sns.scatterplot, s=10, alpha=0.45, linewidth=0)
    g.map_lower(sns.scatterplot, s=10, alpha=0.45, linewidth=0)
    # Single regression line per pair (simple)
    g.map_lower(sns.regplot, scatter=False, ci=None, line_kws={"lw": 1.1, "alpha": 0.9})
    # Diagonal histograms
    g.map_diag(sns.histplot, bins=20, alpha=0.6)

    # Fix overlapping variable names: rotate and shrink; replace underscores
    for ax in g.axes.flatten():
        if ax is None:
            continue
        ax.tick_params(axis="x", labelrotation=45, labelsize=7)
        ax.tick_params(axis="y", labelsize=7)
        if ax.get_xlabel():
            ax.set_xlabel(ax.get_xlabel().replace("_", " "))
            ax.xaxis.label.set_rotation(35)
            ax.xaxis.label.set_ha("right")
        if ax.get_ylabel():
            ax.set_ylabel(ax.get_ylabel().replace("_", " "))
        ax.grid(True, alpha=0.25)

    g.add_legend(title="Sex")
    g.fig.subplots_adjust(left=0.07, right=0.98, top=0.98, bottom=0.10, wspace=0.08, hspace=0.08)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    g.figure.savefig(OUTPUT_PATH, bbox_inches="tight")
    plt.close(g.figure)


if __name__ == "__main__":
    main()
