#!/usr/bin/env python3
"""Fractions Panel A: Violin plots per load case (M vs F) from solver-order Zarr.

Reads:
- results/ref_S1P_fixed_new2/data/mode_energy_fraction.zarr  (solver / rank order)
- results/ref_S1P_fixed_new2/data/demography.xlsx

Produces:
- results/ref_S1P_fixed_new2/analysis/fractions_panel_A.{pdf,png}
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import zarr

from utils.plot_utils import (
    FULL_WIDTH, ROW_H,
    setup_plot_style, MALE_COLOR, FEMALE_COLOR, FILL_ALPHA, BOX_LW,
    panel_label,
)

DATA_DIR = Path("results/ref_S1P_fixed_new2/data")
ANALYSIS_DIR = Path("results/ref_S1P_fixed_new2/analysis")
OUTPUT_STEM = ANALYSIS_DIR / "fractions_panel_A"
PLOT_LOADS = ["SP2leg", "SP1leg", "LAB_phase1", "LAB_phase2", "LAB_phase3"]
DISPLAY_LABEL = {
    "SP2leg": "SP2leg",
    "SP1leg": "SP1leg",
    "LAB_phase1": "LAB\u2081",
    "LAB_phase2": "LAB\u2082",
    "LAB_phase3": "LAB\u2083",
}
NUM_MODES = 12
LOADCASE_INDEX: dict[str, int] = {
    "SP2leg": 0,
    "SP1leg": 1,
    "LAB_phase1": 2,
    "LAB_phase2": 3,
    "LAB_phase3": 4,
}


def plot_violin(ax, df_load: pd.DataFrame) -> None:
    """Plot split violin plots for M vs F."""
    for mode in range(1, NUM_MODES + 1):
        sub = df_load[df_load["mode"] == mode]
        mv = sub.loc[sub["sex"].str.upper() == "M", "fraction"].to_numpy()
        fv = sub.loc[sub["sex"].str.upper() == "F", "fraction"].to_numpy()
        mv = mv[np.isfinite(mv)]
        fv = fv[np.isfinite(fv)]
        
        if mv.size >= 2:
            parts = ax.violinplot([mv], positions=[mode - 1], widths=0.75,
                                  showmeans=False, showmedians=False, showextrema=False)
            for pc in parts["bodies"]:
                pc.set_facecolor(MALE_COLOR)
                pc.set_alpha(FILL_ALPHA)
                pc.set_edgecolor("black")
                pc.set_linewidth(BOX_LW)
                if len(pc.get_paths()) > 0:
                    pc.get_paths()[0].vertices[:, 0] = np.clip(
                        pc.get_paths()[0].vertices[:, 0], -np.inf, (mode - 1) - 0.02
                    )
        
        if fv.size >= 2:
            parts = ax.violinplot([fv], positions=[mode - 1], widths=0.75,
                                  showmeans=False, showmedians=False, showextrema=False)
            for pc in parts["bodies"]:
                pc.set_facecolor(FEMALE_COLOR)
                pc.set_alpha(FILL_ALPHA)
                pc.set_edgecolor("black")
                pc.set_linewidth(BOX_LW)
                if len(pc.get_paths()) > 0:
                    pc.get_paths()[0].vertices[:, 0] = np.clip(
                        pc.get_paths()[0].vertices[:, 0], (mode - 1) + 0.02, np.inf
                    )
    
    ax.set_ylim(0, 1)
    ax.set_xlim(-0.5, NUM_MODES - 0.5)
    ax.set_xticks(range(NUM_MODES))
    ax.set_xticklabels([str(i + 1) for i in range(NUM_MODES)])
    ax.set_ylabel("Energy fraction")
    ax.grid(True, axis="y", linestyle=":", linewidth=0.8, color="0.8", alpha=0.8)


def main() -> None:
    """Generate Panel A figure with violin plots."""
    setup_plot_style()
    OUTPUT_STEM.parent.mkdir(parents=True, exist_ok=True)
    
    # Load solver-order modal energy fractions from Zarr
    frac = np.asarray(zarr.open(str(DATA_DIR / "mode_energy_fraction.zarr"), mode="r")["data"][:], dtype=float)
    if frac.ndim != 3:
        raise ValueError(f"Expected (N_subjects, N_modes, N_loadcases), got {frac.shape}")
    if frac.shape[1] < NUM_MODES or frac.shape[2] < len(LOADCASE_INDEX):
        raise ValueError(f"Data shape {frac.shape} incompatible with expected modes/loads.")

    demo = pd.read_excel(DATA_DIR / "demography.xlsx")
    demo["sex"] = demo["sex"].astype(str).str.strip().str.upper()

    if len(demo) != frac.shape[0]:
        raise ValueError(
            f"demography.xlsx rows ({len(demo)}) != zarr subjects ({frac.shape[0]}). "
            "Check subject ordering / dataset alignment."
        )

    # Build long-form data
    long_records = []
    for load in PLOT_LOADS:
        li = LOADCASE_INDEX[load]
        block = frac[:, :NUM_MODES, li]  # (N, NUM_MODES)
        df_long = pd.DataFrame({
            "sex": demo["sex"].to_numpy(),
        })
        for mode in range(1, NUM_MODES + 1):
            tmp = df_long.copy()
            tmp["load_case"] = load
            tmp["mode"] = mode
            tmp["fraction"] = block[:, mode - 1]
            long_records.append(tmp)

    df_long = pd.concat(long_records, ignore_index=True)

    fig, axes = plt.subplots(1, len(PLOT_LOADS), figsize=(FULL_WIDTH, ROW_H * 1.6),
                             sharey=True, layout="constrained")
    
    for i, load in enumerate(PLOT_LOADS):
        ax = axes[i]
        sub = df_long[df_long["load_case"] == load]
        plot_violin(ax, sub)
        panel_label(ax, f"({chr(97 + i)}) {DISPLAY_LABEL.get(load, load)}")
        ax.set_xlabel("Mode")

    axes[0].set_ylabel("Energy fraction")
    for fmt in ("pdf", "png"):
        out = OUTPUT_STEM.with_suffix(f".{fmt}")
        fig.savefig(out)
    plt.close()
    print(f"Saved: {OUTPUT_STEM}.{{pdf,png}}")


if __name__ == "__main__":
    main()
