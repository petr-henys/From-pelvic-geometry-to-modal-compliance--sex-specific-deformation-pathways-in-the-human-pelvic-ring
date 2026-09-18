from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import zarr

from analysis.publication_style import (
    apply_publication_style,
    panel_label,
    style_axis,
    MALE_COLOR,
    FEMALE_COLOR,
    FULL_WIDTH,
    ROW_H,
)

DATA_DIR = Path("results/ref_S1P_fixed_new2/data")
ANALYSIS_DIR = Path("results/ref_S1P_fixed_new2/analysis")
OUTPUT_STEM = ANALYSIS_DIR / "fractions_panel_A"
PLOT_LOADS = ["SP2leg", "SP1leg", "LAB_phase1", "LAB_phase2", "LAB_phase3"]
DISPLAY_LABEL = {
    "SP2leg": "SP2leg",
    "SP1leg": "SP1leg",
    "LAB_phase1": r"LAB$_1$",
    "LAB_phase2": r"LAB$_2$",
    "LAB_phase3": r"LAB$_3$",
}
NUM_MODES = 15
LOADCASE_INDEX: dict[str, int] = {
    "SP2leg": 0,
    "SP1leg": 1,
    "LAB_phase1": 2,
    "LAB_phase2": 3,
    "LAB_phase3": 4,
}


def plot_violin(ax: plt.Axes, df_load: pd.DataFrame, show_ylabel: bool = False) -> None:
    """Plot split violin plots for M vs F with medians."""
    for mode in range(1, NUM_MODES + 1):
        sub = df_load[df_load["mode"] == mode]
        mv = sub.loc[sub["sex"].str.upper() == "M", "fraction"].to_numpy()
        fv = sub.loc[sub["sex"].str.upper() == "F", "fraction"].to_numpy()
        mv = mv[np.isfinite(mv)]
        fv = fv[np.isfinite(fv)]

        # Male (left half)
        if mv.size >= 2:
            parts = ax.violinplot([mv], positions=[mode - 1], widths=0.8,
                                  showmeans=False, showmedians=False, showextrema=False)
            for pc in parts["bodies"]:
                pc.set_facecolor(MALE_COLOR)
                pc.set_alpha(0.55)
                pc.set_edgecolor(MALE_COLOR)
                pc.set_linewidth(0.6)
                if len(pc.get_paths()) > 0:
                    pc.get_paths()[0].vertices[:, 0] = np.clip(
                        pc.get_paths()[0].vertices[:, 0], -np.inf, (mode - 1) - 0.015
                    )
            # Male median indicator
            med_m = np.median(mv)
            ax.plot([(mode - 1) - 0.28, (mode - 1) - 0.03], [med_m, med_m],
                    color="#003B66", lw=1.2, solid_capstyle="round")

        # Female (right half)
        if fv.size >= 2:
            parts = ax.violinplot([fv], positions=[mode - 1], widths=0.8,
                                  showmeans=False, showmedians=False, showextrema=False)
            for pc in parts["bodies"]:
                pc.set_facecolor(FEMALE_COLOR)
                pc.set_alpha(0.55)
                pc.set_edgecolor(FEMALE_COLOR)
                pc.set_linewidth(0.6)
                if len(pc.get_paths()) > 0:
                    pc.get_paths()[0].vertices[:, 0] = np.clip(
                        pc.get_paths()[0].vertices[:, 0], (mode - 1) + 0.015, np.inf
                    )
            # Female median indicator
            med_f = np.median(fv)
            ax.plot([(mode - 1) + 0.03, (mode - 1) + 0.28], [med_f, med_f],
                    color="#8C3B00", lw=1.2, solid_capstyle="round")

    ax.set_ylim(-0.02, 1.02)
    ax.set_xlim(-0.6, NUM_MODES - 0.4)
    ax.set_xticks(range(NUM_MODES))
    ax.set_xticklabels([str(i + 1) for i in range(NUM_MODES)], fontsize=6.8)
    if show_ylabel:
        ax.set_ylabel("Energy fraction")
    else:
        ax.set_ylabel("")
    ax.set_xlabel("Mode", fontsize=7.5)
    style_axis(ax, x_grid=False, y_grid=True)


def main() -> None:
    """Generate Panel A figure with split violin plots."""
    apply_publication_style()
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

    fig, axes = plt.subplots(
        1, len(PLOT_LOADS),
        figsize=(FULL_WIDTH, ROW_H * 1.55),
        sharey=True,
        constrained_layout=True,
    )

    panel_letters = ["A", "B", "C", "D", "E"]
    for i, load in enumerate(PLOT_LOADS):
        ax = axes[i]
        sub = df_long[df_long["load_case"] == load]
        plot_violin(ax, sub, show_ylabel=(i == 0))
        panel_label(ax, panel_letters[i], offset=(-0.05, 1.05))
        ax.set_title(DISPLAY_LABEL.get(load, load), fontsize=8, pad=4, fontweight="bold")

    # Sex legend on panel A (modes 8-15 have near-zero fraction in SP2leg)
    legend_elements = [
        Patch(facecolor=MALE_COLOR, edgecolor="#003B66", alpha=0.6, label="Male"),
        Patch(facecolor=FEMALE_COLOR, edgecolor="#8C3B00", alpha=0.6, label="Female"),
    ]
    axes[0].legend(
        handles=legend_elements,
        loc="upper right",
        frameon=True,
        facecolor="white",
        edgecolor="none",
        fontsize=6.5,
        borderpad=0.25,
        handlelength=1.0,
        handleheight=0.8,
    )

    for fmt in ("pdf", "png"):
        out = OUTPUT_STEM.with_suffix(f".{fmt}")
        fig.savefig(out, dpi=400, bbox_inches="tight", pad_inches=0.04)
    plt.close()
    print(f"Saved: {OUTPUT_STEM}.{{pdf,png}}")


if __name__ == "__main__":
    main()
