#!/usr/bin/env python3
"""Dashboard for unified material uncertainty propagation.

Seven-panel figure:
  A  Total CoV by sex (bar)
  B  Variance decomposition: ligament / cartilage / bone (stacked bar)
  C  Gap Z-scores by sex (bar + p10 whisker)
  D  Ligament internal decomposition: global / type / asym (stacked bar)
  E  Cartilage per-region importance (stacked bar)
  F  Bone α vs β importance (stacked bar)
  G  Combined parameter importance (top-N, grouped bar)
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.plot_utils import (
    ANNOT_SIZE,
    DEFAULT_FONT_SIZE,
    FULL_WIDTH,
    ROW_H,
    MALE_COLOR,
    FEMALE_COLOR,
    SMALL_ANNOT_SIZE,
    setup_plot_style,
    format_mode_label_short,
    panel_label,
)

# ── Paths ────────────────────────────────────────────────────────────────
DATA_PATH = Path("analysis_outputs/material_uncertainty/population_material_uq.csv")
SUMMARY_PATH = Path("analysis_outputs/material_uncertainty/population_material_summary.csv")
OUTPUT_DIR = Path("analysis_outputs/figures")

# ── Colours ──────────────────────────────────────────────────────────────
COLOR_LIG = "#0173B2"      # blue — ligaments
COLOR_CART = "#DE8F05"     # orange — cartilage
COLOR_BONE = "#029E73"     # green — bone

# Ligament internal levels
COLOR_LIG_GLOBAL = "#56B4E9"   # light blue
COLOR_LIG_TYPE = "#CC78BC"     # lilac
COLOR_LIG_ASYM = "#D55E00"     # vermillion

# Cartilage regions
COLOR_SIJ_L = "#E69F00"
COLOR_SIJ_R = "#F0E442"
COLOR_SYM = "#56B4E9"

# Bone params
COLOR_ALPHA = "#009E73"
COLOR_BETA = "#CC78BC"


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing {DATA_PATH}. Run analysis/material_uncertainty_population.py first."
        )
    return pd.read_csv(DATA_PATH), pd.read_csv(SUMMARY_PATH)


def plot_dashboard(df: pd.DataFrame, summary: pd.DataFrame, output_path: Path) -> None:
    setup_plot_style()

    fig = plt.figure(figsize=(FULL_WIDTH, 4.35 * ROW_H), constrained_layout=False)
    gs = fig.add_gridspec(
        4,
        2,
        hspace=0.36,
        wspace=0.22,
        height_ratios=[1.0, 1.0, 1.0, 0.95],
    )

    ax_cov = fig.add_subplot(gs[0, 0])
    ax_decomp = fig.add_subplot(gs[0, 1])
    ax_gap = fig.add_subplot(gs[1, 0])
    ax_lig = fig.add_subplot(gs[1, 1])
    ax_cart = fig.add_subplot(gs[2, 0])
    ax_bone = fig.add_subplot(gs[2, 1])
    ax_imp = fig.add_subplot(gs[3, :])

    modes = sorted(df["mode"].unique())
    n_modes = len(modes)
    x_pos = np.arange(n_modes)
    labels = [format_mode_label_short(m - 1, one_based=True) for m in modes]
    bar_w = 0.35

    # ── Panel A: Total CoV by sex ────────────────────────────────
    for j, (sex, color) in enumerate([("Female", FEMALE_COLOR), ("Male", MALE_COLOR)]):
        s = summary[summary["sex"] == sex].set_index("mode")
        cov = s.reindex(modes)["mean_CoV_total"] * 100
        ax_cov.bar(x_pos + j * bar_w, cov, width=bar_w, color=color, label=sex, alpha=0.85)

    ax_cov.set_ylabel("CoV (%)")
    panel_label(ax_cov, "A. Total CoV")
    ax_cov.set_xticks(x_pos + bar_w / 2)
    ax_cov.set_xticklabels(labels, rotation=30, ha="right")
    ax_cov.legend(frameon=False, fontsize=ANNOT_SIZE)
    ax_cov.grid(axis="y", ls="--", alpha=0.4)

    # ── Panel B: Three-group variance decomposition ──────────────
    mean_fracs = (
        df.groupby("mode")[["frac_lig", "frac_cart", "frac_bone"]]
        .mean()
        .reindex(modes)
    )
    fl = mean_fracs["frac_lig"].values * 100
    fc = mean_fracs["frac_cart"].values * 100
    fb = mean_fracs["frac_bone"].values * 100

    ax_decomp.bar(x_pos, fl, color=COLOR_LIG, label="Ligaments", width=0.7)
    ax_decomp.bar(x_pos, fc, bottom=fl, color=COLOR_CART, label="Cartilage", width=0.7)
    ax_decomp.bar(x_pos, fb, bottom=fl + fc, color=COLOR_BONE, label="Bone", width=0.7)
    ax_decomp.set_ylabel("Variance Explained (%)")
    panel_label(ax_decomp, "B. Group Variance Share")
    ax_decomp.set_ylim(0, 100)
    ax_decomp.set_xticks(x_pos)
    ax_decomp.set_xticklabels(labels, rotation=30, ha="right")
    ax_decomp.legend(frameon=False, fontsize=ANNOT_SIZE, ncol=3, loc="upper right")
    ax_decomp.grid(axis="y", ls="--", alpha=0.4)

    # ── Panel C: Gap Z-scores ────────────────────────────────────
    for j, (sex, color) in enumerate([("Female", FEMALE_COLOR), ("Male", MALE_COLOR)]):
        s = summary[summary["sex"] == sex].set_index("mode")
        z_mean = s.reindex(modes)["mean_gap_z"]
        z_p10 = s.reindex(modes)["p10_gap_z"]
        offset = (j - 0.5) * bar_w
        ax_gap.bar(x_pos + offset, z_mean, width=bar_w, color=color, alpha=0.8,
                   label=f"{sex} (mean)")
        for k in range(n_modes):
            zm = z_mean.iloc[k] if k < len(z_mean) else np.nan
            zp = z_p10.iloc[k] if k < len(z_p10) else np.nan
            if pd.notna(zm) and pd.notna(zp):
                ax_gap.plot([x_pos[k] + offset] * 2, [zp, zm], color="black", lw=0.8)
                ax_gap.plot(x_pos[k] + offset, zp, "v", color="black", ms=3)

    ax_gap.axhline(2.0, color="red", ls="--", alpha=0.6, lw=0.8)
    ax_gap.text(n_modes - 0.5, 2.2, "Z = 2", color="red", fontsize=ANNOT_SIZE, ha="right")
    ax_gap.axhline(0, color="gray", ls="-", alpha=0.3)

    # Highlight 9-10 gap
    if 9 in modes and 10 in modes:
        i9 = modes.index(9)
        i10 = modes.index(10)
        mid = (x_pos[i9] + x_pos[i10]) / 2
        ax_gap.axvspan(mid - 0.5, mid + 0.5, color="yellow", alpha=0.15)

    ax_gap.set_ylabel(r"Gap $Z$-score")
    panel_label(ax_gap, "C. Gap Z-score")
    ax_gap.set_xticks(x_pos[:-1])
    ax_gap.set_xticklabels([f"{m}–{m + 1}" for m in modes[:-1]], rotation=30, ha="right", fontsize=ANNOT_SIZE)
    ax_gap.set_xlabel("Mode Gap")
    ax_gap.legend(frameon=False, fontsize=ANNOT_SIZE, ncol=2, loc="upper left")
    ax_gap.grid(axis="y", ls="--", alpha=0.4)

    # ── Panel D: Ligament internal (global / type / asym) ────────
    mean_lig = (
        df.groupby("mode")[["lig_frac_global", "lig_frac_type", "lig_frac_asym"]]
        .mean()
        .reindex(modes)
    )
    fg = mean_lig["lig_frac_global"].values * 100
    ft = mean_lig["lig_frac_type"].values * 100
    fa = mean_lig["lig_frac_asym"].values * 100

    ax_lig.bar(x_pos, fg, color=COLOR_LIG_GLOBAL, label="Global Laxity", width=0.7)
    ax_lig.bar(x_pos, ft, bottom=fg, color=COLOR_LIG_TYPE, label="Ligament Type", width=0.7)
    ax_lig.bar(x_pos, fa, bottom=fg + ft, color=COLOR_LIG_ASYM, label="L/R Asym.", width=0.7)
    ax_lig.set_ylabel("Variance (%)")
    panel_label(ax_lig, "D. Ligament Split")
    ax_lig.set_ylim(0, 100)
    ax_lig.set_xticks(x_pos)
    ax_lig.set_xticklabels(labels, rotation=30, ha="right")
    ax_lig.legend(frameon=False, fontsize=ANNOT_SIZE, ncol=3, loc="upper right")
    ax_lig.grid(axis="y", ls="--", alpha=0.4)

    # ── Panel E: Cartilage per-region ────────────────────────────
    cart_cols = ["rel_imp_SIJCartilageLeft", "rel_imp_SIJCartilageRight", "rel_imp_PubicSymphysis"]
    cart_data = summary.groupby("mode")[[c for c in cart_cols if c in summary.columns]].mean().reindex(modes)
    # Stack: all-population mean
    bottom = np.zeros(n_modes)
    for col, color, label in [
        ("rel_imp_SIJCartilageLeft", COLOR_SIJ_L, "SIJ Left"),
        ("rel_imp_SIJCartilageRight", COLOR_SIJ_R, "SIJ Right"),
        ("rel_imp_PubicSymphysis", COLOR_SYM, "Symphysis"),
    ]:
        if col not in cart_data.columns:
            continue
        vals = cart_data[col].fillna(0).values * 100
        ax_cart.bar(x_pos, vals, bottom=bottom, color=color, label=label, width=0.7)
        bottom += vals

    ax_cart.set_ylabel("Rel. Importance (%)")
    panel_label(ax_cart, "E. Cartilage Split")
    ax_cart.set_ylim(0, max(bottom.max() * 1.1, 10))
    ax_cart.set_xticks(x_pos)
    ax_cart.set_xticklabels(labels, rotation=30, ha="right")
    ax_cart.legend(frameon=False, fontsize=ANNOT_SIZE, ncol=3, loc="upper right")
    ax_cart.grid(axis="y", ls="--", alpha=0.4)

    # ── Panel F: Bone α vs β ─────────────────────────────────────
    mean_bone = (
        df.groupby("mode")[["bone_frac_alpha", "bone_frac_beta"]]
        .mean()
        .reindex(modes)
    )
    ba = mean_bone["bone_frac_alpha"].fillna(0).values * 100
    bb = mean_bone["bone_frac_beta"].fillna(0).values * 100

    ax_bone.bar(x_pos, ba, color=COLOR_ALPHA, label=r"$\alpha$ (scale)", width=0.7)
    ax_bone.bar(x_pos, bb, bottom=ba, color=COLOR_BETA, label=r"$\beta$ (exponent)", width=0.7)
    ax_bone.set_ylabel("Variance (%)")
    panel_label(ax_bone, r"F. Bone: $\alpha$ vs $\beta$")
    ax_bone.set_ylim(0, 100)
    ax_bone.set_xticks(x_pos)
    ax_bone.set_xticklabels(labels, rotation=30, ha="right")
    ax_bone.legend(frameon=False, fontsize=ANNOT_SIZE)
    ax_bone.grid(axis="y", ls="--", alpha=0.4)

    # ── Panel G: Top-5 parameters (combined) ─────────────────────
    # Compute population-mean marginal fraction across all params
    marg_cols = [c for c in summary.columns if c.startswith("rel_imp_")]
    mean_imp = summary.groupby("mode")[marg_cols].mean().mean()  # mode-average then param
    top5 = mean_imp.nlargest(5)

    # Pretty labels
    PRETTY = {
        "rel_imp_symphisys": "Symphysis lig.",
        "rel_imp_right_posterior_SIJ_ligaments": "R post SIJ",
        "rel_imp_left_posterior_SIJ_ligaments": "L post SIJ",
        "rel_imp_right_SS": "R SS",
        "rel_imp_left_SS": "L SS",
        "rel_imp_right_ST": "R ST",
        "rel_imp_left_ST": "L ST",
        "rel_imp_right_anterior_SIJ_ligaments": "R ant SIJ",
        "rel_imp_left_anterior_SIJ_ligaments": "L ant SIJ",
        "rel_imp_right_INL": "R Interosseous",
        "rel_imp_left_INL": "L Interosseous",
        "rel_imp_SIJCartilageLeft": "SIJ Cart L",
        "rel_imp_SIJCartilageRight": "SIJ Cart R",
        "rel_imp_PubicSymphysis": "Pub Symph Cart",
        "rel_imp_bone_alpha": r"Bone $\alpha$",
        "rel_imp_bone_beta": r"Bone $\beta$",
    }

    bar_labels = [PRETTY.get(c, c.replace("rel_imp_", "")) for c in top5.index]
    bar_vals = top5.values * 100
    bar_colors = []
    for c in top5.index:
        if "bone" in c:
            bar_colors.append(COLOR_BONE)
        elif "Cartilage" in c or "Symphysis" in c.replace("rel_imp_", ""):
            # distinguish from lig symphysis
            if "SIJ" in c or "Pub" in c:
                bar_colors.append(COLOR_CART)
            else:
                bar_colors.append(COLOR_LIG)
        else:
            bar_colors.append(COLOR_LIG)

    y_idx = np.arange(len(top5))
    ax_imp.barh(y_idx, bar_vals, color=bar_colors, height=0.6)
    ax_imp.set_yticks(y_idx)
    ax_imp.set_yticklabels(bar_labels, fontsize=ANNOT_SIZE)
    ax_imp.set_xlabel("Mean Rel. Importance (%)")
    panel_label(ax_imp, "G. Top Parameters")
    ax_imp.invert_yaxis()
    ax_imp.grid(axis="x", ls="--", alpha=0.4)

    # ── Save ─────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.06, top=0.97)
    fig.set_constrained_layout(False)
    fig.set_layout_engine("none")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches=None)
    fig.set_constrained_layout(False)
    fig.set_layout_engine("none")
    fig.savefig(output_path.with_suffix(".png"), dpi=300, bbox_inches=None)
    plt.close(fig)
    print(f"Dashboard → {output_path}")


def main() -> None:
    df, summary = load_data()
    plot_dashboard(df, summary, OUTPUT_DIR / "material_uncertainty_dashboard")


if __name__ == "__main__":
    main()
