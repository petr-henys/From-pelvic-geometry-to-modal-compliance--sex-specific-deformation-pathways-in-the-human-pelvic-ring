"""Generate publication figures for mode deformation mechanisms.

Produces:
1. Fig 1 Companion: Stacked bar chart of 6 Level A tensor fractions (normal ML/AP/CC,
   shear ML-AP/ML-CC/AP-CC) across all 15 modes for the whole pelvis.
2. Regional tissue distribution: Strain norm distribution across bone bodies and cartilages.
3. Clustered subspace invariance (Modes 9-10).
4. Beam mechanism decomposition on pubic rami.
"""
from __future__ import annotations

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from analysis.spectral_config import OUT_DIR

OUTPUT_DIR = OUT_DIR / "mode_mechanisms"


def plot_tensor_fractions_bar():
    """Plot stacked bar chart of Level A tensor components across all 15 modes."""
    summary_path = OUTPUT_DIR / "mode_mechanisms_summary_solver.csv"
    if not summary_path.exists():
        print(f"Summary file {summary_path} does not exist yet.")
        return

    df = pd.read_csv(summary_path)
    wp = df[df["region"] == "WholePelvis"].sort_values("mode_solver")

    modes = wp["mode_solver"].values
    f_ML = wp["f_ML_mean"].values * 100
    f_AP = wp["f_AP_mean"].values * 100
    f_CC = wp["f_CC_mean"].values * 100
    f_ML_AP = wp["f_ML_AP_mean"].values * 100
    f_ML_CC = wp["f_ML_CC_mean"].values * 100
    f_AP_CC = wp["f_AP_CC_mean"].values * 100

    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)

    # Color palette: Normals (blues/cyans), Shears (oranges/reds/purples)
    c_ML = "#1f77b4"      # Deep blue
    c_AP = "#4ba3e3"      # Sky blue
    c_CC = "#9ecae1"      # Light blue
    c_ML_AP = "#ff7f0e"   # Orange
    c_ML_CC = "#e6550d"   # Dark orange / vermillion
    c_AP_CC = "#9467bd"   # Purple

    x = np.arange(len(modes))
    width = 0.65

    # Bottom accumulators
    b0 = np.zeros_like(f_ML)
    b1 = b0 + f_ML
    b2 = b1 + f_AP
    b3 = b2 + f_CC
    b4 = b3 + f_ML_AP
    b5 = b4 + f_ML_CC

    ax.bar(x, f_ML, width, bottom=b0, label=r"$\varepsilon_{\rm ML}$ (Mediolateral normal)", color=c_ML)
    ax.bar(x, f_AP, width, bottom=b1, label=r"$\varepsilon_{\rm AP}$ (Anteroposterior normal)", color=c_AP)
    ax.bar(x, f_CC, width, bottom=b2, label=r"$\varepsilon_{\rm CC}$ (Craniocaudal normal)", color=c_CC)
    ax.bar(x, f_ML_AP, width, bottom=b3, label=r"$\varepsilon_{\rm ML-AP}$ (In-plane transverse shear)", color=c_ML_AP)
    ax.bar(x, f_ML_CC, width, bottom=b4, label=r"$\varepsilon_{\rm ML-CC}$ (Coronal vertical shear)", color=c_ML_CC)
    ax.bar(x, f_AP_CC, width, bottom=b5, label=r"$\varepsilon_{\rm AP-CC}$ (Sagittal shear)", color=c_AP_CC)

    ax.set_xticks(x)
    ax.set_xticklabels([f"Mode {m}" for m in modes], rotation=45, ha="right", fontsize=10)
    ax.set_ylabel("Kinematic Strain Fraction (%)", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 100)
    ax.set_title("Population Kinematic Strain Tensor Profiles (Whole Pelvis, N=278)", fontsize=12, fontweight="bold", pad=12)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True, fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    # Highlight near-degenerate 9-10 pair
    ax.axvspan(7.5, 9.5, color="gray", alpha=0.15, linestyle=":", label="_nolegend_")
    ax.text(8.5, 102, "Modes 9–10\nCluster", ha="center", va="bottom", fontsize=8, fontweight="bold", color="#333333")

    plt.tight_layout()
    out_file = OUTPUT_DIR / "fig1_tensor_profiles_whole_pelvis.png"
    out_pdf = OUTPUT_DIR / "fig1_tensor_profiles_whole_pelvis.pdf"
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_file} and {out_pdf}")


def plot_subspace_invariance():
    """Plot invariant subspace profile for Modes 9-10 cluster."""
    sub_path = OUTPUT_DIR / "subspace_mechanisms_summary.csv"
    if not sub_path.exists():
        return

    df = pd.read_csv(sub_path)
    sub910 = df[(df["subspace"] == "Subspace_9_10") & (df["region"] == "WholePelvis")]
    if len(sub910) == 0:
        return

    comps = ["f_ML", "f_AP", "f_CC", "f_ML_AP", "f_ML_CC", "f_AP_CC"]
    labels = ["ML\n(Normal)", "AP\n(Normal)", "CC\n(Normal)", "ML-AP\n(Shear)", "ML-CC\n(Shear)", "AP-CC\n(Shear)"]
    means = [sub910[f"{c}_mean"].values[0] * 100 for c in comps]
    stds = [sub910[f"{c}_std"].values[0] * 100 for c in comps]

    fig, ax = plt.subplots(figsize=(7, 4), dpi=300)
    colors = ["#1f77b4", "#4ba3e3", "#9ecae1", "#ff7f0e", "#e6550d", "#9467bd"]
    x = np.arange(len(comps))
    bars = ax.bar(x, means, yerr=stds, capsize=4, color=colors, edgecolor="black", linewidth=0.8, alpha=0.85)

    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, m + 1.5, f"{m:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Basis-Invariant Fraction (%)", fontsize=10, fontweight="bold")
    ax.set_ylim(0, max(means) + 12)
    ax.set_title("Invariant Kinematic Profile: Modes 9–10 Subspace (N=278)", fontsize=11, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    out_file = OUTPUT_DIR / "subspace_9_10_invariant_profile.png"
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_file}")


if __name__ == "__main__":
    plot_tensor_fractions_bar()
    plot_subspace_invariance()

