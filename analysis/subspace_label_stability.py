#!/usr/bin/env python3
"""Quantify label instability vs subspace stability in the 9-10 cluster.

This analysis makes the manuscript's central conceptual point explicit:
for the near-degenerate 9-10 cluster, AP-inlet coupling flips between the
two rank labels when subjects swap, whereas the 2D subspace-level AP
coupling remains largely preserved.

Outputs
-------
- analysis_outputs/figures/subspace_label_stability.pdf
- analysis_outputs/tables/subspace_label_stability.csv
- analysis_outputs/tables/subspace_label_stability.tex
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.spatial import cKDTree

from analysis.spectral_config import DATA_DIR_FULL, EPS_IN_PERCENTILE, FIG_DIR, TAB_DIR
from analysis.spectral_data import (
    load_eigenvalues,
    load_eigenvectors_obj,
    load_fe_mesh_coords,
    load_inlet_landmarks,
    load_mass_matrix,
    load_permutations,
    load_template_and_shapes,
)
from analysis.spectral_metrics import (
    build_patient_measurement_vectors,
    canonical_pair_couplings,
    compute_gap_in,
    coupling_per_1mm_max,
    detect_clusters,
    invert_pairing_permutation,
    orthonormalize_l2,
)
from utils.plot_utils import (
    ANNOT_SIZE,
    FEMALE_COLOR,
    LINE_WIDTH,
    MALE_COLOR,
    NEUTRAL_COLOR,
    ROW_H,
    SCATTER_ALPHA,
    setup_plot_style,
    panel_label,
)


CLUSTER = (8, 10)  # 0-based -> rank modes 9-10


def _rank_biserial(u_stat: float, n_a: int, n_b: int) -> float:
    """Rank-biserial effect size with r > 0 when group B tends larger."""
    return float(1.0 - (2.0 * float(u_stat)) / (int(n_a) * int(n_b)))


def _fmt_p(p_value: float) -> str:
    if not np.isfinite(p_value):
        return "p=NA"
    if p_value < 1e-4:
        return f"p={p_value:.1e}"
    return f"p={p_value:.4f}"


def _write_tex_table(df: pd.DataFrame, out_path: Path) -> None:
    lines = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Metric & $n_0$ & $n_1$ & Median$_0$ & Median$_1$ & $r_{rb}$ \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        lines.append(
            f"{row['metric_label']} & "
            f"{int(row['n_no_swap'])} & {int(row['n_swap'])} & "
            f"{row['median_no_swap']:.3f} & {row['median_swap']:.3f} & "
            f"{row['rank_biserial_r']:.3f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    out_path.write_text("\n".join(lines) + "\n")


def build_analysis_frame() -> tuple[pd.DataFrame, pd.DataFrame]:
    eigenvalues = load_eigenvalues(DATA_DIR_FULL)
    gaps = compute_gap_in(eigenvalues)
    eps_in = float(np.nanpercentile(gaps.ravel(), EPS_IN_PERCENTILE))
    clusters_all = detect_clusters(eigenvalues, eps_in)
    member_mask = np.array([(CLUSTER in clist) for clist in clusters_all], dtype=bool)

    permutations = load_permutations(DATA_DIR_FULL)
    perm_rank_to_ref = invert_pairing_permutation(permutations)
    swap_mask = (perm_rank_to_ref[:, 8] == 9) & (perm_rank_to_ref[:, 9] == 8)

    mesh = load_fe_mesh_coords()
    if mesh is None:
        raise RuntimeError("FE mesh coordinates are required for label-stability analysis.")
    landmarks = load_inlet_landmarks()
    if "AP" not in landmarks or "ML" not in landmarks:
        raise RuntimeError("Inlet AP/ML landmarks are required for label-stability analysis.")

    template_points, shapes = load_template_and_shapes()  # type: ignore[misc]
    tree = cKDTree(mesh)
    n_nodes = mesh.shape[0]
    M0 = load_mass_matrix()

    b_ap_all = build_patient_measurement_vectors(
        tree,
        n_nodes,
        template_points,
        shapes,
        landmarks["AP"][0],
        landmarks["AP"][1],
    )
    b_ml_all = build_patient_measurement_vectors(
        tree,
        n_nodes,
        template_points,
        shapes,
        landmarks["ML"][0],
        landmarks["ML"][1],
    )

    sex = np.load(Path(__file__).resolve().parents[1] / "data" / "sex.npy")
    evecs = load_eigenvectors_obj(DATA_DIR_FULL)

    rows: list[dict[str, object]] = []
    for subject_idx in np.flatnonzero(member_mask):
        block = np.asarray(evecs[subject_idx, 8:10, :, :], dtype=float).reshape(2, -1).T
        Q = orthonormalize_l2(block, M=M0)
        b_ap = b_ap_all[subject_idx]
        b_ml = b_ml_all[subject_idx]
        subspace_ap, _, *_ = canonical_pair_couplings(Q, b_ap, b_ml, M=M0)

        rows.append(
            {
                "subject_idx": int(subject_idx),
                "sex": str(sex[subject_idx]),
                "swap": int(swap_mask[subject_idx]),
                "rank_mode_9_ap": coupling_per_1mm_max(b_ap, block[:, 0]),
                "rank_mode_10_ap": coupling_per_1mm_max(b_ap, block[:, 1]),
                "subspace_ap": subspace_ap,
            }
        )

    subject_df = pd.DataFrame(rows)

    summary_rows: list[dict[str, object]] = []
    metric_map = {
        "rank_mode_9_ap": "Rank mode 9 AP coupling",
        "rank_mode_10_ap": "Rank mode 10 AP coupling",
        "subspace_ap": "Subspace AP coupling",
    }
    for metric_key, metric_label in metric_map.items():
        no_swap = subject_df.loc[subject_df["swap"] == 0, metric_key].to_numpy(dtype=float)
        swap = subject_df.loc[subject_df["swap"] == 1, metric_key].to_numpy(dtype=float)
        u_stat, p_value = stats.mannwhitneyu(no_swap, swap, alternative="two-sided")
        summary_rows.append(
            {
                "metric_key": metric_key,
                "metric_label": metric_label,
                "n_no_swap": int(no_swap.size),
                "n_swap": int(swap.size),
                "median_no_swap": float(np.nanmedian(no_swap)),
                "median_swap": float(np.nanmedian(swap)),
                "u_statistic": float(u_stat),
                "p_value": float(p_value),
                "rank_biserial_r": _rank_biserial(u_stat, no_swap.size, swap.size),
            }
        )

    return subject_df, pd.DataFrame(summary_rows)


def plot_figure(subject_df: pd.DataFrame, summary_df: pd.DataFrame, out_path: Path) -> None:
    setup_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 1.7 * ROW_H), constrained_layout=True)

    metric_map = [
        ("rank_mode_9_ap", "A  Rank mode 9"),
        ("rank_mode_10_ap", "B  Rank mode 10"),
        ("subspace_ap", "C  9-10 subspace"),
    ]
    sex_colors = {"M": MALE_COLOR, "F": FEMALE_COLOR}
    x_positions = np.array([0.0, 1.0])

    for ax, (metric_key, title) in zip(axes, metric_map):
        no_swap = subject_df.loc[subject_df["swap"] == 0, metric_key].to_numpy(dtype=float)
        swap = subject_df.loc[subject_df["swap"] == 1, metric_key].to_numpy(dtype=float)
        bp = ax.boxplot(
            [no_swap, swap],
            positions=x_positions,
            widths=0.55,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "white", "linewidth": LINE_WIDTH},
            boxprops={"linewidth": LINE_WIDTH},
            whiskerprops={"linewidth": LINE_WIDTH},
            capprops={"linewidth": LINE_WIDTH},
        )
        for patch, face in zip(bp["boxes"], [NEUTRAL_COLOR, NEUTRAL_COLOR]):
            patch.set_facecolor(face)
            patch.set_alpha(0.45)
            patch.set_edgecolor("black")

        for j, group_value in enumerate([0, 1]):
            group_df = subject_df.loc[subject_df["swap"] == group_value, ["sex", metric_key]]
            rng = np.random.default_rng(100 + j)
            jitter = rng.uniform(-0.10, 0.10, size=len(group_df))
            for offset, (_, row) in zip(jitter, group_df.iterrows()):
                ax.scatter(
                    x_positions[j] + offset,
                    float(row[metric_key]),
                    s=12,
                    color=sex_colors.get(str(row["sex"]), "0.2"),
                    alpha=SCATTER_ALPHA,
                    linewidths=0.0,
                    zorder=3,
                )

        stat_row = summary_df.loc[summary_df["metric_key"] == metric_key].iloc[0]
        y_max = max(np.nanmax(no_swap), np.nanmax(swap))
        y_min = min(np.nanmin(no_swap), np.nanmin(swap))
        y_span = max(1e-6, y_max - y_min)
        ax.text(
            0.5,
            y_max + 0.08 * y_span,
            f"{_fmt_p(float(stat_row['p_value']))}\n"
            f"med: {float(stat_row['median_no_swap']):.3f} vs {float(stat_row['median_swap']):.3f}",
            ha="center",
            va="bottom",
            fontsize=ANNOT_SIZE,
        )
        ax.set_ylim(y_min - 0.10 * y_span, y_max + 0.24 * y_span)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(["swap=0", "swap=1"])
        ax.set_ylabel("AP coupling (mm/mm)")
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        panel_label(ax, title)

    handles = [
        plt.Line2D([], [], linestyle="", marker="o", color=MALE_COLOR, label="Male"),
        plt.Line2D([], [], linestyle="", marker="o", color=FEMALE_COLOR, label="Female"),
    ]
    axes[-1].legend(handles=handles, frameon=False, loc="lower right", fontsize=ANNOT_SIZE)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)

    subject_df, summary_df = build_analysis_frame()
    summary_path = TAB_DIR / "subspace_label_stability.csv"
    tex_path = TAB_DIR / "subspace_label_stability.tex"
    fig_path = FIG_DIR / "subspace_label_stability.pdf"

    summary_df.to_csv(summary_path, index=False)
    _write_tex_table(summary_df, tex_path)
    plot_figure(subject_df, summary_df, fig_path)

    print(f"Wrote {fig_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {tex_path}")


if __name__ == "__main__":
    main()
