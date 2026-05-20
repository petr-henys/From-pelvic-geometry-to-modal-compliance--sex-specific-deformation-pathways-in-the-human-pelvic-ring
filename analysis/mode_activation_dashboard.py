#!/usr/bin/env python3
"""Mode activation analysis across standing and LAB load cases.

Purpose
-------
Quantify which eigenmodes are functionally activated by each tested load case,
which mode sets are shared across loads, and where routing differs.

Outputs
-------
- analysis_outputs/figures/mode_activation_dashboard.pdf
- analysis_outputs/tables/mode_activation_dominant_sets.{csv,tex}
- analysis_outputs/tables/mode_activation_overlap.{csv,tex}
- analysis_outputs/tables/mode_activation_family_shares.{csv,tex}
- analysis_outputs/mode_activation/mode_activation_summary.csv
- analysis_outputs/mode_activation/mode_activation_overlap_matrix.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from analysis.spectral_config import (
    COL2_WIDTH,
    DATA_DIR_FULL,
    FIG_DPI,
    FIG_DIR,
    OUT_DIR,
    ROW_H,
)
from analysis.spectral_data import _open_zarr_array
from analysis.spectral_plots import _save_fig, _save_table
from utils.plot_utils import (
    ANNOT_SIZE,
    MATERIAL_EFFECT_COLOR,
    NEUTRAL_COLOR,
    SHAPE_EFFECT_COLOR,
    SMALL_ANNOT_SIZE,
    SWAP_LINE_COLOR,
    panel_label,
    setup_plot_style,
)

LOAD_CASES = ["SP2leg", "SP1leg", "LAB_phase1", "LAB_phase2", "LAB_phase3"]
LOAD_LABELS = ["SP2leg", "SP1leg", "LAB$_1$", "LAB$_2$", "LAB$_3$"]
MODE_COUNT = 15
TOP_K = 3
DOMINANT_THRESHOLD = 0.80
N_BOOT = 1000
RNG_SEED = 42

MODE_HINT = {
    1: "CC rotation",
    2: "ML rotation",
    3: "AP rotation",
    9: "inlet opening",
    10: "asymmetric inlet/pubis",
}

MODE_FAMILIES: list[tuple[str, list[int], str]] = [
    ("Rotational backbone (1--3)", [1, 2, 3], NEUTRAL_COLOR),
    ("Mid-rank mixed (4--8)", [4, 5, 6, 7, 8], SHAPE_EFFECT_COLOR),
    ("Inlet routing pair (9--10)", [9, 10], SWAP_LINE_COLOR),
    ("Higher-rank mixed (11--15)", [11, 12, 13, 14, 15], MATERIAL_EFFECT_COLOR),
]


def _mode_tag(mode: int) -> str:
    """Readable mode tag with optional atlas hint."""
    if mode in MODE_HINT:
        return f"{mode} ({MODE_HINT[mode]})"
    return f"{mode}"


def _mode_list_str(modes: list[int]) -> str:
    return ",".join(str(m) for m in modes)


def _bootstrap_mean_ci(values: np.ndarray, boot_idx: np.ndarray) -> tuple[float, float]:
    """Percentile bootstrap 95 % CI for the mean."""
    boot_means = values[boot_idx].mean(axis=1)
    return float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))


def _compute_mode_summary(frac: np.ndarray) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Per-mode/load statistics, including top-k prevalence and CI."""
    n_subj, n_modes, n_load = frac.shape
    rng = np.random.default_rng(RNG_SEED)
    boot_idx = rng.integers(0, n_subj, size=(N_BOOT, n_subj))

    mean_ci_lo = np.full((n_modes, n_load), np.nan)
    mean_ci_hi = np.full((n_modes, n_load), np.nan)
    for li in range(n_load):
        # shape: (N_BOOT, N_subj, N_modes) -> (N_BOOT, N_modes)
        boot_means = frac[boot_idx, :, li].mean(axis=1)
        mean_ci_lo[:, li] = np.percentile(boot_means, 2.5, axis=0)
        mean_ci_hi[:, li] = np.percentile(boot_means, 97.5, axis=0)

    top1 = np.argmax(frac, axis=1)  # (N_subj, N_load)

    rows: list[dict] = []
    for li, load in enumerate(LOAD_CASES):
        load_arr = frac[:, :, li]
        topk = np.argpartition(-load_arr, TOP_K - 1, axis=1)[:, :TOP_K]

        for mi in range(n_modes):
            mode_1 = mi + 1
            vals = load_arr[:, mi]
            top3_prev = float((topk == mi).any(axis=1).mean())
            top1_prev = float((top1[:, li] == mi).mean())

            rows.append({
                "load_case": load,
                "mode": mode_1,
                "mode_tag": _mode_tag(mode_1),
                "mean_fraction": float(np.mean(vals)),
                "mean_fraction_ci_lo": float(mean_ci_lo[mi, li]),
                "mean_fraction_ci_hi": float(mean_ci_hi[mi, li]),
                "median_fraction": float(np.median(vals)),
                "q25_fraction": float(np.percentile(vals, 25)),
                "q75_fraction": float(np.percentile(vals, 75)),
                "top1_prevalence": top1_prev,
                "top3_prevalence": top3_prev,
            })

    summary = pd.DataFrame(rows)
    return summary, mean_ci_lo, mean_ci_hi


def _dominant_sets(mean_profile: np.ndarray) -> tuple[pd.DataFrame, dict[str, set[int]]]:
    """Dominant-mode sets per load: minimal modes reaching 80 % of mean energy."""
    rows: list[dict] = []
    sets: dict[str, set[int]] = {}

    for li, load in enumerate(LOAD_CASES):
        w = mean_profile[:, li]
        order = np.argsort(w)[::-1]
        cum = np.cumsum(w[order])
        k80 = int(np.searchsorted(cum, DOMINANT_THRESHOLD) + 1)
        dom_modes = (order[:k80] + 1).astype(int).tolist()
        top3_modes = (order[:3] + 1).astype(int).tolist()

        sets[load] = set(dom_modes)

        family_shares = []
        for fam_name, fam_modes, _ in MODE_FAMILIES:
            idx = np.array(fam_modes, dtype=int) - 1
            family_shares.append((fam_name, float(w[idx].sum())))
        family_shares.sort(key=lambda x: x[1], reverse=True)

        top_mode = int(order[0] + 1)
        rows.append({
            "load_case": load,
            "load_label_tex": LOAD_LABELS[li],
            "top_mode": top_mode,
            "top_mode_tag": _mode_tag(top_mode),
            "top_mode_share_pct": float(100.0 * w[order[0]]),
            "top3_modes": _mode_list_str(top3_modes),
            "top3_share_pct": float(100.0 * w[order[:3]].sum()),
            "n80": k80,
            "dominant_modes_80": _mode_list_str(dom_modes),
            "dominant_share_80_pct": float(100.0 * cum[k80 - 1]),
            "leading_family": family_shares[0][0],
            "leading_family_share_pct": float(100.0 * family_shares[0][1]),
        })

    return pd.DataFrame(rows), sets


def _pairwise_overlap(mean_profile: np.ndarray, dominant_sets: dict[str, set[int]]) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Pairwise overlap/similarity between load-case activation profiles."""
    n_load = len(LOAD_CASES)
    jacc = np.full((n_load, n_load), np.nan)
    cosine = np.full((n_load, n_load), np.nan)

    rows: list[dict] = []
    for i, a in enumerate(LOAD_CASES):
        va = mean_profile[:, i]
        for j, b in enumerate(LOAD_CASES):
            vb = mean_profile[:, j]

            sa, sb = dominant_sets[a], dominant_sets[b]
            union = sa | sb
            inter = sa & sb
            jac = float(len(inter) / len(union)) if union else np.nan
            jacc[i, j] = jac

            den = float(np.linalg.norm(va) * np.linalg.norm(vb))
            cos = float(np.dot(va, vb) / den) if den > 0 else np.nan
            cosine[i, j] = cos

            rows.append({
                "load_a": a,
                "load_b": b,
                "jaccard_80": jac,
                "cosine_full_profile": cos,
                "shared_modes_80": _mode_list_str(sorted(inter)),
                "shared_mode_count_80": int(len(inter)),
            })

    return pd.DataFrame(rows), jacc, cosine


def _family_shares(frac: np.ndarray) -> pd.DataFrame:
    """Family-level shares per load with bootstrap CIs."""
    n_subj = frac.shape[0]
    rng = np.random.default_rng(RNG_SEED)
    boot_idx = rng.integers(0, n_subj, size=(N_BOOT, n_subj))

    rows: list[dict] = []
    for li, load in enumerate(LOAD_CASES):
        for fam_name, fam_modes, _ in MODE_FAMILIES:
            idx = np.array(fam_modes, dtype=int) - 1
            vals = frac[:, idx, li].sum(axis=1)
            ci_lo, ci_hi = _bootstrap_mean_ci(vals, boot_idx)
            rows.append({
                "load_case": load,
                "family": fam_name,
                "mean_share": float(np.mean(vals)),
                "mean_share_ci_lo": ci_lo,
                "mean_share_ci_hi": ci_hi,
            })
    return pd.DataFrame(rows)


def _plot_dashboard(
    mean_profile: np.ndarray,
    dominant_df: pd.DataFrame,
    jaccard: np.ndarray,
    family_df: pd.DataFrame,
) -> None:
    """Create 2x2 dashboard focused on activation overlap and specificity."""
    setup_plot_style(dpi=FIG_DPI)

    fig, axes = plt.subplots(2, 2, figsize=(COL2_WIDTH, 2.35 * ROW_H), layout="constrained")

    # Panel A: mean fractions heatmap with dominant-set markers
    ax = axes[0, 0]
    im = ax.imshow(mean_profile, aspect="auto", cmap="magma", vmin=0.0, vmax=float(np.max(mean_profile)))
    ax.set_xticks(np.arange(len(LOAD_CASES)))
    ax.set_xticklabels(LOAD_LABELS, fontsize=ANNOT_SIZE)
    ax.set_yticks(np.arange(MODE_COUNT))
    ax.set_yticklabels([str(i) for i in range(1, MODE_COUNT + 1)], fontsize=ANNOT_SIZE)
    ax.set_xlabel("Load case")
    ax.set_ylabel("Mode")

    dom_map = {row["load_case"]: row["dominant_modes_80"] for _, row in dominant_df.iterrows()}
    for li, load in enumerate(LOAD_CASES):
        dom_modes = [int(x) for x in str(dom_map[load]).split(",") if x]
        for m in dom_modes:
            ax.plot(li, m - 1, marker="s", markersize=4.8, markerfacecolor="none", markeredgecolor="white", lw=0.8)

    ax.text(
        1.02,
        0.98,
        "Key mode character:\n"
        "1: CC rotation\n"
        "2: ML rotation\n"
        "3: AP rotation\n"
        "9: inlet opening\n"
        "10: asymmetric inlet/pubis",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=SMALL_ANNOT_SIZE,
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("Mean energy fraction")
    panel_label(ax, "(A) Mean mode activation by load")

    # Panel B: cumulative concentration curves (sorted per load)
    ax = axes[0, 1]
    load_colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B3", "#CCB974"]
    x = np.arange(1, MODE_COUNT + 1)
    for li, load in enumerate(LOAD_CASES):
        w = mean_profile[:, li]
        w_sorted = np.sort(w)[::-1]
        cum = np.cumsum(w_sorted)
        k80 = int(np.searchsorted(cum, DOMINANT_THRESHOLD) + 1)
        ax.plot(x, cum, color=load_colors[li], label=LOAD_LABELS[li], lw=1.2)
        ax.scatter([k80], [cum[k80 - 1]], color=load_colors[li], s=20, zorder=3)

    ax.axhline(DOMINANT_THRESHOLD, color="0.3", ls="--", lw=0.8)
    ax.set_xlim(1, MODE_COUNT)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Mode rank in load-specific ordering")
    ax.set_ylabel("Cumulative mean energy share")
    ax.legend(loc="lower right", fontsize=SMALL_ANNOT_SIZE, frameon=True)
    panel_label(ax, "(B) Concentration of routing")

    # Panel C: Jaccard overlap of dominant (80 %) sets
    ax = axes[1, 0]
    hm = ax.imshow(jaccard, vmin=0.0, vmax=1.0, cmap="Blues")
    ax.set_xticks(np.arange(len(LOAD_CASES)))
    ax.set_xticklabels(LOAD_LABELS, rotation=35, ha="right", fontsize=ANNOT_SIZE)
    ax.set_yticks(np.arange(len(LOAD_CASES)))
    ax.set_yticklabels(LOAD_LABELS, fontsize=ANNOT_SIZE)
    for i in range(len(LOAD_CASES)):
        for j in range(len(LOAD_CASES)):
            val = jaccard[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=SMALL_ANNOT_SIZE)
    cbar = fig.colorbar(hm, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("Jaccard overlap (dominant 80% sets)")
    panel_label(ax, "(C) Shared dominant modes across loads")

    # Panel D: family shares with bootstrap CI
    ax = axes[1, 1]
    x = np.arange(len(LOAD_CASES))
    width = 0.19
    for fi, (fam_name, _, color) in enumerate(MODE_FAMILIES):
        sub = family_df[family_df["family"] == fam_name].set_index("load_case").reindex(LOAD_CASES)
        y = sub["mean_share"].to_numpy(dtype=float)
        lo = sub["mean_share_ci_lo"].to_numpy(dtype=float)
        hi = sub["mean_share_ci_hi"].to_numpy(dtype=float)
        xpos = x + (fi - 1.5) * width
        ax.bar(xpos, y, width=width, color=color, label=fam_name)
        ax.errorbar(xpos, y, yerr=np.vstack([y - lo, hi - y]), fmt="none", ecolor="0.2", lw=0.7, capsize=1.8)

    ax.set_xticks(x)
    ax.set_xticklabels(LOAD_LABELS, rotation=20, ha="right", fontsize=ANNOT_SIZE)
    ax.set_ylim(0.0, 1.02)
    ax.set_ylabel("Mean family energy share")
    ax.legend(loc="upper right", fontsize=SMALL_ANNOT_SIZE, frameon=True)
    panel_label(ax, "(D) Mechanical family decomposition")

    _save_fig(fig, "mode_activation_dashboard")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "mode_activation").mkdir(parents=True, exist_ok=True)

    frac = _open_zarr_array(DATA_DIR_FULL / "mode_energy_fraction.zarr")
    if frac.ndim != 3:
        raise ValueError(f"Expected (N_subjects, N_modes, N_loadcases), got {frac.shape}")
    if frac.shape[1] < MODE_COUNT or frac.shape[2] < len(LOAD_CASES):
        raise ValueError(
            f"Data shape {frac.shape} incompatible with expected modes/loads "
            f"({MODE_COUNT} modes, {len(LOAD_CASES)} loads)."
        )

    frac = np.asarray(frac[:, :MODE_COUNT, : len(LOAD_CASES)], dtype=float)
    mean_profile = frac.mean(axis=0)  # (modes, loads)

    summary_df, _, _ = _compute_mode_summary(frac)
    dominant_df, dominant_sets = _dominant_sets(mean_profile)
    overlap_df, jaccard, cosine = _pairwise_overlap(mean_profile, dominant_sets)
    family_df = _family_shares(frac)

    # Persist machine-readable outputs
    out_dir = OUT_DIR / "mode_activation"
    summary_df.to_csv(out_dir / "mode_activation_summary.csv", index=False)
    overlap_df.to_csv(out_dir / "mode_activation_overlap_long.csv", index=False)

    overlap_matrix_rows = []
    for i, la in enumerate(LOAD_CASES):
        for j, lb in enumerate(LOAD_CASES):
            overlap_matrix_rows.append({
                "load_a": la,
                "load_b": lb,
                "jaccard_80": float(jaccard[i, j]),
                "cosine_full_profile": float(cosine[i, j]),
            })
    pd.DataFrame(overlap_matrix_rows).to_csv(out_dir / "mode_activation_overlap_matrix.csv", index=False)

    # Manuscript-facing tables
    dom_tex = dominant_df[[
        "load_label_tex",
        "top_mode_tag",
        "top_mode_share_pct",
        "top3_modes",
        "top3_share_pct",
        "n80",
        "dominant_modes_80",
        "dominant_share_80_pct",
    ]].copy()
    dom_tex = dom_tex.rename(columns={
        "load_label_tex": "Load",
        "top_mode_tag": "Top mode",
        "top_mode_share_pct": r"Top mode share (\%)",
        "top3_modes": "Top-3 modes",
        "top3_share_pct": r"Top-3 share (\%)",
        "n80": "$N_{80}$",
        "dominant_modes_80": r"Modes in 80\% set",
        "dominant_share_80_pct": r"Cumulative share at $N_{80}$ (\%)",
    })
    _save_table(dom_tex, "mode_activation_dominant_sets")

    pair_rows = []
    for i in range(len(LOAD_CASES)):
        for j in range(i + 1, len(LOAD_CASES)):
            la = LOAD_CASES[i]
            lb = LOAD_CASES[j]
            sub = overlap_df[(overlap_df["load_a"] == la) & (overlap_df["load_b"] == lb)].iloc[0]
            pair_rows.append({
                "Load A": LOAD_LABELS[i],
                "Load B": LOAD_LABELS[j],
                r"Jaccard overlap (80\% sets)": float(sub["jaccard_80"]),
                "Cosine similarity (full profile)": float(sub["cosine_full_profile"]),
                r"Shared modes (80\% sets)": str(sub["shared_modes_80"]),
            })
    overlap_pairs_df = pd.DataFrame(pair_rows)
    _save_table(overlap_pairs_df, "mode_activation_overlap")

    fam_tex = family_df.copy()
    fam_tex["Load"] = fam_tex["load_case"].map(dict(zip(LOAD_CASES, LOAD_LABELS)))
    fam_tex = fam_tex[["Load", "family", "mean_share", "mean_share_ci_lo", "mean_share_ci_hi"]]
    fam_tex = fam_tex.rename(columns={
        "family": "Mode family",
        "mean_share": "Mean share",
        "mean_share_ci_lo": r"95\% CI low",
        "mean_share_ci_hi": r"95\% CI high",
    })
    _save_table(fam_tex, "mode_activation_family_shares")

    _plot_dashboard(mean_profile, dominant_df, jaccard, family_df)

    print("Saved figure: analysis_outputs/figures/mode_activation_dashboard.pdf")
    print("Saved tables: analysis_outputs/tables/mode_activation_*.{csv,tex}")
    print("Saved CSV summaries: analysis_outputs/mode_activation/*.csv")


if __name__ == "__main__":
    main()
