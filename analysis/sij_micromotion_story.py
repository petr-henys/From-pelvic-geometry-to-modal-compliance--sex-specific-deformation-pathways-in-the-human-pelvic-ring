#!/usr/bin/env python3
"""SIJ micromotion subanalysis for manuscript integration.

Builds a compact, reproducible analysis of SIJ kinematics from precomputed
static-load outputs:
1. Load-case micromotion profile (rotations/translations)
2. Swap vs non-swap comparisons at LAB
3. Operating-point / tail correlations at LAB
4. Variance-channel comparison: full vs shape_only vs material_only
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import zarr
from scipy import stats


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from utils.plot_utils import (
    ANNOT_SIZE,
    FULL_WIDTH,
    ROW_H,
    MALE_COLOR,
    FEMALE_COLOR,
    NEUTRAL_COLOR,
    SWAP_LINE_COLOR,
    SHAPE_EFFECT_COLOR,
    MATERIAL_EFFECT_COLOR,
    SEX_EFFECT_COLOR,
    setup_plot_style,
    panel_label,
)

DATA_DIR_FULL = REPO_ROOT / "results" / "ref_S1P_fixed_new2" / "data"
DATA_DIR_SHAPE_ONLY = REPO_ROOT / "results" / "ref_S1P_fixed_new2_shape_only" / "data"
DATA_DIR_MATERIAL_ONLY = REPO_ROOT / "results" / "ref_S1P_fixed_new2_material_only" / "data"
TAIL_METRICS = REPO_ROOT / "analysis_outputs" / "tail_risk" / "metrics_tail_aggregated.csv"
OUT_DIR = REPO_ROOT / "analysis_outputs" / "sij_micromotion"
FIGURE_OUT = REPO_ROOT / "analysis_outputs" / "figures" / "sij_micromotion_dashboard"

LAB_CASES = ["LAB_phase1", "LAB_phase2", "LAB_phase3"]
LOAD_CASE_SOURCE = {
    "SP2leg": "SP2leg",
    "SP1leg": "SP1leg",
    "LAB_phase1": "LAB_phase1",
    "LAB_phase2": "LAB_phase2",
    "LAB_phase3": "LAB_phase3",
}
LOAD_CASES = ["SP2leg", "SP1leg", "LAB_phase1", "LAB_phase2", "LAB_phase3"]
LOAD_CASES_DELTA = ["SP1leg", "LAB_phase1", "LAB_phase2", "LAB_phase3"]

LOAD_LABELS = {
    "SP2leg": "SP2",
    "SP1leg": "SP1",
    "LAB_phase1": "LAB\u2081",
    "LAB_phase2": "LAB\u2082",
    "LAB_phase3": "LAB\u2083",
}


def _load_sij_arrays(data_dir: Path, load_case: str) -> tuple[np.ndarray, np.ndarray]:
    angles = np.asarray(
        zarr.open_group(str(data_dir / f"sij_angles_{load_case}.zarr"), mode="r")["data"][:],
        dtype=float,
    )
    trans = np.asarray(
        zarr.open_group(str(data_dir / f"sij_trans_{load_case}.zarr"), mode="r")["data"][:],
        dtype=float,
    )
    if angles.shape != trans.shape or angles.ndim != 3 or angles.shape[1:] != (2, 3):
        raise ValueError(
            f"Unexpected SIJ array shapes for {load_case}: angles={angles.shape}, trans={trans.shape}"
        )
    return angles, trans


def _build_metric_frame(angles: np.ndarray, trans: np.ndarray) -> pd.DataFrame:
    rot_left = np.linalg.norm(angles[:, 0, :], axis=1)
    rot_right = np.linalg.norm(angles[:, 1, :], axis=1)
    trans_left = np.linalg.norm(trans[:, 0, :], axis=1)
    trans_right = np.linalg.norm(trans[:, 1, :], axis=1)

    return pd.DataFrame(
        {
            "rot_mag_deg": 0.5 * (rot_left + rot_right),
            "trans_mag_mm": 0.5 * (trans_left + trans_right),
            "nut_abs_deg": 0.5 * (np.abs(angles[:, 0, 0]) + np.abs(angles[:, 1, 0])),
            "ap_rot_abs_deg": 0.5 * (np.abs(angles[:, 0, 1]) + np.abs(angles[:, 1, 1])),
            "cc_rot_abs_deg": 0.5 * (np.abs(angles[:, 0, 2]) + np.abs(angles[:, 1, 2])),
            "ml_trans_abs_mm": 0.5 * (np.abs(trans[:, 0, 0]) + np.abs(trans[:, 1, 0])),
            "ap_trans_abs_mm": 0.5 * (np.abs(trans[:, 0, 1]) + np.abs(trans[:, 1, 1])),
            "cc_trans_abs_mm": 0.5 * (np.abs(trans[:, 0, 2]) + np.abs(trans[:, 1, 2])),
            "lr_rot_asym_deg": np.abs(rot_left - rot_right),
            "lr_trans_asym_mm": np.abs(trans_left - trans_right),
        }
    )


def _rank_biserial_from_u(u_stat: float, n_a: int, n_b: int) -> float:
    # Convention: r > 0 when the *second* group (b) tends larger.
    # Matches the rest of the codebase (1 - 2U/(n_a*n_b)) where U is for group a.
    return float(1.0 - (2.0 * u_stat) / (n_a * n_b))


def _load_subject_covariates(n_subjects: int) -> pd.DataFrame:
    sex = np.load(REPO_ROOT / "data" / "sex.npy")
    age = np.load(REPO_ROOT / "data" / "age.npy")
    if len(sex) != n_subjects or len(age) != n_subjects:
        raise ValueError(
            f"sex/age length mismatch with SIJ data: sex={len(sex)}, age={len(age)}, n={n_subjects}"
        )

    tail = pd.read_csv(TAIL_METRICS)
    lab = (
        tail[(tail["group"] == "all") & (tail["category"] == "labour")]
        .sort_values("subject_idx")
        .reset_index(drop=True)
    )
    if len(lab) != n_subjects:
        raise ValueError(f"Expected {n_subjects} labour rows in tail metrics, got {len(lab)}")

    return pd.DataFrame(
        {
            "subject_idx": np.arange(n_subjects, dtype=int),
            "sex": sex,
            "age": age,
            "swap_9_10": lab["swap_9_10"].to_numpy(dtype=float),
            "gap_in_9_10": lab["gap_in_9_10"].to_numpy(dtype=float),
            "dG_9_10": lab["dG_9_10"].to_numpy(dtype=float),
            "P99_SED": lab["P99_SED"].to_numpy(dtype=float),
            "top1pct_mean_SED": lab["top1pct_mean_SED"].to_numpy(dtype=float),
        }
    )


def build_subject_level_table() -> pd.DataFrame:
    angles_ref, trans_ref = _load_sij_arrays(DATA_DIR_FULL, LOAD_CASE_SOURCE[LAB_CASES[0]])
    subject_covariates = _load_subject_covariates(angles_ref.shape[0])

    frames: list[pd.DataFrame] = []
    for load_case in LOAD_CASES:
        angles, trans = _load_sij_arrays(DATA_DIR_FULL, LOAD_CASE_SOURCE[load_case])
        metrics = _build_metric_frame(angles, trans)
        metrics.insert(0, "subject_idx", np.arange(len(metrics), dtype=int))
        metrics.insert(1, "load_case", load_case)
        frame = metrics.merge(subject_covariates, on="subject_idx", how="left")
        frames.append(frame)

    return pd.concat(frames, ignore_index=True)


def summarize_load_profile(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    metrics = [
        "rot_mag_deg",
        "trans_mag_mm",
        "ap_rot_abs_deg",
        "cc_rot_abs_deg",
        "ap_trans_abs_mm",
        "ml_trans_abs_mm",
        "cc_trans_abs_mm",
    ]
    for load_case in LOAD_CASES:
        sub = df[df["load_case"] == load_case]
        row: dict[str, float | str] = {"load_case": load_case, "n": int(len(sub))}
        for metric in metrics:
            vals = sub[metric].to_numpy(dtype=float)
            q10, q50, q90 = np.nanpercentile(vals, [10, 50, 90])
            row[f"{metric}_p10"] = float(q10)
            row[f"{metric}_median"] = float(q50)
            row[f"{metric}_p90"] = float(q90)
        rows.append(row)
    return pd.DataFrame(rows)


def compare_swap_groups_pressure(df: pd.DataFrame, lab_case: str = "LAB_phase1") -> pd.DataFrame:
    sub = df[df["load_case"] == lab_case].copy()
    nonswap = sub["swap_9_10"] < 0.5
    swap = sub["swap_9_10"] >= 0.5
    metrics = [
        "rot_mag_deg",
        "trans_mag_mm",
        "ap_rot_abs_deg",
        "cc_rot_abs_deg",
        "ap_trans_abs_mm",
        "ml_trans_abs_mm",
        "lr_trans_asym_mm",
    ]
    rows: list[dict[str, float | str]] = []
    for metric in metrics:
        a = sub.loc[nonswap, metric].to_numpy(dtype=float)
        b = sub.loc[swap, metric].to_numpy(dtype=float)
        u_stat, p_val = stats.mannwhitneyu(a, b, alternative="two-sided")
        rows.append(
            {
                "metric": metric,
                "median_no_swap": float(np.nanmedian(a)),
                "median_swap": float(np.nanmedian(b)),
                "delta_swap_minus_no_swap": float(np.nanmedian(b) - np.nanmedian(a)),
                "p_value": float(p_val),
                "rank_biserial_r": _rank_biserial_from_u(float(u_stat), len(a), len(b)),
            }
        )
    return pd.DataFrame(rows)


def compare_sex_groups_pressure(df: pd.DataFrame, lab_case: str = "LAB_phase1") -> pd.DataFrame:
    sub = df[df["load_case"] == lab_case].copy()
    male = sub["sex"] == "M"
    female = sub["sex"] == "F"
    metrics = [
        "rot_mag_deg",
        "trans_mag_mm",
        "ap_rot_abs_deg",
        "cc_rot_abs_deg",
        "ap_trans_abs_mm",
        "ml_trans_abs_mm",
    ]
    rows: list[dict[str, float | str]] = []
    for metric in metrics:
        m = sub.loc[male, metric].to_numpy(dtype=float)
        f = sub.loc[female, metric].to_numpy(dtype=float)
        u_stat, p_val = stats.mannwhitneyu(m, f, alternative="two-sided")
        rows.append(
            {
                "metric": metric,
                "median_male": float(np.nanmedian(m)),
                "median_female": float(np.nanmedian(f)),
                "delta_female_minus_male": float(np.nanmedian(f) - np.nanmedian(m)),
                "p_value": float(p_val),
                "rank_biserial_r": _rank_biserial_from_u(float(u_stat), len(m), len(f)),
            }
        )
    return pd.DataFrame(rows)


def paired_load_deltas_vs_sp2(df: pd.DataFrame) -> pd.DataFrame:
    base = (
        df[df["load_case"] == "SP2leg"]
        .sort_values("subject_idx")
        .reset_index(drop=True)
    )
    rows: list[dict[str, float | str]] = []
    metrics = ["rot_mag_deg", "trans_mag_mm", "ap_trans_abs_mm", "ml_trans_abs_mm", "cc_trans_abs_mm"]
    for load_case in LOAD_CASES_DELTA:
        sub = (
            df[df["load_case"] == load_case]
            .sort_values("subject_idx")
            .reset_index(drop=True)
        )
        for metric in metrics:
            delta = sub[metric].to_numpy(dtype=float) - base[metric].to_numpy(dtype=float)
            nz = delta[np.abs(delta) > 1e-12]
            _, p_val = stats.wilcoxon(nz, zero_method="wilcox", alternative="two-sided")
            sign_balance = float((np.sum(delta > 0) - np.sum(delta < 0)) / len(delta))
            rows.append(
                {
                    "load_case": load_case,
                    "metric": metric,
                    "median_delta": float(np.nanmedian(delta)),
                    "p_value": float(p_val),
                    "sign_balance": sign_balance,
                }
            )
    return pd.DataFrame(rows)


def correlate_with_operating_point_pressure(df: pd.DataFrame, lab_case: str = "LAB_phase1") -> pd.DataFrame:
    sub = df[df["load_case"] == lab_case].copy()
    metrics = ["rot_mag_deg", "trans_mag_mm", "ap_trans_abs_mm", "ml_trans_abs_mm", "cc_rot_abs_deg"]
    covariates = ["gap_in_9_10", "dG_9_10", "P99_SED", "top1pct_mean_SED"]
    rows: list[dict[str, float | str]] = []
    for metric in metrics:
        y = sub[metric].to_numpy(dtype=float)
        for cov in covariates:
            x = sub[cov].to_numpy(dtype=float)
            mask = np.isfinite(x) & np.isfinite(y)
            rho, p_val = stats.spearmanr(x[mask], y[mask])
            rows.append(
                {
                    "metric": metric,
                    "covariate": cov,
                    "rho_spearman": float(rho),
                    "p_value": float(p_val),
                    "n": int(mask.sum()),
                }
            )
    return pd.DataFrame(rows)


def adjusted_pressure_models(df: pd.DataFrame, lab_case: str = "LAB_phase1") -> pd.DataFrame:
    sub = df[df["load_case"] == lab_case].copy()
    sub["sex_F"] = (sub["sex"] == "F").astype(float)

    rows: list[dict[str, float | str]] = []
    outcomes = ["rot_mag_deg", "trans_mag_mm", "ml_trans_abs_mm", "ap_trans_abs_mm"]
    for y in outcomes:
        model_df = sub[[y, "swap_9_10", "sex_F", "age", "gap_in_9_10", "dG_9_10", "P99_SED"]].copy()
        for col in ["age", "gap_in_9_10", "dG_9_10"]:
            vals = model_df[col].to_numpy(dtype=float)
            mu = float(np.nanmean(vals))
            sd = float(np.nanstd(vals, ddof=1))
            model_df[col] = (vals - mu) / sd if sd > 0 else 0.0

        m_base = smf.ols(
            f"{y} ~ swap_9_10 + sex_F + age + gap_in_9_10 + dG_9_10",
            data=model_df,
        ).fit(cov_type="HC3")
        m_p99 = smf.ols(
            f"{y} ~ swap_9_10 + sex_F + age + gap_in_9_10 + dG_9_10 + np.log10(P99_SED)",
            data=model_df,
        ).fit(cov_type="HC3")

        rows.append(
            {
                "outcome": y,
                "swap_coef_base": float(m_base.params["swap_9_10"]),
                "swap_p_base": float(m_base.pvalues["swap_9_10"]),
                "sexF_coef_base": float(m_base.params["sex_F"]),
                "sexF_p_base": float(m_base.pvalues["sex_F"]),
                "swap_coef_plus_logP99": float(m_p99.params["swap_9_10"]),
                "swap_p_plus_logP99": float(m_p99.pvalues["swap_9_10"]),
            }
        )
    return pd.DataFrame(rows)


def variance_channel_comparison() -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    data_dirs = {
        "full": DATA_DIR_FULL,
        "shape_only": DATA_DIR_SHAPE_ONLY,
        "material_only": DATA_DIR_MATERIAL_ONLY,
    }
    for load_case in LOAD_CASES:
        metrics_by_channel: dict[str, dict[str, np.ndarray]] = {}
        source_case = LOAD_CASE_SOURCE[load_case]
        for channel, data_dir in data_dirs.items():
            angles, trans = _load_sij_arrays(data_dir, source_case)
            metrics = _build_metric_frame(angles, trans)
            metrics_by_channel[channel] = {
                "rot_mag_deg": metrics["rot_mag_deg"].to_numpy(dtype=float),
                "trans_mag_mm": metrics["trans_mag_mm"].to_numpy(dtype=float),
                "ap_trans_abs_mm": metrics["ap_trans_abs_mm"].to_numpy(dtype=float),
                "cc_rot_abs_deg": metrics["cc_rot_abs_deg"].to_numpy(dtype=float),
            }

        for metric in ["rot_mag_deg", "trans_mag_mm", "ap_trans_abs_mm", "cc_rot_abs_deg"]:
            var_full = float(np.nanvar(metrics_by_channel["full"][metric], ddof=1))
            var_shape = float(np.nanvar(metrics_by_channel["shape_only"][metric], ddof=1))
            var_material = float(np.nanvar(metrics_by_channel["material_only"][metric], ddof=1))
            rows.append(
                {
                    "load_case": load_case,
                    "metric": metric,
                    "var_full": var_full,
                    "var_shape_only": var_shape,
                    "var_material_only": var_material,
                    "shape_over_full_pct": 100.0 * var_shape / var_full if var_full > 0 else np.nan,
                    "material_over_full_pct": 100.0 * var_material / var_full if var_full > 0 else np.nan,
                }
            )
    return pd.DataFrame(rows)


def write_latex_tables(load_summary: pd.DataFrame, swap_tests: dict[str, pd.DataFrame], variance_channels: pd.DataFrame) -> None:
    # Table 1: load profile medians
    load_label = {
        "SP2leg": "SP2leg",
        "SP1leg": "SP1leg",
        "LAB_phase1": r"LAB$_1$",
        "LAB_phase2": r"LAB$_2$",
        "LAB_phase3": r"LAB$_3$",
    }
    load_lines = [
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Load case & Rot. mag. (deg) & Trans. mag. (mm) & AP trans. (mm) & ML trans. (mm) \\",
        r"\midrule",
    ]
    for _, row in load_summary.iterrows():
        label = load_label[str(row["load_case"])]
        load_lines.append(
            f"{label} & "
            f"{row['rot_mag_deg_median']:.3f} & "
            f"{row['trans_mag_mm_median']:.3f} & "
            f"{row['ap_trans_abs_mm_median']:.3f} & "
            f"{row['ml_trans_abs_mm_median']:.3f} \\\\"
        )
    load_lines.extend([r"\bottomrule", r"\end{tabular}"])
    (OUT_DIR / "sij_micromotion_load_summary.tex").write_text("\n".join(load_lines) + "\n")

    # Table 2: swap tests (per LAB phase)
    nice_metric = {
        "rot_mag_deg": "Rotation magnitude (deg)",
        "trans_mag_mm": "Translation magnitude (mm)",
        "ap_rot_abs_deg": "AP rotation (deg)",
        "cc_rot_abs_deg": "CC rotation (deg)",
        "ap_trans_abs_mm": "AP translation (mm)",
        "ml_trans_abs_mm": "ML translation (mm)",
        "lr_trans_asym_mm": "L/R translation asymmetry (mm)",
    }
    for lab_case in LAB_CASES:
        st = swap_tests[lab_case]
        swap_lines = [
            r"\begin{tabular}{lcccc}",
            r"\toprule",
            r"Metric & No-swap median & Swap median & $\Delta$ (swap$-$no-swap) & $p$ \\",
            r"\midrule",
        ]
        for _, row in st.iterrows():
            label = nice_metric[row["metric"]]
            swap_lines.append(
                f"{label} & "
                f"{row['median_no_swap']:.3f} & "
                f"{row['median_swap']:.3f} & "
                f"{row['delta_swap_minus_no_swap']:+.3f} & "
                f"{row['p_value']:.3g} \\\\"
            )
        swap_lines.extend([r"\bottomrule", r"\end{tabular}"])
        (OUT_DIR / f"sij_micromotion_swap_tests_{lab_case}.tex").write_text("\n".join(swap_lines) + "\n")

    # Table 3: variance channels for LAB phases
    nice_metric_v = {
        "rot_mag_deg": "Rotation magnitude",
        "trans_mag_mm": "Translation magnitude",
        "ap_trans_abs_mm": "AP translation",
        "cc_rot_abs_deg": "CC rotation",
    }
    for lab_case in LAB_CASES:
        vc = variance_channels[variance_channels["load_case"] == lab_case].copy()
        vc_lines = [
            r"\begin{tabular}{lcc}",
            r"\toprule",
            r"Metric & Shape-only / full (\%) & Material-only / full (\%) \\",
            r"\midrule",
        ]
        for _, row in vc.iterrows():
            vc_lines.append(
                f"{nice_metric_v[row['metric']]} & "
                f"{row['shape_over_full_pct']:.1f} & "
                f"{row['material_over_full_pct']:.2f} \\\\"
            )
        vc_lines.extend([r"\bottomrule", r"\end{tabular}"])
        (OUT_DIR / f"sij_micromotion_variance_channels_{lab_case}.tex").write_text("\n".join(vc_lines) + "\n")


def make_dashboard_figure(
    subject_df: pd.DataFrame,
    load_summary: pd.DataFrame,
    load_deltas: pd.DataFrame,
    sex_tests: dict[str, pd.DataFrame],
    swap_tests: dict[str, pd.DataFrame],
    corr: dict[str, pd.DataFrame],
    variance_channels: pd.DataFrame,
) -> None:
    setup_plot_style()

    n_lab = len(LAB_CASES)
    fig = plt.figure(figsize=(FULL_WIDTH, 6.5 * ROW_H), constrained_layout=False)
    gs = fig.add_gridspec(4, n_lab, hspace=0.42, wspace=0.28)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1:])

    sex_axes = [fig.add_subplot(gs[1, i]) for i in range(n_lab)]
    swap_axes = [fig.add_subplot(gs[2, i]) for i in range(n_lab)]

    ax_g = fig.add_subplot(gs[3, 0])
    ax_h = fig.add_subplot(gs[3, 1:])

    # Panel A: global load profile (medians with 10--90 percentile band)
    ordered = load_summary.set_index("load_case").reindex(LOAD_CASES)
    x = np.arange(len(LOAD_CASES), dtype=float)
    rot_med = ordered["rot_mag_deg_median"].to_numpy(dtype=float)
    rot_q10 = ordered["rot_mag_deg_p10"].to_numpy(dtype=float)
    rot_q90 = ordered["rot_mag_deg_p90"].to_numpy(dtype=float)
    trans_med = ordered["trans_mag_mm_median"].to_numpy(dtype=float)
    trans_q10 = ordered["trans_mag_mm_p10"].to_numpy(dtype=float)
    trans_q90 = ordered["trans_mag_mm_p90"].to_numpy(dtype=float)

    ax_a.plot(x, rot_med, marker="o", color=NEUTRAL_COLOR, label="Rotation (deg)")
    ax_a.fill_between(x, rot_q10, rot_q90, color=NEUTRAL_COLOR, alpha=0.18)
    ax_a.set_ylabel("Rot. mag. (deg)", labelpad=2)
    ax_a.set_xticks(x)
    ax_a.set_xticklabels([LOAD_LABELS[l] for l in LOAD_CASES])

    ax_a2 = ax_a.twinx()
    ax_a2.plot(x, trans_med, marker="s", color=MATERIAL_EFFECT_COLOR, label="Translation (mm)")
    ax_a2.fill_between(x, trans_q10, trans_q90, color=MATERIAL_EFFECT_COLOR, alpha=0.18)
    ax_a2.set_ylabel("Trans. mag. (mm)", labelpad=2)

    lines_a = ax_a.get_lines() + ax_a2.get_lines()
    labels_a = [line.get_label() for line in lines_a]
    ax_a.legend(lines_a, labels_a, loc="upper right", fontsize=ANNOT_SIZE)
    panel_label(ax_a, "A. Load profile")

    # Panel B: directional translation deltas vs SP2 baseline
    delta_sub = load_deltas[
        load_deltas["metric"].isin(["ap_trans_abs_mm", "ml_trans_abs_mm", "cc_trans_abs_mm"])
        & load_deltas["load_case"].isin(LOAD_CASES_DELTA)
    ].copy()
    metric_order = ["ap_trans_abs_mm", "ml_trans_abs_mm", "cc_trans_abs_mm"]
    metric_name = {
        "ap_trans_abs_mm": "AP",
        "ml_trans_abs_mm": "ML",
        "cc_trans_abs_mm": "CC",
    }
    metric_color = {
        "ap_trans_abs_mm": MALE_COLOR,
        "ml_trans_abs_mm": SEX_EFFECT_COLOR,
        "cc_trans_abs_mm": FEMALE_COLOR,
    }
    x2 = np.arange(len(LOAD_CASES_DELTA), dtype=float)
    bar_w = 0.24
    for idx, metric in enumerate(metric_order):
        vals = (
            delta_sub[delta_sub["metric"] == metric]
            .set_index("load_case")
            .reindex(LOAD_CASES_DELTA)["median_delta"]
            .to_numpy(dtype=float)
        )
        ax_b.bar(x2 + (idx - 1) * bar_w, vals, width=bar_w, color=metric_color[metric], label=metric_name[metric])
    ax_b.axhline(0.0, color="0.3", lw=0.7)
    ax_b.set_xticks(x2)
    ax_b.set_xticklabels([LOAD_LABELS[l] for l in LOAD_CASES_DELTA])
    ax_b.set_ylabel(r"Median $\Delta$ vs SP2 (mm)", labelpad=3)
    ax_b.legend(frameon=False, fontsize=ANNOT_SIZE, ncol=3, loc="upper left")
    panel_label(ax_b, "B. Direction shifts")

    # Panels row 1: sex differences under each LAB phase
    sex_panel_labels = [chr(ord("C") + i) for i in range(n_lab)]
    sex_metric_order = ["rot_mag_deg", "trans_mag_mm"]
    sex_metric_labels = ["Rotation", "Translation"]
    for i_phase, lab_case in enumerate(LAB_CASES):
        ax = sex_axes[i_phase]
        lab = subject_df[subject_df["load_case"] == lab_case].copy()
        x3 = np.arange(len(sex_metric_order), dtype=float)
        bar_w3 = 0.34
        for idx, (sex_key, color) in enumerate([("F", FEMALE_COLOR), ("M", MALE_COLOR)]):
            medians = []
            lows = []
            highs = []
            for metric in sex_metric_order:
                vals = lab.loc[lab["sex"] == sex_key, metric].to_numpy(dtype=float)
                q10, q50, q90 = np.nanpercentile(vals, [10, 50, 90])
                medians.append(q50)
                lows.append(q50 - q10)
                highs.append(q90 - q50)
            yerr = np.vstack([np.asarray(lows, dtype=float), np.asarray(highs, dtype=float)])
            ax.bar(
                x3 + (idx - 0.5) * bar_w3,
                np.asarray(medians, dtype=float),
                yerr=yerr,
                width=bar_w3,
                color=color,
                alpha=0.88,
                label="Female" if sex_key == "F" else "Male",
            )
        ax.set_xticks(x3)
        ax.set_xticklabels(sex_metric_labels)
        ax.set_ylabel("Median (10--90 percentile)")
        ax.legend(frameon=False, fontsize=ANNOT_SIZE, loc="upper right")
        st = sex_tests[lab_case]
        rot_p = float(st.loc[st["metric"] == "rot_mag_deg", "p_value"].iloc[0])
        tr_p = float(st.loc[st["metric"] == "trans_mag_mm", "p_value"].iloc[0])
        ax.text(
            0.02, 0.98, f"p(rot)={rot_p:.2g}\np(trans)={tr_p:.2g}",
            transform=ax.transAxes, va="top", ha="left", fontsize=ANNOT_SIZE,
        )
        panel_label(ax, f"{sex_panel_labels[i_phase]}. Sex contrast ({LOAD_LABELS[lab_case]})")

    # Panels row 2: swap effects under each LAB phase
    swap_panel_start = chr(ord("C") + n_lab)
    swap_panel_labels = [chr(ord(swap_panel_start) + i) for i in range(n_lab)]
    swap_metric_order = ["rot_mag_deg", "trans_mag_mm", "ap_trans_abs_mm"]
    swap_metric_labels = ["Rotation", "Translation", "AP trans."]
    for i_phase, lab_case in enumerate(LAB_CASES):
        ax = swap_axes[i_phase]
        st = swap_tests[lab_case]
        x4 = np.arange(len(swap_metric_order), dtype=float)
        bar_w4 = 0.34
        med_no = (
            st.set_index("metric")
            .reindex(swap_metric_order)["median_no_swap"]
            .to_numpy(dtype=float)
        )
        med_sw = (
            st.set_index("metric")
            .reindex(swap_metric_order)["median_swap"]
            .to_numpy(dtype=float)
        )
        pvals_sw = (
            st.set_index("metric")
            .reindex(swap_metric_order)["p_value"]
            .to_numpy(dtype=float)
        )
        ax.bar(x4 - bar_w4 / 2, med_no, width=bar_w4, color=NEUTRAL_COLOR, alpha=0.85, label="No swap")
        ax.bar(x4 + bar_w4 / 2, med_sw, width=bar_w4, color=SWAP_LINE_COLOR, alpha=0.85, label="Swap")
        ax.set_xticks(x4)
        ax.set_xticklabels(swap_metric_labels)
        ax.set_ylabel("Median value")
        y_top = float(np.nanmax(np.r_[med_no, med_sw]))
        for i, p_val in enumerate(pvals_sw):
            ax.text(i, y_top * 1.06, f"p={p_val:.2g}", ha="center", va="bottom", fontsize=ANNOT_SIZE)
        ax.set_ylim(0.0, y_top * 1.18)
        ax.legend(frameon=False, fontsize=ANNOT_SIZE, loc="upper right")
        panel_label(ax, f"{swap_panel_labels[i_phase]}. Swap contrast ({LOAD_LABELS[lab_case]})")

    # Panel G: operating-point and tail coupling matrix (LAB₁)
    g_label = chr(ord(swap_panel_start) + n_lab)
    h_label = chr(ord(g_label) + 1)
    corr_lab1 = corr[LAB_CASES[0]]
    corr_metrics = ["rot_mag_deg", "trans_mag_mm", "ap_trans_abs_mm", "ml_trans_abs_mm", "cc_rot_abs_deg"]
    corr_covs = ["gap_in_9_10", "dG_9_10", "P99_SED", "top1pct_mean_SED"]
    corr_labels_m = ["Rot", "Trans", "AP trans", "ML trans", "CC rot"]
    corr_labels_c = [r"$gap_{in}$", r"$d_G$", r"$P99$", r"$Top1\%$"]
    rho_mat = (
        corr_lab1.pivot(index="metric", columns="covariate", values="rho_spearman")
        .reindex(index=corr_metrics, columns=corr_covs)
        .to_numpy(dtype=float)
    )
    p_mat = (
        corr_lab1.pivot(index="metric", columns="covariate", values="p_value")
        .reindex(index=corr_metrics, columns=corr_covs)
        .to_numpy(dtype=float)
    )
    im = ax_g.imshow(rho_mat, cmap="coolwarm", vmin=-0.5, vmax=0.5, aspect="auto")
    ax_g.set_xticks(np.arange(len(corr_covs)))
    ax_g.set_xticklabels(corr_labels_c, rotation=0)
    ax_g.set_yticks(np.arange(len(corr_metrics)))
    ax_g.set_yticklabels(corr_labels_m)
    for r in range(rho_mat.shape[0]):
        for c in range(rho_mat.shape[1]):
            star = "*" if p_mat[r, c] < 0.05 else ""
            ax_g.text(c, r, f"{rho_mat[r, c]:.2f}{star}", ha="center", va="center", fontsize=ANNOT_SIZE, color="black")
    cbar = fig.colorbar(im, ax=ax_g, fraction=0.046, pad=0.04)
    cbar.set_label(r"Spearman $\rho$")
    panel_label(ax_g, f"{g_label}. Coupling matrix ({LOAD_LABELS[LAB_CASES[0]]})")

    # Panel H: morphology/material variance channel ratio (all LAB phases)
    vc_metric_order = ["rot_mag_deg", "trans_mag_mm", "ap_trans_abs_mm", "cc_rot_abs_deg"]
    vc_metric_labels = ["Rot", "Trans", "AP trans", "CC rot"]
    n_metrics = len(vc_metric_order)
    n_phases = len(LAB_CASES)
    x6 = np.arange(n_metrics * n_phases, dtype=float)
    tick_labels = []
    shape_vals = []
    material_vals = []
    for metric in vc_metric_order:
        for lab_case in LAB_CASES:
            tick_labels.append(f"{vc_metric_labels[vc_metric_order.index(metric)]}\n{LOAD_LABELS[lab_case]}")
            vc_sub = variance_channels[
                (variance_channels["load_case"] == lab_case) & (variance_channels["metric"] == metric)
            ]
            if len(vc_sub) > 0:
                shape_vals.append(float(vc_sub["shape_over_full_pct"].iloc[0]))
                material_vals.append(float(vc_sub["material_over_full_pct"].iloc[0]))
            else:
                shape_vals.append(np.nan)
                material_vals.append(np.nan)

    shape_arr = np.asarray(shape_vals, dtype=float)
    material_arr = np.asarray(material_vals, dtype=float)
    ax_h.bar(x6, shape_arr, color=SHAPE_EFFECT_COLOR, alpha=0.78, width=0.55, label="Shape-only / full")
    ax_h.axhline(100.0, color="0.35", lw=0.7, ls="--")
    ax_h.set_xticks(x6)
    ax_h.set_xticklabels(tick_labels, fontsize=ANNOT_SIZE - 1)
    ax_h.set_ylabel("Shape-only/full (%)")
    ax_h.set_ylim(98.4, 101.6)

    ax_h2 = ax_h.twinx()
    ax_h2.plot(x6, material_arr, color=MATERIAL_EFFECT_COLOR, marker="o", label="Material-only / full")
    ax_h2.set_ylabel("Material-only/full (%)")
    ax_h2.set_ylim(0.0, max(0.06, float(np.nanmax(material_arr)) * 1.5))

    lines_h = [ax_h.patches[0], ax_h2.get_lines()[0]]
    labels_h = ["Shape-only / full", "Material-only / full"]
    ax_h.legend(lines_h, labels_h, loc="upper right", fontsize=ANNOT_SIZE)
    panel_label(ax_h, f"{h_label}. Variance channels")

    FIGURE_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.05, top=0.975)
    fig.set_constrained_layout(False)
    fig.set_layout_engine("none")
    fig.savefig(FIGURE_OUT.with_suffix(".pdf"), bbox_inches=None)
    fig.set_constrained_layout(False)
    fig.set_layout_engine("none")
    fig.savefig(FIGURE_OUT.with_suffix(".png"), dpi=300, bbox_inches=None)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    subject_df = build_subject_level_table()
    load_summary = summarize_load_profile(subject_df)
    load_deltas = paired_load_deltas_vs_sp2(subject_df)
    variance_channels = variance_channel_comparison()

    # Per-LAB phase analyses
    swap_tests: dict[str, pd.DataFrame] = {}
    sex_tests: dict[str, pd.DataFrame] = {}
    corr: dict[str, pd.DataFrame] = {}
    models: dict[str, pd.DataFrame] = {}
    for lab_case in LAB_CASES:
        swap_tests[lab_case] = compare_swap_groups_pressure(subject_df, lab_case)
        sex_tests[lab_case] = compare_sex_groups_pressure(subject_df, lab_case)
        corr[lab_case] = correlate_with_operating_point_pressure(subject_df, lab_case)
        models[lab_case] = adjusted_pressure_models(subject_df, lab_case)

    # Save CSVs
    subject_df.to_csv(OUT_DIR / "sij_micromotion_subject_level.csv", index=False)
    load_summary.to_csv(OUT_DIR / "sij_micromotion_load_summary.csv", index=False)
    load_deltas.to_csv(OUT_DIR / "sij_micromotion_load_deltas_vs_sp2.csv", index=False)
    variance_channels.to_csv(OUT_DIR / "sij_micromotion_variance_channels.csv", index=False)
    for lab_case in LAB_CASES:
        swap_tests[lab_case].to_csv(OUT_DIR / f"sij_micromotion_swap_tests_{lab_case}.csv", index=False)
        sex_tests[lab_case].to_csv(OUT_DIR / f"sij_micromotion_sex_tests_{lab_case}.csv", index=False)
        corr[lab_case].to_csv(OUT_DIR / f"sij_micromotion_correlations_{lab_case}.csv", index=False)
        models[lab_case].to_csv(OUT_DIR / f"sij_micromotion_models_{lab_case}.csv", index=False)

    write_latex_tables(load_summary, swap_tests, variance_channels)
    make_dashboard_figure(
        subject_df=subject_df,
        load_summary=load_summary,
        load_deltas=load_deltas,
        sex_tests=sex_tests,
        swap_tests=swap_tests,
        corr=corr,
        variance_channels=variance_channels,
    )

    print(f"Wrote SIJ micromotion outputs to {OUT_DIR}")
    print(f"Wrote SIJ dashboard to {FIGURE_OUT.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
