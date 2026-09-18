#!/usr/bin/env python3
"""Reproducible SIJ micromotion analysis for a Journal of Anatomy manuscript.

This script consumes precomputed FE outputs and metadata, then generates:
- analysis-ready tables
- statistical model outputs (robust regression with multiplicity control)
- publication figures
- LaTeX-ready tables
- a data dictionary
"""

from __future__ import annotations

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import zarr
from scipy.stats import wilcoxon
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests


REPO_ROOT = Path(__file__).resolve().parents[3]
PAPER_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR_FULL = REPO_ROOT / "results" / "ref_S1P_fixed_new2" / "data"
DATA_DIR_SHAPE_ONLY = REPO_ROOT / "results" / "ref_S1P_fixed_new2_shape_only" / "data"
DATA_DIR_MATERIAL_ONLY = REPO_ROOT / "results" / "ref_S1P_fixed_new2_material_only" / "data"

DEMOGRAPHY_XLSX = DATA_DIR_FULL / "demography.xlsx"
ANATOMY_XLSX = DATA_DIR_FULL / "anatomy_data.xlsx"
ALLOMETRY_XLSX = DATA_DIR_FULL / "allometry.xlsx"

OUT_TABLE_DIR = PAPER_ROOT / "tables" / "generated"
OUT_FIG_DIR = PAPER_ROOT / "figures"
OUT_DATA_DICT = PAPER_ROOT / "data_dictionary.md"
OUT_RESULTS_JSON = OUT_TABLE_DIR / "analysis_summary.json"

LOAD_ORDER = ["SP2leg", "SP1leg", "LAB_phase1", "LAB_phase2", "LAB_phase3"]
LAB_LOADS = ["LAB_phase1", "LAB_phase2", "LAB_phase3"]
LOAD_LABEL = {
    "SP2leg": "SP2leg",
    "SP1leg": "SP1leg",
    "LAB_phase1": "LAB1",
    "LAB_phase2": "LAB2",
    "LAB_phase3": "LAB3",
}

MORPH_COLS = [
    "AP",
    "BiacetabularWidth",
    "BiischiadicWidth",
    "BituberousWidth",
    "IliopectinealEminenceWidth",
    "PIT",
    "SacralWidth",
    "SubpubicAngle",
]
LINEAR_DIM_COLS = [c for c in MORPH_COLS if c != "SubpubicAngle"]

PRIMARY_OUTCOMES = ["rot_mag_deg", "trans_mag_mm"]
SECONDARY_OUTCOMES = ["ap_trans_abs_mm", "ml_trans_abs_mm", "cc_trans_abs_mm"]

METRIC_LABELS = {
    "rot_mag_deg": "Rotation magnitude (deg)",
    "trans_mag_mm": "Translation magnitude (mm)",
    "nut_abs_deg": "Nutation component (deg)",
    "ap_rot_abs_deg": "AP rotation component (deg)",
    "cc_rot_abs_deg": "CC rotation component (deg)",
    "ap_trans_abs_mm": "AP translation component (mm)",
    "ml_trans_abs_mm": "ML translation component (mm)",
    "cc_trans_abs_mm": "CC translation component (mm)",
    "lr_rot_asym_deg": "Left-right rotation asymmetry (deg)",
    "lr_trans_asym_mm": "Left-right translation asymmetry (mm)",
}


def ensure_dirs() -> None:
    OUT_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    OUT_FIG_DIR.mkdir(parents=True, exist_ok=True)


def zscore(series: pd.Series) -> pd.Series:
    sd = float(series.std(ddof=1))
    if sd == 0 or np.isnan(sd):
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - float(series.mean())) / sd


def fmt_median_iqr(values: pd.Series) -> str:
    q25, q50, q75 = np.percentile(values.to_numpy(dtype=float), [25, 50, 75])
    return f"{q50:.3f} [{q25:.3f}, {q75:.3f}]"


def partial_r2_from_t(t_value: float, df_resid: float) -> float:
    t2 = float(t_value) ** 2
    return t2 / (t2 + float(df_resid)) if np.isfinite(t2) and np.isfinite(df_resid) else np.nan


def bootstrap_ci(data: np.ndarray, statistic_fn, n_boot: int = 2000, seed: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    data = np.asarray(data, dtype=float)
    n = len(data)
    if n == 0:
        return np.nan, np.nan
    samples = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        samples[i] = statistic_fn(data[idx])
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def load_core_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    demography = pd.read_excel(DEMOGRAPHY_XLSX)
    morphology = pd.read_excel(ANATOMY_XLSX, sheet_name="Morphology")
    allometry = pd.read_excel(ALLOMETRY_XLSX, sheet_name="allometry")

    for frame in [demography, morphology, allometry]:
        frame["patient_id"] = frame["patient_id"].astype(str)

    return demography, morphology, allometry


def build_base_subject_table(
    demography: pd.DataFrame,
    morphology: pd.DataFrame,
    allometry: pd.DataFrame,
) -> pd.DataFrame:
    base = demography.merge(morphology, on="patient_id", how="inner", validate="one_to_one")
    base = base.merge(allometry, on="patient_id", how="inner", validate="one_to_one")

    base["sex_F"] = (base["sex"] == "F").astype(float)
    base["subject_idx"] = np.arange(len(base), dtype=int)

    base["geom_mean_linear_mm"] = np.exp(np.log(base[LINEAR_DIM_COLS]).mean(axis=1))
    base["centroid_linear_mm"] = np.sqrt((base[LINEAR_DIM_COLS] ** 2).sum(axis=1))
    base["log_total_volume"] = np.log(base["total_volume"])
    base["log_total_surface"] = np.log(base["total_surface"])
    base["log_scale"] = np.log(base["scale"])

    for col in ["age", "log_total_volume", "AP", "BiischiadicWidth", "SubpubicAngle", "log_scale"]:
        base[f"{col}_z"] = zscore(base[col])

    return base


def _extract_lr_sij_frame(anatomy_xlsx: Path, load_case: str) -> pd.DataFrame:
    left = pd.read_excel(anatomy_xlsx, sheet_name=f"SIJ_L_{load_case}")
    right = pd.read_excel(anatomy_xlsx, sheet_name=f"SIJ_R_{load_case}")

    left["patient_id"] = left["patient_id"].astype(str)
    right["patient_id"] = right["patient_id"].astype(str)
    df = left.merge(right, on="patient_id", how="inner", validate="one_to_one")

    la = df[
        [
            f"sij_ang_{load_case}_L_NUT",
            f"sij_ang_{load_case}_L_AP",
            f"sij_ang_{load_case}_L_CC",
        ]
    ].to_numpy(dtype=float)
    ra = df[
        [
            f"sij_ang_{load_case}_R_NUT",
            f"sij_ang_{load_case}_R_AP",
            f"sij_ang_{load_case}_R_CC",
        ]
    ].to_numpy(dtype=float)
    lt = df[
        [
            f"sij_trans_{load_case}_L_ML",
            f"sij_trans_{load_case}_L_AP",
            f"sij_trans_{load_case}_L_CC",
        ]
    ].to_numpy(dtype=float)
    rt = df[
        [
            f"sij_trans_{load_case}_R_ML",
            f"sij_trans_{load_case}_R_AP",
            f"sij_trans_{load_case}_R_CC",
        ]
    ].to_numpy(dtype=float)

    rot_l = np.linalg.norm(la, axis=1)
    rot_r = np.linalg.norm(ra, axis=1)
    tr_l = np.linalg.norm(lt, axis=1)
    tr_r = np.linalg.norm(rt, axis=1)

    out = pd.DataFrame(
        {
            "patient_id": df["patient_id"],
            "load_case": load_case,
            "rot_mag_deg": 0.5 * (rot_l + rot_r),
            "trans_mag_mm": 0.5 * (tr_l + tr_r),
            "nut_abs_deg": 0.5 * (np.abs(la[:, 0]) + np.abs(ra[:, 0])),
            "ap_rot_abs_deg": 0.5 * (np.abs(la[:, 1]) + np.abs(ra[:, 1])),
            "cc_rot_abs_deg": 0.5 * (np.abs(la[:, 2]) + np.abs(ra[:, 2])),
            "ml_trans_abs_mm": 0.5 * (np.abs(lt[:, 0]) + np.abs(rt[:, 0])),
            "ap_trans_abs_mm": 0.5 * (np.abs(lt[:, 1]) + np.abs(rt[:, 1])),
            "cc_trans_abs_mm": 0.5 * (np.abs(lt[:, 2]) + np.abs(rt[:, 2])),
            "ml_trans_signed_mm": 0.5 * (lt[:, 0] + rt[:, 0]),
            "ap_trans_signed_mm": 0.5 * (lt[:, 1] + rt[:, 1]),
            "cc_trans_signed_mm": 0.5 * (lt[:, 2] + rt[:, 2]),
            "lr_rot_asym_deg": np.abs(rot_l - rot_r),
            "lr_trans_asym_mm": np.abs(tr_l - tr_r),
        }
    )
    return out


def build_subject_level(base_subject: pd.DataFrame) -> pd.DataFrame:
    all_frames = [_extract_lr_sij_frame(ANATOMY_XLSX, load) for load in LOAD_ORDER]
    kinematics = pd.concat(all_frames, ignore_index=True)

    subject = kinematics.merge(base_subject, on="patient_id", how="inner", validate="many_to_one")
    subject["load_case"] = pd.Categorical(subject["load_case"], categories=LOAD_ORDER, ordered=True)

    # z-scored outcomes for standardized coefficients
    for outcome in PRIMARY_OUTCOMES + SECONDARY_OUTCOMES + ["ap_rot_abs_deg", "cc_rot_abs_deg"]:
        subject[f"{outcome}_z"] = zscore(subject[outcome])

    return subject


def create_table_1_cohort(base_subject: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, str]] = []

    def stat(series: pd.Series) -> str:
        return f"{series.mean():.2f} ± {series.std(ddof=1):.2f}"

    n_total = len(base_subject)
    n_f = int((base_subject["sex"] == "F").sum())
    n_m = int((base_subject["sex"] == "M").sum())

    rows.append(
        {
            "Variable": "N",
            "Overall": str(n_total),
            "Female": str(n_f),
            "Male": str(n_m),
            "Missing": "0",
        }
    )

    summary_vars = [
        ("age", "Age (years)"),
        ("AP", "Inlet AP diameter (mm)"),
        ("BiacetabularWidth", "Biacetabular width (mm)"),
        ("BiischiadicWidth", "Biischiadic width (mm)"),
        ("BituberousWidth", "Bituberous width (mm)"),
        ("IliopectinealEminenceWidth", "Iliopectineal eminence width (mm)"),
        ("PIT", "Pelvic inlet transverse (mm)"),
        ("SacralWidth", "Sacral width (mm)"),
        ("SubpubicAngle", "Subpubic angle (deg)"),
        ("scale", "Global scale factor (-)"),
        ("total_volume", "Total pelvic volume (cm$^3$)"),
        ("total_surface", "Total pelvic surface (cm$^2$)"),
    ]

    for col, label in summary_vars:
        s = base_subject[col]
        sf = base_subject.loc[base_subject["sex"] == "F", col]
        sm = base_subject.loc[base_subject["sex"] == "M", col]
        rows.append(
            {
                "Variable": label,
                "Overall": stat(s),
                "Female": stat(sf),
                "Male": stat(sm),
                "Missing": str(int(s.isna().sum())),
            }
        )

    return pd.DataFrame(rows)


def create_table_2_load_definitions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Load case": "SP2leg",
                "Mechanical proxy": "Bilateral standing",
                "Applied total forces": "[0,0,+400] N on right AC notch and [0,0,+400] N on left AC notch",
                "Resultant": "800 N cranio-caudal",
            },
            {
                "Load case": "SP1leg",
                "Mechanical proxy": "Unilateral standing",
                "Applied total forces": "[0,0,+800] N on right AC notch",
                "Resultant": "800 N cranio-caudal",
            },
            {
                "Load case": "LAB1",
                "Mechanical proxy": "Mediolateral ring compression proxy",
                "Applied total forces": "[+400,0,0] N on ring_contact_left and [-400,0,0] N on ring_contact_right",
                "Resultant": "0 N net; bilateral ML pair",
            },
            {
                "Load case": "LAB2",
                "Mechanical proxy": "Ischial loading / SIJ distraction proxy",
                "Applied total forces": "[+400,0,0] N on left_ischium_tuber and [-400,0,0] N on right_ischium_tuber",
                "Resultant": "0 N net; bilateral ML pair",
            },
            {
                "Load case": "LAB3",
                "Mechanical proxy": "Outlet AP distraction proxy",
                "Applied total forces": "[0,-400,0] N on pubis_ins and [0,+400,0] N on SCJ",
                "Resultant": "0 N net; AP pair",
            },
        ]
    )


def create_table_3_micromotion_summary(subject_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    metrics = [
        "rot_mag_deg",
        "trans_mag_mm",
        "ap_trans_abs_mm",
        "ml_trans_abs_mm",
        "cc_trans_abs_mm",
    ]

    for load in LOAD_ORDER:
        sub = subject_df.loc[subject_df["load_case"] == load]
        row: dict[str, str] = {
            "Load case": LOAD_LABEL[load],
            "N": str(len(sub)),
        }
        for m in metrics:
            row[METRIC_LABELS[m]] = fmt_median_iqr(sub[m])
        rows.append(row)

    return pd.DataFrame(rows)


def compute_directional_rerouting(subject_df: pd.DataFrame) -> pd.DataFrame:
    base = (
        subject_df.loc[subject_df["load_case"] == "SP2leg", ["patient_id"] + SECONDARY_OUTCOMES]
        .rename(columns={m: f"{m}_sp2" for m in SECONDARY_OUTCOMES})
        .copy()
    )

    rows: list[dict[str, float | str]] = []
    for load in [l for l in LOAD_ORDER if l != "SP2leg"]:
        sub = subject_df.loc[subject_df["load_case"] == load, ["patient_id"] + SECONDARY_OUTCOMES].merge(
            base,
            on="patient_id",
            how="inner",
            validate="one_to_one",
        )
        for metric in SECONDARY_OUTCOMES:
            delta = sub[metric].to_numpy(dtype=float) - sub[f"{metric}_sp2"].to_numpy(dtype=float)
            ci_low, ci_high = bootstrap_ci(delta, np.median, n_boot=3000, seed=11)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                nz = delta[np.abs(delta) > 1e-12]
                p_val = float(wilcoxon(nz).pvalue) if len(nz) else np.nan
            dz = float(np.mean(delta) / np.std(delta, ddof=1)) if np.std(delta, ddof=1) > 0 else np.nan
            rows.append(
                {
                    "load_case": load,
                    "metric": metric,
                    "median_delta": float(np.median(delta)),
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                    "p_value": p_val,
                    "cohen_dz": dz,
                }
            )

    out = pd.DataFrame(rows)
    mask = out["p_value"].notna().to_numpy()
    out["p_fdr_bh"] = np.nan
    out.loc[mask, "p_fdr_bh"] = multipletests(out.loc[mask, "p_value"], method="fdr_bh")[1]
    return out


def fit_cluster_robust_model(subject_df: pd.DataFrame, outcome: str) -> tuple[pd.DataFrame, object]:
    formula = (
        f"{outcome} ~ C(load_case, Treatment(reference='SP2leg')) + sex_F + age_z + "
        "log_total_volume_z + AP_z + BiischiadicWidth_z + SubpubicAngle_z + "
        "C(load_case, Treatment(reference='SP2leg')):log_total_volume_z + sex_F:log_total_volume_z"
    )
    model = smf.ols(formula, data=subject_df).fit(
        cov_type="cluster", cov_kwds={"groups": subject_df["patient_id"]}
    )

    # standardized counterpart for standardized coefficients
    z_formula = formula.replace(outcome, f"{outcome}_z", 1)
    model_z = smf.ols(z_formula, data=subject_df).fit(
        cov_type="cluster", cov_kwds={"groups": subject_df["patient_id"]}
    )

    terms = []
    ci = model.conf_int()
    for term in model.params.index:
        if term == "Intercept":
            continue
        t_value = float(model.tvalues.get(term, np.nan))
        terms.append(
            {
                "outcome": outcome,
                "term": term,
                "beta": float(model.params[term]),
                "ci95_low": float(ci.loc[term, 0]),
                "ci95_high": float(ci.loc[term, 1]),
                "p_value": float(model.pvalues[term]),
                "beta_std": float(model_z.params.get(term, np.nan)),
                "partial_r2": partial_r2_from_t(t_value, float(model.df_resid)),
            }
        )

    out = pd.DataFrame(terms)
    out["p_fdr_bh"] = multipletests(out["p_value"], method="fdr_bh")[1]
    return out, model


def fit_sex_models_lab(subject_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []

    for load in LAB_LOADS:
        sub = subject_df.loc[subject_df["load_case"] == load].copy()
        for outcome in PRIMARY_OUTCOMES:
            formula = (
                f"{outcome} ~ sex_F + age_z + log_total_volume_z + AP_z + "
                "BiischiadicWidth_z + SubpubicAngle_z + sex_F:log_total_volume_z"
            )
            model = smf.ols(formula, data=sub).fit(cov_type="HC3")
            model_z = smf.ols(formula.replace(outcome, f"{outcome}_z", 1), data=sub).fit(cov_type="HC3")
            term = "sex_F"
            ci = model.conf_int().loc[term]
            rows.append(
                {
                    "load_case": load,
                    "outcome": outcome,
                    "beta_female_minus_male": float(model.params[term]),
                    "ci95_low": float(ci[0]),
                    "ci95_high": float(ci[1]),
                    "p_value": float(model.pvalues[term]),
                    "beta_std": float(model_z.params[term]),
                    "partial_r2": partial_r2_from_t(float(model.tvalues[term]), float(model.df_resid)),
                }
            )

    out = pd.DataFrame(rows)
    out["p_fdr_bh"] = multipletests(out["p_value"], method="fdr_bh")[1]
    return out


def fit_allometry_models(subject_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[tuple[str, str], object]]:
    rows: list[dict[str, float | str]] = []
    models: dict[tuple[str, str], object] = {}

    for load in LOAD_ORDER:
        sub = subject_df.loc[subject_df["load_case"] == load].copy()
        for outcome in PRIMARY_OUTCOMES:
            formula = (
                f"np.log({outcome}) ~ log_total_volume + sex_F + age_z + AP_z + "
                "BiischiadicWidth_z + SubpubicAngle_z + log_total_volume:sex_F"
            )
            model = smf.ols(formula, data=sub).fit(cov_type="HC3")
            models[(load, outcome)] = model

            b_male = float(model.params["log_total_volume"])
            b_inter = float(model.params.get("log_total_volume:sex_F", 0.0))
            b_female = b_male + b_inter

            cov = model.cov_params()
            var_m = float(cov.loc["log_total_volume", "log_total_volume"])
            se_m = np.sqrt(max(var_m, 0.0))

            var_f = float(
                cov.loc["log_total_volume", "log_total_volume"]
                + cov.loc["log_total_volume:sex_F", "log_total_volume:sex_F"]
                + 2.0 * cov.loc["log_total_volume", "log_total_volume:sex_F"]
            )
            se_f = np.sqrt(max(var_f, 0.0))

            rows.append(
                {
                    "load_case": load,
                    "outcome": outcome,
                    "male_exponent": b_male,
                    "male_ci95_low": b_male - 1.96 * se_m,
                    "male_ci95_high": b_male + 1.96 * se_m,
                    "female_exponent": b_female,
                    "female_ci95_low": b_female - 1.96 * se_f,
                    "female_ci95_high": b_female + 1.96 * se_f,
                    "sex_slope_interaction_beta": b_inter,
                    "sex_slope_interaction_p": float(model.pvalues.get("log_total_volume:sex_F", np.nan)),
                    "adj_r2": float(model.rsquared_adj),
                    "n": int(len(sub)),
                }
            )

    out = pd.DataFrame(rows)
    out["interaction_p_fdr_bh"] = multipletests(out["sex_slope_interaction_p"], method="fdr_bh")[1]
    return out, models


def fit_allometry_models_scale(subject_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []

    for load in LOAD_ORDER:
        sub = subject_df.loc[subject_df["load_case"] == load].copy()
        for outcome in PRIMARY_OUTCOMES:
            formula = (
                f"np.log({outcome}) ~ log_scale + sex_F + age_z + AP_z + "
                "BiischiadicWidth_z + SubpubicAngle_z + log_scale:sex_F"
            )
            model = smf.ols(formula, data=sub).fit(cov_type="HC3")

            b_male = float(model.params["log_scale"])
            b_inter = float(model.params.get("log_scale:sex_F", 0.0))
            b_female = b_male + b_inter

            cov = model.cov_params()
            var_m = float(cov.loc["log_scale", "log_scale"])
            se_m = np.sqrt(max(var_m, 0.0))

            var_f = float(
                cov.loc["log_scale", "log_scale"]
                + cov.loc["log_scale:sex_F", "log_scale:sex_F"]
                + 2.0 * cov.loc["log_scale", "log_scale:sex_F"]
            )
            se_f = np.sqrt(max(var_f, 0.0))

            rows.append(
                {
                    "load_case": load,
                    "outcome": outcome,
                    "male_exponent_scale": b_male,
                    "male_ci95_low": b_male - 1.96 * se_m,
                    "male_ci95_high": b_male + 1.96 * se_m,
                    "female_exponent_scale": b_female,
                    "female_ci95_low": b_female - 1.96 * se_f,
                    "female_ci95_high": b_female + 1.96 * se_f,
                    "sex_slope_interaction_beta_scale": b_inter,
                    "sex_slope_interaction_p_scale": float(model.pvalues.get("log_scale:sex_F", np.nan)),
                    "adj_r2": float(model.rsquared_adj),
                    "n": int(len(sub)),
                }
            )

    out = pd.DataFrame(rows)
    out["interaction_p_fdr_bh_scale"] = multipletests(out["sex_slope_interaction_p_scale"], method="fdr_bh")[1]
    return out


def _metric_frame_from_arrays(angles: np.ndarray, trans: np.ndarray) -> pd.DataFrame:
    rot_l = np.linalg.norm(angles[:, 0, :], axis=1)
    rot_r = np.linalg.norm(angles[:, 1, :], axis=1)
    tr_l = np.linalg.norm(trans[:, 0, :], axis=1)
    tr_r = np.linalg.norm(trans[:, 1, :], axis=1)

    return pd.DataFrame(
        {
            "rot_mag_deg": 0.5 * (rot_l + rot_r),
            "trans_mag_mm": 0.5 * (tr_l + tr_r),
            "ap_trans_abs_mm": 0.5 * (np.abs(trans[:, 0, 1]) + np.abs(trans[:, 1, 1])),
            "ml_trans_abs_mm": 0.5 * (np.abs(trans[:, 0, 0]) + np.abs(trans[:, 1, 0])),
            "cc_trans_abs_mm": 0.5 * (np.abs(trans[:, 0, 2]) + np.abs(trans[:, 1, 2])),
        }
    )


def compute_variance_channels() -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []

    channels = {
        "full": DATA_DIR_FULL,
        "shape_only": DATA_DIR_SHAPE_ONLY,
        "material_only": DATA_DIR_MATERIAL_ONLY,
    }

    for load in LOAD_ORDER:
        metrics = {}
        n_subjects_per_channel: dict[str, int] = {}
        for ch, data_dir in channels.items():
            angles = np.asarray(zarr.open_group(str(data_dir / f"sij_angles_{load}.zarr"), mode="r")["data"][:], dtype=float)
            trans = np.asarray(zarr.open_group(str(data_dir / f"sij_trans_{load}.zarr"), mode="r")["data"][:], dtype=float)
            n_subjects_per_channel[ch] = angles.shape[0]
            metrics[ch] = _metric_frame_from_arrays(angles, trans)

        # Ensure subject alignment across variance channels
        assert len(set(n_subjects_per_channel.values())) == 1, (
            f"Subject count mismatch across variance channels for {load}: {n_subjects_per_channel}"
        )

        for metric in ["rot_mag_deg", "trans_mag_mm", "ap_trans_abs_mm", "ml_trans_abs_mm", "cc_trans_abs_mm"]:
            x_full = metrics["full"][metric].to_numpy(dtype=float)
            x_shape = metrics["shape_only"][metric].to_numpy(dtype=float)
            x_mat = metrics["material_only"][metric].to_numpy(dtype=float)

            var_full = float(np.var(x_full, ddof=1))
            var_shape = float(np.var(x_shape, ddof=1))
            var_mat = float(np.var(x_mat, ddof=1))

            shape_ratio = 100.0 * var_shape / var_full if var_full > 0 else np.nan
            mat_ratio = 100.0 * var_mat / var_full if var_full > 0 else np.nan

            # bootstrap ratio CI
            n = len(x_full)
            rng = np.random.default_rng(123)
            bs_shape = np.empty(2000, dtype=float)
            bs_mat = np.empty(2000, dtype=float)
            for i in range(2000):
                idx = rng.integers(0, n, n)
                vf = np.var(x_full[idx], ddof=1)
                vs = np.var(x_shape[idx], ddof=1)
                vm = np.var(x_mat[idx], ddof=1)
                bs_shape[i] = 100.0 * vs / vf if vf > 0 else np.nan
                bs_mat[i] = 100.0 * vm / vf if vf > 0 else np.nan

            rows.append(
                {
                    "load_case": load,
                    "metric": metric,
                    "var_full": var_full,
                    "var_shape_only": var_shape,
                    "var_material_only": var_mat,
                    "shape_over_full_pct": shape_ratio,
                    "shape_over_full_ci95_low": float(np.nanpercentile(bs_shape, 2.5)),
                    "shape_over_full_ci95_high": float(np.nanpercentile(bs_shape, 97.5)),
                    "material_over_full_pct": mat_ratio,
                    "material_over_full_ci95_low": float(np.nanpercentile(bs_mat, 2.5)),
                    "material_over_full_ci95_high": float(np.nanpercentile(bs_mat, 97.5)),
                }
            )

    return pd.DataFrame(rows)


def write_latex_table(df: pd.DataFrame, out_path: Path, float_fmt: str = "%.3f") -> None:
    safe_df = df.copy()
    safe_df.columns = [str(c).replace("_", r"\_") for c in safe_df.columns]
    for col in safe_df.columns:
        if safe_df[col].dtype == object:
            safe_df[col] = safe_df[col].map(
                lambda x: x.replace("_", r"\_") if isinstance(x, str) else x
            )
    text = safe_df.to_latex(index=False, escape=False, float_format=float_fmt.__mod__)
    out_path.write_text(text)


def write_data_dictionary(base_subject: pd.DataFrame, subject_df: pd.DataFrame, variance_channels: pd.DataFrame) -> None:
    miss = subject_df.isna().sum()
    lines = [
        "# Data Dictionary",
        "",
        "## Source files",
        f"- `results/ref_S1P_fixed_new2/data/demography.xlsx`: patient-level age and sex metadata.",
        f"- `results/ref_S1P_fixed_new2/data/anatomy_data.xlsx`: morphology sheet (8 pelvic dimensions) and SIJ kinematics sheets for 5 load cases x 2 sides.",
        f"- `results/ref_S1P_fixed_new2/data/allometry.xlsx`: true-size metrics (scale, surface, volume, mass).",
        f"- `results/ref_S1P_fixed_new2[_shape_only|_material_only]/data/sij_angles_*.zarr`, `sij_trans_*.zarr`: variance-channel decomposition inputs.",
        "",
        "## Cohort",
        f"- Subjects: {len(base_subject)}",
        f"- Sex counts: F={(base_subject['sex'] == 'F').sum()}, M={(base_subject['sex'] == 'M').sum()}",
        f"- Age range: {base_subject['age'].min():.0f}-{base_subject['age'].max():.0f} years",
        "",
        "## Load cases",
        "- SP2leg: bilateral standing load",
        "- SP1leg: unilateral standing load",
        "- LAB_phase1, LAB_phase2, LAB_phase3: parturition-motivated loading proxies",
        "",
        "## SIJ metrics (per subject x load)",
        "- `rot_mag_deg`: mean left-right 3D rotation magnitude (deg)",
        "- `trans_mag_mm`: mean left-right 3D translation magnitude (mm)",
        "- `nut_abs_deg`, `ap_rot_abs_deg`, `cc_rot_abs_deg`: absolute rotational components (deg)",
        "- `ap_trans_abs_mm`, `ml_trans_abs_mm`, `cc_trans_abs_mm`: absolute translation components (mm)",
        "- `lr_rot_asym_deg`, `lr_trans_asym_mm`: left-right asymmetry magnitudes",
        "",
        "## Size and morphometric covariates",
        "- `total_volume` (cm^3), `total_surface` (cm^2), `scale` (-) from allometry workbook",
        "- `log_total_volume`, `log_scale`: log-transformed true-size proxies",
        "- Morphology dimensions (mm unless angle): AP, BiacetabularWidth, BiischiadicWidth, BituberousWidth, IliopectinealEminenceWidth, PIT, SacralWidth, SubpubicAngle (deg)",
        "",
        "## Missingness",
    ]
    for col in [
        "age",
        "sex",
        "AP",
        "BiacetabularWidth",
        "BiischiadicWidth",
        "BituberousWidth",
        "IliopectinealEminenceWidth",
        "PIT",
        "SacralWidth",
        "SubpubicAngle",
        "scale",
        "total_volume",
        "rot_mag_deg",
        "trans_mag_mm",
    ]:
        lines.append(f"- `{col}`: {int(miss.get(col, 0))} missing values")

    lines.extend(
        [
            "",
            "## Variance-channel metrics",
            "- `shape_over_full_pct`: variance ratio (%) using shape-only model relative to full model",
            "- `material_over_full_pct`: variance ratio (%) using material-only model relative to full model",
            "- 95% CIs estimated by bootstrap resampling across subjects",
        ]
    )
    OUT_DATA_DICT.write_text("\n".join(lines) + "\n")


def plot_figure_1_loads_and_coordinates() -> None:
    from matplotlib.patches import Ellipse, FancyArrowPatch

    fig = plt.figure(figsize=(11.0, 4.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 1.3], wspace=0.28)

    # Panel A: stylized anatomy + coordinate frame
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.set_title("A. Pelvic coordinate frame and SIJ metrics", fontsize=11, loc="left")
    ax0.add_patch(Ellipse((0.32, 0.50), 0.45, 0.60, angle=15, fc="#cbd5e8", ec="#324c7a", lw=1.4))
    ax0.add_patch(Ellipse((0.68, 0.50), 0.45, 0.60, angle=-15, fc="#cbd5e8", ec="#324c7a", lw=1.4))
    ax0.add_patch(Ellipse((0.50, 0.50), 0.24, 0.42, angle=0, fc="#f1d2b8", ec="#7a4a2d", lw=1.2))

    ax0.plot([0.40, 0.42], [0.50, 0.58], "o", color="#8c2d04", ms=5)
    ax0.plot([0.60, 0.58], [0.50, 0.58], "o", color="#8c2d04", ms=5)
    ax0.text(0.18, 0.90, "SIJ rotation: |θ| (deg)", fontsize=9)
    ax0.text(0.18, 0.84, "SIJ translation: |d| (mm)", fontsize=9)

    origin = (0.50, 0.17)
    ax0.add_patch(FancyArrowPatch(origin, (0.76, 0.17), arrowstyle="-|>", mutation_scale=12, lw=1.2, color="black"))
    ax0.add_patch(FancyArrowPatch(origin, (0.50, 0.42), arrowstyle="-|>", mutation_scale=12, lw=1.2, color="black"))
    ax0.add_patch(FancyArrowPatch(origin, (0.40, 0.07), arrowstyle="-|>", mutation_scale=12, lw=1.2, color="black"))
    ax0.text(0.78, 0.17, "ML (+x)", va="center", fontsize=9)
    ax0.text(0.50, 0.44, "AP (+y)", ha="center", fontsize=9)
    ax0.text(0.38, 0.05, "CC (+z)", ha="center", fontsize=9)
    ax0.set_xlim(0.0, 1.0)
    ax0.set_ylim(0.0, 1.0)
    ax0.axis("off")

    # Panel B: load vectors (x=ML, y=AP, z=CC)
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.set_title("B. Standardized loading vectors", fontsize=11, loc="left")

    vectors = {
        "SP2leg-L": (0, 0, 400),
        "SP2leg-R": (0, 0, 400),
        "SP1leg": (0, 0, 800),
        "LAB1-L": (+400, 0, 0),
        "LAB1-R": (-400, 0, 0),
        "LAB2-L": (+400, 0, 0),
        "LAB2-R": (-400, 0, 0),
        "LAB3-pubis": (0, -400, 0),
        "LAB3-SCJ": (0, +400, 0),
    }

    y = np.arange(len(vectors))[::-1]
    ml = np.array([vectors[k][0] for k in vectors], dtype=float)
    ap = np.array([vectors[k][1] for k in vectors], dtype=float)
    cc = np.array([vectors[k][2] for k in vectors], dtype=float)

    ax1.barh(y - 0.2, ml, height=0.18, color="#2b8cbe", label="ML component")
    ax1.barh(y, ap, height=0.18, color="#a1d99b", label="AP component")
    ax1.barh(y + 0.2, cc, height=0.18, color="#fdae6b", label="CC component")

    ax1.set_yticks(y)
    ax1.set_yticklabels(list(vectors.keys()), fontsize=8)
    ax1.set_xlabel("Force component (N)")
    ax1.axvline(0, color="0.35", lw=0.8)
    ax1.legend(frameon=False, fontsize=8, ncol=3, loc="upper right")

    fig.tight_layout()
    fig.savefig(OUT_FIG_DIR / "Fig1_anatomy_coordinates_loads.pdf")
    fig.savefig(OUT_FIG_DIR / "Fig1_anatomy_coordinates_loads.png", dpi=300)
    plt.close(fig)


def plot_figure_2_primary_by_load(subject_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharex=True)

    for ax, metric, ylabel, letter in [
        (axes[0], "rot_mag_deg", "Rotation magnitude (deg)", "A"),
        (axes[1], "trans_mag_mm", "Translation magnitude (mm)", "B"),
    ]:
        summary = (
            subject_df.groupby("load_case", observed=True)[metric]
            .agg(median="median", q25=lambda x: np.percentile(x, 25), q75=lambda x: np.percentile(x, 75))
            .reindex(LOAD_ORDER)
        )
        x = np.arange(len(LOAD_ORDER), dtype=float)
        med = summary["median"].to_numpy(dtype=float)
        low = med - summary["q25"].to_numpy(dtype=float)
        high = summary["q75"].to_numpy(dtype=float) - med
        ax.errorbar(x, med, yerr=np.vstack([low, high]), fmt="o-", lw=1.4, capsize=3, color="#225ea8")
        ax.set_xticks(x)
        ax.set_xticklabels([LOAD_LABEL[l] for l in LOAD_ORDER])
        ax.set_ylabel(ylabel)
        ax.set_title(f"{letter}. {ylabel}", loc="left", fontsize=11)
        ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(OUT_FIG_DIR / "Fig2_primary_load_profiles.pdf")
    fig.savefig(OUT_FIG_DIR / "Fig2_primary_load_profiles.png", dpi=300)
    plt.close(fig)


def plot_figure_3_directional_rerouting(delta_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    metrics = ["ap_trans_abs_mm", "ml_trans_abs_mm", "cc_trans_abs_mm"]
    colors = {
        "ap_trans_abs_mm": "#2b8cbe",
        "ml_trans_abs_mm": "#238b45",
        "cc_trans_abs_mm": "#cb181d",
    }

    x = np.arange(len(LAB_LOADS) + 1)  # include SP1leg
    loads = ["SP1leg"] + LAB_LOADS
    bar_w = 0.22

    for i, m in enumerate(metrics):
        sub = delta_df.loc[delta_df["metric"] == m].set_index("load_case").reindex(loads)
        vals = sub["median_delta"].to_numpy(dtype=float)
        yerr = np.vstack(
            [
                vals - sub["ci95_low"].to_numpy(dtype=float),
                sub["ci95_high"].to_numpy(dtype=float) - vals,
            ]
        )
        ax.bar(x + (i - 1) * bar_w, vals, width=bar_w, color=colors[m], label=METRIC_LABELS[m].replace(" component", ""))
        ax.errorbar(x + (i - 1) * bar_w, vals, yerr=yerr, fmt="none", ecolor="0.2", elinewidth=0.8, capsize=2)

    ax.axhline(0, color="0.25", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([LOAD_LABEL[l] for l in loads])
    ax.set_ylabel("Median difference vs SP2leg (mm)")
    ax.set_title("Directional translation re-routing relative to SP2leg", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8, ncol=3)
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(OUT_FIG_DIR / "Fig3_directional_rerouting.pdf")
    fig.savefig(OUT_FIG_DIR / "Fig3_directional_rerouting.png", dpi=300)
    plt.close(fig)


def plot_figure_4_sex_effects(sex_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.6), sharey=True)

    for ax, outcome, title in [
        (axes[0], "rot_mag_deg", "A. Rotation magnitude"),
        (axes[1], "trans_mag_mm", "B. Translation magnitude"),
    ]:
        sub = sex_df.loc[sex_df["outcome"] == outcome].set_index("load_case").reindex(LAB_LOADS)
        y = np.arange(len(LAB_LOADS))
        beta = sub["beta_female_minus_male"].to_numpy(dtype=float)
        low = beta - sub["ci95_low"].to_numpy(dtype=float)
        high = sub["ci95_high"].to_numpy(dtype=float) - beta

        ax.errorbar(beta, y, xerr=np.vstack([low, high]), fmt="o", color="#7a0177", ecolor="#7a0177", capsize=3)
        ax.axvline(0, color="0.3", lw=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels([LOAD_LABEL[l] for l in LAB_LOADS])
        ax.invert_yaxis()
        ax.set_title(title, loc="left", fontsize=11)
        ax.set_xlabel("Adjusted female-male effect")
        ax.grid(axis="x", alpha=0.25)

    fig.tight_layout()
    fig.savefig(OUT_FIG_DIR / "Fig4_sex_effects_lab.pdf")
    fig.savefig(OUT_FIG_DIR / "Fig4_sex_effects_lab.png", dpi=300)
    plt.close(fig)


def plot_figure_5_variance_channels(variance_df: pd.DataFrame) -> None:
    metrics = ["rot_mag_deg", "trans_mag_mm", "ap_trans_abs_mm", "ml_trans_abs_mm", "cc_trans_abs_mm"]
    row_labels = [METRIC_LABELS[m].replace(" component", "") for m in metrics]

    shape_mat = np.vstack(
        [
            variance_df.loc[variance_df["metric"] == m]
            .set_index("load_case")
            .reindex(LOAD_ORDER)["shape_over_full_pct"]
            .to_numpy(dtype=float)
            for m in metrics
        ]
    )
    material_mat = np.vstack(
        [
            variance_df.loc[variance_df["metric"] == m]
            .set_index("load_case")
            .reindex(LOAD_ORDER)["material_over_full_pct"]
            .to_numpy(dtype=float)
            for m in metrics
        ]
    )

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2), constrained_layout=True)

    im0 = axes[0].imshow(shape_mat, aspect="auto", cmap="YlGnBu", vmin=np.nanmin(shape_mat), vmax=np.nanmax(shape_mat))
    axes[0].set_title("A. Shape-only / full variance (%)", loc="left", fontsize=11)
    axes[0].set_xticks(np.arange(len(LOAD_ORDER)))
    axes[0].set_xticklabels([LOAD_LABEL[l] for l in LOAD_ORDER])
    axes[0].set_yticks(np.arange(len(metrics)))
    axes[0].set_yticklabels(row_labels, fontsize=8)
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(material_mat, aspect="auto", cmap="OrRd", vmin=np.nanmin(material_mat), vmax=np.nanmax(material_mat))
    axes[1].set_title("B. Material-only / full variance (%)", loc="left", fontsize=11)
    axes[1].set_xticks(np.arange(len(LOAD_ORDER)))
    axes[1].set_xticklabels([LOAD_LABEL[l] for l in LOAD_ORDER])
    axes[1].set_yticks(np.arange(len(metrics)))
    axes[1].set_yticklabels(row_labels, fontsize=8)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    fig.savefig(OUT_FIG_DIR / "Fig5_variance_channels.pdf")
    fig.savefig(OUT_FIG_DIR / "Fig5_variance_channels.png", dpi=300)
    plt.close(fig)


def plot_figure_6_allometry(subject_df: pd.DataFrame, models: dict[tuple[str, str], object]) -> None:
    fig, axes = plt.subplots(2, len(LOAD_ORDER), figsize=(15.0, 6.5), sharex=True)
    colors = {0.0: "#08519c", 1.0: "#cb181d"}
    sex_names = {0.0: "Male", 1.0: "Female"}

    for i_row, outcome in enumerate(PRIMARY_OUTCOMES):
        for i_col, load in enumerate(LOAD_ORDER):
            ax = axes[i_row, i_col]
            sub = subject_df.loc[subject_df["load_case"] == load].copy()
            model = models[(load, outcome)]

            for sex_f in [0.0, 1.0]:
                ss = sub.loc[sub["sex_F"] == sex_f]
                ax.scatter(
                    ss["total_volume"],
                    ss[outcome],
                    s=11,
                    alpha=0.35,
                    color=colors[sex_f],
                    label=sex_names[sex_f] if (i_row == 0 and i_col == 0) else None,
                )

                x = np.linspace(sub["total_volume"].min(), sub["total_volume"].max(), 120)
                pred_df = pd.DataFrame(
                    {
                        "log_total_volume": np.log(x),
                        "sex_F": sex_f,
                        "age_z": 0.0,
                        "AP_z": 0.0,
                        "BiischiadicWidth_z": 0.0,
                        "SubpubicAngle_z": 0.0,
                    }
                )
                y_hat = np.exp(model.predict(pred_df))
                ax.plot(x, y_hat, color=colors[sex_f], lw=1.6)

            ax.set_xscale("log")
            ax.set_yscale("log")
            if i_row == 0:
                ax.set_title(LOAD_LABEL[load], fontsize=9)
            if i_col == 0:
                ax.set_ylabel(METRIC_LABELS[outcome], fontsize=9)
            if i_row == 1:
                ax.set_xlabel("Total pelvic volume (cm$^3$)", fontsize=8)
            ax.grid(alpha=0.18)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle("Allometry of SIJ micromotion (log-log; sex-specific fits)", y=0.995, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT_FIG_DIR / "Fig6_allometry_loglog.pdf")
    fig.savefig(OUT_FIG_DIR / "Fig6_allometry_loglog.png", dpi=300)
    plt.close(fig)


def save_latex_tables(
    table1: pd.DataFrame,
    table2: pd.DataFrame,
    table3: pd.DataFrame,
    pooled_models: pd.DataFrame,
    allometry: pd.DataFrame,
    directional: pd.DataFrame,
    variance_channels: pd.DataFrame,
    sex_models: pd.DataFrame,
    allometry_scale: pd.DataFrame,
) -> None:
    # Table 1
    write_latex_table(table1, OUT_TABLE_DIR / "table1_cohort.tex")
    # Table 2
    write_latex_table(table2, OUT_TABLE_DIR / "table2_loads.tex")
    # Table 3
    write_latex_table(table3, OUT_TABLE_DIR / "table3_micromotion_summary.tex")

    # Table 4 (focused allometry + key pooled terms)
    keep_terms = pooled_models["term"].str.contains(
        r"C\(load_case|sex_F|log_total_volume_z|age_z|AP_z|BiischiadicWidth_z|SubpubicAngle_z",
        regex=True,
    )
    pooled_small = pooled_models.loc[keep_terms].copy()
    pooled_small = pooled_small[
        [
            "outcome",
            "term",
            "beta",
            "ci95_low",
            "ci95_high",
            "beta_std",
            "partial_r2",
            "p_fdr_bh",
        ]
    ]

    allometry_small = allometry[
        [
            "load_case",
            "outcome",
            "male_exponent",
            "male_ci95_low",
            "male_ci95_high",
            "female_exponent",
            "female_ci95_low",
            "female_ci95_high",
            "sex_slope_interaction_beta",
            "interaction_p_fdr_bh",
            "adj_r2",
        ]
    ]

    write_latex_table(pooled_small, OUT_TABLE_DIR / "table4a_primary_models.tex")
    write_latex_table(allometry_small, OUT_TABLE_DIR / "table4b_allometry.tex")

    # Supplementary tables
    write_latex_table(directional, OUT_TABLE_DIR / "supp_directional_rerouting.tex")
    write_latex_table(variance_channels, OUT_TABLE_DIR / "supp_variance_channels.tex")
    write_latex_table(sex_models, OUT_TABLE_DIR / "supp_sex_models.tex")
    write_latex_table(allometry_scale, OUT_TABLE_DIR / "supp_allometry_scale.tex")


def main() -> None:
    ensure_dirs()
    sns.set_theme(style="whitegrid", context="paper")

    demography, morphology, allometry_raw = load_core_data()
    base_subject = build_base_subject_table(demography, morphology, allometry_raw)
    subject_df = build_subject_level(base_subject)

    # Tables and statistics
    table1 = create_table_1_cohort(base_subject)
    table2 = create_table_2_load_definitions()
    table3 = create_table_3_micromotion_summary(subject_df)
    directional = compute_directional_rerouting(subject_df)

    pooled_frames = []
    for outcome in PRIMARY_OUTCOMES:
        frame, _ = fit_cluster_robust_model(subject_df, outcome)
        pooled_frames.append(frame)
    pooled_models = pd.concat(pooled_frames, ignore_index=True)

    sex_models = fit_sex_models_lab(subject_df)
    allometry, allometry_model_objects = fit_allometry_models(subject_df)
    allometry_scale = fit_allometry_models_scale(subject_df)
    variance_channels = compute_variance_channels()

    # Persist machine-readable outputs
    base_subject.to_csv(OUT_TABLE_DIR / "subject_base.csv", index=False)
    subject_df.to_csv(OUT_TABLE_DIR / "subject_level.csv", index=False)
    table1.to_csv(OUT_TABLE_DIR / "table1_cohort.csv", index=False)
    table2.to_csv(OUT_TABLE_DIR / "table2_loads.csv", index=False)
    table3.to_csv(OUT_TABLE_DIR / "table3_micromotion_summary.csv", index=False)
    directional.to_csv(OUT_TABLE_DIR / "directional_rerouting.csv", index=False)
    pooled_models.to_csv(OUT_TABLE_DIR / "primary_models_cluster_robust.csv", index=False)
    sex_models.to_csv(OUT_TABLE_DIR / "sex_models_lab.csv", index=False)
    allometry.to_csv(OUT_TABLE_DIR / "allometry_models.csv", index=False)
    allometry_scale.to_csv(OUT_TABLE_DIR / "allometry_models_scale.csv", index=False)
    variance_channels.to_csv(OUT_TABLE_DIR / "variance_channels.csv", index=False)

    save_latex_tables(
        table1,
        table2,
        table3,
        pooled_models,
        allometry,
        directional,
        variance_channels,
        sex_models,
        allometry_scale,
    )
    write_data_dictionary(base_subject, subject_df, variance_channels)

    # Figures
    plot_figure_1_loads_and_coordinates()
    plot_figure_2_primary_by_load(subject_df)
    plot_figure_3_directional_rerouting(directional)
    plot_figure_4_sex_effects(sex_models)
    plot_figure_5_variance_channels(variance_channels)
    plot_figure_6_allometry(subject_df, allometry_model_objects)

    # concise summary for manuscript writing
    summary_payload = {
        "n_subjects": int(base_subject.shape[0]),
        "sex_counts": {
            "F": int((base_subject["sex"] == "F").sum()),
            "M": int((base_subject["sex"] == "M").sum()),
        },
        "age": {
            "mean": float(base_subject["age"].mean()),
            "sd": float(base_subject["age"].std(ddof=1)),
            "min": float(base_subject["age"].min()),
            "max": float(base_subject["age"].max()),
        },
        "rot_trans_medians": table3.to_dict(orient="records"),
    }
    OUT_RESULTS_JSON.write_text(json.dumps(summary_payload, indent=2))


if __name__ == "__main__":
    main()
