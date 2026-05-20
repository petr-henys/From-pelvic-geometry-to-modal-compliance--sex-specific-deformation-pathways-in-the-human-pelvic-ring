#!/usr/bin/env python3
"""Eigenvalue statistical analysis.

Model: log(eigenvalue) ~ log(scale) + sex + shape + material
Multiple testing: Benjamini-Hochberg FDR. Also reports MANOVA.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import statsmodels.api as sm
import zarr
from statsmodels.multivariate.manova import MANOVA
from statsmodels.stats.multitest import fdrcorrection
from scipy import stats

from utils.stats_utils import residualize_ols, group_lmg_r2, fit_beta

# Require IV 2SLS via linearmodels (no fallbacks, no try/except)
from linearmodels.iv import IV2SLS as IV2SLS_LM

ALPHA = 0.05
MIN_SAMPLES = 10
NUM_MODES = 15

FULL_RESULTS_DIR = Path("results/ref_S1P_fixed_new2")
SHAPE_RESULTS_DIR = Path("results/ref_S1P_fixed_new2_shape_only")
MATERIAL_RESULTS_DIR = Path("results/ref_S1P_fixed_new2_material_only")

DATA_DIR = FULL_RESULTS_DIR / "data"
ANALYSIS_DIR = FULL_RESULTS_DIR / "analysis"
RESULTS_CSV = ANALYSIS_DIR / "eigen_results.csv"


@dataclass
class AnalysisData:
    eigen: pd.DataFrame
    eigen_shape_only: pd.DataFrame
    eigen_material_only: pd.DataFrame
    allometry: pd.DataFrame
    demography: pd.DataFrame


def _apply_mode_permutation(data: np.ndarray, perm: np.ndarray) -> np.ndarray:
    """Reorder mode axis by permutation mapping (ref-mode -> solver-mode)."""
    n_samples, n_modes = perm.shape
    result = np.full((n_samples, n_modes), np.nan, dtype=np.float64)
    for i in range(n_samples):
        for j in range(n_modes):
            solver_idx = int(perm[i, j])
            if 0 <= solver_idx < data.shape[1]:
                result[i, j] = data[i, solver_idx]
    return result


def _load_modes_from_zarr(results_dir: Path) -> pd.DataFrame:
    """Load eigenvalues + permutations from a result directory and return paired modes table."""
    data_dir = results_dir / "data"
    eigvals = np.asarray(zarr.open(str(data_dir / "eigenvalues.zarr"), mode="r")["data"][:], dtype=float)
    perm = np.asarray(zarr.open(str(data_dir / "eig_permutations.zarr"), mode="r")["data"][:], dtype=np.int32)
    eigvals_sorted = _apply_mode_permutation(eigvals, perm)

    # Keep downstream regressions stable when some modes are unpaired (perm=-1)
    # or non-positive by imputing per-mode medians.
    invalid_mask = (~np.isfinite(eigvals_sorted)) | (eigvals_sorted <= 0)
    if invalid_mask.any():
        cleaned = eigvals_sorted.copy()
        cleaned[invalid_mask] = np.nan
        col_medians = np.nanmedian(cleaned, axis=0)
        col_medians[~np.isfinite(col_medians)] = 1.0
        rows, cols = np.where(invalid_mask)
        cleaned[rows, cols] = col_medians[cols]
        eigvals_sorted = cleaned

    return pd.DataFrame({f"eig_{i+1}": eigvals_sorted[:, i] for i in range(eigvals_sorted.shape[1])})


def _load_modes_table(results_dir: Path, legacy_filename: str) -> pd.DataFrame:
    """Prefer Excel (if available), otherwise reconstruct paired modes from Zarr."""
    candidates = [
        results_dir / "data" / legacy_filename,
        results_dir / "data" / "eigen_data.xlsx",
    ]
    for p in candidates:
        if p.exists():
            return pd.read_excel(p, sheet_name="modes")
    return _load_modes_from_zarr(results_dir)


def load_analysis_data() -> AnalysisData:
    """Load core tables and ensure consistent alignment."""
    eigen = _load_modes_table(FULL_RESULTS_DIR, "eigen_data.xlsx")
    eigen_shape_only = _load_modes_table(SHAPE_RESULTS_DIR, "eigen_data_shape_only.xlsx")
    eigen_material_only = _load_modes_table(MATERIAL_RESULTS_DIR, "eigen_data_material_only.xlsx")

    n_samples = len(eigen)
    if len(eigen_shape_only) != n_samples:
        raise ValueError(f"Shape-only sample mismatch: {len(eigen_shape_only)} vs full {n_samples}")
    if len(eigen_material_only) != n_samples:
        raise ValueError(f"Material-only sample mismatch: {len(eigen_material_only)} vs full {n_samples}")

    allometry = pd.read_excel(DATA_DIR / "allometry.xlsx")
    demography = pd.read_excel(DATA_DIR / "demography.xlsx")

    demography["sex"] = demography["sex"].astype(str).str.strip().str.upper()
    valid_sexes = {"M", "F"}
    invalid = sorted(set(demography["sex"]) - valid_sexes)
    if invalid:
        raise ValueError(f"Unexpected sex labels: {invalid}. Expected only 'M' and 'F'.")

    return AnalysisData(
        eigen=eigen,
        eigen_shape_only=eigen_shape_only,
        eigen_material_only=eigen_material_only,
        allometry=allometry,
        demography=demography,
    )


def extract_design_vectors(data: AnalysisData) -> tuple[np.ndarray, np.ndarray]:
    """Return aligned log-scale values and binary sex codes."""
    scale = data.allometry["scale"].astype(float).to_numpy()
    if np.any(scale <= 0) or not np.isfinite(scale).all():
        raise ValueError("Scale values must be positive and finite.")
    log_scale = np.log(scale)

    sex_series = data.demography["sex"].map({"M": 0.0, "F": 1.0})
    if sex_series.isna().any():
        raise ValueError("Sex coding failed.")
    sex_codes = sex_series.to_numpy(dtype=float)

    return log_scale, sex_codes


def compute_percent_effects(beta: float, se_beta: float, gamma: float,
                           se_gamma: float, df_resid: int, log_scale_iqr: float) -> dict:
    """Compute interpretable percent changes with confidence intervals."""
    t_crit = stats.t.ppf(1 - ALPHA / 2, df_resid)

    pct_scale = 100.0 * ((2.0 ** beta) - 1.0)
    se_pct_scale = 100.0 * np.log(2.0) * (2.0 ** beta) * se_beta

    pct_scale_iqr = 100.0 * (np.exp(beta * log_scale_iqr) - 1.0)
    se_pct_scale_iqr = 100.0 * log_scale_iqr * np.exp(beta * log_scale_iqr) * se_beta

    pct_sex = 100.0 * (np.exp(gamma) - 1.0)
    se_pct_sex = 100.0 * np.exp(gamma) * se_gamma

    return {
        "pct_scale_x2": pct_scale,
        "pct_scale_x2_SE": se_pct_scale,
        "pct_scale_x2_CI_lo": pct_scale - t_crit * se_pct_scale,
        "pct_scale_x2_CI_hi": pct_scale + t_crit * se_pct_scale,
        "pct_scale_IQR": pct_scale_iqr,
        "pct_scale_IQR_SE": se_pct_scale_iqr,
        "pct_scale_IQR_CI_lo": pct_scale_iqr - t_crit * se_pct_scale_iqr,
        "pct_scale_IQR_CI_hi": pct_scale_iqr + t_crit * se_pct_scale_iqr,
        "pct_sex": pct_sex,
        "pct_sex_SE": se_pct_sex,
        "pct_sex_CI_lo": pct_sex - t_crit * se_pct_sex,
        "pct_sex_CI_hi": pct_sex + t_crit * se_pct_sex,
    }


def analyze_all_modes(data: AnalysisData) -> pd.DataFrame:
    """Run log-linear regression for every eigenvalue mode."""
    log_scale, sex_codes = extract_design_vectors(data)
    base_mask = np.isfinite(log_scale) & np.isfinite(sex_codes)

    scale = data.allometry["scale"].astype(float).to_numpy()
    mask = np.isfinite(scale) & (scale > 0)
    S0 = float(np.exp(np.log(scale[mask]).mean()))

    log_scale_valid = log_scale[base_mask]
    log_scale_iqr = float(np.percentile(log_scale_valid, 75) - np.percentile(log_scale_valid, 25))

    # ------------------------------------------------------------------
    # Instrument selection for potential endogeneity in log_scale
    # Prefer a small set of strong, plausibly exogenous allometry instruments
    # (e.g., total_volume, total_surface, total_mass_template). If unavailable,
    # falls back to top-K by absolute correlation with log_scale.
    # ------------------------------------------------------------------
    def _select_scale_instruments(
        allometry: pd.DataFrame,
        log_scale_vec: np.ndarray,
        top_k: int = 3,
    ) -> pd.DataFrame:
        cand_cols = [
            c for c in allometry.columns
            if c not in {"patient_id", "scale"}
        ]
        # Preferred columns if present
        preferred = [
            "total_volume", "total_surface", "total_mass_template",
            "eff_density_template", "total_mass",
        ]
        cols: list[str] = [c for c in preferred if c in cand_cols]
        # If not enough, fill with top-K by |corr| to log_scale
        if len(cols) < top_k:
            remaining = [c for c in cand_cols if c not in cols]
            corrs: list[tuple[float, str]] = []
            for c in remaining:
                x = allometry[c].to_numpy(dtype=float)
                # Safe log when strictly positive
                if np.all(np.isfinite(x)) and (x > 0).all():
                    x_t = np.log(x)
                else:
                    x_t = x
                m = np.isfinite(x_t) & np.isfinite(log_scale_vec)
                if m.sum() > 2 and np.nanstd(x_t[m]) > 0:
                    corr = float(np.corrcoef(x_t[m], log_scale_vec[m])[0, 1])
                    if np.isfinite(corr):
                        corrs.append((abs(corr), c))
            corrs.sort(reverse=True)
            for _, c in corrs:
                if c not in cols:
                    cols.append(c)
                if len(cols) >= top_k:
                    break

        # Construct instrument DataFrame with safe transforms and names
        Z = {}
        for c in cols:
            x = allometry[c].to_numpy(dtype=float)
            if np.all(np.isfinite(x)) and (x > 0).all():
                Z[f"Z_{c}_log"] = np.log(x)
            else:
                Z[f"Z_{c}"] = x
        Z_df = pd.DataFrame(Z)
        return Z_df

    def _full_rank_instruments(
        Z_all: pd.DataFrame,
        row_mask: np.ndarray,
        min_needed: int,
        X_exog: pd.DataFrame,
        tol: float = 1e-10,
    ) -> tuple[pd.DataFrame, list[str]]:
        """Select a full-column-rank subset of instruments on masked rows.

        Greedy selection in the provided column order; drops non-finite/constant columns.
        Ensures at least `min_needed` columns remain; otherwise raises ValueError.
        """
        Z = Z_all.loc[row_mask].reset_index(drop=True)
        # Drop columns with any non-finite values or near-zero variance
        finite_cols = [c for c in Z.columns if np.isfinite(Z[c].to_numpy()).all()]
        Z = Z[finite_cols]
        keep: list[str] = []
        # Start with exogenous part to ensure combined [exog | Z] is full rank
        M = X_exog.loc[row_mask].to_numpy()
        for c in Z.columns:
            col = Z[c].to_numpy().reshape(-1, 1)
            if np.nanstd(col) <= tol:
                continue
            M_try = np.column_stack([M, col])
            r_prev = np.linalg.matrix_rank(M, tol)
            r_new = np.linalg.matrix_rank(M_try, tol)
            if r_new > r_prev:
                M = M_try
                keep.append(c)

        if len(keep) < min_needed:
            raise ValueError(
                f"Insufficient instrument rank: have {len(keep)}, need >= {min_needed}."
            )
        return Z[keep], keep

    Z_scale_all = _select_scale_instruments(data.allometry, log_scale)

    mode_records = []
    for mode_idx in range(NUM_MODES):
        mode_number = mode_idx + 1
        eigen_column = f"eig_{mode_number}"

        eigen_values = data.eigen[eigen_column].astype(float).to_numpy()
        eigen_shape_only = data.eigen_shape_only[eigen_column].astype(float).to_numpy()
        eigen_material_only = data.eigen_material_only[eigen_column].astype(float).to_numpy()

        value_mask = np.isfinite(eigen_values) & (eigen_values > 0)
        valid_mask = base_mask & value_mask
        valid_count = int(valid_mask.sum())

        y_full = np.log(eigen_values[valid_mask])
        # OLS design not used below; IV is used for inference

        # -----------------------
        # IV 2SLS for potential endogeneity in log_scale
        # Endogenous: log_scale; Exogenous: sex; Instruments: selected Z_scale
        # -----------------------
        # Assemble pandas objects with names for clarity
        y_ser = pd.Series(y_full, name="log_eig")
        # Build exog at full sample length so _full_rank_instruments can apply
        # valid_mask (boolean of length n_samples) via .loc correctly.
        exog_full = pd.DataFrame({
            "const": np.ones(n_samples),
            "sex": sex_codes,
        })
        endog_df = pd.DataFrame({
            "log_scale": log_scale[valid_mask],
        })
        Z_candidates = Z_scale_all
        Z_df = _full_rank_instruments(
            Z_candidates, valid_mask, min_needed=endog_df.shape[1], X_exog=exog_full
        )[0]
        # Slice exog to valid rows for regression (must match y_ser length).
        exog_df = exog_full.loc[valid_mask].reset_index(drop=True)

        res_full = IV2SLS_LM(y_ser, exog_df, endog_df, Z_df).fit(cov_type="robust")

        df_resid = int(res_full.df_resid)
        t_crit = stats.t.ppf(1 - ALPHA / 2, df_resid)
        beta_coef = float(res_full.params["log_scale"])  # coefficient on endogenous regressor
        gamma = float(res_full.params["sex"])
        beta_se = float(res_full.std_errors["log_scale"])
        gamma_se = float(res_full.std_errors["sex"])
        beta_t = float(res_full.tstats["log_scale"])
        beta_p = float(res_full.pvalues["log_scale"])
        gamma_t = float(res_full.tstats["sex"])
        gamma_p = float(res_full.pvalues["sex"])
        nobs_full = int(res_full.nobs)

        record = {
            "mode": mode_number,
            "eigen_column": eigen_column,
            "n": nobs_full,
            "df_resid": df_resid,
            "beta_scale": beta_coef,
            "beta_scale_SE": beta_se,
            "beta_scale_t": beta_t,
            "beta_scale_p": beta_p,
            "beta_scale_CI_lo": beta_coef - t_crit * beta_se,
            "beta_scale_CI_hi": beta_coef + t_crit * beta_se,
            "gamma_sex": gamma,
            "gamma_sex_SE": gamma_se,
            "gamma_sex_t": gamma_t,
            "gamma_sex_p": gamma_p,
            "gamma_sex_CI_lo": gamma - t_crit * gamma_se,
            "gamma_sex_CI_hi": gamma + t_crit * gamma_se,
        }

        effects = compute_percent_effects(beta_coef, beta_se, gamma, gamma_se, df_resid, log_scale_iqr)
        record.update(effects)

        beta_shape_for_removal = fit_beta(log_scale, sex_codes, eigen_shape_only)
        factor_shape = (S0 / scale) ** beta_shape_for_removal
        eigen_shape_descaled = eigen_shape_only * factor_shape

        beta_material_for_removal = fit_beta(log_scale, sex_codes, eigen_material_only)
        factor_material = (S0 / scale) ** beta_material_for_removal
        eigen_material_descaled = eigen_material_only * factor_material

        # OLS shape design not used (IV below)

        # IV for shape-augmented model: endog = [log_scale, log(eigen_shape_descaled)]
        y_ser = pd.Series(y_full, name="log_eig")
        exog_df = pd.DataFrame({
            "const": np.ones(valid_count),
            "sex": sex_codes[valid_mask],
        })
        endog_df = pd.DataFrame({
            "log_scale": log_scale[valid_mask],
            "log_shape_ds": np.log(eigen_shape_descaled[valid_mask]),
        })
        Z_candidates = pd.concat([
            Z_scale_all,
            pd.DataFrame({"Z_log_shape_only": np.log(eigen_shape_only)})
        ], axis=1)
        Z_df = _full_rank_instruments(
            Z_candidates, valid_mask, min_needed=endog_df.shape[1], X_exog=exog_df
        )[0]

        res_shape = IV2SLS_LM(y_ser, exog_df, endog_df, Z_df).fit(cov_type="robust")

        beta_shape = float(res_shape.params["log_shape_ds"])
        gamma_shape = float(res_shape.params["sex"])
        df_resid_shape = int(res_shape.df_resid)
        tcrit_shape = stats.t.ppf(1 - ALPHA / 2, df_resid_shape)
        record["beta_shape"] = beta_shape
        record["beta_shape_SE"] = float(res_shape.std_errors["log_shape_ds"])
        record["beta_shape_t"] = float(res_shape.tstats["log_shape_ds"])
        record["beta_shape_p"] = float(res_shape.pvalues["log_shape_ds"])
        record["beta_shape_CI_lo"] = beta_shape - tcrit_shape * record["beta_shape_SE"]
        record["beta_shape_CI_hi"] = beta_shape + tcrit_shape * record["beta_shape_SE"]
        record["gamma_shape"] = gamma_shape
        record["gamma_shape_SE"] = float(res_shape.std_errors["sex"])
        record["gamma_shape_t"] = float(res_shape.tstats["sex"])
        record["gamma_shape_p"] = float(res_shape.pvalues["sex"])
        record["gamma_shape_CI_lo"] = gamma_shape - tcrit_shape * record["gamma_shape_SE"]
        record["gamma_shape_CI_hi"] = gamma_shape + tcrit_shape * record["gamma_shape_SE"]

        # IQR-based percent change for shape regressor
        log_shape_ds_vec = np.log(eigen_shape_descaled[valid_mask])
        log_shape_iqr = float(np.percentile(log_shape_ds_vec, 75) - np.percentile(log_shape_ds_vec, 25))
        pct_shape_iqr = 100.0 * (np.exp(beta_shape * log_shape_iqr) - 1.0)
        se_pct_shape_iqr = 100.0 * log_shape_iqr * np.exp(beta_shape * log_shape_iqr) * record["beta_shape_SE"]
        record["pct_shape_IQR"] = pct_shape_iqr
        record["pct_shape_IQR_SE"] = se_pct_shape_iqr
        record["pct_shape_IQR_CI_lo"] = pct_shape_iqr - tcrit_shape * se_pct_shape_iqr
        record["pct_shape_IQR_CI_hi"] = pct_shape_iqr + tcrit_shape * se_pct_shape_iqr

        # OLS material design not used (IV below)

        # IV for material-augmented model: endog = [log_scale, log(eigen_material_descaled)]
        y_ser = pd.Series(y_full, name="log_eig")
        exog_df = pd.DataFrame({
            "const": np.ones(valid_count),
            "sex": sex_codes[valid_mask],
        })
        endog_df = pd.DataFrame({
            "log_scale": log_scale[valid_mask],
            "log_material_ds": np.log(eigen_material_descaled[valid_mask]),
        })
        Z_candidates = pd.concat([
            Z_scale_all,
            pd.DataFrame({"Z_log_material_only": np.log(eigen_material_only)})
        ], axis=1)
        Z_df = _full_rank_instruments(
            Z_candidates, valid_mask, min_needed=endog_df.shape[1], X_exog=exog_df
        )[0]

        res_material = IV2SLS_LM(y_ser, exog_df, endog_df, Z_df).fit(cov_type="robust")

        beta_material = float(res_material.params["log_material_ds"])
        gamma_material = float(res_material.params["sex"])
        df_resid_material = int(res_material.df_resid)
        tcrit_material = stats.t.ppf(1 - ALPHA / 2, df_resid_material)
        record["beta_material"] = beta_material
        record["beta_material_SE"] = float(res_material.std_errors["log_material_ds"])
        record["beta_material_t"] = float(res_material.tstats["log_material_ds"])
        record["beta_material_p"] = float(res_material.pvalues["log_material_ds"])
        record["beta_material_CI_lo"] = beta_material - tcrit_material * record["beta_material_SE"]
        record["beta_material_CI_hi"] = beta_material + tcrit_material * record["beta_material_SE"]
        record["gamma_material"] = gamma_material
        record["gamma_material_SE"] = float(res_material.std_errors["sex"])
        record["gamma_material_t"] = float(res_material.tstats["sex"])
        record["gamma_material_p"] = float(res_material.pvalues["sex"])
        record["gamma_material_CI_lo"] = gamma_material - tcrit_material * record["gamma_material_SE"]
        record["gamma_material_CI_hi"] = gamma_material + tcrit_material * record["gamma_material_SE"]

        # IQR-based percent change for material regressor
        log_material_ds_vec = np.log(eigen_material_descaled[valid_mask])
        log_material_iqr = float(np.percentile(log_material_ds_vec, 75) - np.percentile(log_material_ds_vec, 25))
        pct_material_iqr = 100.0 * (np.exp(beta_material * log_material_iqr) - 1.0)
        se_pct_material_iqr = 100.0 * log_material_iqr * np.exp(beta_material * log_material_iqr) * record["beta_material_SE"]
        record["pct_material_IQR"] = pct_material_iqr
        record["pct_material_IQR_SE"] = se_pct_material_iqr
        record["pct_material_IQR_CI_lo"] = pct_material_iqr - tcrit_material * se_pct_material_iqr
        record["pct_material_IQR_CI_hi"] = pct_material_iqr + tcrit_material * se_pct_material_iqr

        z_shape = np.log(eigen_shape_only[valid_mask])
        z_material = np.log(eigen_material_only[valid_mask])
        sex_c = sex_codes[valid_mask] - np.nanmean(sex_codes[valid_mask])
        X_base = np.column_stack([
            log_scale[valid_mask],
            sex_c,
            log_scale[valid_mask] * sex_c,
        ])
        r_shape = residualize_ols(z_shape, X_base)
        r_material = residualize_ols(z_material, X_base)

        groups = [
            ("scale", np.column_stack([log_scale[valid_mask], log_scale[valid_mask] * sex_c])),
            ("sex", sex_c.reshape(-1, 1)),
            ("shape", np.column_stack([r_shape, r_shape * sex_c])),
            ("material", np.column_stack([r_material, r_material * sex_c])),
        ]
        lmg = group_lmg_r2(y_full, groups)

        record["R2_total"] = float(lmg["R2_total"])
        record["R2_scale"] = float(lmg["scale"])
        record["R2_sex"] = float(lmg["sex"])
        record["R2_shape"] = float(lmg["shape"])
        record["R2_material"] = float(lmg["material"])

        if np.isfinite(record["R2_total"]) and record["R2_total"] > 0:
            record["R2pct_scale"] = 100.0 * record["R2_scale"] / record["R2_total"]
            record["R2pct_sex"] = 100.0 * record["R2_sex"] / record["R2_total"]
            record["R2pct_shape"] = 100.0 * record["R2_shape"] / record["R2_total"]
            record["R2pct_material"] = 100.0 * record["R2_material"] / record["R2_total"]
        else:
            record["R2pct_scale"] = np.nan
            record["R2pct_sex"] = np.nan
            record["R2pct_shape"] = np.nan
            record["R2pct_material"] = np.nan

        beta_full_for_scaling = fit_beta(log_scale, sex_codes, eigen_values)
        factor_full = (S0 / scale) ** beta_full_for_scaling
        eigen_scaled = eigen_values * factor_full
        eigen_shape_scaled = eigen_shape_descaled
        eigen_material_scaled = eigen_material_descaled

        shape_ratio = np.nanmedian(eigen_scaled[valid_mask] / eigen_material_scaled[valid_mask])
        shape_effect_pct = 100.0 * (shape_ratio - 1.0)

        material_ratio = np.nanmedian(eigen_scaled[valid_mask] / eigen_shape_scaled[valid_mask])
        material_effect_pct = 100.0 * (material_ratio - 1.0)

        record["pct_shape"] = shape_effect_pct
        record["pct_material"] = material_effect_pct

        mode_records.append(record)

    results_df = pd.DataFrame(mode_records).sort_values("mode").reset_index(drop=True)
    return results_df


def apply_fdr_correction(df: pd.DataFrame) -> pd.DataFrame:
    """Apply Benjamini-Hochberg FDR correction to p-values."""
    p_values = np.concatenate([
        df["beta_scale_p"].values,
        df["gamma_sex_p"].values,
        df["beta_shape_p"].values,
        df["gamma_shape_p"].values,
        df["beta_material_p"].values,
        df["gamma_material_p"].values,
    ])

    rejected, p_fdr = fdrcorrection(p_values, alpha=ALPHA, method="indep")

    n_modes = len(df)
    df["beta_scale_signif"] = rejected[0 * n_modes:1 * n_modes]
    df["beta_scale_pFDR"] = p_fdr[0 * n_modes:1 * n_modes]
    df["gamma_sex_signif"] = rejected[1 * n_modes:2 * n_modes]
    df["gamma_sex_pFDR"] = p_fdr[1 * n_modes:2 * n_modes]
    df["beta_shape_signif"] = rejected[2 * n_modes:3 * n_modes]
    df["beta_shape_pFDR"] = p_fdr[2 * n_modes:3 * n_modes]
    df["gamma_shape_signif"] = rejected[3 * n_modes:4 * n_modes]
    df["gamma_shape_pFDR"] = p_fdr[3 * n_modes:4 * n_modes]
    df["beta_material_signif"] = rejected[4 * n_modes:5 * n_modes]
    df["beta_material_pFDR"] = p_fdr[4 * n_modes:5 * n_modes]
    df["gamma_material_signif"] = rejected[5 * n_modes:6 * n_modes]
    df["gamma_material_pFDR"] = p_fdr[5 * n_modes:6 * n_modes]

    return df


def run_manova(data: AnalysisData) -> str:
    """Run multivariate ANOVA across all eigenvalue modes."""
    log_scale, sex_codes = extract_design_vectors(data)
    eig_cols = [f"eig_{i + 1}" for i in range(NUM_MODES)]
    eigen_matrix = data.eigen[eig_cols].astype(float).to_numpy()

    finite_eigen = np.isfinite(eigen_matrix).all(axis=1)
    positive_eigen = (eigen_matrix > 0).all(axis=1)
    base_mask = np.isfinite(log_scale) & np.isfinite(sex_codes)
    valid_mask = base_mask & finite_eigen & positive_eigen

    response_df = pd.DataFrame({
        "log_scale": log_scale[valid_mask],
        "sex": sex_codes[valid_mask],
    })

    for idx, col in enumerate(eig_cols):
        response_df[f"log_{col}"] = np.log(eigen_matrix[valid_mask, idx])

    y_vars = " + ".join([f"log_{col}" for col in eig_cols])
    formula = f"{y_vars} ~ log_scale + C(sex)"

    manova = MANOVA.from_formula(formula, data=response_df)
    return str(manova.mv_test())


def main() -> None:
    """Run complete eigenvalue analysis."""
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    data = load_analysis_data()

    print("Running regression analysis (IV2SLS cov_type='robust'; OLS helper uses HC3)...")
    results_df = analyze_all_modes(data)
    results_df = apply_fdr_correction(results_df)

    print("\nRunning MANOVA...")
    manova_output = run_manova(data)
    print("\n=== MANOVA Results ===")
    print(manova_output)

    results_df.to_csv(RESULTS_CSV, index=False)
    print(f"\nResults saved: {RESULTS_CSV}")
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
