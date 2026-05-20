#!/usr/bin/env python3
"""Energy fraction analysis.

Estimation follows eigen_analysis principles and writes CSV outputs only:
- Transform fractions with logit and fit IV 2SLS: logit(p) ~ log(scale) + sex
- Treat log(scale) as endogenous; sex as exogenous
- Use instruments derived from allometry features; robust (heteroskedasticity-robust) SEs
- Report percent-point (pp) changes for doubling scale and sex difference with 95% CIs

No plotting is performed here. Separate plotting scripts consume the CSV outputs.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataclasses import dataclass

import numpy as np
import pandas as pd
import zarr
from linearmodels.iv import IV2SLS as IV2SLS_LM
from scipy import stats
from statsmodels.stats.multitest import fdrcorrection

from utils.data_utils import normalize_load_name, find_sheet_for_load

ALPHA = 0.05
MIN_SAMPLES = 10

DATA_DIR = Path("results/ref_S1P_fixed_new2/data")
ANALYSIS_DIR = Path("results/ref_S1P_fixed_new2/analysis")
LOAD_CASES = ["SP2leg", "SP1leg", "LAB_phase1", "LAB_phase2", "LAB_phase3"]
ZARR_LOAD_ORDER = ["SP2leg", "SP1leg", "LAB_phase1", "LAB_phase2", "LAB_phase3"]


def _first_existing_path(candidates: list[Path], label: str) -> Path:
    """Return first existing candidate path or raise with helpful context."""
    for p in candidates:
        if p.exists():
            return p
    candidate_list = "\n".join(f"  - {p}" for p in candidates)
    raise FileNotFoundError(f"Could not find {label}. Checked:\n{candidate_list}")


def _logit(p: np.ndarray) -> np.ndarray:
    """Logit transform with clipping to avoid infinities."""
    return np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))

@dataclass
class Fit:
    """IV regression result container."""
    beta: float
    beta_se: float
    beta_p: float
    gamma: float
    gamma_se: float
    gamma_p: float
    df: int


def _select_scale_instruments(allometry: pd.DataFrame, log_scale: np.ndarray, top_k: int = 3) -> pd.DataFrame:
    """Build instrument candidates from allometry with safe transform."""
    cand_cols = [c for c in allometry.columns if c not in {"patient_id", "scale"}]
    preferred = ["total_volume", "total_surface", "total_mass_template", "eff_density_template", "total_mass"]
    cols = [c for c in preferred if c in cand_cols]
    
    if len(cols) < top_k:
        rem = [c for c in cand_cols if c not in cols]
        ranked: list[tuple[float, str]] = []
        for c in rem:
            x = allometry[c].to_numpy(float)
            xt = np.log(x) if np.all(np.isfinite(x)) and (x > 0).all() else x
            m = np.isfinite(xt) & np.isfinite(log_scale)
            if m.sum() > 2 and np.nanstd(xt[m]) > 0:
                r = float(np.corrcoef(xt[m], log_scale[m])[0, 1])
                if np.isfinite(r):
                    ranked.append((abs(r), c))
        ranked.sort(reverse=True)
        for _, c in ranked:
            if c not in cols:
                cols.append(c)
            if len(cols) >= top_k:
                break
    
    Z = {}
    for c in cols:
        x = allometry[c].to_numpy(float)
        if np.all(np.isfinite(x)) and (x > 0).all():
            Z[f"Z_{c}_log"] = np.log(x)
        else:
            Z[f"Z_{c}"] = x
    return pd.DataFrame(Z)


def _full_rank_instruments(Z_all: pd.DataFrame, row_mask: np.ndarray, X_exog_full: pd.DataFrame, 
                           min_needed: int = 1, tol: float = 1e-10) -> pd.DataFrame:
    """Greedy select a full-rank subset of instruments relative to exogenous design."""
    Z = Z_all.loc[row_mask].reset_index(drop=True)
    finite = [c for c in Z.columns if np.isfinite(Z[c].to_numpy()).all()]
    Z = Z[finite]
    M = X_exog_full.loc[row_mask].to_numpy()
    keep: list[str] = []
    
    for c in Z.columns:
        col = Z[c].to_numpy().reshape(-1, 1)
        if np.nanstd(col) <= tol:
            continue
        M_try = np.column_stack([M, col])
        if np.linalg.matrix_rank(M_try, tol) > np.linalg.matrix_rank(M, tol):
            M = M_try
            keep.append(c)
    
    if len(keep) < min_needed:
        raise ValueError(f"Insufficient instrument rank: have {len(keep)}, need >= {min_needed}.")
    return Z[keep]


def fit_iv_logit_fraction(y: np.ndarray, log_scale: np.ndarray, sex: np.ndarray, 
                          Z_all: pd.DataFrame, row_mask: np.ndarray) -> Fit:
    """Fit IV 2SLS model for logit-transformed energy fractions."""
    v = np.isfinite(y) & np.isfinite(log_scale) & np.isfinite(sex)
    v = v & row_mask
    
    if v.sum() < MIN_SAMPLES:
        return Fit(np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0)
    
    Y = _logit(y[v])
    exog_full = pd.DataFrame({"const": np.ones_like(sex, dtype=float), "sex": sex})
    exog_df = exog_full.loc[v].reset_index(drop=True)
    endog_df = pd.DataFrame({"log_scale": log_scale[v]})
    Z_df = _full_rank_instruments(Z_all, v, X_exog_full=exog_full, min_needed=endog_df.shape[1])
    
    res = IV2SLS_LM(pd.Series(Y, name="logit_frac"), exog_df, endog_df, Z_df).fit(cov_type="robust")
    
    return Fit(
        beta=float(res.params["log_scale"]),
        beta_se=float(res.std_errors["log_scale"]),
        beta_p=float(res.pvalues["log_scale"]),
        gamma=float(res.params["sex"]),
        gamma_se=float(res.std_errors["sex"]),
        gamma_p=float(res.pvalues["sex"]),
        df=int(res.df_resid)
    )


def pp_effect(beta: float, se: float, df: int, p_ref: float) -> tuple[float, float, float]:
    """Percent-point effect for doubling scale."""
    t_crit = stats.t.ppf(1 - ALPHA / 2, df)
    lo = float(_logit(np.array([p_ref]))[0])
    z = lo + beta * np.log(2.0)
    p2 = 1.0 / (1.0 + np.exp(-z))
    dp = p2 - p_ref
    d = p2 * (1 - p2) * np.log(2.0)
    se_dp = abs(d) * se
    return 100 * dp, 100 * (dp - t_crit * se_dp), 100 * (dp + t_crit * se_dp)


def pp_effect_delta(beta: float, se: float, df: int, p_ref: float, delta: float) -> tuple[float, float, float]:
    """Percent-point effect for a change of size `delta` in predictor."""
    t_crit = stats.t.ppf(1 - ALPHA / 2, df)
    lo = float(_logit(np.array([p_ref]))[0])
    z = lo + beta * delta
    p2 = 1.0 / (1.0 + np.exp(-z))
    dp = p2 - p_ref
    d = p2 * (1 - p2) * delta
    se_dp = abs(d) * se
    return 100 * dp, 100 * (dp - t_crit * se_dp), 100 * (dp + t_crit * se_dp)


def analyze(load: str, eigen: pd.ExcelFile, allometry: pd.DataFrame,
            scale: np.ndarray, sex: np.ndarray,
            z_shape_data: np.ndarray | None = None,
            z_material_data: np.ndarray | None = None) -> pd.DataFrame:
    """Analyze energy fractions for a given load case."""
    meta = pd.read_excel(eigen, sheet_name="metadata")
    sheet = find_sheet_for_load(load, meta)
    frac_un = pd.read_excel(eigen, sheet_name=sheet)
    cols = [c for c in frac_un.columns if c.startswith("energy_frac_mode_")]
    
    m = np.isfinite(scale) & (scale > 0) & np.isfinite(sex)
    ls = np.log(scale)
    log_scale_iqr = float(np.percentile(ls[m], 75) - np.percentile(ls[m], 25))
    Z_all = _select_scale_instruments(allometry, ls)
    
    res = []
    li = ZARR_LOAD_ORDER.index(load) if z_shape_data is not None and z_material_data is not None else None
    
    for i, col in enumerate(cols):
        y_full = frac_un[col].astype(float).values
        fit = fit_iv_logit_fraction(y_full, ls, sex, Z_all, row_mask=m)
        pref = float(np.nanmean(np.clip(y_full[m], 1e-6, 1 - 1e-6)))
        pp_s, pp_s_lo, pp_s_hi = pp_effect(fit.beta, fit.beta_se, fit.df, pref)
        pp_s_iqr, pp_s_iqr_lo, pp_s_iqr_hi = pp_effect_delta(fit.beta, fit.beta_se, fit.df, pref, log_scale_iqr)
        pp_x, pp_x_lo, pp_x_hi = pp_effect(fit.gamma, fit.gamma_se, fit.df, pref)
        t_crit = stats.t.ppf(1 - ALPHA / 2, fit.df) if fit.df > 0 else np.nan
        rec = {
            "load_case": load,
            "mode": i + 1,
            "beta_scale": fit.beta,
            "beta_scale_SE": fit.beta_se,
            "beta_scale_p": fit.beta_p,
            "beta_scale_CI_lo": fit.beta - t_crit * fit.beta_se if np.isfinite(t_crit) else np.nan,
            "beta_scale_CI_hi": fit.beta + t_crit * fit.beta_se if np.isfinite(t_crit) else np.nan,
            "gamma_sex": fit.gamma,
            "gamma_sex_SE": fit.gamma_se,
            "gamma_sex_p": fit.gamma_p,
            "gamma_sex_CI_lo": fit.gamma - t_crit * fit.gamma_se if np.isfinite(t_crit) else np.nan,
            "gamma_sex_CI_hi": fit.gamma + t_crit * fit.gamma_se if np.isfinite(t_crit) else np.nan,
            "pp_scale_x2": pp_s,
            "pp_scale_x2_CI_lo": pp_s_lo,
            "pp_scale_x2_CI_hi": pp_s_hi,
            "pp_scale_IQR": pp_s_iqr,
            "pp_scale_IQR_CI_lo": pp_s_iqr_lo,
            "pp_scale_IQR_CI_hi": pp_s_iqr_hi,
            "pp_sex": pp_x,
            "pp_sex_CI_lo": pp_x_lo,
            "pp_sex_CI_hi": pp_x_hi,
        }

        # Shape-only augmented model
        if li is not None:
            y_shape = z_shape_data[:, i, li]
            z_shape = _logit(np.clip(y_shape, 1e-6, 1 - 1e-6))
            v = np.isfinite(y_full) & np.isfinite(ls) & np.isfinite(sex) & np.isfinite(z_shape)
            
            if v.sum() >= MIN_SAMPLES:
                Y = _logit(np.clip(y_full[v], 1e-6, 1 - 1e-6))
                exog_full = pd.DataFrame({"const": np.ones_like(sex, dtype=float), "sex": sex})
                exog_df = exog_full.loc[v].reset_index(drop=True)
                endog_df = pd.DataFrame({
                    "log_scale": ls[v],
                    "logit_shape_only": z_shape[v],
                })
                Z_candidates = pd.concat([Z_all, pd.DataFrame({"Z_logit_shape_only": z_shape})], axis=1)
                Z_df = _full_rank_instruments(Z_candidates, v, X_exog_full=exog_full, min_needed=endog_df.shape[1])
                res_shape = IV2SLS_LM(pd.Series(Y, name="logit_frac"), exog_df, endog_df, Z_df).fit(cov_type="robust")
                
                bsh = float(res_shape.params.get("logit_shape_only", np.nan))
                bsh_se = float(res_shape.std_errors.get("logit_shape_only", np.nan))
                bsh_p = float(res_shape.pvalues.get("logit_shape_only", np.nan))
                dfsh = int(res_shape.df_resid)
                t_crit_sh = stats.t.ppf(1 - ALPHA / 2, dfsh)
                
                rec["beta_shape"] = bsh
                rec["beta_shape_SE"] = bsh_se
                rec["beta_shape_p"] = bsh_p
                rec["beta_shape_CI_lo"] = bsh - t_crit_sh * bsh_se
                rec["beta_shape_CI_hi"] = bsh + t_crit_sh * bsh_se
                
                z_iqr = float(np.percentile(z_shape[v], 75) - np.percentile(z_shape[v], 25))
                pp_sh, pp_sh_lo, pp_sh_hi = pp_effect_delta(bsh, bsh_se, dfsh, pref, z_iqr)
                rec["pp_shape_IQR"] = pp_sh
                rec["pp_shape_IQR_CI_lo"] = pp_sh_lo
                rec["pp_shape_IQR_CI_hi"] = pp_sh_hi

        # Material-only augmented model
        if li is not None:
            y_mat = z_material_data[:, i, li]
            z_mat = _logit(np.clip(y_mat, 1e-6, 1 - 1e-6))
            v = np.isfinite(y_full) & np.isfinite(ls) & np.isfinite(sex) & np.isfinite(z_mat)
            
            if v.sum() >= MIN_SAMPLES:
                Y = _logit(np.clip(y_full[v], 1e-6, 1 - 1e-6))
                exog_full = pd.DataFrame({"const": np.ones_like(sex, dtype=float), "sex": sex})
                exog_df = exog_full.loc[v].reset_index(drop=True)
                endog_df = pd.DataFrame({
                    "log_scale": ls[v],
                    "logit_material_only": z_mat[v],
                })
                Z_candidates = pd.concat([Z_all, pd.DataFrame({"Z_logit_material_only": z_mat})], axis=1)
                Z_df = _full_rank_instruments(Z_candidates, v, X_exog_full=exog_full, min_needed=endog_df.shape[1])
                res_mat = IV2SLS_LM(pd.Series(Y, name="logit_frac"), exog_df, endog_df, Z_df).fit(cov_type="robust")
                
                bmt = float(res_mat.params.get("logit_material_only", np.nan))
                bmt_se = float(res_mat.std_errors.get("logit_material_only", np.nan))
                bmt_p = float(res_mat.pvalues.get("logit_material_only", np.nan))
                dfmt = int(res_mat.df_resid)
                t_crit_mt = stats.t.ppf(1 - ALPHA / 2, dfmt)
                
                rec["beta_material"] = bmt
                rec["beta_material_SE"] = bmt_se
                rec["beta_material_p"] = bmt_p
                rec["beta_material_CI_lo"] = bmt - t_crit_mt * bmt_se
                rec["beta_material_CI_hi"] = bmt + t_crit_mt * bmt_se
                
                z_iqr = float(np.percentile(z_mat[v], 75) - np.percentile(z_mat[v], 25))
                pp_mt, pp_mt_lo, pp_mt_hi = pp_effect_delta(bmt, bmt_se, dfmt, pref, z_iqr)
                rec["pp_material_IQR"] = pp_mt
                rec["pp_material_IQR_CI_lo"] = pp_mt_lo
                rec["pp_material_IQR_CI_hi"] = pp_mt_hi

        res.append(rec)
    return pd.DataFrame(res)

def main() -> None:
    """Run complete energy fraction analysis."""
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    
    print("Loading data...")
    eigen = pd.ExcelFile(DATA_DIR / "eigen_data.xlsx")
    allo = pd.read_excel(DATA_DIR / "allometry.xlsx")
    demo = pd.read_excel(DATA_DIR / "demography.xlsx")
    
    demo["sex"] = demo["sex"].astype(str).str.strip().str.upper()
    valid_sexes = {"M", "F"}
    invalid = sorted(set(demo["sex"]) - valid_sexes)
    if invalid:
        raise ValueError(f"Unexpected sex labels: {invalid}. Expected only 'M' and 'F'.")
    
    scale = allo["scale"].astype(float).values
    if np.any(scale <= 0) or not np.isfinite(scale).all():
        raise ValueError("Scale values must be positive and finite.")
    
    sex = demo["sex"].map({"M": 0.0, "F": 1.0}).astype(float).values
    
    base_results = DATA_DIR.parents[1]
    shape_path = _first_existing_path(
        [
            base_results / "ref_S1P_fixed_new2_shape_only" / "data" / "mode_energy_fraction.zarr",
        ],
        "shape-only mode energy data",
    )
    material_path = _first_existing_path(
        [
            base_results / "ref_S1P_fixed_new2_material_only" / "data" / "mode_energy_fraction.zarr",
        ],
        "material-only mode energy data",
    )
    z_shape_root = zarr.open(str(shape_path), mode='r')
    z_material_root = zarr.open(str(material_path), mode='r')
    z_shape_data = z_shape_root["data"][:]
    z_material_data = z_material_root["data"][:]

    print("Running IV regression analysis...")
    all_dfs = []
    for load in LOAD_CASES:
        res_df = analyze(load, eigen, allo, scale, sex, z_shape_data=z_shape_data, z_material_data=z_material_data)
        all_dfs.append(res_df)
    
    comb = pd.concat(all_dfs, ignore_index=True)
    
    print("Applying FDR correction...")
    p_values = np.concatenate([
        comb["beta_scale_p"].values,
        comb["gamma_sex_p"].values,
    ])
    rejected, p_fdr = fdrcorrection(p_values, alpha=ALPHA, method="indep")
    
    n = len(comb)
    comb["beta_scale_signif"] = rejected[:n]
    comb["beta_scale_pFDR"] = p_fdr[:n]
    comb["gamma_sex_signif"] = rejected[n:]
    comb["gamma_sex_pFDR"] = p_fdr[n:]
    
    comb.to_csv(ANALYSIS_DIR / "fractions_results.csv", index=False)
    print(f"Results: {ANALYSIS_DIR / 'fractions_results.csv'}")
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
