#!/usr/bin/env python3
"""Remove scale effect from eigenvalues and energy fractions.

Adjusts all data sheets (modes, energy fractions) by removing scale effect:
    y_adj = y * (S0/scale)^beta_k
where beta_k is estimated via log(y) ~ log(scale) + sex

Outputs: eigen_data_scaled.xlsx with same structure as eigen_data.xlsx
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from utils.stats_utils import fit_beta

PROJECT_ROOT = Path(__file__).resolve().parent
SIMULATION_DIR = PROJECT_ROOT / "results" / "ref_S1P_fixed_new2"
DATA_DIR = SIMULATION_DIR / "data"
EIGEN_DATA_PATH = DATA_DIR / "eigen_data.xlsx"
EIGEN_SCALED_PATH = DATA_DIR / "eigen_data_scaled.xlsx"
ALLOMETRY_PATH = DATA_DIR / "allometry.xlsx"
DEMOGRAPHY_PATH = DATA_DIR / "demography.xlsx"


def remove_scale_effect(df: pd.DataFrame, scale: np.ndarray, sex: np.ndarray, S0: float) -> pd.DataFrame:
    """Remove scale effect from all numeric columns in dataframe."""
    log_scale = np.log(scale)
    df_scaled = df.copy()
    
    for col in df.columns:
        if col == 'patient_id':
            continue
        y = df[col].astype(float).values
        beta = fit_beta(log_scale, sex, y)
        
        if np.isfinite(beta):
            factor = (S0 / scale) ** beta
            df_scaled[col] = y * factor
    
    return df_scaled


def main() -> None:
    # Load data
    allom = pd.read_excel(ALLOMETRY_PATH)
    demo = pd.read_excel(DEMOGRAPHY_PATH)
    eigen_file = EIGEN_DATA_PATH
    
    scale = allom["scale"].values
    sex = demo["sex"].map({"M": 0, "F": 1}).values
    
    # Reference size S0 = geometric mean
    mask = np.isfinite(scale) & (scale > 0)
    S0 = float(np.exp(np.log(scale[mask]).mean()))
    
    # Load all sheets
    xl = pd.ExcelFile(eigen_file)
    skip_sheets = {'permutations', 'imputed', 'metadata'}
    
    # Process each sheet
    output_sheets = {}
    for sheet_name in xl.sheet_names:
        df = pd.read_excel(eigen_file, sheet_name=sheet_name)
        
        if sheet_name in skip_sheets:
            output_sheets[sheet_name] = df
        else:
            # Remove scale effect from data sheets (modes, energy fractions)
            df_scaled = remove_scale_effect(df, scale, sex, S0)
            frac_cols = [c for c in df_scaled.columns if c.startswith("energy_frac_mode_")]
            if frac_cols:
                row_sum = df_scaled[frac_cols].sum(axis=1)
                # Avoid division by zero
                row_sum_safe = row_sum.replace(0, np.nan)
                print(f"Normalizing energy fractions in sheet '{sheet_name}'")
                df_scaled[frac_cols] = df_scaled[frac_cols].div(row_sum_safe, axis=0)

            output_sheets[sheet_name] = df_scaled
    
    # Update metadata
    meta_df = output_sheets['metadata']
    meta_df = pd.concat([
        meta_df,
        pd.DataFrame([
            ("", ""),
            ("SCALING_INFO", ""),
            ("scaling_variable", "scale (from allometry.xlsx)"),
            ("reference_size_S0", f"{S0:.6f}"),
            ("method", "y_scaled = y * (S0/scale)^beta"),
        ], columns=meta_df.columns)
    ], ignore_index=True)
    output_sheets['metadata'] = meta_df
    
    # Save as eigen_data_scaled.xlsx
    with pd.ExcelWriter(EIGEN_SCALED_PATH, engine="openpyxl") as writer:
        for sheet_name, df in output_sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    
    print(f"Saved: {EIGEN_SCALED_PATH}")
    print(f"Reference size S0: {S0:.6f}")
    print(f"Processed sheets: {[s for s in xl.sheet_names if s not in skip_sheets]}")


if __name__ == "__main__":
    main()
