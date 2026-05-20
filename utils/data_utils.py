"""Data loading and processing utilities.

Simple, explicit data loading functions for patient metadata and analysis results.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd


def normalize_load_name(s: str) -> str:
    """Normalize load case string for matching (lowercase, alphanumeric only)."""
    import re
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def find_sheet_for_load(load: str, meta: pd.DataFrame) -> str:
    """Find Excel sheet name matching a given load case label."""
    key = normalize_load_name(load)
    m = meta.copy()
    m["_k"] = m["load_case"].astype(str).map(normalize_load_name)
    return str(m.loc[m["_k"] == key].iloc[0]["sheet_name"])


def load_patient_metadata(data_dir: Path) -> pd.DataFrame:
    """Load patient demographic and allometry metadata.
    
    Parameters
    ----------
    data_dir : Path
        Directory containing allometry.xlsx and demography.xlsx
        
    Returns
    -------
    pd.DataFrame
        Merged patient metadata with columns: patient_id, sex, age, scale, etc.
    """
    allometry = pd.read_excel(data_dir / "allometry.xlsx")
    demography = pd.read_excel(data_dir / "demography.xlsx")
    
    # Standardize sex column
    demography["sex"] = demography["sex"].astype(str).str.strip().str.upper()
    
    # Merge on patient_id
    metadata = pd.merge(allometry, demography, on="patient_id", how="inner")
    
    return metadata


def merge_with_metadata(
    df: pd.DataFrame,
    data_dir: Path,
    metadata_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Merge analysis DataFrame with patient metadata.
    
    Parameters
    ----------
    df : pd.DataFrame
        Analysis results with patient_id column
    data_dir : Path
        Directory containing metadata files
    metadata_cols : list[str] | None
        Specific metadata columns to include (None = all)
        
    Returns
    -------
    pd.DataFrame
        Merged DataFrame with metadata columns added
    """
    metadata = load_patient_metadata(data_dir)
    
    if metadata_cols is not None:
        cols_to_merge = ["patient_id"] + [
            c for c in metadata_cols if c != "patient_id" and c in metadata.columns
        ]
        metadata = metadata[cols_to_merge]
    
    return pd.merge(df, metadata, on="patient_id", how="inner")


def standardize_sex_column(df: pd.DataFrame, col: str = "sex") -> pd.DataFrame:
    """Standardize sex column to uppercase M/F.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing a sex column.
    col : str
        Name of the sex column.

    Returns
    -------
    pd.DataFrame
        DataFrame with standardized sex column.
    """
    df = df.copy()
    df[col] = df[col].astype(str).str.strip().str.upper()
    return df


def add_age_groups(
    df: pd.DataFrame,
    age_col: str = "age",
    bins: list[int] | None = None,
    labels: list[str] | None = None,
) -> pd.DataFrame:
    """Add an age_group column based on binned age ranges.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with an age column.
    age_col : str
        Name of the age column.
    bins : list[int]
        Bin edges for pd.cut.
    labels : list[str]
        Labels for the resulting bins.

    Returns
    -------
    pd.DataFrame
        DataFrame with added 'age_group' column.
    """
    if bins is None:
        bins = [0, 30, 45, 60, 80, 110]
    if labels is None:
        labels = ["<30", "30-45", "45-60", "60-80", "80+"]
    df = df.copy()
    df["age_group"] = pd.cut(df[age_col], bins=bins, labels=labels, right=True)
    return df


def load_eigen_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load eigenvalue data from Excel.
    
    Parameters
    ----------
    data_dir : Path
        Directory containing eigen_data.xlsx
        
    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (modes_df, permutations_df)
    """
    eigen_file = pd.ExcelFile(data_dir / "eigen_data.xlsx")
    modes = pd.read_excel(eigen_file, sheet_name="modes")
    permutations = pd.read_excel(eigen_file, sheet_name="permutations")
    
    return modes, permutations
