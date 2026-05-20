#!/usr/bin/env python3
"""Batch processing utilities for pelvic anatomical data collection.

Collects morphological dimensions and sacroiliac joint (SIJ) kinematics:
- Pelvic dimensions from Michal's measurements (8 linear/angular measures)
- SIJ angles (rotation) for 4 load cases × 2 joints (left/right) × 3 DOF
- SIJ translations for 4 load cases × 2 joints (left/right) × 3 DOF

Outputs anatomy_data.xlsx file with 11 sheets organized by load case and side:
- Sheet 1 "Morphology": Patient ID + 8 morphology dimensions
- Sheets 2-11: SIJ data organized as "SIJ_{L/R}_{load_case}"
  - Each sheet: patient_id + 3 angles + 3 translations (7 columns total)
  - Example: "SIJ_L_SP2leg", "SIJ_R_SP2leg", "SIJ_L_SP1leg", etc.

Sheet organization by load case and side:
- SIJ_L_SP2leg, SIJ_R_SP2leg (two-leg standing)
- SIJ_L_SP1leg, SIJ_R_SP1leg (one-leg standing)
- SIJ_L_LAB_phase1, SIJ_R_LAB_phase1 (labor phase 1)
- SIJ_L_LAB_phase2, SIJ_R_LAB_phase2 (labor phase 2)

Anatomical DOF naming:
- Angles (degrees): NUT (Nutation/α_x), AP (Anterior-Posterior/β_y), CC (Cranio-Caudal/γ_z)
- Translations (mm): ML (Medial-Lateral/dx), AP (Anterior-Posterior/dy), CC (Cranio-Caudal/dz)

Column format examples (within each SIJ sheet):
- sij_ang_SP2leg_L_NUT, sij_ang_SP2leg_L_AP, sij_ang_SP2leg_L_CC
- sij_trans_SP2leg_L_ML, sij_trans_SP2leg_L_AP, sij_trans_SP2leg_L_CC
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import logging_config
from database import collect_patient_info, get_patientid_as_numpy

# Setup logging
logger = logging.getLogger(__name__)

# Explicit configuration
SIMULATION_DIR = PROJECT_ROOT / "results" / "ref_S1P_fixed_new2"
SIMULATION_DATA_DIR = SIMULATION_DIR / "data"
DATA_DIR = PROJECT_ROOT / "data"
DATABASE_DIR = Path("/mnt/database/BoneDat/raw")
OUTPUT_PATH = SIMULATION_DATA_DIR / "anatomy_data.xlsx"
MORPHOLOGY_ZARR_PATH = DATA_DIR / "dimensions_michal.zarr"

# Load case names (must match Zarr file naming)
LOAD_CASES = ["SP2leg", "SP1leg", "LAB_phase1", "LAB_phase2", "LAB_phase3"]
JOINT_NAMES = ["L", "R"]  # Left, Right

# Anatomical names for degrees of freedom
# Angles (rotations in degrees)
ANGLE_DOF_NAMES = ["NUT", "AP", "CC"]  # Nutation (α_x), Anterior-Posterior (β_y), Cranio-Caudal (γ_z)
# Translations (displacements in mm)
TRANS_DOF_NAMES = ["ML", "AP", "CC"]  # Medial-Lateral (dx), Anterior-Posterior (dy), Cranio-Caudal (dz)

# Morphology dimension names from dimensions_michal.zarr
MORPHOLOGY_DIMS = [
    "AP",
    "BiacetabularWidth",
    "BiischiadicWidth",
    "BituberousWidth",
    "IliopectinealEminenceWidth",
    "PIT",
    "SacralWidth",
    "SubpubicAngle",
]


def load_morphology_data(zarr_path: Path = MORPHOLOGY_ZARR_PATH) -> dict[str, np.ndarray]:
    """Load pelvic morphological dimensions from Zarr.
    
    Parameters
    ----------
    zarr_path : Path
        Path to dimensions_michal.zarr
        
    Returns
    -------
    dict[str, np.ndarray]
        Dictionary mapping dimension names to arrays of shape (n_samples,)
    """
    logger.debug(f"Loading morphology from {zarr_path}")
    
    if not zarr_path.exists():
        raise FileNotFoundError(f"Morphology data not found: {zarr_path}")
    
    root = zarr.open(str(zarr_path), mode='r')
    
    morph_data = {}
    for dim_name in MORPHOLOGY_DIMS:
        if dim_name not in root:
            logger.warning(f"Missing dimension: {dim_name}")
            continue
        morph_data[dim_name] = np.array(root[dim_name])
    
    logger.debug(f"Loaded {len(morph_data)} morphology dimensions")
    return morph_data


def load_sij_data(
    results_dir: Path,
    data_type: str,
    load_case: str,
) -> np.ndarray:
    """Load SIJ angles or translations for a specific load case.
    
    Parameters
    ----------
    results_dir : Path
        Directory containing SIJ Zarr files
    data_type : str
        Either 'angles' or 'trans' (translations)
    load_case : str
        Load case name (e.g., 'SP2leg', 'LAB_phase1')
        
    Returns
    -------
    np.ndarray
        Array of shape (n_samples, 2, 3) where:
        - dim 0: patients
        - dim 1: joints (0=Left, 1=Right)
        - dim 2: DOF (X, Y, Z)
    """
    zarr_path = results_dir / f"sij_{data_type}_{load_case}.zarr"
    
    if not zarr_path.exists():
        raise FileNotFoundError(f"SIJ data not found: {zarr_path}")
    
    logger.debug(f"Loading SIJ {data_type} from {zarr_path}")
    root = zarr.open(str(zarr_path), mode='r')
    
    if 'data' not in root:
        raise ValueError(f"Missing 'data' array in {zarr_path}")
    
    data = np.array(root['data'])
    
    # Validate shape
    if data.ndim != 3 or data.shape[1] != 2 or data.shape[2] != 3:
        raise ValueError(
            f"Expected shape (n_samples, 2, 3), got {data.shape} for {zarr_path}"
        )
    
    return data


def flatten_sij_data(
    sij_data: np.ndarray,
    prefix: str,
    is_angle: bool = True,
) -> dict[str, np.ndarray]:
    """Flatten SIJ data into columnar format with anatomical names.
    
    Parameters
    ----------
    sij_data : np.ndarray
        Array of shape (n_samples, 2, 3)
    prefix : str
        Column name prefix (e.g., 'sij_ang_SP2leg')
    is_angle : bool
        If True, use angle DOF names (NUT, AP, CC), otherwise translation names (ML, AP, CC)
        
    Returns
    -------
    dict[str, np.ndarray]
        Dictionary with keys like 'sij_ang_SP2leg_L_NUT', 'sij_trans_SP2leg_R_ML', etc.
    """
    flattened = {}
    dof_names = ANGLE_DOF_NAMES if is_angle else TRANS_DOF_NAMES
    
    for joint_idx, joint_name in enumerate(JOINT_NAMES):
        for dof_idx, dof_name in enumerate(dof_names):
            col_name = f"{prefix}_{joint_name}_{dof_name}"
            flattened[col_name] = sij_data[:, joint_idx, dof_idx]
    
    return flattened


def collect_all_sij_data(results_dir: Path) -> dict[str, np.ndarray]:
    """Collect all SIJ angles and translations for all load cases.
    
    Parameters
    ----------
    results_dir : Path
        Directory containing SIJ Zarr files
        
    Returns
    -------
    dict[str, np.ndarray]
        Dictionary with flattened SIJ data columns
    """
    all_data = {}
    
    for load_case in LOAD_CASES:
        # Load angles
        angles = load_sij_data(results_dir, "angles", load_case)
        angle_cols = flatten_sij_data(angles, f"sij_ang_{load_case}", is_angle=True)
        all_data.update(angle_cols)
        logger.debug(f"Loaded SIJ angles for {load_case}")
        
        # Load translations
        trans = load_sij_data(results_dir, "trans", load_case)
        trans_cols = flatten_sij_data(trans, f"sij_trans_{load_case}", is_angle=False)
        all_data.update(trans_cols)
        logger.debug(f"Loaded SIJ translations for {load_case}")
    
    return all_data


def validate_data_consistency(data_dict: dict[str, np.ndarray], n_expected: int) -> None:
    """Validate that all arrays have consistent length.
    
    Parameters
    ----------
    data_dict : dict[str, np.ndarray]
        Dictionary of data arrays
    n_expected : int
        Expected number of samples
        
    Raises
    ------
    ValueError
        If any array has inconsistent length
    """
    for key, arr in data_dict.items():
        if len(arr) != n_expected:
            raise ValueError(
                f"Inconsistent data length: {key} has {len(arr)} samples, "
                f"expected {n_expected}"
            )


def main() -> None:
    """Generate pelvic anatomy workbook from simulation outputs."""
    lm = logging_config.LoggingManager(str(SIMULATION_DIR.resolve()))
    lm.add_module(__name__, logging.DEBUG)
    lm.setup()
    lm.reconfigure_console("INFO")

    logger.info("Starting anatomy data collection")
    logger.info(f"Simulation directory: {SIMULATION_DIR}")
    logger.info(f"Output file: {OUTPUT_PATH}")
    logger.info(f"Database directory: {DATABASE_DIR}")

    # Load patient IDs from database
    logger.info("Loading patient IDs from database")
    patient_info = collect_patient_info(DATABASE_DIR)
    patient_ids = get_patientid_as_numpy(patient_info)
    n_samples = len(patient_ids)
    logger.info(f"Found {n_samples} patients in database")

    # Load morphology data
    logger.info("Loading pelvic morphology dimensions")
    morph_data = load_morphology_data()
    
    # Validate morphology data
    validate_data_consistency(morph_data, n_samples)
    logger.debug(f"Loaded morphology dimensions: {list(morph_data.keys())}")

    # Load SIJ data
    logger.info("Loading SIJ kinematics (angles and translations)")
    sij_data = collect_all_sij_data(SIMULATION_DATA_DIR)
    
    # Validate SIJ data
    validate_data_consistency(sij_data, n_samples)
    logger.debug(f"Loaded {len(sij_data)} SIJ columns")

    # Prepare morphology dataframe
    logger.info("Preparing morphology dataframe")
    morphology_data = {
        "patient_id": patient_ids,
        **morph_data,
    }
    
    morphology_df = pd.DataFrame(morphology_data)
    
    # Organize SIJ data by load case and side (combining angles + translations)
    logger.info("Preparing SIJ dataframes (organized by load case and side)")
    
    sij_sheets = {}
    
    for load_case in LOAD_CASES:
        for joint_name in JOINT_NAMES:
            # Create sheet for this load case + joint combination
            sheet_name = f"SIJ_{joint_name}_{load_case}"
            
            # Collect angles and translations for this load+joint
            sheet_data = {"patient_id": patient_ids}
            
            # Add angles (3 DOF)
            for k, v in sij_data.items():
                if f"sij_ang_{load_case}_{joint_name}_" in k:
                    sheet_data[k] = v
            
            # Add translations (3 DOF)
            for k, v in sij_data.items():
                if f"sij_trans_{load_case}_{joint_name}_" in k:
                    sheet_data[k] = v
            
            sij_sheets[sheet_name] = pd.DataFrame(sheet_data)
    
    # Log summary
    logger.info("Anatomy data summary:")
    logger.info(f"  Total patients: {len(morphology_df)}")
    logger.info(f"  Morphology: {len(morphology_df.columns)} columns")
    logger.info(f"  SIJ sheets: {len(sij_sheets)} sheets (5 load cases × 2 sides)")
    for sheet_name, df in sij_sheets.items():
        logger.debug(f"    {sheet_name}: {len(df.columns)} columns")
    
    # Display sample statistics for morphology
    logger.info("Morphology statistics:")
    for dim_name in MORPHOLOGY_DIMS:
        if dim_name in morph_data:
            values = morph_data[dim_name]
            logger.info(
                f"  {dim_name}: {values.mean():.2f} ± {values.std():.2f} "
                f"(range: {values.min():.2f} - {values.max():.2f})"
            )

    # Save results to Excel with multiple sheets
    SIMULATION_DATA_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving results to {SIMULATION_DATA_DIR}")
    logger.debug(f"Writing anatomy data to {OUTPUT_PATH}")
    with pd.ExcelWriter(OUTPUT_PATH, engine='openpyxl') as writer:
        # Write morphology first
        morphology_df.to_excel(writer, sheet_name='Morphology', index=False)
        
        # Write SIJ sheets in organized order (by load case, then L/R)
        for load_case in LOAD_CASES:
            for joint_name in JOINT_NAMES:
                sheet_name = f"SIJ_{joint_name}_{load_case}"
                sij_sheets[sheet_name].to_excel(writer, sheet_name=sheet_name, index=False)
    
    logger.info("Anatomy data collection completed successfully")
    logger.info(f"Output: {OUTPUT_PATH}")
    logger.info(f"  Total sheets: {1 + len(sij_sheets)} (1 morphology + {len(sij_sheets)} SIJ)")
    logger.info("  SIJ organization: 5 load cases × 2 sides (L/R), 7 columns each (patient_id + 3 angles + 3 translations)")


if __name__ == "__main__":
    main()
