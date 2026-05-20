#!/usr/bin/env python3
"""Batch processing utilities for patient demographic data collection.

Collects basic demographic information across datasets:
- Patient ID
- Age (computed from birth date and CT scan date)
- Sex

Outputs demography.xlsx file with one row per patient.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import logging_config
from database import collect_patient_info, get_patientid_as_numpy, get_as_numpy

# Setup logging
logger = logging.getLogger(__name__)

# Explicit configuration
SIMULATION_DIR = PROJECT_ROOT / "results" / "ref_S1P_fixed_new2"
SIMULATION_DATA_DIR = SIMULATION_DIR / "data"
DATA_DIR = PROJECT_ROOT / "data"
DATABASE_DIR = Path("/mnt/database/BoneDat/raw")
OUTPUT_PATH = SIMULATION_DATA_DIR / "demography.xlsx"
COORDINATES_PATH = DATA_DIR / "X.npy"


def validate_patient_count(
    patient_ids: np.ndarray,
    n_samples: int,
) -> None:
    """Validate that patient ID count matches data sample count.
    
    Parameters
    ----------
    patient_ids : np.ndarray
        Array of patient IDs from database
    n_samples : int
        Number of samples in the dataset
        
    Raises
    ------
    ValueError
        If counts don't match
    """
    if patient_ids.shape[0] != n_samples:
        raise ValueError(
            f"Patient ID count mismatch: database has {patient_ids.shape[0]} patients, "
            f"but data has {n_samples} samples"
        )


def main() -> None:
    """Generate patient demographic workbook."""
    lm = logging_config.LoggingManager(str(SIMULATION_DIR.resolve()))
    lm.add_module(__name__, logging.DEBUG)
    lm.setup()
    lm.reconfigure_console("INFO")

    logger.info("Starting patient demographics collection")
    logger.info(f"Simulation directory: {SIMULATION_DIR}")
    logger.info(f"Output file: {OUTPUT_PATH}")
    logger.info(f"Database directory: {DATABASE_DIR}")

    # Load sample count from coordinates array
    logger.info("Loading data to determine sample count")
    if not COORDINATES_PATH.exists():
        raise FileNotFoundError(f"Coordinates file not found: {COORDINATES_PATH}")
    
    coordinates = np.load(COORDINATES_PATH)
    n_samples = coordinates.shape[0]
    logger.info(f"Dataset contains {n_samples} samples")
    
    # Load patient information from database
    logger.info("Loading patient information from database")
    patient_info = collect_patient_info(DATABASE_DIR)
    
    # Extract arrays
    patient_ids = get_patientid_as_numpy(patient_info)
    sex_values = get_as_numpy(patient_info, 'sex')
    age_values = get_as_numpy(patient_info, 'age')
    
    logger.debug(f"Collected data for {len(patient_ids)} patients")
    
    # Validate counts match
    validate_patient_count(patient_ids, n_samples)
    
    # Prepare demographics dataframe
    logger.info("Preparing demographics dataframe")
    demography_data = {
        "patient_id": patient_ids,
        "age": age_values,
        "sex": sex_values,
    }
    
    demography_df = pd.DataFrame(demography_data)
    
    # Log summary statistics
    logger.info("Demographics summary:")
    logger.info(f"  Total patients: {len(demography_df)}")
    logger.info(f"  Sex distribution: {demography_df['sex'].value_counts().to_dict()}")
    logger.info(f"  Age range: {age_values.min():.1f} - {age_values.max():.1f} years")
    logger.info(f"  Age mean ± std: {age_values.mean():.1f} ± {age_values.std():.1f} years")
    
    # Save results
    SIMULATION_DATA_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving results to {SIMULATION_DATA_DIR}")
    logger.debug(f"Writing demographics to {OUTPUT_PATH}")
    demography_df.to_excel(OUTPUT_PATH, index=False)
    
    logger.info("Demographics collection completed successfully")
    logger.info(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
