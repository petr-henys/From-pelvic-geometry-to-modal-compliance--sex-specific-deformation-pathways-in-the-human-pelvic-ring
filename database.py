"""Patient metadata collection from database directories.

Collects patient sex and age from metadata Excel files in patient directories.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np
import logging
from typing import Any

logger = logging.getLogger(__name__)

def collect_patient_info(
    root_directory: str | Path, 
    metadata_file: str = 'metadata.xlsx'
) -> dict[str, dict[str, Any]]:
    """Collects patient sex and age from metadata files in patient directories.

    Parameters
    ----------
    root_directory : str | Path
        Path to the directory containing patient folders.
    metadata_file : str
        Name of the metadata file (default: 'metadata.xlsx').

    Returns
    -------
    dict[str, dict[str, Any]]
        Dictionary mapping patient IDs to dictionaries containing sex and age.
    """
    patient_data: dict[str, dict[str, Any]] = {}

    for patient_folder in Path(root_directory).iterdir():
        if patient_folder.is_dir():
            metadata_path = patient_folder / metadata_file

            if metadata_path.exists():
                metadata = pd.read_excel(metadata_path)
                try:
                    patient_id = patient_folder.name  # Use folder name as ID
                    sex = metadata['sex'][0]  # Assuming single-row metadata
                    age = metadata['CT date'][0] - metadata['born'][0]
                except KeyError:
                    logger.warning(f"Missing 'sex' or date columns in {metadata_path}")
                    continue  # Skip this patient if data is incomplete
                except IndexError:
                    logger.warning(f"Empty metadata file in {metadata_path}")
                    continue
                patient_data[patient_id] = {'sex': sex, 'age': age}

    return patient_data

def get_as_numpy(patient_info: dict[str, dict[str, Any]], item: str) -> np.ndarray:
    """Extracts a specific data item and returns as NumPy array.

    Parameters
    ----------
    patient_info : dict[str, dict[str, Any]]
        Dictionary containing patient data.
    item : str
        Key of the item to extract.

    Returns
    -------
    np.ndarray
        Array containing the extracted data.
    """
    return np.array([info[item] for info in patient_info.values()])

def get_patientid_as_numpy(patient_info: dict[str, dict[str, Any]]) -> np.ndarray:
    """Extracts patient IDs as NumPy array.

    Parameters
    ----------
    patient_info : dict[str, dict[str, Any]]
        Dictionary containing patient data.

    Returns
    -------
    np.ndarray
        Array containing the patient IDs.
    """
    return np.array(list(patient_info.keys()))

if __name__ == '__main__':
    # Example usage - replace with actual database path
    DATABASE_DIRECTORY = '/mnt/database/BoneDat/raw'
    patient_info = collect_patient_info(DATABASE_DIRECTORY)
    sex = get_as_numpy(patient_info, 'sex')
    age = get_as_numpy(patient_info, 'age')
    patient_ids = get_patientid_as_numpy(patient_info)
    print(f"Loaded {len(patient_info)} patients from {DATABASE_DIRECTORY}")
