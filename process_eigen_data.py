#!/usr/bin/env python3
"""Process eigenvalue data from modal analysis into Excel format.

Generates eigen_data.xlsx with:
- modes: patient_id, eig_1..eig_N
- permutations: patient_id, perm_1..perm_N
- imputed: patient_id, eig_imputed (if imputation occurred)
- load_cases: metadata with imputation summary
- 1_sp2leg, 2_sp1leg, etc.: energy fractions per load case
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import logging_config
from database import collect_patient_info, get_patientid_as_numpy

logger = logging.getLogger(__name__)

# Explicit configuration
SIMULATION_DIR = PROJECT_ROOT / "results" / "ref_S1P_fixed_new2"
SIMULATION_DATA_DIR = SIMULATION_DIR / "data"
DATABASE_DIR = Path("/mnt/database/BoneDat/raw")
SIMULATION_METADATA_PATH = SIMULATION_DIR / "simulation_metadata.json"
EIGENVALUES_ZARR = SIMULATION_DATA_DIR / "eigenvalues.zarr"
PERMUTATIONS_ZARR = SIMULATION_DATA_DIR / "eig_permutations.zarr"
ENERGY_ZARR = SIMULATION_DATA_DIR / "mode_energy_fraction.zarr"
LIGAMENT_SENS_ZARR = SIMULATION_DATA_DIR / "lig_sensitivities.zarr"
PRETENSION_SENS_ZARR = SIMULATION_DATA_DIR / "pretension_sensitivities.zarr"
CARTILAGE_SENS_ZARR = SIMULATION_DATA_DIR / "cartilage_sensitivities.zarr"
EIGEN_OUTPUT_PATH = SIMULATION_DATA_DIR / "eigen_data.xlsx"
LIGAMENT_OUTPUT_PATH = SIMULATION_DATA_DIR / "ligament_sensitivity.xlsx"
PRETENSION_OUTPUT_PATH = SIMULATION_DATA_DIR / "pretension_sensitivity.xlsx"
CARTILAGE_OUTPUT_PATH = SIMULATION_DATA_DIR / "cartilage_sensitivity.xlsx"


def apply_mode_permutation(data: np.ndarray, perm: np.ndarray) -> np.ndarray:
    """Reorder modes according to permutation indices.
    
    Negative permutation indices indicate unpaired modes (filled with NaN).
    """
    n_samples, n_modes = perm.shape
    result_shape = (n_samples, n_modes) + data.shape[2:]
    result = np.full(result_shape, np.nan, dtype=np.float64)
    
    for i in range(n_samples):
        for j in range(n_modes):
            solver_idx = int(perm[i, j])
            if solver_idx >= 0:
                result[i, j] = data[i, solver_idx]
    
    return result


def impute_invalid_eigenvalues(eigvals: np.ndarray) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Impute invalid eigenvalues with column medians."""
    invalid_mask = (~np.isfinite(eigvals)) | (eigvals <= 0)
    
    if not invalid_mask.any():
        return eigvals, None, None
    
    cleaned = eigvals.copy()
    cleaned[invalid_mask] = np.nan
    column_medians = np.nanmedian(cleaned, axis=0)
    
    rows, cols = np.where(invalid_mask)
    cleaned[rows, cols] = column_medians[cols]
    
    return cleaned, np.unique(rows), invalid_mask


def load_patient_ids(database_dir: Path) -> np.ndarray:
    """Load patient IDs from database metadata files."""
    patient_info = collect_patient_info(database_dir)
    return get_patientid_as_numpy(patient_info).astype(str)


def _sanitize_sheet_name(name: str) -> str:
    """Sanitize to valid Excel sheet name (max 31 chars)."""
    sanitized = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in name)
    return sanitized[:31]


def create_eigen_excel(
    output_path: Path,
    patient_ids: np.ndarray,
    eigvals: np.ndarray,
    perm: np.ndarray,
    energy_sorted: np.ndarray,
    load_case_names: list[str],
    imputed_rows: np.ndarray | None,
    invalid_mask: np.ndarray | None,
) -> None:
    """Create eigen_data.xlsx with eigenvalues and energy fractions."""
    n_modes = eigvals.shape[1]
    
    # Eigenvalues sheet
    eigen_df = pd.DataFrame({
        "patient_id": patient_ids,
        **{f"eig_{i+1}": eigvals[:, i] for i in range(n_modes)},
    })
    
    # Permutations sheet
    perm_df = pd.DataFrame({
        "patient_id": patient_ids,
        **{f"perm_{i+1}": perm[:, i] for i in range(n_modes)},
    })
    
    # Imputation sheet (optional)
    imputed_df = None
    if imputed_rows is not None:
        eig_imputed = np.zeros(len(patient_ids), dtype=bool)
        eig_imputed[imputed_rows] = True
        imputed_df = pd.DataFrame({
            "patient_id": patient_ids,
            "eig_imputed": eig_imputed,
        })
    
    # Energy fraction sheets
    energy_sheets = []
    for load_idx, load_name in enumerate(load_case_names):
        sheet_name = _sanitize_sheet_name(f"{load_idx+1}_{load_name}")
        energy_df = pd.DataFrame({
            "patient_id": patient_ids,
            **{f"energy_frac_mode_{i+1}": energy_sorted[:, i, load_idx] for i in range(n_modes)},
        })
        energy_sheets.append((sheet_name, energy_df))
    
    # Build metadata
    metadata_records = [(sheet, load_case_names[i]) for i, (sheet, _) in enumerate(energy_sheets)]
    metadata_records.append(("", ""))
    metadata_records.append(("IMPUTATION_SUMMARY", ""))
    
    if imputed_rows is not None:
        metadata_records.append(("n_patients_imputed", str(len(imputed_rows))))
        metadata_records.append(("patients_imputed", ", ".join(patient_ids[imputed_rows])))
        
        if invalid_mask is not None:
            for mode_idx in range(n_modes):
                n_imputed = int(invalid_mask[:, mode_idx].sum())
                if n_imputed > 0:
                    metadata_records.append((f"eig_{mode_idx+1}_imputed", str(n_imputed)))
            metadata_records.append(("total_imputed_values", str(int(invalid_mask.sum()))))
    else:
        metadata_records.append(("n_patients_imputed", "0"))
    
    metadata_df = pd.DataFrame(metadata_records, columns=["sheet_name", "load_case"])
    
    # Write Excel
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        eigen_df.to_excel(writer, sheet_name="modes", index=False)
        perm_df.to_excel(writer, sheet_name="permutations", index=False)
        if imputed_df is not None:
            imputed_df.to_excel(writer, sheet_name="imputed", index=False)
        metadata_df.to_excel(writer, sheet_name="metadata", index=False)
        for sheet_name, df_energy in energy_sheets:
            df_energy.to_excel(writer, sheet_name=sheet_name, index=False)


def create_sensitivity_workbook(
    output_path: Path,
    patient_ids: np.ndarray,
    sens_sorted: np.ndarray,
    param_names: list[str],
) -> None:
    """Write sensitivity workbook with one sheet per parameter."""
    n_modes = sens_sorted.shape[1]
    n_params = sens_sorted.shape[2]
    
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for j in range(n_params):
            sheet_name = _sanitize_sheet_name(param_names[j])
            df = pd.DataFrame({
                "patient_id": patient_ids,
                **{f"sens_mode_{i+1}": sens_sorted[:, i, j] for i in range(n_modes)},
            })
            df.to_excel(writer, sheet_name=sheet_name, index=False)


def main() -> None:
    """Generate eigenvalue and sensitivity workbooks from simulation outputs."""
    lm = logging_config.LoggingManager(str(SIMULATION_DIR.resolve()))
    lm.add_module(__name__, logging.DEBUG)
    lm.setup()
    lm.reconfigure_console("INFO")

    logger.info("Starting eigen data processing")
    logger.info(f"Simulation directory: {SIMULATION_DIR}")
    logger.info(f"Output file: {EIGEN_OUTPUT_PATH}")
    logger.info(f"Database directory: {DATABASE_DIR}")

    # Load raw data
    eigvals_raw = np.asarray(zarr.open(str(EIGENVALUES_ZARR))["data"][:])
    perm = np.asarray(zarr.open(str(PERMUTATIONS_ZARR))["data"][:], dtype=np.int32)
    energy_raw = np.asarray(zarr.open(str(ENERGY_ZARR))["data"][:])

    n_modes = perm.shape[1]
    patient_ids = load_patient_ids(DATABASE_DIR)

    # Load case names from metadata
    metadata = json.loads(SIMULATION_METADATA_PATH.read_text())
    load_case_names = list(metadata["mode_load_cases"])

    # Permute and impute
    eigvals_sorted = apply_mode_permutation(eigvals_raw, perm)
    eigvals_sorted, imputed_rows, invalid_mask = impute_invalid_eigenvalues(eigvals_sorted)

    if imputed_rows is not None:
        logger.warning(
            "Imputed eigenvalues for %s patients: %s",
            len(imputed_rows),
            ", ".join(patient_ids[imputed_rows]),
        )
        for mode_idx in range(n_modes):
            n_imputed = int(invalid_mask[:, mode_idx].sum())
            if n_imputed > 0:
                logger.warning("  Mode %d: %d imputed values", mode_idx + 1, n_imputed)

    energy_sorted = apply_mode_permutation(energy_raw, perm)

    # Generate eigen_data.xlsx
    SIMULATION_DATA_DIR.mkdir(parents=True, exist_ok=True)
    create_eigen_excel(
        EIGEN_OUTPUT_PATH,
        patient_ids,
        eigvals_sorted,
        perm,
        energy_sorted,
        load_case_names,
        imputed_rows,
        invalid_mask,
    )

    # Ligament sensitivities
    if LIGAMENT_SENS_ZARR.exists():
        lig_group = zarr.open(str(LIGAMENT_SENS_ZARR))
        lig_sens = apply_mode_permutation(np.asarray(lig_group["data"][:]), perm)
        lig_names = list(lig_group["data"].attrs["ligament_names"])
        create_sensitivity_workbook(
            LIGAMENT_OUTPUT_PATH,
            patient_ids,
            lig_sens,
            lig_names,
        )

    # Pretension sensitivities
    if PRETENSION_SENS_ZARR.exists():
        pretension_group = zarr.open(str(PRETENSION_SENS_ZARR))
        pretension_sens = apply_mode_permutation(np.asarray(pretension_group["data"][:]), perm)
        pretension_names = list(pretension_group["data"].attrs["ligament_names"])
        create_sensitivity_workbook(
            PRETENSION_OUTPUT_PATH,
            patient_ids,
            pretension_sens,
            pretension_names,
        )

    # Cartilage sensitivities
    if CARTILAGE_SENS_ZARR.exists():
        cart_group = zarr.open(str(CARTILAGE_SENS_ZARR))
        cart_sens = apply_mode_permutation(np.asarray(cart_group["data"][:]), perm)
        cart_names = list(cart_group["data"].attrs["parameter_names"])
        create_sensitivity_workbook(
            CARTILAGE_OUTPUT_PATH,
            patient_ids,
            cart_sens,
            cart_names,
        )


if __name__ == "__main__":
    main()
