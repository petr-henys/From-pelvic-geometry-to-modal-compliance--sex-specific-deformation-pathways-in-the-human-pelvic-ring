"""Batch processing utilities for pelvic morphometric analysis.

Computes per-sample morphological metrics across datasets:
- Geometric properties (volumes, surface areas) for pelvic subregions
- Mass distributions from bone density fields

Outputs allometry.xlsx file.
For eigenvalue/modal analysis data, see process_eigen_data.py.

Supports parallel processing for large datasets with progress tracking.

User Configuration Required:
----------------------------
BODY_INDICES: List of mesh body indices after split_bodies() corresponding to [LI, RI, S]
              Example: [0, 1, 2] means body 0=LI, body 1=RI, body 2=Sacrum
"""

from __future__ import annotations

import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Sequence, Any

import numpy as np
import pandas as pd
import pyvista as pv
from tqdm.auto import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import logging_config
from simulation.utils import ensure_nonnegative
from database import collect_patient_info, get_patientid_as_numpy

MM3_TO_CM3 = 1e-3  # 1 mm^3 = 0.001 cm^3
MM2_TO_CM2 = 1e-2  # 1 mm^2 = 0.01 cm^2

# Setup logging
logger = logging.getLogger(__name__)

# Explicit configuration
SIMULATION_DIR = PROJECT_ROOT / "results" / "ref_S1P_fixed_new2"
SIMULATION_DATA_DIR = SIMULATION_DIR / "data"
DATA_DIR = PROJECT_ROOT / "data"
ANATOMY_DIR = PROJECT_ROOT / "anatomy_data"
MESH_PATH = ANATOMY_DIR / "birth_canal" / "pelvic.vtk"
DATABASE_DIR = Path("/mnt/database/BoneDat/raw")
BODY_INDICES = [0, 1, 2]  # LI, RI, S
BODY_NAMES = ["LI", "RI", "S"]
N_WORKERS = 8
OUTPUT_PATH = SIMULATION_DATA_DIR / "allometry.xlsx"


def _extract_mesh_metrics(
    body: pv.UnstructuredGrid,
) -> dict[str, float]:
    """Extract geometric and mass properties from mesh body.
    
    Computes volume (cm³) and surface area (cm²). If the point-data field
    'RHO' is present, also computes total mass (g) via cell-wise ρ × V summation.
    
    Returns
    -------
    dict
        Keys: 'volume_cm3', 'area_cm2', optionally 'mass_g'
    """
    metrics = {
        "volume_cm3": float(body.volume * MM3_TO_CM3),
        "area_cm2": float(body.extract_surface().area * MM2_TO_CM2),
    }

    if "RHO" in body.point_data:
        body_cell = body.point_data_to_cell_data(pass_point_data=False).compute_cell_sizes(
            length=False, area=False, volume=True
        )
        rho = body_cell.cell_data["RHO"]
        vol_cm3 = body_cell.cell_data["Volume"] * MM3_TO_CM3
        metrics["mass_g"] = float((rho * vol_cm3).sum())

    return metrics


def split_mesh_bodies(mesh: pv.UnstructuredGrid) -> list[pv.UnstructuredGrid]:
    """Split mesh into separate bodies."""
    return list(mesh.split_bodies())


def select_bodies(parts: list[pv.UnstructuredGrid], indices: Sequence[int]) -> list[pv.UnstructuredGrid]:
    """Select specific bodies by index.
    
    Parameters
    ----------
    parts : list[pv.UnstructuredGrid]
        All mesh bodies from split_bodies()
    indices : Sequence[int]
        Indices of bodies to select (e.g., [0, 1, 2] for LI, RI, S)
        
    Returns
    -------
    list[pv.UnstructuredGrid]
        Selected bodies in the order specified by indices
    """
    return [parts[i] for i in indices]


def compute_template_metrics(
    mesh: pv.UnstructuredGrid,
) -> dict[str, list[float]]:
    """Compute mesh metrics (volumes, areas, masses) for all bodies."""

    parts = split_mesh_bodies(mesh)
    volumes, areas, masses = [], [], []

    for body in parts:
        metrics = _extract_mesh_metrics(body)
        volumes.append(metrics["volume_cm3"])
        areas.append(metrics["area_cm2"])
        if "mass_g" in metrics:
            masses.append(metrics["mass_g"])

    result = {"volumes": volumes, "areas": areas}
    if masses:
        result["masses"] = masses
    return result


def _compute_patient_metrics_worker(
    args: tuple[int, pv.UnstructuredGrid, np.ndarray, np.ndarray | None],
) -> tuple[int, dict[str, list[float]]]:
    """Worker function for parallel patient metric computation."""
    idx, base_mesh, coords, rho_data = args

    mesh = base_mesh.copy(deep=True)
    mesh.points[:] = coords
    if rho_data is not None:
        mesh.point_data["RHO"] = rho_data

    return idx, compute_template_metrics(mesh)


def _compute_patients_metrics_batch(
    mesh: pv.UnstructuredGrid,
    coordinates: np.ndarray,
    rho_patients: np.ndarray | None = None,
    *,
    n_workers: int = 8,
) -> dict[str, np.ndarray]:
    """Compute metrics for all patients (always parallel)."""
    samples = coordinates.shape[0]
    n_parts = len(split_mesh_bodies(mesh))

    # Initialize arrays
    result_arrays = {
        "volumes": np.empty((samples, n_parts), dtype=float),
        "areas": np.empty((samples, n_parts), dtype=float),
    }
    if rho_patients is not None:
        result_arrays["masses"] = np.empty((samples, n_parts), dtype=float)

    # Always parallel processing
    pbar = tqdm(total=samples, desc="Computing metrics", unit="patient")

    # Prepare work items
    work_items = [
        (i, mesh, coordinates[i], rho_patients[i] if rho_patients is not None else None)
        for i in range(samples)
    ]

    # Collect results in order
    collected_results: list[dict[str, list[float]] | None] = [None] * samples

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(_compute_patient_metrics_worker, item) for item in work_items]

        for future in as_completed(futures):
            idx, metrics = future.result()
            collected_results[idx] = metrics
            pbar.update(1)

    pbar.close()

    # Fill result arrays
    for idx in range(samples):
        metrics = collected_results[idx]
        assert metrics is not None
        result_arrays["volumes"][idx] = metrics["volumes"]
        result_arrays["areas"][idx] = metrics["areas"]
        if "masses" in metrics and "masses" in result_arrays:
            result_arrays["masses"][idx] = metrics["masses"]

    return result_arrays


def compute_patients_masses_on_template_geometry(
    mesh: pv.UnstructuredGrid,
    n_samples: int,
    rho_patients: np.ndarray,
    *,
    n_workers: int = 8,
) -> np.ndarray:
    """Compute patient masses using fixed template geometry (density-only effect)."""
    template_coords = np.broadcast_to(
        mesh.points[None, :, :], (n_samples,) + mesh.points.shape
    ).copy()
    metrics = _compute_patients_metrics_batch(
        mesh,
        template_coords,
        rho_patients,
        n_workers=n_workers,
    )
    return metrics["masses"]


    


def compute_isotropic_scales(
    patient_volumes: np.ndarray,
    template_volumes: Sequence[float],
) -> np.ndarray:
    """Compute isotropic scaling factors based on volume ratios."""
    template_total = float(np.sum(template_volumes))
    patient_totals = patient_volumes.sum(axis=1)
    return np.cbrt(patient_totals / template_total)


def load_data(data_dir: Path) -> dict[str, Any]:
    """Load required arrays: coordinates and density."""
    coordinates = np.load(data_dir / "X.npy")
    rho = ensure_nonnegative((np.load(data_dir / "I.npy") + 1024.0) / 1000.0, 0.1)
    return {"coordinates": coordinates, "rho": rho, "n_samples": coordinates.shape[0]}


# ---------------- Excel/sheet utilities ----------------

def main() -> None:
    """Regenerate patient allometry metrics workbook."""
    lm = logging_config.LoggingManager(str(SIMULATION_DIR.resolve()))
    lm.add_module(__name__, logging.DEBUG)
    lm.setup()
    lm.reconfigure_console("INFO")

    logger.info("Starting patient metrics collection")
    logger.info(f"Simulation directory: {SIMULATION_DIR}")
    logger.info(f"Output file: {OUTPUT_PATH}")
    logger.info(f"Body indices [LI, RI, S]: {BODY_INDICES}")
    logger.info(f"Database directory: {DATABASE_DIR}")

    data_dir = DATA_DIR

    # Load mesh
    logger.debug(f"Loading mesh from {MESH_PATH}")
    mesh = pv.read(MESH_PATH)

    # Load data
    logger.info("Loading data arrays")
    data = load_data(data_dir)
    n_samples = data["n_samples"]
    
    # Load patient IDs from database
    logger.info("Loading patient IDs from database")
    patient_info = collect_patient_info(DATABASE_DIR)
    patient_ids = get_patientid_as_numpy(patient_info)
    
    if patient_ids.shape[0] != n_samples:
        raise ValueError(
            f"Patient ID count mismatch: database has {patient_ids.shape[0]} patients, "
            f"but data has {n_samples} samples"
        )
    
    # Assign mean density to template mesh
    mesh.point_data["RHO"] = data["rho"].mean(0)

    # Split mesh into bodies and select the specified ones
    all_bodies = split_mesh_bodies(mesh)
    logger.info(f"Mesh split into {len(all_bodies)} bodies")
    
    selected_bodies = select_bodies(all_bodies, BODY_INDICES)
    logger.info(f"Selected bodies at indices {BODY_INDICES} for {BODY_NAMES}")

    # Compute template metrics for selected bodies only
    logger.info("Computing template metrics (scale=1)")
    template_volumes = []
    template_areas = []
    template_masses = []
    
    for body_idx, body in zip(BODY_INDICES, selected_bodies):
        metrics = _extract_mesh_metrics(body)
        template_volumes.append(metrics["volume_cm3"])
        template_areas.append(metrics["area_cm2"])
        if "mass_g" in metrics:
            template_masses.append(metrics["mass_g"])
        logger.debug(
            f"Body {body_idx}: volume={metrics['volume_cm3']:.2f} cm³, "
            f"area={metrics['area_cm2']:.2f} cm²"
        )

    # Compute patient-specific metrics
    logger.info(f"Computing patient-specific geometry metrics ({n_samples} samples, parallel)")
    patient_results = _compute_patients_metrics_batch(
        mesh,
        data["coordinates"],
        data["rho"],
        n_workers=N_WORKERS,
    )

    # Compute masses on fixed template geometry (density-only effect)
    logger.info("Computing masses on fixed template geometry (density-only)")
    patient_masses_templgeom_all = compute_patients_masses_on_template_geometry(
        mesh,
        n_samples,
        data["rho"],
        n_workers=N_WORKERS,
    )
    
    # Extract and select only the specified bodies
    patient_volumes_all = patient_results["volumes"]  # (n_samples, n_bodies)
    patient_areas_all = patient_results["areas"]
    patient_masses_all = patient_results["masses"]
    
    # Select only the bodies at BODY_INDICES
    patient_volumes = patient_volumes_all[:, BODY_INDICES]  # (n_samples, 3)
    patient_areas = patient_areas_all[:, BODY_INDICES]
    patient_masses = patient_masses_all[:, BODY_INDICES]
    patient_masses_templgeom = patient_masses_templgeom_all[:, BODY_INDICES]
    
    logger.debug(f"Selected patient metrics shape: {patient_volumes.shape}")

    # Compute isotropic scales
    isotropic_scales = compute_isotropic_scales(patient_volumes, template_volumes)

    # Prepare allometry dataframe
    logger.info("Preparing output dataframes")
    total_surface = patient_areas.sum(axis=1)
    total_volume = patient_volumes.sum(axis=1)
    total_mass = patient_masses.sum(axis=1)
    total_mass_templgeom = patient_masses_templgeom.sum(axis=1)
    template_total_volume = float(np.sum(template_volumes))
    allometry_data = {
        "patient_id": patient_ids,
        "scale": isotropic_scales,
        **{f"surf_{name}": patient_areas[:, i] for i, name in enumerate(BODY_NAMES)},
        "total_surface": total_surface,
        **{f"vol_{name}": patient_volumes[:, i] for i, name in enumerate(BODY_NAMES)},
        "total_volume": total_volume,
        "eff_thick": total_volume / total_surface,
        **{f"mass_{name}": patient_masses[:, i] for i, name in enumerate(BODY_NAMES)},
        "total_mass": total_mass,
        "eff_density": total_mass / total_volume,
        # Masses computed on fixed template geometry (density-only effect)
        **{f"mass_template_{name}": patient_masses_templgeom[:, i] for i, name in enumerate(BODY_NAMES)},
        "total_mass_template": total_mass_templgeom,
        "eff_density_template": total_mass_templgeom / template_total_volume,
        "Asymetry_Index": np.abs(patient_masses[:, 0] - patient_masses[:, 1]) / (
            (patient_masses[:, 0] + patient_masses[:, 1]) / 2
        )
    }

    # Create dataframes
    patient_allometry_df = pd.DataFrame(allometry_data)

    # Save results
    SIMULATION_DATA_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving results to {SIMULATION_DATA_DIR}")
    patient_allometry_path = OUTPUT_PATH
    logger.debug(f"Writing patient allometry and template metrics to {patient_allometry_path}")

    # Prepare template reference sheet (scale=1) for selected bodies
    ref_template_data = {
        "scale": [1.0],
        **{f"surf_{name}": [template_areas[i]] for i, name in enumerate(BODY_NAMES)},
        "total_surface": [float(np.sum(template_areas))],
        **{f"vol_{name}": [template_volumes[i]] for i, name in enumerate(BODY_NAMES)},
        "total_volume": [template_total_volume],
    }

    # If masses for template (using mean rho) are available, include them
    if len(template_masses) == len(BODY_NAMES):
        ref_template_data.update({
            **{f"mass_{name}": [template_masses[i]] for i, name in enumerate(BODY_NAMES)},
            "total_mass": [float(np.sum(template_masses))],
            "eff_density": [float(np.sum(template_masses)) / template_total_volume],
        })

    ref_template_df = pd.DataFrame(ref_template_data)

    with pd.ExcelWriter(patient_allometry_path, engine="openpyxl") as writer:
        # Write patient data first to keep compatibility with readers using default sheet
        patient_allometry_df.to_excel(writer, sheet_name="allometry", index=False)
        # Add template reference metrics
        ref_template_df.to_excel(writer, sheet_name="ref_template", index=False)
    
    logger.info("Patient metrics collection completed successfully")


if __name__ == "__main__":
    main()
