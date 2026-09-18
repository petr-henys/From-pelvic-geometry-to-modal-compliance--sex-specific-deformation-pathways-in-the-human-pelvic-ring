"""Data loading for spectral panel analysis.

All I/O (Zarr, NumPy, JSON landmarks, FEBio mesh) lives here.
"""
from __future__ import annotations

import warnings
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from analysis.spectral_config import (
    DIMS_ZARR,
    DIMS_ZARR_PALPATION,
    LANDMARK_DIR,
    RAW_DATA_DIR,
    REPO_ROOT,
)


# ============================================================
# Zarr helpers
# ============================================================
def _open_zarr_array(zarr_path: Path, key: str = "data") -> np.ndarray:
    """Load a Zarr array, handling v2/v3 API differences."""
    import zarr

    store = zarr.open_group(str(zarr_path), mode="r")
    if key in store:
        return np.asarray(store[key][:])
    arr = zarr.open_array(str(zarr_path), mode="r")
    return np.asarray(arr[:])


def _open_zarr_array_obj(zarr_path: Path, key: str = "data"):
    """Open a Zarr array object without loading it into memory."""
    import zarr

    store = zarr.open_group(str(zarr_path), mode="r")
    if key in store:
        return store[key]
    return zarr.open_array(str(zarr_path), mode="r")


# ============================================================
# Domain data loaders
# ============================================================
def load_eigenvalues(data_dir: Path) -> np.ndarray:
    """Load eigenvalues array (N_subjects x N_modes)."""
    return _open_zarr_array(data_dir / "eigenvalues.zarr")


def load_permutations(data_dir: Path) -> np.ndarray:
    """Load mode permutation array (N_subjects x N_modes)."""
    return _open_zarr_array(data_dir / "eig_permutations.zarr")


def load_eigenvectors_obj(data_dir: Path):
    """Open eigenvectors Zarr array (lazy, not loaded into memory)."""
    return _open_zarr_array_obj(data_dir / "eigenvectors.zarr")


def load_metadata() -> tuple[np.ndarray, np.ndarray]:
    """Load sex and age arrays."""
    return np.load(RAW_DATA_DIR / "sex.npy"), np.load(RAW_DATA_DIR / "age.npy")


def load_pelvic_dimensions() -> dict[str, np.ndarray]:
    """Load pelvic dimensions with source-aware aliases.

    Source policy
    -------------
    - Inlet AP / inlet transverse remain sourced from ``dimensions_ref.zarr``
      (birth-canal landmark set).
    - Outlet AP / outlet transverse / biischiadic width are sourced from
      ``dimensions_michal.zarr`` (palpation landmark set).

    Raw source-specific keys are kept when available, and standardised aliases
    are overlaid for downstream analyses:
      ``BispinousWidth`` -> ``BiischiadicWidth``
      ``TransverseOutletDiameter`` -> ``BituberousWidth``
      ``AnteriorPosteriorOutletDiameter`` -> ``OutletAP``
    """
    import zarr

    dims: dict[str, np.ndarray] = {}

    if DIMS_ZARR.exists():
        store = zarr.open_group(str(DIMS_ZARR), mode="r")
        dims.update({k: np.asarray(store[k][:]) for k in store})
    else:
        warnings.warn(f"Missing pelvic dimensions store: {DIMS_ZARR}", stacklevel=2)

    if DIMS_ZARR_PALPATION.exists():
        store_pal = zarr.open_group(str(DIMS_ZARR_PALPATION), mode="r")
        dims_pal = {k: np.asarray(store_pal[k][:]) for k in store_pal}
        dims.update(dims_pal)

        # Preserve the birth-canal outlet metrics under explicit names before
        # replacing the standard aliases with the palpation-based definitions.
        for raw_key in (
            "BispinousWidth",
            "TransverseOutletDiameter",
            "AnteriorPosteriorOutletDiameter",
        ):
            if raw_key in dims and f"{raw_key}_BirthCanal" not in dims:
                dims[f"{raw_key}_BirthCanal"] = np.asarray(dims[raw_key])

        alias_map = {
            "BiischiadicWidth": "BispinousWidth",
            "BituberousWidth": "TransverseOutletDiameter",
            "OutletAP": "AnteriorPosteriorOutletDiameter",
        }
        for src_key, alias_key in alias_map.items():
            if src_key in dims_pal:
                dims[alias_key] = dims_pal[src_key]
    else:
        warnings.warn(
            f"Missing palpation dimensions store: {DIMS_ZARR_PALPATION}",
            stacklevel=2,
        )

    return dims


def load_inlet_landmarks() -> dict[str, np.ndarray]:
    """Load AP and ML inlet landmark coordinates from 3D Slicer markup JSON."""
    from simulation.utils import load_json_points

    landmarks: dict[str, np.ndarray] = {}
    target_files = {
        "AP": "anterior_posterior_inlet_diameter.mrk.json",
        "ML": "pelvis_inlet_transverse.mrk.json",
    }
    for key, fname in target_files.items():
        fpath = LANDMARK_DIR / fname
        if not fpath.exists():
            warnings.warn(f"Missing landmark file: {fpath}", stacklevel=2)
            continue
        pts = np.asarray(load_json_points(fpath), dtype=float)
        if pts.shape[0] < 2:
            warnings.warn(f"Landmark file has <2 points: {fpath}", stacklevel=2)
            continue
        landmarks[key] = pts[:2]
    return landmarks


def load_outlet_landmarks() -> dict[str, np.ndarray]:
    """Load biischiadic, bituberous, and outlet AP landmark coordinates from 3D Slicer markup JSON."""
    from simulation.utils import load_json_points
    from analysis.spectral_config import OUTLET_LANDMARK_DIR

    landmarks: dict[str, np.ndarray] = {}
    target_files = {
        "BIS": "biischiadic.mrk.json",
        "BIT": "bituberous.mrk.json",
        "OUTLETAP": "outlet_AP.mrk.json",
    }
    for key, fname in target_files.items():
        fpath = OUTLET_LANDMARK_DIR / fname
        if not fpath.exists():
            warnings.warn(f"Missing landmark file: {fpath}", stacklevel=2)
            continue
        pts = np.asarray(load_json_points(fpath), dtype=float)
        if pts.shape[0] < 2:
            warnings.warn(f"Landmark file has <2 points: {fpath}", stacklevel=2)
            continue
        landmarks[key] = pts[:2]
    return landmarks


@lru_cache(maxsize=1)
def load_reference_space():
    """Rebuild the serial FE space in the ordering of the stored solver vectors.

    Never substitute FEBio file order for DOLFINx degree-of-freedom order.
    The mesh path is taken from the simulation metadata, not a plotting default.
    """
    from dolfinx import fem
    from mpi4py import MPI
    from simulation.febio_parser import FEBio2Dolfinx
    if MPI.COMM_WORLD.size != 1:
        raise RuntimeError("Stored cohort arrays require serial postprocessing")
    metadata = json.loads((REPO_ROOT / "results/ref_S1P_fixed_new2/simulation_metadata.json").read_text())
    parser = FEBio2Dolfinx(str(REPO_ROOT / metadata["config"]["DOMAIN_MESH_PATH"]))
    domain = parser.mesh_dolfinx
    V = fem.functionspace(domain, ("P", 1, (domain.geometry.dim,)))
    if V.tabulate_dof_coordinates().shape[0] != metadata["reference_eigvec_shape"][1]:
        raise ValueError("Reconstructed FE space does not match stored eigenvectors")
    return parser, V


def load_fe_mesh_coords() -> np.ndarray:
    """Coordinates in the solver DOF ordering of eigenvectors.zarr."""
    _, V = load_reference_space()
    return np.asarray(V.tabulate_dof_coordinates(), dtype=float)


def load_template_and_shapes() -> tuple[np.ndarray, np.ndarray] | None:
    """Load template point cloud and SSM shapes array.

    Returns (template_points, shapes) with shapes (N_subjects, N_points, 3)
    or None if files are missing.
    """
    import pyvista as pv

    template_path = REPO_ROOT / "data" / "pelvic.vtk"
    shapes_path = REPO_ROOT / "data" / "X.npy"
    for p in (template_path, shapes_path):
        if not p.exists():
            warnings.warn(f"Missing file for patient-specific b: {p}", stacklevel=2)
            return None
    tpl = pv.read(str(template_path))
    shapes = np.load(str(shapes_path))
    return np.asarray(tpl.points, dtype=float), shapes


@lru_cache(maxsize=1)
def load_mass_matrix():
    """Consistent reference L2 mass matrix in stored solver DOF ordering.

    Missing DOLFINx is an error: silently changing the metric would alter the
    statistical estimand and make published subspace results irreproducible.
    """
    from dolfinx import fem
    from dolfinx.fem import petsc
    import ufl
    from scipy.sparse import csr_matrix
    _, V = load_reference_space()
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    form = fem.form(ufl.inner(u, v) * ufl.dx)
    M = petsc.assemble_matrix(form)
    M.assemble()
    ai, aj, av = M.getValuesCSR()
    result = csr_matrix((av.copy(), aj.copy(), ai.copy()), shape=M.getSize())
    M.destroy()
    return result
