#!/usr/bin/env python3
"""Render mode shapes side-by-side using PyVista.

Loads reference and sample eigenvectors, applies Zarr permutation reordering,
and renders galleries with undeformed mesh colored by magnitude and deformed wireframe overlay.

View definitions:
  AP: anterior -> posterior (camera at -Y looking +Y, up +Z)
  ML: medial/lateral (camera at -X looking +X, up +Z)
  CC: cranial -> caudal (camera at +Z looking -Z, up +Y)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pyvista as pv
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parent))

from logging_config import LoggingManager
from config import SimulationConfig

logger = logging.getLogger("render_modes")


def _load_mesh(mesh_path: Path) -> pv.DataSet:
    """Read visualization mesh from disk."""
    return pv.read(str(mesh_path))


def _load_reference_modes(results_dir: Path) -> np.ndarray:
    """Load reference eigenvectors (modes, nodes, 3)."""
    npz_path = results_dir / "reference_solution.npz"
    data = np.load(str(npz_path))
    return np.asarray(data["eigenvectors"])


def _load_sample_modes(results_dir: Path):
    """Load sample eigenvectors lazily (samples, modes, nodes, 3)."""
    evc_path = results_dir / "data" / "eigenvectors.zarr"
    evc = zarr.open(str(evc_path), mode="r")["data"]
    return evc


def _load_permutations(results_dir: Path) -> np.ndarray:
    """Load permutation array (samples, ref_modes) where perm[sample, i_ref] = j_sample."""
    perm_path = results_dir / "data" / "eig_permutations.zarr"
    perm = zarr.open(str(perm_path), mode="r")["data"]
    return np.asarray(perm, dtype=np.int32)


def _compute_summary_modes(
    sample_vecs,
    perm: np.ndarray,
) -> list[tuple[str, np.ndarray]]:
    """Compute percentile summary modes without materializing full reordered tensor.

    Processes one reference mode at a time to keep memory bounded.
    """
    n_samples, n_modes, n_nodes, n_dim = sample_vecs.shape
    n_ref_modes = perm.shape[1]
    if n_ref_modes > n_modes:
        raise ValueError(f"Permutation ref modes ({n_ref_modes}) exceed available modes ({n_modes})")

    q25 = np.zeros((n_ref_modes, n_nodes, n_dim), dtype=np.float64)
    median = np.zeros((n_ref_modes, n_nodes, n_dim), dtype=np.float64)
    q75 = np.zeros((n_ref_modes, n_nodes, n_dim), dtype=np.float64)

    for i_ref in range(n_ref_modes):
        logger.info("Computing sample summaries for reference mode %d/%d", i_ref + 1, n_ref_modes)
        stack = np.full((n_samples, n_nodes, n_dim), np.nan, dtype=np.float64)
        for i_sample in range(n_samples):
            j_sample = int(perm[i_sample, i_ref])
            if 0 <= j_sample < n_modes:
                stack[i_sample] = np.asarray(sample_vecs[i_sample, j_sample], dtype=np.float64)
        q = np.nanpercentile(stack, [25.0, 50.0, 75.0], axis=0)
        q25[i_ref] = np.nan_to_num(q[0], nan=0.0)
        median[i_ref] = np.nan_to_num(q[1], nan=0.0)
        q75[i_ref] = np.nan_to_num(q[2], nan=0.0)

    return [("median", median), ("Q25", q25), ("Q75", q75)]


def _compute_parallel_scale(bounds: tuple[float, ...], aspect: float) -> float:
    """Return VTK parallel scale to fit bounds in viewport with generous margin."""
    xmin, xmax, _, _, zmin, zmax = bounds
    width = xmax - xmin
    height = zmax - zmin
    half_h_required = max(0.5 * height, 0.5 * width / max(aspect, 1e-6))
    return 1.1 * half_h_required


def _camera_pose(center: np.ndarray, distance: float, view: str) -> tuple:
    """Return camera position, focal point, and up vector."""
    c = tuple(center.tolist())
    views = {
        "AP": ((center[0], center[1] - distance, center[2]), c, (0.0, 0.0, 1.0)),
        "ML": ((center[0] - distance, center[1], center[2]), c, (0.0, 0.0, 1.0)),
        "CC": ((center[0], center[1], center[2] + distance), c, (0.0, 1.0, 0.0)),
    }
    return views[view.upper()]


def render_modes_gallery(
    mesh: pv.DataSet,
    eigenvectors: np.ndarray,
    output_path: Path,
    scale: float,
    modes_per_row: int,
    camera_view: str,
) -> None:
    """Render tiled gallery without borders by composing per-mode tiles.

    Renders each mode in its own off-screen, square plotter (no subplot grid),
    then stitches the 12 tiles into a 2×6 (or specified) mosaic. This avoids
    any internal borders or spacing added by multi-viewport rendering.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_modes = eigenvectors.shape[0]
    n_cols = min(modes_per_row, n_modes)
    n_rows = int(np.ceil(n_modes / n_cols))

    tile = 480  # px per tile (square)
    logger.info("Rendering %d modes (tiles %dx%d): %s", n_modes, n_rows, n_cols, output_path)

    center = np.asarray(mesh.center)
    distance = mesh.length * 2.0
    parallel_scale = _compute_parallel_scale(mesh.bounds, 1.0)

    from analysis.publication_rendering import create_publication_plotter

    tiles: list[np.ndarray] = []
    for mode_idx, vec in enumerate(eigenvectors):
        p = create_publication_plotter(
            window_size=(tile, tile),
            enable_ssaa=True,
            enable_depth_peeling=True,
            enable_ssao=False,
        )

        # Undeformed mesh colored by magnitude
        magnitudes = np.linalg.norm(vec, axis=-1)
        base = mesh.copy(deep=True)
        base.point_data.clear()
        base.point_data["magnitude"] = magnitudes
        surf_base = base.extract_surface(algorithm='dataset_surface').compute_normals(point_normals=True, feature_angle=60.0)
        p.add_mesh(
            surf_base,
            scalars="magnitude",
            cmap="rainbow",
            smooth_shading=True,
            ambient=0.28,
            diffuse=0.80,
            specular=0.12,
            show_scalar_bar=False,
        )

        # Deformed mesh as subtle wireframe
        deformed = mesh.copy(deep=True)
        deformed.points = mesh.points + scale * vec
        surf_deformed = deformed.extract_surface(algorithm='dataset_surface')
        p.add_mesh(
            surf_deformed,
            color="#222222",
            style="wireframe",
            line_width=1.0,
            opacity=0.20,
        )

        cam_pos = _camera_pose(center, distance, camera_view)
        p.camera_position = [cam_pos[0], cam_pos[1], cam_pos[2]]
        p.camera.ParallelProjectionOn()
        p.camera.parallel_scale = parallel_scale

        p.add_text(
            f"Mode {mode_idx+1}", position="upper_left", font_size=14, shadow=False, color="#111827"
        )

        img = p.screenshot(return_img=True)
        p.close()
        tiles.append(img)

    # Stitch tiles into mosaic (row-major), padding last row if needed
    rows = []
    for r in range(n_rows):
        row_tiles = tiles[r * n_cols:(r + 1) * n_cols]
        if not row_tiles:
            continue
        
        # Pad incomplete rows with white tiles
        n_missing = n_cols - len(row_tiles)
        if n_missing > 0:
            blank_tile = np.full((tile, tile, 3), 255, dtype=row_tiles[0].dtype)
            row_tiles.extend([blank_tile] * n_missing)
        
        rows.append(np.concatenate(row_tiles, axis=1))
    mosaic = np.concatenate(rows, axis=0)

    import matplotlib.pyplot as _plt  # local import to avoid global state
    _plt.imsave(str(output_path), mosaic)


def _main() -> None:
    """Render mode galleries for reference and sample statistics."""
    preferred_results_dir = (Path(__file__).resolve().parent / "results" / "ref_S1P_fixed_new2").resolve()
    if preferred_results_dir.exists():
        results_dir = preferred_results_dir
    else:
        cfg = SimulationConfig()
        results_dir = Path(cfg["RESULTS_DIR"]).resolve()
    logger.info("Using results directory: %s", results_dir)
    LoggingManager(str(results_dir)).setup()
    
    # Load data
    mesh_path = results_dir / "paraview" / "reference_fields.vtk"
    mesh = _load_mesh(mesh_path)
    
    ref_vecs = _load_reference_modes(results_dir)
    logger.info("Reference modes: %s", ref_vecs.shape)
    
    sample_vecs = _load_sample_modes(results_dir)
    logger.info("Sample modes (raw): %s", sample_vecs.shape)
    
    # Compute sample summaries in reference-mode ordering (streaming to avoid OOM)
    perm = _load_permutations(results_dir)
    logger.info("Permutations: %s", perm.shape)
    summaries = _compute_summary_modes(sample_vecs, perm)
    
    # Render configuration
    scale = 2e4
    modes_per_row = 3
    camera_view = "AP"  # Options: AP, ML, CC
    
    vis_folder = results_dir / "modes_visualisation"
    vis_folder.mkdir(parents=True, exist_ok=True)
    
    # Render galleries
    galleries = [("reference", ref_vecs), *summaries]

    for label, vecs in galleries:
        out_path = vis_folder / f"modes_{label}_{camera_view}.png"
        logger.info("Rendering %s: %s", label, out_path)
        render_modes_gallery(mesh, vecs, out_path, scale, modes_per_row, camera_view)
        logger.info("Saved: %s", out_path)


if __name__ == "__main__":
    _main()
