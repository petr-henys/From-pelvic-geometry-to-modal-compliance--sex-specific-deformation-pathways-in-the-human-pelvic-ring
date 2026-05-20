"""Zarr-based storage management for eigenstiffness simulation results.

Handles chunked array storage with Blosc compression for:
- Eigenvalues (samples × num_modes)
- Eigenvectors (samples × num_modes × num_dofs × gdim)
- Deformation fields and SIJ metrics (various shapes)

Chunk sizes optimized for memory-mapped access and parallel writes.
"""

from pathlib import Path

import logging
import zarr
import numpy as np
from typing import Tuple, Dict, Any, Sequence

logger = logging.getLogger(__name__)

class StorageManager:
    """Zarr array management for simulation datasets with automatic chunking.
    
    Creates and manages compressed Zarr stores for large-scale simulation outputs.
    Chunk sizes computed from config parameters (ZARR_*_CHUNK_MB) to balance
    memory overhead vs I/O efficiency.
    
    Storage layout:
    - results_dir/data/eigenvalues.zarr/data: (samples, num_modes) float64
    - results_dir/data/eigenvectors.zarr/data: (samples, num_modes, ...) float64
    - Additional arrays for deformations, SIJ metrics, modal analysis
    """
    
    def __init__(self, config):  # Avoid strict typing; config object is user-defined
        self.config = config
        self.results_dir = Path(config["RESULTS_DIR"]).resolve()
        self._data_dir = self.results_dir / "data"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self._data_dir.mkdir(parents=True, exist_ok=True)
    
    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    # setup_zarr_storage removed; use add_field_storage for all datasets
    
    def save_metadata(self, metadata: Dict[str, Any]) -> None:
        """Save simulation metadata to JSON.
        
        Stores complete metadata dict to simulation_metadata.json.
        """
        import json
        metadata_path = self.results_dir / "simulation_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        logger.info("Metadata saved to: %s", metadata_path)
    
    def validate_storage_access(self, eigvals_dataset: zarr.Array, eigvecs_store: zarr.Group) -> bool:
        """Validate Zarr store shapes and accessibility.

        Returns False (and logs an error) if the provided datasets do not
        match the expected dimensionality or required arrays are missing.
        """
        if eigvals_dataset.ndim != 2:
            logger.error(
                "Invalid eigenvalue dataset shape: %s, expected 2D array",
                eigvals_dataset.shape,
            )
            return False

        if 'data' not in eigvecs_store:
            logger.error("Eigenvector data array not found in store")
            return False

        eigvecs_dataset = eigvecs_store['data']
        if eigvecs_dataset.ndim < 3:
            logger.error(
                "Invalid eigenvector dataset shape: %s, expected at least 3D array",
                eigvecs_dataset.shape,
            )
            return False

        logger.debug("Storage OK: eigenvalues%s eigenvectors%s", eigvals_dataset.shape, eigvecs_dataset.shape)
        return True
    
    # ------------------------------------------------------------------
    # New flexible API
    # ------------------------------------------------------------------
    def add_field_storage(
        self,
        field_name: str,
        shape: Tuple[int, ...],
        dims_description: str,
        dtype: Any = np.float64,
        max_chunk_mb: int | None = None,
    ) -> zarr.Array:
        """Create (or recreate) a Zarr store for an arbitrary field.
        Parameters:
          field_name: logical name (creates <field_name>.zarr)
          shape: full dataset shape
          dims_description: textual description of axes
          dtype: numeric dtype (default float64)
          max_chunk_mb: override chunk target (MB) else inferred from config
        Returns: zarr.Array dataset
        """
        if len(shape) == 0:
            raise ValueError("Shape cannot be empty")
        samples = shape[0]
        self._validate_positive(samples, 'shape[0] (samples)')
        bytes_per_element = np.dtype(dtype).itemsize
        # Estimate per-sample size (assume axis 0 = samples)
        per_sample_elems = int(np.prod(shape[1:])) if len(shape) > 1 else 1
        bytes_per_sample = per_sample_elems * bytes_per_element
        if max_chunk_mb is None:
            # Fallback config keys; allow generic override
            max_chunk_mb = self.config.get(
                f"ZARR_{field_name.upper()}_MAX_CHUNK_MB",
                self.config.get("ZARR_GENERIC_MAX_CHUNK_MB", 16),
            )
        chunk_samples = self._samples_per_chunk(bytes_per_sample, max_chunk_mb, samples)
        chunks = (chunk_samples,) + shape[1:]
        group = self._open_group(f"{field_name}.zarr")
        ds = self._create_or_replace_dataset(
            group,
            name='data',
            shape=shape,
            chunks=chunks,
            dtype=dtype,
        )
        group.attrs.update({
            'field_name': field_name,
            'shape_description': dims_description,
            'config_snapshot': self._config_snapshot(),
            'num_samples': samples,
            'dtype': str(np.dtype(dtype)),
        })
        logger.info(
            "Field '%s' storage shape=%s chunks=%s (~%.2fMB total)",
            field_name,
            ds.shape,
            chunks,
            float(np.prod(shape) * bytes_per_element) / (1024 ** 2),
        )
        return ds
    
    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _open_group(self, name: str) -> zarr.Group:
        """Open or create (overwrite) a Zarr group inside data directory."""
        path = self._data_dir / name
        return zarr.open_group(str(path), mode='w')  # Always recreate for clean runs
    
    def _create_or_replace_dataset(self, group: zarr.Group, name: str, *, shape: Tuple[int, ...],
                                   chunks: Tuple[int, ...], dtype: Any) -> zarr.Array:
        if name in group:
            del group[name]
        return group.zeros(name=name, shape=shape, chunks=chunks, dtype=dtype)
    
    def _config_snapshot(self) -> Dict[str, Any]:
        """Get complete config dict with ligament names ordered near stiffness entries."""
        if isinstance(self.config, dict):
            base = dict(self.config)
        elif hasattr(self.config, "config") and isinstance(self.config.config, dict):
            base = dict(self.config.config)
        else:
            base = dict(self.config)

        ligament_names = base.get("LIGAMENT_NAMES")
        if ligament_names is None:
            return base

        ordered: Dict[str, Any] = {}
        names_inserted = False
        for key, value in base.items():
            if key == "LIGAMENT_NAMES":
                continue
            if (
                not names_inserted
                and key in {"LIGAMENT_STIFFNESSES", "LIGAMENT_PRETENSIONS"}
            ):
                ordered["LIGAMENT_NAMES"] = ligament_names
                names_inserted = True
            ordered[key] = value

        if not names_inserted:
            ordered["LIGAMENT_NAMES"] = ligament_names

        return ordered
    
    def _validate_positive(self, value: int, name: str) -> None:
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
    
    def _calculate_optimal_chunks(self, num_samples: int, eigvec_shape: Tuple[int, ...]) -> Tuple[Tuple[int, int], Tuple[int, ...]]:
        n_eigvals = self.config["NUM_EIGENVALUES"]
        max_eigval_chunk = self.config.get("ZARR_EIGVAL_CHUNK_SIZE", 100)
        eigval_chunks = (min(max_eigval_chunk, num_samples), n_eigvals)
        if len(eigvec_shape) < 2:
            raise ValueError(f"Invalid eigenvector shape: {eigvec_shape}")
        nodes_per_mode = int(np.prod(eigvec_shape[1:]))
        bytes_per_mode_per_sample = nodes_per_mode * 8  # float64 size
        bytes_per_sample = n_eigvals * bytes_per_mode_per_sample
        max_chunk_mb = self.config.get("ZARR_EIGVEC_MAX_CHUNK_MB", 16)
        eigvec_chunk_samples = self._samples_per_chunk(bytes_per_sample, max_chunk_mb, num_samples)
        eigvec_chunks = (eigvec_chunk_samples, n_eigvals) + eigvec_shape[1:]
        logger.debug("Chunks eigvals=%s eigvecs=%s", eigval_chunks, eigvec_chunks)
        return eigval_chunks, eigvec_chunks
    
    def _samples_per_chunk(self, bytes_per_sample: int, max_chunk_mb: int, num_samples: int) -> int:
        max_chunk_bytes = max_chunk_mb * 1024 * 1024
        samples = max(1, int(max_chunk_bytes / max(1, bytes_per_sample)))
        return min(samples, num_samples)
    
    def _infer_space_layout(self, V: Any) -> Tuple[int, int]:
        gdim = V.mesh.geometry.dim
        n_scalar = V.dofmap.index_map.size_global * V.dofmap.index_map_bs
        if n_scalar % gdim != 0:
            raise ValueError(
                f"Scalar DoF count ({n_scalar}) not divisible by gdim ({gdim}); "
                "reshaping is inconsistent. Function space configuration error."
            )
        n_nodes = n_scalar // gdim
        return gdim, n_nodes
    
    def _log_estimated_sizes(self, name_shape_map: Dict[str, Sequence[int]]) -> None:
        total = 0.0
        for name, shape in name_shape_map.items():
            size_mb = float(np.prod(shape)) * 8 / (1024 ** 2)
            total += size_mb
            logger.info("  Estimated storage: %s=%.1fMB", name, size_mb)
        if len(name_shape_map) > 1:
            logger.info("  Estimated storage total=%.1fMB", total)
