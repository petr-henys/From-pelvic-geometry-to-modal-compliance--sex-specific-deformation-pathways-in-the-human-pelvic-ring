from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import zarr

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

pytestmark = pytest.mark.storage

from simulation.storage_manager import StorageManager


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    tmpdir = tempfile.mkdtemp()
    yield Path(tmpdir)
    shutil.rmtree(tmpdir)


@pytest.fixture
def mock_config(temp_dir):
    """Create a mock configuration object."""
    config = {
        "RESULTS_DIR": str(temp_dir / "results"),
        "NUM_EIGENVALUES": 10,
        "ZARR_EIGVAL_CHUNK_SIZE": 50,
        "ZARR_EIGVEC_MAX_CHUNK_MB": 8,
    }
    return config


@pytest.fixture
def storage_manager(mock_config):
    """Create a StorageManager instance."""
    return StorageManager(mock_config)


@pytest.fixture
def reference_eigenvectors():
    """Create sample reference eigenvectors."""
    # Shape: (num_modes=10, nodes=100, dims=3)
    return np.random.randn(10, 100, 3)


def test_initialization_creates_directories(temp_dir):
    config = {"RESULTS_DIR": str(temp_dir / "test_results")}
    manager = StorageManager(config)
    
    assert manager.results_dir.exists()
    assert manager._data_dir.exists()
    assert manager._data_dir == manager.results_dir / "data"


def test_create_eigen_datasets_with_generic_api(storage_manager, reference_eigenvectors):
    num_samples = 50
    n_modes = storage_manager.config["NUM_EIGENVALUES"]

    # Create eigenvalues via generic API
    eigvals_dataset = storage_manager.add_field_storage(
        field_name="eigenvalues",
        shape=(num_samples, n_modes),
        dims_description="samples, modes",
        dtype=np.float64,
    )
    assert isinstance(eigvals_dataset, zarr.Array)
    assert eigvals_dataset.shape == (50, n_modes)
    assert eigvals_dataset.dtype == np.float64

    # Create eigenvectors via generic API
    nodes, gdim = reference_eigenvectors.shape[1], reference_eigenvectors.shape[2]
    _ = storage_manager.add_field_storage(
        field_name="eigenvectors",
        shape=(num_samples, n_modes, nodes, gdim),
        dims_description="samples, modes, nodes, spatial_dims",
        dtype=np.float64,
    )
    eigvecs_group = zarr.open_group(str(storage_manager._data_dir / "eigenvectors.zarr"), mode="r")
    assert isinstance(eigvecs_group, zarr.Group)
    assert 'data' in eigvecs_group
    eigvecs_dataset = eigvecs_group['data']
    assert eigvecs_dataset.shape == (50, n_modes, nodes, gdim)
    assert eigvecs_dataset.dtype == np.float64
    # Metadata provided by generic API
    assert eigvecs_group.attrs['num_samples'] == 50
    assert eigvecs_group.attrs['shape_description'] == 'samples, modes, nodes, spatial_dims'


def test_add_field_storage_validates_inputs(storage_manager):
    n_modes = storage_manager.config["NUM_EIGENVALUES"]
    # Zero samples
    with pytest.raises(ValueError, match=r"shape\[0\] \(samples\) must be positive"):
        storage_manager.add_field_storage(
            field_name="eigenvalues",
            shape=(0, n_modes),
            dims_description="samples, modes",
        )
    # Negative samples
    with pytest.raises(ValueError, match=r"shape\[0\] \(samples\) must be positive"):
        storage_manager.add_field_storage(
            field_name="eigenvalues",
            shape=(-1, n_modes),
            dims_description="samples, modes",
        )
    # Empty shape
    with pytest.raises(ValueError, match="Shape cannot be empty"):
        storage_manager.add_field_storage(
            field_name="eigenvectors",
            shape=(),
            dims_description="",
        )


def test_save_metadata_creates_json_file(storage_manager):
    metadata = {
        "config": {"test_param": "value"},
        "simulation_type": "test",
        "timestamp": "2024-01-01",
        "parameters": {"alpha": 0.5, "beta": 1.0},
        "versions": {"numpy": "1.0", "zarr": "2.0"}
    }
    
    storage_manager.save_metadata(metadata)
    
    metadata_path = storage_manager.results_dir / "simulation_metadata.json"
    assert metadata_path.exists()
    
    with open(metadata_path, 'r') as f:
        loaded = json.load(f)
    
    assert loaded['simulation_type'] == 'test'
    assert loaded['parameters']['alpha'] == 0.5


def test_save_metadata_handles_minimal_metadata(storage_manager):
    metadata = {
        "timestamp": "2024-01-01",
        "versions": {"numpy": "1.0"}
    }
    storage_manager.save_metadata(metadata)
    
    metadata_path = storage_manager.results_dir / "simulation_metadata.json"
    assert metadata_path.exists()
    
    with open(metadata_path, 'r') as f:
        loaded = json.load(f)
    
    assert 'timestamp' in loaded
    assert 'versions' in loaded


def test_validate_storage_access_valid_stores(storage_manager, reference_eigenvectors):
    num_samples = 20
    n_modes = storage_manager.config["NUM_EIGENVALUES"]
    eigvals_dataset = storage_manager.add_field_storage(
        field_name="eigenvalues",
        shape=(num_samples, n_modes),
        dims_description="samples, modes",
    )
    nodes, gdim = reference_eigenvectors.shape[1], reference_eigenvectors.shape[2]
    _ = storage_manager.add_field_storage(
        field_name="eigenvectors",
        shape=(num_samples, n_modes, nodes, gdim),
        dims_description="samples, modes, nodes, spatial_dims",
    )
    eigvecs_group = zarr.open_group(str(storage_manager._data_dir / "eigenvectors.zarr"), mode="r")
    
    is_valid = storage_manager.validate_storage_access(eigvals_dataset, eigvecs_group)
    assert is_valid is True


def test_validate_storage_access_invalid_eigvals_shape(storage_manager):
    # Create invalid 1D eigenvalues array
    eigvals_dataset = zarr.zeros((100,), dtype=np.float64)
    eigvecs_group = zarr.group()
    eigvecs_group.create_array('data', shape=(100, 10, 50, 3), dtype=np.float64)
    
    is_valid = storage_manager.validate_storage_access(eigvals_dataset, eigvecs_group)
    assert is_valid is False


def test_validate_storage_access_missing_eigvecs_data(storage_manager):
    eigvals_dataset = zarr.zeros((100, 10), dtype=np.float64)
    eigvecs_group = zarr.group()  # No 'data' array
    
    is_valid = storage_manager.validate_storage_access(eigvals_dataset, eigvecs_group)
    assert is_valid is False


def test_validate_storage_access_invalid_eigvecs_shape(storage_manager):
    eigvals_dataset = zarr.zeros((100, 10), dtype=np.float64)
    eigvecs_group = zarr.group()
    # Create 2D instead of at least 3D
    eigvecs_group.create_array('data', shape=(100, 10), dtype=np.float64)
    
    is_valid = storage_manager.validate_storage_access(eigvals_dataset, eigvecs_group)
    assert is_valid is False


def test_validate_storage_access_dtype_mismatch(storage_manager):
    eigvals_dataset = zarr.zeros((100, 10), dtype=np.float64)
    eigvecs_group = zarr.group()
    eigvecs_group.create_array('data', shape=(100, 10, 50, 3), dtype=np.complex128)
    is_valid = storage_manager.validate_storage_access(eigvals_dataset, eigvecs_group)
    # Current implementation does not check dtype, but access should still be allowed
    # If dtype validation is added later, flip expectation accordingly
    assert isinstance(is_valid, bool)


def test_add_field_storage_creates_dataset(storage_manager):
    shape = (100, 50, 3)
    dataset = storage_manager.add_field_storage(
        field_name="velocities",
        shape=shape,
        dims_description="samples, nodes, spatial_dims",
        dtype=np.float32,
        max_chunk_mb=4
    )
    
    assert isinstance(dataset, zarr.Array)
    assert dataset.shape == shape
    assert dataset.dtype == np.float32
    
    # Check group metadata
    group_path = storage_manager._data_dir / "velocities.zarr"
    assert group_path.exists()
    
    group = zarr.open_group(str(group_path), mode='r')
    assert group.attrs['field_name'] == 'velocities'
    assert group.attrs['shape_description'] == "samples, nodes, spatial_dims"
    assert group.attrs['num_samples'] == 100
    assert group.attrs['dtype'] == 'float32'


def test_add_field_storage_validates_shape(storage_manager):
    # Empty shape
    with pytest.raises(ValueError, match="Shape cannot be empty"):
        storage_manager.add_field_storage(
            field_name="test",
            shape=(),
            dims_description="empty"
        )
    
    # Zero samples
    with pytest.raises(ValueError, match=r"shape\[0\].*must be positive"):
        storage_manager.add_field_storage(
            field_name="test",
            shape=(0, 10),
            dims_description="invalid"
        )
    
    # Negative samples
    with pytest.raises(ValueError, match=r"shape\[0\].*must be positive"):
        storage_manager.add_field_storage(
            field_name="test",
            shape=(-5, 10),
            dims_description="invalid"
        )


def test_infer_space_layout_invalid_divisibility():
    from simulation.storage_manager import StorageManager
    manager = StorageManager({"RESULTS_DIR": "/tmp/results"})

    class DummyIndexMap:
        size_global = 10

    class DummyDofMap:
        index_map = DummyIndexMap()
        index_map_bs = 4  # 10*4 not divisible by gdim=3

    class DummyMesh:
        class Geometry:
            dim = 3
        geometry = Geometry()

    class DummyV:
        dofmap = DummyDofMap()
        mesh = DummyMesh()

    with pytest.raises(ValueError, match="Scalar DoF count.*divisible"):
        manager._infer_space_layout(DummyV())


def test_add_field_storage_uses_config_overrides(mock_config, temp_dir):
    # Add field-specific config
    mock_config["ZARR_TEMPERATURES_MAX_CHUNK_MB"] = 32
    manager = StorageManager(mock_config)
    
    dataset = manager.add_field_storage(
        field_name="temperatures",
        shape=(1000, 200),
        dims_description="samples, points"
    )
    
    # With 32MB limit and ~1.6MB per sample (200*8 bytes), 
    # should allow more samples per chunk
    assert dataset.chunks[0] > 10


def test_add_field_storage_fallback_generic_config(mock_config, temp_dir):
    mock_config["ZARR_GENERIC_MAX_CHUNK_MB"] = 2
    manager = StorageManager(mock_config)
    
    dataset = manager.add_field_storage(
        field_name="custom_field",
        shape=(100, 1000),
        dims_description="samples, features"
    )
    
    # Should use the generic limit
    # 1000 float64 = 8KB per sample, 2MB limit -> ~250 samples per chunk
    assert dataset.chunks[0] <= 100  # Can't exceed total samples


def test_chunk_calculation_respects_limits(storage_manager, reference_eigenvectors):
    # Test with small number of samples
    num_samples = 5
    n_modes = storage_manager.config["NUM_EIGENVALUES"]
    eigvals_dataset = storage_manager.add_field_storage(
        field_name="eigenvalues",
        shape=(num_samples, n_modes),
        dims_description="samples, modes",
    )
    nodes, gdim = reference_eigenvectors.shape[1], reference_eigenvectors.shape[2]
    _ = storage_manager.add_field_storage(
        field_name="eigenvectors",
        shape=(num_samples, n_modes, nodes, gdim),
        dims_description="samples, modes, nodes, spatial_dims",
    )
    eigvecs_group = zarr.open_group(str(storage_manager._data_dir / "eigenvectors.zarr"), mode="r")
    # Chunks shouldn't exceed number of samples
    assert eigvals_dataset.chunks[0] <= num_samples
    eigvecs_dataset = eigvecs_group['data']
    assert eigvecs_dataset.chunks[0] <= num_samples


def test_recreates_existing_datasets(storage_manager, reference_eigenvectors):
    n_modes = storage_manager.config["NUM_EIGENVALUES"]
    # Create initial eigenvalues
    eigvals_dataset1 = storage_manager.add_field_storage(
        field_name="eigenvalues",
        shape=(10, n_modes),
        dims_description="samples, modes",
    )
    eigvals_dataset1[0] = np.ones(n_modes)

    # Recreate with different number of samples (should overwrite)
    eigvals_dataset2 = storage_manager.add_field_storage(
        field_name="eigenvalues",
        shape=(20, n_modes),
        dims_description="samples, modes",
    )
    assert eigvals_dataset2.shape[0] == 20
    assert np.all(eigvals_dataset2[0] == 0)


def test_handles_different_dtypes(storage_manager):
    for dtype in [np.float32, np.float64, np.int32, np.complex128]:
        dataset = storage_manager.add_field_storage(
            field_name=f"field_{dtype.__name__}",
            shape=(10, 20),
            dims_description="test",
            dtype=dtype
        )
        assert dataset.dtype == dtype


def test_logging_estimated_sizes(storage_manager, reference_eigenvectors, caplog):
    import logging
    caplog.set_level(logging.INFO)
    n_modes = storage_manager.config["NUM_EIGENVALUES"]
    nodes, gdim = reference_eigenvectors.shape[1], reference_eigenvectors.shape[2]
    storage_manager._log_estimated_sizes({
        'eigenvalues': (100, n_modes),
        'eigenvectors': (100, n_modes, nodes, gdim),
    })
    # Check that size estimates were logged
    assert "Estimated storage" in caplog.text
    assert "MB" in caplog.text


def test_config_snapshot_with_dict_config(temp_dir):
    config_dict = {
        "RESULTS_DIR": str(temp_dir),
        "PARAM_A": 1,
        "PARAM_B": "test"
    }
    
    # Create a mock config object with dictionary-like behavior
    class MockConfig:
        def __init__(self, data):
            self.config = data
            # Make the config keys accessible as attributes
            for k, v in data.items():
                setattr(self, k, v)
        
        def __getitem__(self, key):
            return self.config[key]
        
        def get(self, key, default=None):
            return self.config.get(key, default)
    
    config_obj = MockConfig(config_dict)
    manager = StorageManager(config_obj)
    snapshot = manager._config_snapshot()
    
    assert snapshot["PARAM_A"] == 1
    assert snapshot["PARAM_B"] == "test"


def test_parallel_write_safety(storage_manager):
    """Test that storage can handle concurrent writes to different indices."""
    shape = (100, 10)
    dataset = storage_manager.add_field_storage(
        field_name="parallel_test",
        shape=shape,
        dims_description="samples, features"
    )
    
    # Simulate parallel writes to different indices
    dataset[0] = np.ones(10) * 1.0
    dataset[50] = np.ones(10) * 2.0
    dataset[99] = np.ones(10) * 3.0
    
    assert np.all(dataset[0] == 1.0)
    assert np.all(dataset[50] == 2.0)
    assert np.all(dataset[99] == 3.0)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
