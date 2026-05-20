"""Tests for the parameter sweep configuration utilities."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

pytestmark = pytest.mark.configuration


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_output_dir(tmp_path):
    """Create temporary output directory for sweep results."""
    output_dir = tmp_path / "sweep_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


@pytest.fixture
def simple_params():
    """Simple parameter dictionary for basic tests."""
    return {
        "param_a": [1, 2, 3],
        "param_b": [0.5, 1.0],
    }


@pytest.fixture
def ligament_params():
    """Realistic ligament stiffness multiplier parameters."""
    return {
        "ligament_symphysis_multiplier": [0.8, 1.0, 1.2],
        "ligament_anterior_sij_multiplier": [0.8, 1.0, 1.2],
        "ligament_posterior_sij_multiplier": [1.0],
    }


@pytest.fixture
def mock_callable():
    """Simple mock callable that creates output directory."""
    def _callable(param_point: Dict[str, Any], output_path: Path) -> None:
        output_path.mkdir(parents=True, exist_ok=True)
    return _callable


@pytest.fixture
def mock_failing_callable():
    """Mock callable that raises an exception."""
    def _callable(param_point: Dict[str, Any], output_path: Path) -> None:
        raise RuntimeError("Simulated failure")
    return _callable


@pytest.fixture
def mock_file_writing_callable():
    """Mock callable that writes files to output_path."""
    def _callable(param_point: Dict[str, Any], output_path: Path) -> None:
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Write a dummy config file
        config_file = output_path / "config.json"
        with open(config_file, "w") as f:
            json.dump(param_point, f, indent=2)
        
        # Write a dummy result file
        result_file = output_path / "results.txt"
        with open(result_file, "w") as f:
            f.write(f"Results for {param_point}\n")
    return _callable


# ============================================================================
# ParameterSweep Tests
# ============================================================================

class TestParameterSweep:
    """Test ParameterSweep configuration and generation."""
    
    def test_initialization(self, simple_params, temp_output_dir):
        """Test basic ParameterSweep initialization."""
        from parametrizer import ParameterSweep
        
        sweep = ParameterSweep(
            params=simple_params,
            base_output_dir=temp_output_dir,
            metadata={"description": "test sweep"}
        )
        
        assert sweep.params == simple_params
        assert sweep.base_output_dir == temp_output_dir
        assert sweep.metadata["description"] == "test sweep"
    
    def test_empty_params_raises_error(self, temp_output_dir):
        """Test that empty params dictionary raises ValueError."""
        from parametrizer import ParameterSweep
        
        with pytest.raises(ValueError, match="params dictionary cannot be empty"):
            ParameterSweep(params={}, base_output_dir=temp_output_dir)
    
    def test_single_value_conversion_to_list(self, temp_output_dir):
        """Test that single values are converted to lists."""
        from parametrizer import ParameterSweep
        
        params = {"param_a": 1, "param_b": [2, 3]}
        sweep = ParameterSweep(params=params, base_output_dir=temp_output_dir)
        
        assert sweep.params["param_a"] == [1]
        assert sweep.params["param_b"] == [2, 3]
    
    def test_generate_points_cartesian_product(self, simple_params, temp_output_dir):
        """Test that generate_points produces correct Cartesian product."""
        from parametrizer import ParameterSweep
        
        sweep = ParameterSweep(params=simple_params, base_output_dir=temp_output_dir)
        points = sweep.generate_points()
        
        # 3 values of param_a × 2 values of param_b = 6 points
        assert len(points) == 6
        
        # Check all combinations present
        expected_combinations = [
            {"param_a": 1, "param_b": 0.5},
            {"param_a": 1, "param_b": 1.0},
            {"param_a": 2, "param_b": 0.5},
            {"param_a": 2, "param_b": 1.0},
            {"param_a": 3, "param_b": 0.5},
            {"param_a": 3, "param_b": 1.0},
        ]
        
        for expected in expected_combinations:
            assert expected in points
    
    def test_total_runs_calculation(self, simple_params, temp_output_dir):
        """Test total_runs() matches product of param list lengths."""
        from parametrizer import ParameterSweep
        
        sweep = ParameterSweep(params=simple_params, base_output_dir=temp_output_dir)
        assert sweep.total_runs() == 6  # 3 × 2
    
    def test_hash_based_output_path(self, simple_params, temp_output_dir):
        """Test that format_output_path generates consistent hashes."""
        from parametrizer import ParameterSweep
        
        sweep = ParameterSweep(params=simple_params, base_output_dir=temp_output_dir)
        
        param_point = {"param_a": 1, "param_b": 0.5}
        path1 = sweep.format_output_path(param_point)
        path2 = sweep.format_output_path(param_point)
        
        # Same parameters should produce same hash
        assert path1 == path2
        assert path1.parent == temp_output_dir
        assert len(path1.name) == 8  # 8-character hash
    
    def test_hash_uniqueness(self, simple_params, temp_output_dir):
        """Test that different parameter combinations produce different hashes."""
        from parametrizer import ParameterSweep
        
        sweep = ParameterSweep(params=simple_params, base_output_dir=temp_output_dir)
        
        point1 = {"param_a": 1, "param_b": 0.5}
        point2 = {"param_a": 2, "param_b": 0.5}
        
        path1 = sweep.format_output_path(point1)
        path2 = sweep.format_output_path(point2)
        
        assert path1 != path2
    
    def test_hash_order_independence(self, temp_output_dir):
        """Test that parameter order doesn't affect hash (uses sorted keys)."""
        from parametrizer import ParameterSweep
        
        sweep = ParameterSweep(
            params={"a": [1], "b": [2]},
            base_output_dir=temp_output_dir
        )
        
        # JSON serialization uses sort_keys=True, so order shouldn't matter
        path1 = sweep.format_output_path({"a": 1, "b": 2})
        path2 = sweep.format_output_path({"b": 2, "a": 1})
        
        assert path1 == path2


# ============================================================================
# SweepResult Tests
# ============================================================================

class TestSweepResult:
    """Test SweepResult data structures and serialization."""
    
    def test_initialization(self):
        """Test basic SweepResult initialization."""
        from parametrizer import SweepResult
        
        param_points = [{"a": 1}, {"a": 2}]
        output_paths = [Path("/tmp/run1"), Path("/tmp/run2")]
        
        result = SweepResult(
            param_points=param_points,
            output_paths=output_paths,
            metadata={"test": "value"},
            base_output_dir=Path("/tmp")
        )
        
        assert len(result.param_points) == 2
        assert len(result.output_paths) == 2
        assert result.metadata["test"] == "value"
    
    def test_save_json_structure(self, temp_output_dir):
        """Test JSON serialization structure."""
        from parametrizer import SweepResult
        
        param_points = [{"a": 1, "b": 0.5}, {"a": 2, "b": 1.0}]
        output_paths = [temp_output_dir / "run1", temp_output_dir / "run2"]
        
        result = SweepResult(
            param_points=param_points,
            output_paths=output_paths,
            metadata={"description": "test"},
            base_output_dir=temp_output_dir
        )
        
        json_file = temp_output_dir / "test_result.json"
        result.save_json(json_file)
        
        assert json_file.exists()
        
        with open(json_file) as f:
            data = json.load(f)
        
        assert "total_runs" in data
        assert "runs" in data
        assert data["total_runs"] == 2
        assert len(data["runs"]) == 2
        assert data["runs"][0]["parameters"] == {"a": 1, "b": 0.5}
        assert "output_path" in data["runs"][0]
    
    def test_save_csv_structure(self, temp_output_dir):
        """Test CSV serialization structure."""
        from parametrizer import SweepResult
        import csv
        
        param_points = [{"a": 1, "b": 0.5}, {"a": 2, "b": 1.0}]
        output_paths = [temp_output_dir / "run1", temp_output_dir / "run2"]
        
        result = SweepResult(
            param_points=param_points,
            output_paths=output_paths,
            base_output_dir=temp_output_dir,
            metadata={}
        )
        
        csv_file = temp_output_dir / "test_result.csv"
        result.save_csv(csv_file)
        
        assert csv_file.exists()
        
        with open(csv_file, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        assert len(rows) == 2
        # Parameters should be in columns
        assert "a" in rows[0]
        assert "b" in rows[0]
        assert "output_path" in rows[0]
        assert rows[0]["a"] == "1"
        assert rows[0]["b"] == "0.5"
    
    def test_save_csv_empty_data(self, temp_output_dir):
        """Test CSV save with empty data doesn't crash."""
        from parametrizer import SweepResult
        
        result = SweepResult(
            param_points=[],
            output_paths=[],
            base_output_dir=temp_output_dir,
            metadata={}
        )
        
        csv_file = temp_output_dir / "empty.csv"
        result.save_csv(csv_file)  # Should not raise


# ============================================================================
# Parametrizer Tests
# ============================================================================

class TestParametrizer:
    """Test Parametrizer sweep execution."""
    
    def test_initialization(self, simple_params, temp_output_dir, mock_callable):
        """Test basic Parametrizer initialization."""
        from parametrizer import ParameterSweep, Parametrizer
        
        sweep = ParameterSweep(params=simple_params, base_output_dir=temp_output_dir)
        parametrizer = Parametrizer(sweep=sweep, callable_func=mock_callable, verbose=False)
        
        assert parametrizer.sweep == sweep
        assert parametrizer.callable_func == mock_callable
        assert not parametrizer._show_progress
    
    def test_run_simple_sweep(self, simple_params, temp_output_dir, mock_callable):
        """Test running a simple parameter sweep."""
        from parametrizer import ParameterSweep, Parametrizer
        
        sweep = ParameterSweep(params=simple_params, base_output_dir=temp_output_dir)
        parametrizer = Parametrizer(sweep=sweep, callable_func=mock_callable, verbose=False)
        
        result = parametrizer.run()
        
        assert len(result.param_points) == 6  # 3 × 2
        assert len(result.output_paths) == 6
        
        # Check CSV and JSON were created
        assert (temp_output_dir / "sweep_results.csv").exists()
        assert (temp_output_dir / "sweep_summary.json").exists()
    
    def test_run_single(self, simple_params, temp_output_dir, mock_callable):
        """Test run_single() method."""
        from parametrizer import ParameterSweep, Parametrizer
        
        sweep = ParameterSweep(params=simple_params, base_output_dir=temp_output_dir)
        parametrizer = Parametrizer(sweep=sweep, callable_func=mock_callable, verbose=False)
        
        param_point = {"param_a": 1, "param_b": 0.5}
        parametrizer.run_single(param_point)
        
        # Callable returns None now
        output_path = sweep.format_output_path(param_point)
        assert output_path.exists()
    
    def test_run_single_custom_output_path(self, simple_params, temp_output_dir, mock_callable):
        """Test run_single() with custom output path."""
        from parametrizer import ParameterSweep, Parametrizer
        
        sweep = ParameterSweep(params=simple_params, base_output_dir=temp_output_dir)
        parametrizer = Parametrizer(sweep=sweep, callable_func=mock_callable, verbose=False)
        
        custom_path = temp_output_dir / "custom_run"
        param_point = {"param_a": 2, "param_b": 1.0}
        
        parametrizer.run_single(param_point, output_path=custom_path)
        
        assert custom_path.exists()
    
    def test_callable_receives_correct_parameters(self, simple_params, temp_output_dir):
        """Test that callable receives correct param_point and output_path."""
        from parametrizer import ParameterSweep, Parametrizer
        
        captured_calls = []
        
        def tracking_callable(param_point: Dict[str, Any], output_path: Path) -> None:
            captured_calls.append((param_point.copy(), output_path))
            output_path.mkdir(parents=True, exist_ok=True)
        
        sweep = ParameterSweep(params=simple_params, base_output_dir=temp_output_dir)
        parametrizer = Parametrizer(sweep=sweep, callable_func=tracking_callable, verbose=False)
        
        result = parametrizer.run()
        
        assert len(captured_calls) == 6
        
        # Verify each call had unique parameters
        param_sets = [call[0] for call in captured_calls]
        assert len(param_sets) == len(set(str(p) for p in param_sets))
    
    def test_file_writing_callable(self, simple_params, temp_output_dir, mock_file_writing_callable):
        """Test that callable can write files to output directories."""
        from parametrizer import ParameterSweep, Parametrizer
        
        sweep = ParameterSweep(params=simple_params, base_output_dir=temp_output_dir)
        parametrizer = Parametrizer(sweep=sweep, callable_func=mock_file_writing_callable, verbose=False)
        
        result = parametrizer.run()
        
        # Check that files were created in each output directory
        for output_path in result.output_paths:
            assert (output_path / "config.json").exists()
            assert (output_path / "results.txt").exists()


# ============================================================================
# LigamentSweepRunnable Tests - REMOVED (ligament-specific, not in parametrizer)
# ============================================================================

# LigamentSweepRunnable was removed from parametrizer.py as it's domain-specific.
# Ligament sensitivity analysis should be implemented as a separate callable
# in analysis/ligament/ module.


# ============================================================================
# Helper Function Tests - REMOVED (ligament-specific, not in parametrizer)
# ============================================================================

# Helper functions for ligament sweeps were removed from parametrizer.py.
# These belong in domain-specific analysis modules, not in the generic framework.


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for complete workflows."""
    
    def test_simple_parameter_sweep_workflow(self, temp_output_dir):
        """Test complete parameter sweep workflow."""
        from parametrizer import ParameterSweep, Parametrizer
        
        # Create parameter sweep
        params = {
            "learning_rate": [0.01, 0.1],
            "batch_size": [32, 64],
        }
        
        sweep = ParameterSweep(
            params=params,
            base_output_dir=temp_output_dir / "simple_sweep",
            metadata={"test": "integration"}
        )
        
        # Total runs: 2 learning rates × 2 batch sizes = 4 runs
        assert sweep.total_runs() == 4
        
        # Mock callable
        def mock_callable(param_point: Dict[str, Any], output_path: Path) -> None:
            output_path.mkdir(parents=True, exist_ok=True)
        
        # Run sweep
        parametrizer = Parametrizer(
            sweep=sweep,
            callable_func=mock_callable,
            verbose=False
        )
        
        result = parametrizer.run()
        
        assert len(result.param_points) == 4
        assert len(result.output_paths) == 4
        
        # Verify outputs were created
        csv_file = temp_output_dir / "simple_sweep" / "sweep_results.csv"
        json_file = temp_output_dir / "simple_sweep" / "sweep_summary.json"
        
        assert csv_file.exists()
        assert json_file.exists()
    
    def test_large_sweep_performance(self, temp_output_dir):
        """Test sweep with many parameter combinations (stress test)."""
        from parametrizer import ParameterSweep, Parametrizer
        
        # Create 5 parameters with 3 values each = 243 combinations
        params = {f"param_{i}": [0.8, 1.0, 1.2] for i in range(5)}
        
        sweep = ParameterSweep(
            params=params,
            base_output_dir=temp_output_dir / "large_sweep"
        )
        
        assert sweep.total_runs() == 243
        
        def fast_callable(param_point: Dict[str, Any], output_path: Path) -> None:
            output_path.mkdir(parents=True, exist_ok=True)
        
        parametrizer = Parametrizer(
            sweep=sweep,
            callable_func=fast_callable,
            verbose=False
        )
        
        result = parametrizer.run()
        
        assert len(result.param_points) == 243
        assert len(result.output_paths) == 243
