#!/usr/bin/env python3
"""Shared pytest configuration."""
from __future__ import annotations

import matplotlib
import pytest
try:
    from mpi4py import MPI  # type: ignore
except Exception:  # pragma: no cover
    MPI = None

try:
    from dolfinx import mesh, fem  # type: ignore
except Exception:  # pragma: no cover
    mesh = None
    fem = None

matplotlib.use("Agg", force=True)


# ======================== Test execution control ========================


def _require_single_rank():
    """Skip test if running under MPI with multiple ranks."""
    if MPI is not None and MPI.COMM_WORLD.size != 1:
        pytest.skip("Tests require single MPI rank")


# ======================== Mesh resolution ========================

MESH_N = 10  # Unit cube mesh resolution (NxNxN elements)


def create_unit_cube():
    """Create unit cube mesh with configurable resolution."""
    if mesh is None or MPI is None:  # pragma: no cover
        pytest.skip("dolfinx/mpi4py not available")
    return mesh.create_unit_cube(MPI.COMM_WORLD, MESH_N, MESH_N, MESH_N)


def create_zero_phi(V: fem.FunctionSpace) -> fem.Function:
    """Create zero-valued geometry mapping function."""
    if fem is None:  # pragma: no cover
        pytest.skip("dolfinx not available")
    phi = fem.Function(V, name="phi")
    phi.x.array[:] = 0.0
    return phi


# ======================== Tolerance constants ========================

ABS_TOL = 1e-12
REL_TOL = 1e-11
ANGLE_TOL = 1e-11
DISPLACEMENT_TOL = 1e-11
MAPPING_ERR_TOL = 1e-11
SHEAR_TOL = 1e-11
NEG_TOL = 1e-11
DIST_MAX_TOL = 1e-7
MAC_TOL = 1e-12
EPS_TOL = 5e-13


# ======================== Test configuration ========================


def make_test_config(overrides: dict | None = None) -> dict:
    """Create minimal test configuration with optional overrides.
    
    Args:
        overrides: Optional dict to override default values
    
    Returns:
        Configuration dict for tests
    """
    config = {
        "QUADRATURE_DEGREE": 4,
        "SOLVER_RTOL": 1e-10,
        "SOLVER_ATOL": 1e-12,
        "SOLVER_MAX_ITERS": 1000,
        "RIGID_MODE_TOLERANCE": 1e-8,
        "RBF_SOLVER_SMOOTHING": 0.0,
        "RBF_SOLVER_NEIGHBORS": 10,
        "RBF_DATA_SMOOTHING": 50.0,
        "RBF_DATA_NEIGHBORS": 10,
        "LIGAMENT_MAPPING_TOLERANCE": 1e-6,
    }
    if overrides:
        config.update(overrides)
    return config


@pytest.fixture
def test_config():
    """Pytest fixture providing test configuration dict."""
    return make_test_config()
# Memory thresholds for tracemalloc-based tests. Tightened to catch growth.
MEMORY_RATIO_LIMIT = 1.6
MEMORY_BASELINE_MIN = int(1e5)
MEMORY_ABS_LIMIT = int(2e5)
