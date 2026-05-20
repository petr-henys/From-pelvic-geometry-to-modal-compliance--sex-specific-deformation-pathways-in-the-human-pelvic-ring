#!/usr/bin/env python3
"""Memory management tests for PETSc/SLEPc objects."""

from __future__ import annotations

import gc
import sys
from pathlib import Path

import numpy as np
import pytest
from mpi4py import MPI
from dolfinx import mesh, fem

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from conftest import (
    make_test_config,
    _require_single_rank,
    create_zero_phi,
    test_config,
    MEMORY_RATIO_LIMIT,
    MEMORY_BASELINE_MIN,
    MEMORY_ABS_LIMIT,
)
from simulation.eigensolver import ElasticEigenSolver
from simulation.elasticsolver import ElasticSolver

pytestmark = pytest.mark.regression


def get_petsc_objects_count():
    """Get count of active PETSc objects (if possible)."""
    return 0  # Not implemented


def test_eigensolver_matrix_cleanup(test_config):
    """Test that ElasticEigenSolver properly destroys K and M matrices.
    
    Memory: Each matrix can be >100MB for large meshes. Leaked matrices
    accumulate linearly with solve iterations, causing OOM.
    
    This verifies explicit .destroy() calls.
    """
    _require_single_rank()
    import ufl

    from conftest import create_unit_cube
    domain = create_unit_cube()
    gdim = domain.geometry.dim
    V = fem.functionspace(domain, ("Lagrange", 1, (gdim,)))
    W = fem.functionspace(domain, ("DG", 0))

    E = fem.Function(W)
    E.x.array[:] = 1000.0
    nu = fem.Function(W)
    nu.x.array[:] = 0.3

    ds = ufl.ds(domain=domain)
    
    # Create solver
    phi = create_zero_phi(V)

    solver = ElasticEigenSolver(V, E, nu, phi, config=test_config)
    
    # Verify matrices exist
    assert solver.K is not None, "Stiffness matrix not created"
    assert solver.M is not None, "Mass matrix not created"
    assert solver.M0 is not None, "Base mass matrix not created"

    K_ptr = id(solver.K)
    M_ptr = id(solver.M)

    # Destroy matrices (incl. M0 base mass kept by ElasticEigenSolver)
    solver.K.destroy()
    solver.M.destroy()
    solver.M0.destroy()
    # Encourage Python-side cleanup
    del solver
    gc.collect()
    
    # Python objects still exist but should be invalidated
    # (PETSc objects are destroyed on C side)
    # We can't easily check this without PETSc internals access
    
    # Attempting to use destroyed matrix should fail or give errors
    # (but we don't test this as it's undefined behavior)
    
    # This test primarily documents the required cleanup pattern


def test_repeated_solve_memory_stability(test_config):
    """Test that memory usage remains stable across multiple solves.
    
    Memory: Each solve should reuse existing sparsity patterns.
    Memory growth indicates leaked temporaries or accumulating allocations.
    
    This uses tracemalloc to detect leaks.
    """
    _require_single_rank()
    import tracemalloc
    from mpi4py import MPI
    from dolfinx import mesh, fem
    from simulation.eigensolver import ElasticEigenSolver
    import ufl

    from conftest import create_unit_cube
    domain = create_unit_cube()
    gdim = domain.geometry.dim
    V = fem.functionspace(domain, ("Lagrange", 1, (gdim,)))
    W = fem.functionspace(domain, ("DG", 0))

    E = fem.Function(W)
    E.x.array[:] = 1000.0
    nu = fem.Function(W)
    nu.x.array[:] = 0.3

    ds = ufl.ds(domain=domain)
    
    # Start memory tracking
    tracemalloc.start()
    
    # Baseline solve
    phi = create_zero_phi(V)

    solver = ElasticEigenSolver(V, E, nu, phi, config=test_config)
    lam1, vecs1 = solver.solve(nev=6, target=1e-6)
    
    snapshot1 = tracemalloc.take_snapshot()
    current1, peak1 = tracemalloc.get_traced_memory()
    
    # Additional solves (should reuse memory)
    for i in range(5):
        # Update material slightly (forces reassembly)
        E.x.array[:] = 1000.0 + i * 10.0
        solver.update()
        lam, vecs = solver.solve(nev=6, target=1e-6)
    
    snapshot2 = tracemalloc.take_snapshot()
    current2, peak2 = tracemalloc.get_traced_memory()
    
    tracemalloc.stop()
    
    baseline = float(current1)
    delta = float(max(current2 - current1, 0))

    if baseline < MEMORY_BASELINE_MIN:
        assert current2 <= MEMORY_ABS_LIMIT, (
            "Excessive memory growth from near-zero baseline: "
            f"{current1/1e6:.1f} MB → {current2/1e6:.1f} MB (Δ={delta/1e6:.2f} MB)"
        )
    else:
        growth_ratio = current2 / baseline
        assert growth_ratio <= MEMORY_RATIO_LIMIT, (
            f"Excessive memory growth: {current1/1e6:.1f} MB → {current2/1e6:.1f} MB "
            f"(ratio={growth_ratio:.2f})"
        )
    
    # Cleanup
    solver.K.destroy(); solver.M.destroy(); solver.M0.destroy()
    del solver
    gc.collect()


def test_ligament_system_update_memory_reuse(test_config):
    """Test that ligament system updates reuse existing matrix structure.
    
    Memory: Updating ligament stiffnesses should only modify values,
    not allocate new matrices.
    
    This verifies that sparsity pattern is frozen after first assembly.
    """
    _require_single_rank()
    import pyvista as pv
    from mpi4py import MPI
    from dolfinx import mesh, fem
    from simulation.ligamentassembler import LigamentSpringSystem

    domain = mesh.create_unit_cube(MPI.COMM_WORLD, 10, 10, 10)
    gdim = domain.geometry.dim
    V = fem.functionspace(domain, ("Lagrange", 1, (gdim,)))
    
    X = V.tabulate_dof_coordinates()
    
    # Create simple ligament
    pts = X[[0, 1, 2]].astype(float)
    poly = pv.PolyData(pts)
    lines = [2, 0, 1, 2, 1, 2]
    poly.lines = np.array(lines, dtype=np.int32)
    
    # Initial assembly
    stiffness_initial = 10.0
    phi = create_zero_phi(V)
    system = LigamentSpringSystem(
        V, [poly], [stiffness_initial], phi,
        make_test_config({"LIGAMENT_MAPPING_TOLERANCE": 1.0})
    )
    
    K_initial_ptr = id(system.K)
    
    # Update stiffness multiple times
    for stiffness_new in [15.0, 20.0, 5.0]:
        system.update(stiffnesses=[stiffness_new])
        
        # Matrix object should be reused (same Python id)
        assert id(system.K) == K_initial_ptr, \
            "Ligament update created new matrix (should reuse existing)"


def test_elasticsolver_load_vector_accumulation(test_config):
    """Test that clearing loads properly resets RHS vector.
    
    Memory: Accumulated load vectors should be cleared, not appended.
    
    This verifies proper cleanup of internal _L_terms list.
    """
    _require_single_rank()
    import ufl

    domain = mesh.create_unit_cube(MPI.COMM_WORLD, 10, 10, 10)
    gdim = domain.geometry.dim
    V = fem.functionspace(domain, ("Lagrange", 1, (gdim,)))
    W = fem.functionspace(domain, ("DG", 0))

    E = fem.Function(W)
    E.x.array[:] = 1000.0
    nu = fem.Function(W)
    nu.x.array[:] = 0.3

    phi = create_zero_phi(V)


    solver = ElasticSolver(V, E, nu, phi, config=test_config)
    
    # Apply load
    ds = ufl.ds(domain=domain)
    traction = np.array([1.0, 0.0, 0.0])
    solver.apply_load(traction, ds, defined_on="template", input_type="intensity")
    
    # Check internal state
    assert len(solver._L_terms) == 1, "Load not added"
    
    # Clear loads
    solver.clear_loads()
    
    # Internal list should be empty
    assert len(solver._L_terms) == 0, "Loads not cleared properly"
    assert solver._L_form is None, "Load form not reset"
    
    # Cleanup
    solver.K.destroy()


def test_petsc_vector_temporary_cleanup(test_config):
    """Test that temporary PETSc vectors are destroyed in computations.
    
    Memory: Functions like mode_bundle_quadratic_forms create temporary
    vectors. These must be destroyed to avoid leaks.
    
    This test verifies cleanup in tight loops.
    """
    _require_single_rank()
    import pyvista as pv
    from mpi4py import MPI
    from dolfinx import mesh, fem
    from simulation.ligamentassembler import LigamentSpringSystem

    domain = mesh.create_unit_cube(MPI.COMM_WORLD, 10, 10, 10)
    gdim = domain.geometry.dim
    V = fem.functionspace(domain, ("Lagrange", 1, (gdim,)))
    
    X = V.tabulate_dof_coordinates()
    
    pts = X[[0, 1, 2]].astype(float)
    poly = pv.PolyData(pts)
    lines = [2, 0, 1, 2, 1, 2]
    poly.lines = np.array(lines, dtype=np.int32)
    
    phi = create_zero_phi(V)
    system = LigamentSpringSystem(
        V, [poly], [10.0], phi,
        make_test_config({"LIGAMENT_MAPPING_TOLERANCE": 1.0})
    )
    
    # Create random modes
    num_modes = 10
    num_nodes = X.shape[0]
    modes = np.random.randn(num_modes, num_nodes, gdim)
    
    # This function creates and should destroy temporary PETSc vectors
    # We can't directly check destructor calls, but memory tracking would show leaks
    
    import tracemalloc
    tracemalloc.start()
    
    snapshot1 = tracemalloc.take_snapshot()
    
    # Call multiple times
    for _ in range(10):
        quad_forms = system.mode_bundle_quadratic_forms(modes)
    
    snapshot2 = tracemalloc.take_snapshot()
    
    # Compare memory - should be stable
    stats = snapshot2.compare_to(snapshot1, 'lineno')
    
    # Total allocated memory should not grow significantly
    total_diff = sum(stat.size_diff for stat in stats)
    
    tracemalloc.stop()
    
    # Allow some growth but flag if excessive (> 1MB for this small problem)
    assert total_diff < 1e6, \
        f"Memory leak detected: {total_diff/1e6:.2f} MB growth in quadratic forms"


def test_garbage_collection_after_solve(test_config):
    """Test that Python garbage collector can reclaim unused objects.
    
    Memory: After destroying PETSc objects, Python should be able to
    reclaim the wrapper objects via garbage collection.
    
    This verifies that circular references are avoided.
    """
    _require_single_rank()
    import ufl
    
    # Force garbage collection before test
    gc.collect()
    initial_objects = len(gc.get_objects())
    
    def create_and_destroy_solver():
        domain = mesh.create_unit_cube(MPI.COMM_WORLD, 10, 10, 10)
        gdim = domain.geometry.dim
        V = fem.functionspace(domain, ("Lagrange", 1, (gdim,)))
        W = fem.functionspace(domain, ("DG", 0))

        E = fem.Function(W)
        E.x.array[:] = 1000.0
        nu = fem.Function(W)
        nu.x.array[:] = 0.3

        ds = ufl.ds(domain=domain)
        
        phi = create_zero_phi(V)

        
        solver = ElasticEigenSolver(V, E, nu, phi, config=test_config)
        lam, vecs = solver.solve(nev=4, target=1e-6)
        
        # Explicit cleanup
        solver.K.destroy()
        solver.M.destroy()
        solver.M0.destroy()

        return lam  # Return something to prevent over-optimization
    
    # Create and destroy multiple times
    for _ in range(3):
        create_and_destroy_solver()
        gc.collect()  # Force collection
    
    # Check object count hasn't grown excessively
    final_objects = len(gc.get_objects())
    growth = final_objects - initial_objects
    
    # Allow some growth for cached objects, but not unbounded
    assert growth < 1000, \
        f"Potential garbage collection issue: {growth} new objects after 3 iterations"


def test_zarr_chunk_memory_footprint(test_config):
    """Test that Zarr array operations stay within expected memory bounds.
    
    Memory: Loading/writing Zarr chunks should use O(chunk_size) memory,
    not O(array_size).
    
    This verifies chunking strategy is effective.
    """
    _require_single_rank()
    import zarr
    import tempfile
    import shutil
    from pathlib import Path
    
    # Create temporary directory
    tmpdir = Path(tempfile.mkdtemp())
    
    # Create large Zarr array with chunking (Zarr 3.x API)
    from zarr.storage import LocalStore
    store = LocalStore(str(tmpdir / "test.zarr"))
    root = zarr.group(store=store)
    
    # Shape: (1000 samples, 12 modes, 500 nodes, 3 dims)
    # Total: ~72 MB uncompressed
    # Chunk: (10 samples, ...) = ~720 KB per chunk
    shape = (1000, 12, 500, 3)
    chunks = (10, 12, 500, 3)
    
    arr = root.create_array(
        name='eigenvectors',
        shape=shape,
        chunks=chunks,
        dtype=np.float64
    )
    
    import tracemalloc
    tracemalloc.start()
    
    # Write one chunk - should not load entire array
    chunk_data = np.random.randn(*chunks)
    arr[0:10, :, :, :] = chunk_data
    
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    # Peak memory should be << total array size
    # chunk ~720 KB, allow 10 MB for overhead
    assert peak < 10 * 1024 * 1024, \
        f"Zarr chunk write used excessive memory: {peak/1e6:.1f} MB"
    
    # Cleanup
    shutil.rmtree(tmpdir)


def test_function_space_reuse_across_solvers(test_config):
    """Test that multiple solvers can share function spaces without conflicts.
    
    Memory: Function spaces should be shareable. Creating new spaces for
    each solver wastes memory.
    
    This verifies that DOLFINx objects are properly reference-counted.
    """
    _require_single_rank()
    import ufl

    domain = mesh.create_unit_cube(MPI.COMM_WORLD, 10, 10, 10)
    gdim = domain.geometry.dim
    V = fem.functionspace(domain, ("Lagrange", 1, (gdim,)))
    W = fem.functionspace(domain, ("DG", 0))

    E = fem.Function(W)
    E.x.array[:] = 1000.0
    nu = fem.Function(W)
    nu.x.array[:] = 0.3

    ds = ufl.ds(domain=domain)
    
    # Create two solvers sharing same function space
    phi_1 = create_zero_phi(V)

    solver1 = ElasticEigenSolver(V, E, nu, phi_1, test_config)
    solver1.fixed_dirichlet(gamma=1e6)
    
    phi_2 = create_zero_phi(V)

    
    solver2 = ElasticEigenSolver(V, E, nu, phi_2, test_config)
    solver2.fixed_dirichlet(gamma=1e6)
    
    # Both should reference same V object
    assert solver1.V is solver2.V, \
        "Solvers should share function space reference"
    
    # Cleanup one solver
    solver1.K.destroy()
    solver1.M.destroy()
    solver1.M0.destroy()
    del solver1
    gc.collect()

    # Other solver should still work
    lam, vecs = solver2.solve(nev=4, target=100.0)
    assert len(lam) >= 3, f"Solver2 failed after solver1 cleanup, got {len(lam)} eigenvalues"

    # Cleanup
    solver2.K.destroy()
    solver2.M.destroy()
    solver2.M0.destroy()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
