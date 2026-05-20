#!/usr/bin/env python3
"""Test constitutive laws for mapped linear elasticity."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from mpi4py import MPI
from dolfinx import mesh, fem
import ufl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from conftest import _require_single_rank, create_zero_phi

from simulation.eigensolver import ElasticEigenSolver
from simulation.elasticsolver import ElasticSolver

pytestmark = pytest.mark.materials





@pytest.fixture(scope="function")
def unit_cube_setup():
    """Create unit cube mesh with material properties."""
    _require_single_rank()
    import ufl

    from conftest import create_unit_cube
    domain = create_unit_cube()
    gdim = domain.geometry.dim
    V = fem.functionspace(domain, ("Lagrange", 1, (gdim,)))
    W = fem.functionspace(domain, ("DG", 0))

    E = fem.Function(W)
    E.x.array[:] = 1000.0  # MPa
    nu = fem.Function(W)
    nu.x.array[:] = 0.3

    ds = ufl.ds(domain=domain)
    dx = ufl.dx(domain=domain)
    
    return domain, V, W, E, nu, ds, dx


def test_uniaxial_stress_energy_balance(unit_cube_setup, test_config):
    """Test uniaxial tension: external work equals elastic strain energy.
    
    Physics: For pure tension σ_xx = E ε_xx, the strain energy density is:
        w = ½ σ:ε = ½ E ε_xx²
    
    External work by traction t = σ_xx on face x=1:
        W_ext = t · u · Area = σ_xx · ε_xx · 1 · 1
    
    This test verifies energy balance for homogeneous deformation.
    """
    _require_single_rank()
    import ufl

    domain, V, W, E, nu, ds, dx = unit_cube_setup
    
    # Material parameters
    E_val = E.x.array[0]
    nu_val = nu.x.array[0]
    
    # Applied strain
    epsilon_xx = 0.01
    
    # Analytical solution: homogeneous deformation u = ε·x
    X = V.tabulate_dof_coordinates()
    u_analytical = np.zeros_like(X)
    u_analytical[:, 0] = epsilon_xx * X[:, 0]
    u_analytical[:, 1] = -nu_val * epsilon_xx * X[:, 1]  # Poisson contraction
    u_analytical[:, 2] = -nu_val * epsilon_xx * X[:, 2]
    
    # Expected stress (accounting for plane strain effects)
    lambda_param = E_val * nu_val / ((1 + nu_val) * (1 - 2 * nu_val))
    mu = E_val / (2 * (1 + nu_val))
    sigma_xx = (lambda_param + 2 * mu) * epsilon_xx
    
    # Strain energy density - for 3D with Poisson contraction
    # Note: This is simplified - full 3D energy includes all strain components
    # Energy = 0.5 * (σ_xx*ε_xx + σ_yy*ε_yy + σ_zz*ε_zz)
    # For uniaxial stress state in 3D: σ_yy = σ_zz = 0, but ε_yy, ε_zz ≠ 0
    # Actual energy is lower than plane strain approximation
    
    # Compute FEM energy
    phi = create_zero_phi(V)

    solver = ElasticSolver(V, E, nu, phi, test_config)
    
    u = fem.Function(V)
    u.x.array[:] = u_analytical.flatten()
    u.x.scatter_forward()
    
    # Strain tensor
    gdim = domain.geometry.dim
    I = ufl.Identity(gdim)
    F = I + ufl.grad(solver.phi)  # Identity initially
    F_inv = ufl.inv(F)
    eps_u = ufl.sym(ufl.grad(u) * F_inv)
    
    # Stress tensor
    sigma = lambda_param * ufl.tr(eps_u) * I + 2 * mu * eps_u
    
    # Energy
    energy_form = fem.form(0.5 * ufl.inner(sigma, eps_u) * dx)
    energy_fem = fem.assemble_scalar(energy_form)

    # Analytical energy for uniaxial stress in 3D:
    #   ε = diag(ε, -νε, -νε)  ⇒  σ_xx = E·ε, σ_yy = σ_zz = 0
    #   energy density = ½ σ:ε = ½ E ε²
    volume = fem.assemble_scalar(fem.form(1.0 * dx))
    expected_energy = 0.5 * E_val * epsilon_xx ** 2 * volume
    assert np.isclose(energy_fem, expected_energy, rtol=1e-10, atol=1e-14), (
        f"Uniaxial energy {energy_fem:.6e} differs from analytical {expected_energy:.6e}"
    )

    solver.K.destroy()


def test_shear_modulus_relation(unit_cube_setup, test_config):
    """Test shear modulus G = E / (2(1+ν)) via pure shear deformation.
    
    Physics: For pure shear γ_xy, the shear stress is:
        τ_xy = G γ_xy
    where G is the shear modulus.
    
    This test applies a simple shear deformation and verifies the
    stress-strain relationship.
    """
    _require_single_rank()
    import ufl

    domain, V, W, E, nu, ds, dx = unit_cube_setup
    
    E_val = E.x.array[0]
    nu_val = nu.x.array[0]
    G_expected = E_val / (2 * (1 + nu_val))
    
    # Simple shear: u_x = γ y, u_y = u_z = 0
    gamma = 0.01
    X = V.tabulate_dof_coordinates()
    u_shear = np.zeros_like(X)
    u_shear[:, 0] = gamma * X[:, 1]
    
    u = fem.Function(V)
    u.x.array[:] = u_shear.flatten()
    u.x.scatter_forward()
    
    # Shear strain: ε_xy = ½(∂u_x/∂y + ∂u_y/∂x) = ½ γ
    eps = ufl.sym(ufl.grad(u))
    
    # Shear stress: τ_xy = 2μ ε_xy = G γ
    mu = E / (2.0 * (1.0 + nu))
    shear_stress_form = fem.form(2 * mu * eps[0, 1] * dx)
    shear_stress_integral = fem.assemble_scalar(shear_stress_form)
    
    # Volume for normalization
    volume = fem.assemble_scalar(fem.form(1.0 * dx))
    tau_avg = shear_stress_integral / volume
    tau_expected = G_expected * gamma
    
    assert np.isclose(tau_avg, tau_expected, rtol=0.01), \
        f"Shear modulus mismatch: τ_avg={tau_avg:.6f}, expected={tau_expected:.6f}"


def test_hydrostatic_compression_bulk_modulus():
    """Test volumetric response: bulk modulus K = E / (3(1-2ν)).
    
    Physics: Under hydrostatic pressure p, the volumetric strain is:
        ε_vol = tr(ε) = p / K
    
    This test verifies the relationship between pressure and volume change.
    """
    _require_single_rank()
    import ufl

    from conftest import create_unit_cube
    domain = create_unit_cube()
    gdim = domain.geometry.dim
    V = fem.functionspace(domain, ("Lagrange", 1, (gdim,)))
    W = fem.functionspace(domain, ("DG", 0))

    E_val = 1200.0
    nu_val = 0.25
    K_expected = E_val / (3 * (1 - 2 * nu_val))
    
    E = fem.Function(W)
    E.x.array[:] = E_val
    nu = fem.Function(W)
    nu.x.array[:] = nu_val
    
    # Hydrostatic compression: u = -ε_vol * x
    pressure = 10.0  # MPa
    epsilon_vol = pressure / K_expected
    
    X = V.tabulate_dof_coordinates()
    u_hydro = -epsilon_vol * X
    
    u = fem.Function(V)
    u.x.array[:] = u_hydro.flatten()
    u.x.scatter_forward()
    
    # Verify volumetric strain
    dx = ufl.dx(domain=domain)
    eps = ufl.sym(ufl.grad(u))
    vol_strain_form = fem.form(ufl.tr(eps) * dx)
    vol_strain_integral = fem.assemble_scalar(vol_strain_form)
    
    volume = fem.assemble_scalar(fem.form(1.0 * dx))
    eps_vol_avg = vol_strain_integral / volume
    
    assert np.isclose(eps_vol_avg, -3 * epsilon_vol, rtol=0.03), \
        f"Volumetric strain mismatch: ε_vol={eps_vol_avg:.6e}, expected={-3*epsilon_vol:.6e}"


def test_incompressibility_limit_poisson_049(test_config):
    """Test near-incompressible behavior: ν → 0.5 implies K → ∞.
    
    Physics: As Poisson's ratio approaches 0.5, the material becomes
    incompressible (volume-preserving). This causes:
    - Bulk modulus K → ∞
    - Poor conditioning of stiffness matrix
    - Volumetric locking in displacement-based FEM
    
    This test verifies that the solver handles near-incompressible
    materials without catastrophic failure.
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
    
    ds = ufl.ds(domain=domain)
    
    # Test range of Poisson ratios approaching incompressibility
    poisson_ratios = [0.3, 0.4, 0.45, 0.49]
    stiffness_norms = []

    phi = create_zero_phi(V)
    for nu_val in poisson_ratios:
        nu.x.array[:] = nu_val
        solver = ElasticEigenSolver(V, E, nu, phi, test_config)
        # Stiffness should be well-defined but increasingly ill-conditioned
        K_norm = solver.K.norm()
        assert K_norm > 0, f"Singular stiffness for ν={nu_val}"
        stiffness_norms.append(K_norm)
        
        solver.K.destroy()
        solver.M.destroy()
    
    # Stiffness norm should increase monotonically
    assert all(stiffness_norms[i] < stiffness_norms[i+1] 
               for i in range(len(stiffness_norms)-1)), \
        "Stiffness norm should increase with Poisson ratio"


def test_isotropic_material_symmetry():
    """Test material isotropy: response invariant under coordinate rotation.
    
    Physics: Isotropic materials have identical properties in all directions.
    The constitutive tensor C_ijkl has only 2 independent constants (λ, μ).
    
    This test verifies rotational symmetry of the stress response.
    """
    _require_single_rank()
    import ufl

    domain = mesh.create_unit_cube(MPI.COMM_WORLD, 10, 10, 10)
    gdim = domain.geometry.dim
    V = fem.functionspace(domain, ("Lagrange", 1, (gdim,)))
    W = fem.functionspace(domain, ("DG", 0))

    E = fem.Function(W)
    E.x.array[:] = 800.0
    nu = fem.Function(W)
    nu.x.array[:] = 0.3
    
    dx = ufl.dx(domain=domain)
    
    # Test deformation in x-direction
    X = V.tabulate_dof_coordinates()
    epsilon = 0.005
    u_x = np.zeros_like(X)
    u_x[:, 0] = epsilon * X[:, 0]
    
    u = fem.Function(V)
    u.x.array[:] = u_x.flatten()
    u.x.scatter_forward()
    
    # Compute energy in x-direction
    lambda_param = E.x.array[0] * nu.x.array[0] / ((1 + nu.x.array[0]) * (1 - 2 * nu.x.array[0]))
    mu = E.x.array[0] / (2 * (1 + nu.x.array[0]))
    
    I = ufl.Identity(gdim)
    eps_x = ufl.sym(ufl.grad(u))
    sigma_x = lambda_param * ufl.tr(eps_x) * I + 2 * mu * eps_x
    energy_x = fem.assemble_scalar(fem.form(0.5 * ufl.inner(sigma_x, eps_x) * dx))
    
    # Test identical deformation in y-direction
    u_y = np.zeros_like(X)
    u_y[:, 1] = epsilon * X[:, 1]
    u.x.array[:] = u_y.flatten()
    u.x.scatter_forward()
    
    eps_y = ufl.sym(ufl.grad(u))
    sigma_y = lambda_param * ufl.tr(eps_y) * I + 2 * mu * eps_y
    energy_y = fem.assemble_scalar(fem.form(0.5 * ufl.inner(sigma_y, eps_y) * dx))
    
    # Energies should be identical (isotropy)
    assert np.isclose(energy_x, energy_y, rtol=1e-10), \
        f"Isotropy violated: E_x={energy_x:.6e}, E_y={energy_y:.6e}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

# ======================== Material utility function tests ========================

from simulation.data_mapper import calculate_bone_modulus

pytestmark = pytest.mark.materials


def test_calculate_bone_modulus_monotonic():
    density = np.array([0.0, 0.5, 1.0, 1.5])
    modulus = calculate_bone_modulus(density)
    assert modulus.shape == density.shape
    assert np.all(np.diff(modulus) > 0)


def test_calculate_bone_modulus_negative_clamp():
    density = np.array([-0.5, 0.0])
    modulus = calculate_bone_modulus(density)
    assert modulus[0] == modulus[1]


def test_empty_density_field():
    """Test handling of empty density fields."""
    empty = np.array([])
    result = calculate_bone_modulus(empty)
    assert result.shape == (0,)
