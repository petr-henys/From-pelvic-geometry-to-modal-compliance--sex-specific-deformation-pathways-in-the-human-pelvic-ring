"""Tests for ``simulation.data_mapper``: bone modulus law, RBF mapping, masks."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import pyvista as pv
from mpi4py import MPI
from dolfinx import fem, mesh as dmesh

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from conftest import _require_single_rank, make_test_config
from simulation.data_mapper import (
    MaterialMapper,
    ShapeMapper,
    calculate_bone_modulus,
)

pytestmark = pytest.mark.simulation


# ----------------- calculate_bone_modulus -----------------

def test_calculate_bone_modulus_threshold_branch_low():
    """Below ash-density threshold: returns the constant low branch (2398)."""
    # threshold default 0.486 ⇒ 1.14 * 0.486 - 0.09 ≈ 0.464 hydroxyapatite density
    density = np.array([0.0, 0.1, 0.4])  # ρ_ash well below threshold
    out = calculate_bone_modulus(density)
    assert np.allclose(out, 2398.0)


def test_calculate_bone_modulus_high_branch_matches_power_law():
    """Above threshold: out = α · ρ_ash^β with default α=10200, β=2."""
    density = np.array([1.0, 1.5])
    rho_ash = (density + 0.09) / 1.14
    expected = 10200.0 * rho_ash ** 2
    out = calculate_bone_modulus(density, alpha=10200.0, beta=2.0, threshold=0.486)
    assert np.allclose(out, expected)


def test_calculate_bone_modulus_clamps_negative_density():
    """Negative density is clamped to zero (low branch)."""
    out = calculate_bone_modulus(np.array([-0.5, -1.0]))
    assert np.all(out == 2398.0)


def test_calculate_bone_modulus_alpha_scaling():
    """Doubling α doubles the high-branch modulus."""
    density = np.array([1.2])
    base = calculate_bone_modulus(density, alpha=10200.0, beta=2.0)
    scaled = calculate_bone_modulus(density, alpha=20400.0, beta=2.0)
    assert np.isclose(scaled[0], 2.0 * base[0])


# ----------------- ShapeMapper -----------------

def _make_unit_cube_V():
    domain = dmesh.create_unit_cube(MPI.COMM_WORLD, 6, 6, 6)
    V = fem.functionspace(domain, ("Lagrange", 1, (domain.geometry.dim,)))
    return domain, V


def _make_template_polydata(rng_seed: int = 0) -> pv.PolyData:
    # Sparse cloud roughly covering [0,1]^3
    rng = np.random.default_rng(rng_seed)
    pts = rng.uniform(0.0, 1.0, size=(40, 3))
    return pv.PolyData(pts)


def test_shape_mapper_apply_reference_zeros_phi():
    _require_single_rank()
    _, V = _make_unit_cube_V()
    template = _make_template_polydata()
    shapes = template.points[np.newaxis, ...].copy()  # one sample, identical to template
    phi = fem.Function(V)
    phi.x.array[:] = 1.234

    mapper = ShapeMapper(phi=phi, template=template, shapes=shapes, config=make_test_config({}))
    mapper.apply_reference()
    assert np.allclose(phi.x.array, 0.0)


def test_shape_mapper_apply_sample_recovers_uniform_translation():
    """A pure translation of the template points produces a uniform φ field."""
    _require_single_rank()
    _, V = _make_unit_cube_V()
    template = _make_template_polydata(rng_seed=1)
    translation = np.array([0.05, -0.03, 0.07])
    sample = template.points + translation
    shapes = sample[np.newaxis, ...]

    phi = fem.Function(V)
    cfg = make_test_config({"RBF_DATA_SMOOTHING": 0.0, "RBF_DATA_NEIGHBORS": 20})
    mapper = ShapeMapper(phi=phi, template=template, shapes=shapes, config=cfg)
    mapper.apply_sample(0)

    gdim = V.mesh.geometry.dim
    interior = (V.tabulate_dof_coordinates() > 0.05).all(axis=1) & (
        V.tabulate_dof_coordinates() < 0.95
    ).all(axis=1)
    field = phi.x.array.reshape(-1, gdim)
    # Interior DoFs should reproduce the translation reasonably well.
    assert np.allclose(field[interior], translation, atol=5e-3)


# ----------------- MaterialMapper -----------------

def _make_cube_dg0():
    domain = dmesh.create_unit_cube(MPI.COMM_WORLD, 5, 5, 5)
    W = fem.functionspace(domain, ("DG", 0))
    E = fem.Function(W)
    nu = fem.Function(W)
    return domain, W, E, nu


def _material_config() -> dict:
    return make_test_config({
        "BONE_MODULUS_ALPHA": 10200.0,
        "BONE_MODULUS_BETA": 2.0,
        "BONE_MODULUS_THRESHOLD": 0.486,
        "BONE_POISSON_RATIO": 0.3,
        "SIJ_CARTILAGE_MODULUS": 1.0,
        "SIJ_CARTILAGE_POISSON_RATIO": 0.45,
        "SYMPHYSIS_MODULUS": 2.0,
        "SYMPHYSIS_POISSON_RATIO": 0.45,
    })


def test_material_mapper_masks_classify_regions():
    """Mask construction: bone vs cartilage vs symphysis from FEBio labels."""
    _require_single_rank()
    domain, W, E, nu = _make_cube_dg0()
    n_cells = E.x.array.size

    # One material label per cell; mix of bone / cartilage / symphysis.
    labels = np.array(["PelvisBone"] * n_cells, dtype=object)
    cartilage_idx = np.arange(0, n_cells, 7)
    # Choose symphysis indices disjoint from cartilage.
    symphysis_idx = np.array(
        [i for i in np.arange(2, n_cells, 11) if i not in set(cartilage_idx.tolist())]
    )
    labels[cartilage_idx] = "SIJCartilageLeft"
    labels[symphysis_idx] = "PubicSymphysis"

    template = _make_template_polydata()
    densities = np.full((1, template.n_points), 0.6)  # above threshold
    febio_stub = SimpleNamespace(material_labels=labels)

    mapper = MaterialMapper(
        E_func=E, nu_func=nu, template=template, densities=densities,
        config=_material_config(), febio_parser=febio_stub,
    )

    assert mapper.cartilage_mask.sum() == cartilage_idx.size
    assert mapper.symphysis_mask.sum() == symphysis_idx.size
    # Bone = neither cartilage nor symphysis (and label contains 'bone')
    assert mapper.bone_mask.sum() == n_cells - cartilage_idx.size - symphysis_idx.size
    # Disjoint masks
    assert not (mapper.bone_mask & mapper.cartilage_mask).any()
    assert not (mapper.bone_mask & mapper.symphysis_mask).any()
    assert not (mapper.cartilage_mask & mapper.symphysis_mask).any()


def test_material_mapper_apply_reference_overrides_cart_and_symph():
    """After ``apply_reference``, cartilage/symphysis cells carry constant config moduli,
    and bone cells take the RBF-interpolated bone modulus."""
    _require_single_rank()
    domain, W, E, nu = _make_cube_dg0()
    n_cells = E.x.array.size

    labels = np.array(["PelvisBone"] * n_cells, dtype=object)
    labels[0] = "SIJCartilageLeft"
    labels[1] = "PubicSymphysis"

    template = _make_template_polydata(rng_seed=2)
    densities = np.full((3, template.n_points), 0.8)  # uniform, above threshold
    febio_stub = SimpleNamespace(material_labels=labels)

    cfg = _material_config()
    mapper = MaterialMapper(
        E_func=E, nu_func=nu, template=template, densities=densities,
        config=cfg, febio_parser=febio_stub,
    )
    mapper.apply_reference()

    # Cartilage / symphysis cells get the config constants.
    assert E.x.array[0] == pytest.approx(cfg["SIJ_CARTILAGE_MODULUS"])
    assert nu.x.array[0] == pytest.approx(cfg["SIJ_CARTILAGE_POISSON_RATIO"])
    assert E.x.array[1] == pytest.approx(cfg["SYMPHYSIS_MODULUS"])
    assert nu.x.array[1] == pytest.approx(cfg["SYMPHYSIS_POISSON_RATIO"])

    # Bone cells: RBF on a uniform density should give a roughly constant bone modulus.
    bone_idx = np.where(mapper.bone_mask)[0]
    rho_ash = (0.8 + 0.09) / 1.14
    expected_bone_E = 10200.0 * rho_ash ** 2
    assert np.allclose(E.x.array[bone_idx], expected_bone_E, rtol=5e-3)
    assert np.allclose(nu.x.array[bone_idx], cfg["BONE_POISSON_RATIO"])


def test_material_mapper_alpha_weight_inverts_alpha():
    """Sensitivity weight for α equals 1/α on active bone cells, zero elsewhere."""
    _require_single_rank()
    domain, W, E, nu = _make_cube_dg0()
    n_cells = E.x.array.size

    labels = np.array(["PelvisBone"] * n_cells, dtype=object)
    labels[0] = "SIJCartilageLeft"

    template = _make_template_polydata(rng_seed=3)
    # Density above threshold so α-weight is nonzero on bone cells.
    densities = np.full((1, template.n_points), 0.8)
    febio_stub = SimpleNamespace(material_labels=labels)

    cfg = _material_config()
    mapper = MaterialMapper(
        E_func=E, nu_func=nu, template=template, densities=densities,
        config=cfg, febio_parser=febio_stub,
    )
    weights = mapper.compute_bone_param_sensitivity_weights()

    bone_idx = np.where(mapper.bone_mask)[0]
    cart_idx = np.where(mapper.cartilage_mask)[0]
    assert np.allclose(weights["alpha"][bone_idx], 1.0 / cfg["BONE_MODULUS_ALPHA"])
    assert np.all(weights["alpha"][cart_idx] == 0.0)
    # β-weight = ln(ρ_ash); ρ_ash uniform ≈ 0.78, so ln ≈ -0.246.
    rho_ash = (0.8 + 0.09) / 1.14
    assert np.allclose(weights["beta"][bone_idx], np.log(rho_ash), rtol=1e-4)
    assert np.all(weights["beta"][cart_idx] == 0.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
