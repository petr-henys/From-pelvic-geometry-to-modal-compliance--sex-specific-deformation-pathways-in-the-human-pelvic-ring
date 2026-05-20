from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.spectral_data import (
    load_inlet_landmarks,
    load_outlet_landmarks,
    load_pelvic_dimensions,
)
from simulation.utils import load_json_points


def test_inlet_landmarks_use_birth_canal_files():
    lm = load_inlet_landmarks()

    expected = {
        "AP": Path("anatomy_data/birth_canal/anterior_posterior_inlet_diameter.mrk.json"),
        "ML": Path("anatomy_data/birth_canal/pelvis_inlet_transverse.mrk.json"),
    }
    assert set(expected).issubset(lm)
    for key, rel_path in expected.items():
        pts = np.asarray(load_json_points(rel_path), dtype=float)[:2]
        assert np.allclose(lm[key], pts)


def test_outlet_landmarks_use_palpation_files():
    lm = load_outlet_landmarks()

    expected = {
        "BIS": Path("anatomy_data/palpace/biischiadic.mrk.json"),
        "BIT": Path("anatomy_data/palpace/bituberous.mrk.json"),
        "OUTLETAP": Path("anatomy_data/palpace/outlet_AP.mrk.json"),
    }
    assert set(expected).issubset(lm)
    for key, rel_path in expected.items():
        pts = np.asarray(load_json_points(rel_path), dtype=float)[:2]
        assert np.allclose(lm[key], pts)


def test_pelvic_dimensions_use_birth_canal_for_inlet_and_palpation_for_outlet():
    dims = load_pelvic_dimensions()
    ref = zarr.open_group("data/dimensions_ref.zarr", mode="r")
    pal = zarr.open_group("data/dimensions_michal.zarr", mode="r")

    assert np.allclose(
        dims["AnteriorPosteriorInletDiameter"],
        np.asarray(ref["AnteriorPosteriorInletDiameter"][:]),
    )
    assert np.allclose(
        dims["PelvisInletTransverse"],
        np.asarray(ref["PelvisInletTransverse"][:]),
    )

    assert np.allclose(
        dims["BispinousWidth"],
        np.asarray(pal["BiischiadicWidth"][:]),
    )
    assert np.allclose(
        dims["TransverseOutletDiameter"],
        np.asarray(pal["BituberousWidth"][:]),
    )
    assert np.allclose(
        dims["AnteriorPosteriorOutletDiameter"],
        np.asarray(pal["OutletAP"][:]),
    )

    assert "BispinousWidth_BirthCanal" in dims
    assert "TransverseOutletDiameter_BirthCanal" in dims
    assert "AnteriorPosteriorOutletDiameter_BirthCanal" in dims
