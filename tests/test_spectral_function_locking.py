from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.spectral_metrics import canonical_pair_couplings, single_functional_coupling


def test_canonical_pair_couplings_scores_both_axes():
    q = np.column_stack(
        [
            np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
        ]
    )
    b_primary = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    b_secondary = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])

    primary, secondary, psi1, psi2, sigmas = canonical_pair_couplings(
        q, b_primary, b_secondary,
    )

    assert primary == pytest.approx(1.0)
    assert secondary == pytest.approx(1.0)
    assert np.linalg.norm(psi1) == pytest.approx(1.0)
    assert np.linalg.norm(psi2) == pytest.approx(1.0)
    assert sigmas.tolist() == pytest.approx([1.0, 1.0])


def test_single_functional_coupling_scores_scalar_observable():
    q = np.column_stack(
        [
            np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
        ]
    )
    b_vec = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0])

    coupling, psi, sigma = single_functional_coupling(q, b_vec)

    assert coupling == pytest.approx(np.sqrt(2.0))
    assert np.linalg.norm(psi) == pytest.approx(1.0)
    assert sigma == pytest.approx(np.sqrt(2.0))


def _load_spectral_function_locking():
    try:
        import analysis.spectral_function_locking as mod
    except FileNotFoundError as exc:
        pytest.skip(str(exc))
    return mod


def test_select_best_axis_uses_largest_absolute_value():
    mod = _load_spectral_function_locking()

    best = mod._select_best_axis(
        {"AP": 0.21, "ML": -0.63, "BIS": 0.44, "BIT": np.nan}
    )

    assert best == "ML"


def test_select_best_axis_can_choose_bit():
    mod = _load_spectral_function_locking()

    best = mod._select_best_axis(
        {"AP": -0.11, "ML": 0.18, "BIS": -0.54, "BIT": 0.72}
    )

    assert best == "BIT"


def test_select_best_axis_can_choose_outlet_ap():
    mod = _load_spectral_function_locking()

    best = mod._select_best_axis(
        {"AP": -0.11, "ML": 0.18, "BIS": -0.54, "BIT": 0.72, "OUTLETAP": -0.83}
    )

    assert best == "OUTLETAP"
