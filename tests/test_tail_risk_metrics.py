from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.spectral_metrics import ref_pair_swap_indicator
from analysis.tail_risk_metrics import compute_tail_metrics, weighted_percentile_robust

pytestmark = pytest.mark.analysis


def test_ref_pair_swap_indicator_identity_and_swap():
    perm = np.array(
        [
            [0, 1, 2, 3],   # identity
            [0, 2, 1, 3],   # ref 1 and 2 swapped
            [0, -1, 2, 3],  # unpaired ref 1
            [0, 1, 1, 3],   # invalid: both map to same rank
            [0, 99, 2, 3],  # invalid: out of range
        ],
        dtype=int,
    )
    out = ref_pair_swap_indicator(perm, 1, 2)
    assert out[0] == 0.0
    assert out[1] == 1.0
    assert np.isnan(out[2])
    assert np.isnan(out[3])
    assert np.isnan(out[4])


def test_weighted_percentile_robust_nan_when_no_valid():
    values = np.array([np.nan, np.inf])
    weights = np.array([1.0, 1.0])
    assert np.isnan(weighted_percentile_robust(values, weights, 50.0))


def test_compute_tail_metrics_top1pct_includes_crossing_element():
    # Construct a case where the second element crosses the 1 % cumulative-volume threshold.
    sed = np.array([100.0, 50.0, 1.0] + [0.0] * 10, dtype=float)
    eps1 = np.zeros_like(sed)
    eps3 = np.zeros_like(sed)
    volumes = np.array([0.006, 0.006, 0.988] + [0.001] * 10, dtype=float)
    mask = np.ones_like(sed, dtype=bool)

    metrics = compute_tail_metrics(sed, eps1, eps3, volumes, mask)

    # Top-1 % volume = 0.01 * total_vol. After sorting by SED, cumulative volumes:
    # 0.006, 0.012, ... so the top-1 % prefix must include the first TWO elements.
    expected = (100.0 * 0.006 + 50.0 * 0.006) / (0.006 + 0.006)
    assert metrics["top1pct_mean_SED"] == pytest.approx(expected, rel=0, abs=1e-12)


def test_compute_tail_metrics_returns_nan_for_too_few_elements():
    sed = np.arange(9, dtype=float)
    eps1 = np.zeros_like(sed)
    eps3 = np.zeros_like(sed)
    volumes = np.ones_like(sed)
    mask = np.ones_like(sed, dtype=bool)

    with pytest.warns(UserWarning):
        metrics = compute_tail_metrics(sed, eps1, eps3, volumes, mask)

    assert np.isnan(metrics["P99_SED"])
    assert np.isnan(metrics["P95_SED"])

