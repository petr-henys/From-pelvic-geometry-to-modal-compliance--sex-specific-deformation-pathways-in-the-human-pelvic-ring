from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.fig_utils import nearest_neighbor_match, write_stats_summary


def test_nearest_neighbor_match_returns_pair_id_and_is_paired():
    df = pd.DataFrame(
        {
            "swap_9_10": [1, 1, 0, 0],
            "gap_in_9_10": [0.10, 0.11, 0.10, 0.11],
            "dG_9_10": [0.20, 0.21, 0.20, 0.21],
            "age": [30, 40, 31, 41],
            "P99_SED": [1.0, 2.0, 1.1, 2.1],
        }
    )
    matched = nearest_neighbor_match(df, "swap_9_10", ["gap_in_9_10", "dG_9_10", "age"], caliper=10.0)
    assert "_pair_id" in matched.columns
    pair_ids = matched["_pair_id"].to_numpy()
    assert pair_ids.size % 2 == 0
    # Each pair_id should appear exactly twice, and contain one control and one treated.
    for pid in np.unique(pair_ids):
        sub = matched[matched["_pair_id"] == pid]
        assert len(sub) == 2
        assert set(sub["swap_9_10"].astype(int).tolist()) == {0, 1}


def test_fig1_uses_domain_in_title(monkeypatch, tmp_path):
    import analysis.make_killer_figures as mk

    captured_titles: list[str] = []
    captured_ylabels: list[str] = []

    def _fake_savefig(fig, fig_dir, name, dpi):
        ax = fig.axes[0]
        captured_titles.append(ax.get_title())
        captured_ylabels.append(ax.get_ylabel())
        return Path(fig_dir) / f"{name}.png"

    monkeypatch.setattr(mk, "_savefig", _fake_savefig)

    df = pd.DataFrame(
        {
            "swap_9_10": [0, 1, 0, 1],
            "sex": ["M", "M", "F", "F"],
            "P99_SED": [1e-3, 2e-3, 1.5e-3, 1.8e-3],
            "top1pct_mean_SED": [1e-2, 2e-2, 1.2e-2, 1.6e-2],
        }
    )
    stats_rows: list[dict] = []
    out = mk.fig1_swap_tail(df, tmp_path, dpi=80, stats_rows=stats_rows, domain="cartilage")

    assert len(captured_titles) == 2
    assert all("cartilage" in t.lower() for t in captured_titles)
    assert all("bone" not in t.lower() for t in captured_titles)
    assert out["purpose"].lower().find("cartilage") >= 0
    assert all("log" in y.lower() for y in captured_ylabels)


def test_write_stats_summary_adds_fdr_column(tmp_path):
    rows = [
        {
            "domain": "bone",
            "figure": "T",
            "description": "p1",
            "N": 10,
            "effect_size": 0.0,
            "effect_label": "x",
            "p_value": 0.01,
            "method": "m",
            "notes": "",
        },
        {
            "domain": "bone",
            "figure": "T",
            "description": "p2",
            "N": 10,
            "effect_size": 0.0,
            "effect_label": "x",
            "p_value": 0.04,
            "method": "m",
            "notes": "",
        },
        {
            "domain": "bone",
            "figure": "T",
            "description": "blank",
            "N": 10,
            "effect_size": 0.0,
            "effect_label": "x",
            "p_value": "",
            "method": "m",
            "notes": "",
        },
    ]
    p = write_stats_summary(tmp_path, rows)
    df = pd.read_csv(p)
    assert "p_fdr_bh" in df.columns

    # BH on [0.01, 0.04] => [0.02, 0.04]
    assert df.loc[df["description"] == "p1", "p_fdr_bh"].iloc[0] == pytest.approx(0.02)
    assert df.loc[df["description"] == "p2", "p_fdr_bh"].iloc[0] == pytest.approx(0.04)
    assert np.isnan(df.loc[df["description"] == "blank", "p_fdr_bh"].iloc[0])
