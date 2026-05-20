#!/usr/bin/env python3
"""Candidate caption
Population variability preserves a gap-protected backbone while destabilizing
rank identity inside a clustered higher-rank pair. In the toy cohort, hidden
pathway effects push a subset of subjects toward near-degeneracy, so label-level
coupling to an inlet-like function exchanges between neighboring modes even
though coupling to the shared two-dimensional subspace remains high. Standing-
like loads route predominantly through the backbone, whereas task-like loads
recruit the clustered subspace. The toy model summarizes the paper's central
interpretation of the pelvic data: variability can break mode labels without
destroying mechanical function.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patheffects as pe

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.plot_utils import (
    ANNOT_SIZE,
    FEMALE_COLOR,
    FULL_WIDTH,
    LINE_WIDTH,
    MALE_COLOR,
    ROW_H,
    SCATTER_ALPHA,
    SEX_EFFECT_COLOR,
    SMALL_ANNOT_SIZE,
    panel_label,
    setup_plot_style,
)

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "analysis_outputs" / "figures"
PNG_PATH = FIG_DIR / "toy_subspace_dashboard.png"
PDF_PATH = FIG_DIR / "toy_subspace_dashboard.pdf"

BACKBONE_COLOR = "0.28"
PAIR_LOW_COLOR = MALE_COLOR
PAIR_HIGH_COLOR = FEMALE_COLOR
SUBSPACE_COLOR = "0.10"
SUMMARY_COLOR = SEX_EFFECT_COLOR


@dataclass(frozen=True)
class ToyConfig:
    tau: float = 2.0
    k_bar: float = 12.0
    alpha: float = 1.6
    c0: float = 0.22
    s0: float = 5.2
    observable: tuple[float, float, float] = (1.0, 0.0, 0.2)
    standing_load: tuple[float, float, float] = (0.10, 0.05, 1.0)
    task_load: tuple[float, float, float] = (1.0, 0.0, 0.2)


def _normalize(vec: np.ndarray) -> np.ndarray:
    return vec / np.linalg.norm(vec)

def _smooth(values: np.ndarray, window: int = 15) -> np.ndarray:
    if window <= 1:
        return values.copy()
    pad = window // 2
    padded = np.pad(values, pad_width=pad, mode="edge")
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(padded, kernel, mode="valid")


def _nuisance_params(eta: float) -> dict[str, float]:
    return {
        "sigma_delta": 0.05 + 0.22 * eta,
        "sigma_c": 0.008 + 0.018 * eta,
        "sigma_log_s": 0.02 + 0.12 * eta,
        "sigma_d": 0.01 + 0.08 * eta,
        "d1_mean": 0.28 + 0.42 * eta,
        "d2_mean": 0.24 + 0.40 * eta,
    }


def _draw_subject(mu_i: float, eta: float, rng: np.random.Generator, cfg: ToyConfig) -> dict[str, np.ndarray | float]:
    nuisance = _nuisance_params(eta)
    observable = _normalize(np.array(cfg.observable, dtype=float))
    standing_load = _normalize(np.array(cfg.standing_load, dtype=float))
    task_load = _normalize(np.array(cfg.task_load, dtype=float))

    for _ in range(200):
        delta = cfg.alpha * mu_i + rng.normal(0.0, nuisance["sigma_delta"])
        c = abs(cfg.c0 + rng.normal(0.0, nuisance["sigma_c"]))
        s = cfg.s0 * np.exp(rng.normal(0.0, nuisance["sigma_log_s"]))
        d1 = nuisance["d1_mean"] + rng.normal(0.0, nuisance["sigma_d"])
        d2 = nuisance["d2_mean"] + rng.normal(0.0, nuisance["sigma_d"])

        k1 = cfg.k_bar + 0.5 * delta
        k2 = cfg.k_bar - 0.5 * delta
        K = np.array([[k1, c, d1], [c, k2, d2], [d1, d2, s]], dtype=float)
        evals, evecs = np.linalg.eigh(K)
        if np.any(evals <= 1e-10):
            continue
        if int(np.argmax(np.abs(evecs[2, :]))) != 0:
            continue

        proj_observable = evecs.T @ observable
        label_low = float(proj_observable[1] ** 2)
        label_high = float(proj_observable[2] ** 2)
        subspace = float(np.sum(proj_observable[1:] ** 2))

        def routing(load: np.ndarray) -> np.ndarray:
            proj = evecs.T @ load
            energy = (proj ** 2) / evals
            return energy / np.sum(energy)

        standing = routing(standing_load)
        task = routing(task_load)

        k1_eff = k1 - d1 * d1 / s
        k2_eff = k2 - d2 * d2 / s
        c_eff = c - d1 * d2 / s
        z_eff = abs(k1_eff - k2_eff) / abs(c_eff)

        return {
            "mu": mu_i,
            "evals": evals,
            "label_low": label_low,
            "label_high": label_high,
            "subspace": subspace,
            "standing": standing,
            "task": task,
            "z_eff": z_eff,
        }

    raise RuntimeError("Could not draw a valid toy subject with a rank-locked backbone.")


def sample_cohort(mu: np.ndarray, eta: float, seed: int, cfg: ToyConfig) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    rows = [_draw_subject(float(mu_i), eta, rng, cfg) for mu_i in mu]

    return {
        "mu": np.array([row["mu"] for row in rows], dtype=float),
        "evals": np.vstack([row["evals"] for row in rows]).astype(float),
        "label_low": np.array([row["label_low"] for row in rows], dtype=float),
        "label_high": np.array([row["label_high"] for row in rows], dtype=float),
        "subspace": np.array([row["subspace"] for row in rows], dtype=float),
        "standing": np.vstack([row["standing"] for row in rows]).astype(float),
        "task": np.vstack([row["task"] for row in rows]).astype(float),
        "z_eff": np.array([row["z_eff"] for row in rows], dtype=float),
    }


def make_main_cohort(cfg: ToyConfig) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(11)
    mu = np.linspace(-1.10, 1.10, 240) + rng.normal(0.0, 0.035, 240)
    mu = np.sort(mu)
    return sample_cohort(mu, eta=0.85, seed=19, cfg=cfg)


def make_summary_curves(cfg: ToyConfig) -> dict[str, np.ndarray]:
    eta_grid = np.linspace(0.0, 1.6, 9)
    mu_grid = np.linspace(-0.35, 0.35, 480)
    baseline = sample_cohort(mu_grid, eta=0.0, seed=31, cfg=cfg)
    reference_label = baseline["label_low"] >= baseline["label_high"]

    mismatch = []
    subspace_retained = []
    task_cluster = []
    for eta in eta_grid:
        cohort = sample_cohort(mu_grid, eta=float(eta), seed=31, cfg=cfg)
        current_label = cohort["label_low"] >= cohort["label_high"]
        mismatch.append(np.mean(current_label != reference_label))
        subspace_retained.append(np.median(cohort["subspace"]))
        task_cluster.append(np.median(np.sum(cohort["task"][:, 1:], axis=1)))

    return {
        "eta": eta_grid,
        "mismatch": np.array(mismatch, dtype=float),
        "subspace_retained": np.array(subspace_retained, dtype=float),
        "task_cluster": np.array(task_cluster, dtype=float),
    }


def _label_last(ax: plt.Axes, x: np.ndarray, y: np.ndarray, text: str, color: str, dy: float = 0.0) -> None:
    ax.text(
        x[-1] + 0.02 * (x.max() - x.min()),
        y[-1] + dy,
        text,
        color=color,
        va="center",
        fontsize=ANNOT_SIZE,
        clip_on=False,
        bbox={
            "boxstyle": "round,pad=0.16",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.88,
        },
    )

def _boxed_note(
    ax: plt.Axes,
    x: float,
    y: float,
    text: str,
    *,
    transform=None,
    ha: str = "left",
    va: str = "center",
    fontsize: int = ANNOT_SIZE,
    color: str = "0.25",
) -> None:
    ax.text(
        x,
        y,
        text,
        transform=transform if transform is not None else ax.transData,
        ha=ha,
        va=va,
        color=color,
        fontsize=fontsize,
        bbox={
            "boxstyle": "round,pad=0.22",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.90,
        },
        zorder=6,
    )


def _bar_text(ax: plt.Axes, x: float, y: float, text: str) -> None:
    ax.text(
        x,
        y,
        text,
        color="white",
        ha="center",
        va="center",
        fontsize=ANNOT_SIZE,
        path_effects=[pe.withStroke(linewidth=1.2, foreground="0.2")],
    )


def build_figure(main: dict[str, np.ndarray], summary: dict[str, np.ndarray], cfg: ToyConfig) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(FULL_WIDTH, 2.55 * ROW_H), layout="constrained")

    mu = main["mu"]
    mix_mask = main["z_eff"] <= cfg.tau
    mix_mu = mu[mix_mask]
    mix_low = np.percentile(mix_mu, 10) if mix_mu.size else -0.1
    mix_high = np.percentile(mix_mu, 90) if mix_mu.size else 0.1

    # Panel A
    ax = axes[0, 0]
    panel_label(ax, "A  Spectrum and clustered pair")
    ax.axvspan(mix_low, mix_high, color="0.92", zorder=0)
    ax.scatter(mu, main["evals"][:, 0], s=9, color=BACKBONE_COLOR, alpha=0.45, edgecolors="none")
    ax.scatter(mu, main["evals"][:, 1], s=9, color=PAIR_LOW_COLOR, alpha=SCATTER_ALPHA * 0.65, edgecolors="none")
    ax.scatter(mu, main["evals"][:, 2], s=9, color=PAIR_HIGH_COLOR, alpha=SCATTER_ALPHA * 0.65, edgecolors="none")
    ax.plot(mu, _smooth(main["evals"][:, 0], 17), color=BACKBONE_COLOR, lw=LINE_WIDTH * 1.8)
    ax.plot(mu, _smooth(main["evals"][:, 1], 17), color=PAIR_LOW_COLOR, lw=LINE_WIDTH * 1.8)
    ax.plot(mu, _smooth(main["evals"][:, 2], 17), color=PAIR_HIGH_COLOR, lw=LINE_WIDTH * 1.8)
    ax.set_xlabel("Morphology proxy $\\mu$")
    ax.set_ylabel("Eigenvalue")
    ax.set_xlim(mu.min() - 0.06, mu.max() + 0.35)
    _boxed_note(ax, mix_low + 0.03, ax.get_ylim()[1] - 0.25, "near-degenerate zone", va="top")
    _label_last(ax, mu, _smooth(main["evals"][:, 0], 17), "backbone mode", BACKBONE_COLOR)
    _label_last(ax, mu, _smooth(main["evals"][:, 2], 17), "cluster label 2", PAIR_HIGH_COLOR, dy=0.08)
    _label_last(ax, mu, _smooth(main["evals"][:, 1], 17), "cluster label 1", PAIR_LOW_COLOR, dy=-0.08)
    _boxed_note(
        ax,
        0.03,
        0.10,
        "backbone rank stays fixed across subjects",
        transform=ax.transAxes,
        va="center",
    )

    # Panel B
    ax = axes[0, 1]
    panel_label(ax, "B  Label-level vs subspace-level function")
    ax.axvspan(mix_low, mix_high, color="0.92", zorder=0)
    ax.scatter(mu, main["label_low"], s=7, color=PAIR_LOW_COLOR, alpha=0.18, edgecolors="none")
    ax.scatter(mu, main["label_high"], s=7, color=PAIR_HIGH_COLOR, alpha=0.18, edgecolors="none")
    ax.scatter(mu, main["subspace"], s=7, color=SUBSPACE_COLOR, alpha=0.06, edgecolors="none")
    y_low = _smooth(main["label_low"], 19)
    y_high = _smooth(main["label_high"], 19)
    y_sub = _smooth(main["subspace"], 19)
    ax.plot(mu, y_low, color=PAIR_LOW_COLOR, lw=LINE_WIDTH * 1.8)
    ax.plot(mu, y_high, color=PAIR_HIGH_COLOR, lw=LINE_WIDTH * 1.8)
    ax.plot(mu, y_sub, color=SUBSPACE_COLOR, lw=LINE_WIDTH * 2.1)
    ax.set_xlabel("Morphology proxy $\\mu$")
    ax.set_ylabel("Function capture")
    ax.set_ylim(0.0, 1.05)
    ax.set_xlim(mu.min() - 0.06, mu.max() + 0.35)
    _boxed_note(
        ax,
        0.58,
        0.92,
        "shared 2D subspace",
        transform=ax.transAxes,
        ha="center",
        color=SUBSPACE_COLOR,
    )
    _label_last(ax, mu, y_high, "label 2", PAIR_HIGH_COLOR, dy=-0.005)
    _label_last(ax, mu, y_low, "label 1", PAIR_LOW_COLOR, dy=-0.01)
    _boxed_note(
        ax,
        0.04,
        0.12,
        "function changes carrier inside the pair",
        transform=ax.transAxes,
        va="center",
    )

    # Panel C
    ax = axes[1, 0]
    panel_label(ax, "C  Standing vs task routing")
    standing_median = np.median(main["standing"], axis=0)
    task_median = np.median(main["task"], axis=0)
    x = np.array([0.0, 1.05])
    width = 0.46
    ax.bar(x, [standing_median[0], task_median[0]], width=width, color=BACKBONE_COLOR)
    ax.bar(x, [standing_median[1], task_median[1]], bottom=[standing_median[0], task_median[0]], width=width, color=PAIR_LOW_COLOR)
    ax.bar(
        x,
        [standing_median[2], task_median[2]],
        bottom=[standing_median[0] + standing_median[1], task_median[0] + task_median[1]],
        width=width,
        color=PAIR_HIGH_COLOR,
    )
    ax.set_xticks(x, ["standing-like", "task-like"])
    ax.set_ylabel("Median routing fraction")
    ax.set_ylim(0.0, 1.04)
    _bar_text(ax, x[0], standing_median[0] / 2, "backbone")
    _bar_text(ax, x[1], task_median[0] / 2, "backbone")
    _bar_text(ax, x[1], task_median[0] + 0.5 * task_median[1], "label 1")
    _bar_text(ax, x[1], task_median[0] + task_median[1] + 0.5 * task_median[2], "label 2")
    ax.text(x[0], 1.01, f"{standing_median[0]:.0%} backbone", ha="center", va="bottom", color=BACKBONE_COLOR, fontsize=ANNOT_SIZE)
    ax.text(
        x[1],
        1.01,
        f"{(task_median[1] + task_median[2]):.0%} clustered block",
        ha="center",
        va="bottom",
        color=SUBSPACE_COLOR,
        fontsize=ANNOT_SIZE,
    )
    _boxed_note(
        ax,
        0.03,
        0.14,
        "task-level split moves between labels,\nblock-level recruitment stays high",
        transform=ax.transAxes,
        va="center",
    )

    # Panel D
    ax = axes[1, 1]
    panel_label(ax, "D  Variability summary")
    eta = summary["eta"]
    mismatch = summary["mismatch"]
    subspace_retained = summary["subspace_retained"]
    task_cluster = summary["task_cluster"]
    ax.plot(eta, mismatch, marker="o", color=PAIR_HIGH_COLOR, label="Label carrier changed")
    ax.plot(eta, subspace_retained, marker="o", color=SUBSPACE_COLOR, label="Subspace coupling retained")
    ax.plot(eta, task_cluster, marker="o", color=SUMMARY_COLOR, label="Task routing through clustered block")
    ax.set_xlabel("Variability / hidden-pathway scale $\\eta$")
    ax.set_ylabel("Fraction or retained share")
    ax.set_ylim(0.0, 1.05)
    ax.set_xlim(eta.min() - 0.03, eta.max() + 0.02)
    ax.legend(loc="upper left", frameon=False, fontsize=SMALL_ANNOT_SIZE)
    _boxed_note(
        ax,
        0.03,
        0.10,
        "same morphology band; only nuisance variability increases",
        transform=ax.transAxes,
        va="center",
    )

    return fig


def main() -> None:
    setup_plot_style(dpi=400)
    cfg = ToyConfig()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    main_cohort = make_main_cohort(cfg)
    summary = make_summary_curves(cfg)
    fig = build_figure(main_cohort, summary, cfg)
    fig.savefig(PNG_PATH, dpi=400, bbox_inches=None)
    fig.savefig(PDF_PATH, bbox_inches=None)
    plt.close(fig)

    mix_fraction = np.mean(main_cohort["z_eff"] <= cfg.tau)
    task_cluster = np.median(np.sum(main_cohort["task"][:, 1:], axis=1))
    print(f"[toy-subspace-dashboard] wrote {PNG_PATH}")
    print(f"[toy-subspace-dashboard] wrote {PDF_PATH}")
    print(f"[toy-subspace-dashboard] mixing_fraction={mix_fraction:.3f}  task_cluster_median={task_cluster:.3f}")


if __name__ == "__main__":
    main()
