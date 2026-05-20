#!/usr/bin/env python3
"""Deterministic 2-DOF toy model figure: stable ordering vs near-degenerate mixing.

Generates the 3×2 panel figure (a–f) illustrating:
  (a,b) Stable-ordering scenario: well-separated eigenvalues + small θ
  (c,d) Near-degenerate scenario: avoided crossing with gap 2|c|, θ → 45°
  (e,f) Diagnostic: |Δ(μ)| vs τ|c(μ)| for both scenarios

Output: analysis_outputs/figures/toy_model_figure_grid.pdf
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.spectral_config import COL2_WIDTH, GOLDEN
from analysis.spectral_plots import _save_fig
from utils.plot_utils import (
    ANNOT_SIZE,
    LINE_WIDTH,
    MIXING_COLOR,
    NEUTRAL_COLOR,
    SMALL_ANNOT_SIZE,
    VEERING_COLOR,
    panel_label,
    setup_plot_style,
)

# ── Toy-model parameters ──────────────────────────────────────
TAU = 2.0                       # safety factor for near-degeneracy criterion
K_BAR = 10.0                    # mean diagonal stiffness
C_STABLE = 0.15                 # coupling in stable-ordering scenario
C_DEGEN = 0.15                  # coupling in near-degenerate scenario
MU = np.linspace(-2.0, 2.0, 500)

# Detuning maps: Δ(μ) = slope * μ
SLOPE_STABLE = 2.5              # large slope → Δ stays above τ|c|
SLOPE_DEGEN = 0.12              # small slope → Δ crosses zero → avoided crossing

# Colors for the two eigenvalue branches
COLOR_PLUS = NEUTRAL_COLOR      # λ₊
COLOR_MINUS = MIXING_COLOR      # λ₋
COLOR_GAP = '0.60'              # gap annotation


def _eigenvalues(delta: np.ndarray, c: float) -> tuple[np.ndarray, np.ndarray]:
    """Compute λ± for a 2×2 symmetric stiffness matrix."""
    mean = K_BAR
    disc = np.sqrt((delta / 2) ** 2 + c ** 2)
    return mean + disc, mean - disc


def _mixing_angle(delta: np.ndarray, c: float) -> np.ndarray:
    """Mixing angle θ ∈ [0, 90] degrees; θ = 45° at Δ = 0."""
    return np.degrees(0.5 * np.arctan2(2 * c, delta)) % 90


def _plot_eigenvalues(ax: plt.Axes, mu: np.ndarray, lam_p: np.ndarray,
                      lam_m: np.ndarray, *, label: str, c_val: float,
                      show_gap: bool) -> None:
    """Plot λ±(μ) with optional gap annotation."""
    ax.plot(mu, lam_p, color=COLOR_PLUS, lw=1.2, label=r'$\lambda_+$')
    ax.plot(mu, lam_m, color=COLOR_MINUS, lw=1.2, label=r'$\lambda_-$')
    ax.set_xlabel(r'morphology parameter $\mu$')
    ax.set_ylabel(r'eigenvalue $\lambda$')
    ax.legend(loc='upper right')

    if show_gap:
        # Annotate the minimum gap at Δ = 0
        idx_0 = np.argmin(np.abs(mu))
        gap = lam_p[idx_0] - lam_m[idx_0]
        mid = 0.5 * (lam_p[idx_0] + lam_m[idx_0])
        ax.annotate(
            '', xy=(mu[idx_0], lam_p[idx_0]),
            xytext=(mu[idx_0], lam_m[idx_0]),
            arrowprops=dict(arrowstyle='<->', color=COLOR_GAP,
                            lw=LINE_WIDTH, shrinkA=1, shrinkB=1),
        )
        ax.text(mu[idx_0] + 0.15, mid, rf'$2|c|$',
                fontsize=ANNOT_SIZE, color=COLOR_GAP, va='center')


def _plot_theta(ax: plt.Axes, mu: np.ndarray, theta: np.ndarray,
                 *, near_degen: bool) -> None:
    """Plot mixing angle θ(μ)."""
    ax.plot(mu, theta, color='0.20', lw=1.2)
    ax.axhline(45, color=COLOR_GAP, lw=LINE_WIDTH, ls='--')
    ax.set_xlabel(r'morphology parameter $\mu$')
    ax.set_ylabel(r'mixing angle $\theta$ (deg)')
    ax.set_ylim(-2, 92)
    ax.set_yticks([0, 15, 30, 45, 60, 75, 90])

    if near_degen:
        ax.text(0.05, 0.88, r'$\theta\!\approx\!45°$: identity exchange',
                transform=ax.transAxes, fontsize=ANNOT_SIZE, color=COLOR_GAP)
    else:
        ax.text(0.05, 0.88, r'$\theta$ remains small',
                transform=ax.transAxes, fontsize=ANNOT_SIZE, color='0.45')


def _plot_diagnostic(ax: plt.Axes, mu: np.ndarray, delta: np.ndarray,
                      c: float, *, near_degen: bool) -> None:
    r"""Plot |Δ(μ)| vs τ|c(μ)| diagnostic."""
    abs_delta = np.abs(delta)
    tau_c = TAU * np.abs(c) * np.ones_like(mu)

    ax.plot(mu, abs_delta, color='0.20', lw=1.2, label=r'$|\Delta(\mu)|$')
    ax.plot(mu, tau_c, color=MIXING_COLOR, lw=1.2, ls='--',
            label=rf'$\tau|c|$ ($\tau={TAU:.0f}$)')

    # Shade mixing region where |Δ| ≤ τ|c|
    mask = abs_delta <= tau_c
    ymax = max(float(abs_delta.max()), float(tau_c.max())) * 1.1
    ax.set_ylim(0, ymax)
    if np.any(mask):
        ax.fill_between(mu, 0, ymax,
                         where=mask, alpha=0.10, color=MIXING_COLOR,
                         label='mixing zone')

    ax.set_xlabel(r'morphology parameter $\mu$')
    ax.set_ylabel('value')
    ax.legend(loc='upper right')

    if near_degen:
        ax.text(0.05, 0.55, 'near-degenerate:\nintersection present',
                transform=ax.transAxes, fontsize=ANNOT_SIZE, color=MIXING_COLOR)
    else:
        ax.text(0.05, 0.55, 'stable ordering:\nno intersection',
                transform=ax.transAxes, fontsize=ANNOT_SIZE, color=VEERING_COLOR)


def make_figure() -> plt.Figure:
    """Build the 2×3 panel figure (3 columns per row)."""
    setup_plot_style()

    fig, axes = plt.subplots(2, 3, figsize=(COL2_WIDTH, COL2_WIDTH * 0.55),
                             constrained_layout=True)

    # ── Compute both scenarios ─────────────────────────────────
    delta_stable = SLOPE_STABLE * MU
    lp_s, lm_s = _eigenvalues(delta_stable, C_STABLE)
    theta_s = _mixing_angle(delta_stable, C_STABLE)

    delta_degen = SLOPE_DEGEN * MU
    lp_d, lm_d = _eigenvalues(delta_degen, C_DEGEN)
    theta_d = _mixing_angle(delta_degen, C_DEGEN)

    # ── Row 0: (a) eig stable, (c) eig near-degen, (e) diag stable
    panel_label(axes[0, 0], '(a) Eigenvalues — stable')
    _plot_eigenvalues(axes[0, 0], MU, lp_s, lm_s,
                      label='stable', c_val=C_STABLE, show_gap=False)

    panel_label(axes[0, 1], '(c) Eigenvalues — near-degenerate')
    _plot_eigenvalues(axes[0, 1], MU, lp_d, lm_d,
                      label='near-degen', c_val=C_DEGEN, show_gap=True)

    panel_label(axes[0, 2], r'(e) Diagnostic — stable')
    _plot_diagnostic(axes[0, 2], MU, delta_stable, C_STABLE, near_degen=False)

    # ── Row 1: (b) θ stable, (d) θ near-degen, (f) diag near-degen
    panel_label(axes[1, 0], '(b) Mixing angle — stable')
    _plot_theta(axes[1, 0], MU, theta_s, near_degen=False)

    panel_label(axes[1, 1], '(d) Mixing angle — near-degenerate')
    _plot_theta(axes[1, 1], MU, theta_d, near_degen=True)

    panel_label(axes[1, 2], r'(f) Diagnostic — near-degenerate')
    _plot_diagnostic(axes[1, 2], MU, delta_degen, C_DEGEN, near_degen=True)

    # Match y-limits for eigenvalue panels (row 0, cols 0–1)
    y0 = min(axes[0, 0].get_ylim()[0], axes[0, 1].get_ylim()[0])
    y1 = max(axes[0, 0].get_ylim()[1], axes[0, 1].get_ylim()[1])
    axes[0, 0].set_ylim(y0, y1)
    axes[0, 1].set_ylim(y0, y1)

    # Match y-limits for mixing angle panels (row 1, cols 0–1)
    y0 = min(axes[1, 0].get_ylim()[0], axes[1, 1].get_ylim()[0])
    y1 = max(axes[1, 0].get_ylim()[1], axes[1, 1].get_ylim()[1])
    axes[1, 0].set_ylim(y0, y1)
    axes[1, 1].set_ylim(y0, y1)

    return fig


def main() -> None:
    fig = make_figure()
    out = _save_fig(fig, 'toy_model_figure_grid')
    print(f'[toy-grid] wrote {out}')


if __name__ == '__main__':
    main()
