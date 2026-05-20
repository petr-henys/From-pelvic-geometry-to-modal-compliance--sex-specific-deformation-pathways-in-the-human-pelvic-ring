#!/usr/bin/env python3
"""Population-variability extension of the 2/3-DOF toy models.

This script turns the analytical avoided-crossing toy model (2-DOF) and its
Schur-complement extension (3-DOF) into a *stochastic cohort*.

Goal
----
Provide an intuitive, population-level analogue of the pelvic FE results:
  - a robust regime vs a near-degenerate (mixing) regime,
  - swap as a discrete marker of label fragility (not a functional unit),
  - task-dependent routing under standing-like vs parturition-like loads,
  - within-subspace routing that modulates an upper-tail localization proxy.

Outputs
-------
Saves a multi-panel dashboard figure into analysis_outputs/figures/.
No new FE simulations are run; this is purely a conceptual model.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Project root on sys.path so `analysis.*` and `utils.*` are importable when
# executing this file directly (mirrors other analysis scripts).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.spectral_plots import _save_fig
from analysis.spectral_config import COL2_WIDTH, GOLDEN
from utils.plot_utils import (
    ANNOT_SIZE,
    FEMALE_COLOR,
    MALE_COLOR,
    NEUTRAL_COLOR,
    SCATTER_ALPHA,
    setup_plot_style,
    panel_label,
)


@dataclass(frozen=True)
class ToyParams:
    """Distributional hyperparameters for the stochastic toy cohort."""

    # Number of samples per sex
    n_per_sex: int = 5000

    # Near-degenerate safety factor in |Δ| <= τ|c|
    tau: float = 2.0

    # Latent morphology axis p ~ N(mu, sigma^2)
    mu_male: float = 0.10
    mu_female: float = -0.15
    sigma_male: float = 0.55
    sigma_female: float = 0.35

    # Map p -> detuning Δ = alpha*(p - p0) + noise
    alpha: float = 2.2
    p0: float = 0.0
    sigma_delta: float = 0.35

    # Mean stiffness level and coupling in the 2x2 block
    k_bar: float = 12.0
    c0: float = 0.28
    sigma_c: float = 0.06

    # 3rd pathway parameters (Schur complement)
    # s is log-normal: s = s0 * exp(N(0, sigma_log_s^2))
    s0: float = 7.5
    sigma_log_s: float = 0.35
    d10: float = 1.25
    d20: float = 1.15
    sigma_d: float = 0.25

    # Proxy weights for "upper-tail localization" under parturition-like load
    # L = w1*u1^2 + w2*u2^2 (computed from K_eff^{-1} f_part)
    w1: float = 0.25
    w2: float = 1.0


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _sample_population(params: ToyParams, seed: int) -> dict[str, np.ndarray]:
    """Sample a stochastic cohort and compute per-subject toy metrics."""
    gen = _rng(seed)

    # ------------------------------------------------------------------
    # Sample latent morphology axis p with mild sex dimorphism
    # ------------------------------------------------------------------
    n_m = params.n_per_sex
    n_f = params.n_per_sex
    sex = np.array(["M"] * n_m + ["F"] * n_f)
    p = np.empty(n_m + n_f, dtype=float)
    p[:n_m] = gen.normal(params.mu_male, params.sigma_male, size=n_m)
    p[n_m:] = gen.normal(params.mu_female, params.sigma_female, size=n_f)

    # Detuning and coupling for the 2x2 block
    delta = params.alpha * (p - params.p0) + gen.normal(0.0, params.sigma_delta, size=p.shape)
    c = np.abs(params.c0 + gen.normal(0.0, params.sigma_c, size=p.shape))

    # 3rd pathway (q3): log-normal stiffness and linear couplings to q1/q2
    s = params.s0 * np.exp(gen.normal(0.0, params.sigma_log_s, size=p.shape))
    d1 = params.d10 + gen.normal(0.0, params.sigma_d, size=p.shape)
    d2 = params.d20 + gen.normal(0.0, params.sigma_d, size=p.shape)

    # ------------------------------------------------------------------
    # Build per-subject stiffness matrices and compute metrics
    # ------------------------------------------------------------------
    k1 = params.k_bar + 0.5 * delta
    k2 = params.k_bar - 0.5 * delta

    n = len(p)
    # Storage
    z_bare = np.full(n, np.nan)
    z_eff = np.full(n, np.nan)
    theta_deg = np.full(n, np.nan)
    swap = np.full(n, np.nan)
    ratio = np.full(n, np.nan)
    logL = np.full(n, np.nan)
    f_backbone_stand = np.full(n, np.nan)
    f_cluster_stand = np.full(n, np.nan)
    f_backbone_part = np.full(n, np.nan)
    f_cluster_part = np.full(n, np.nan)

    f_part_2d = np.array([0.0, 1.0])
    f_stand_3d = np.array([0.0, 0.0, 1.0])
    f_part_3d = np.array([0.0, 1.0, 0.0])

    # Conservative rejection sampling to ensure SPD matrices.
    # With the defaults above, rejection is rare.
    i = 0
    while i < n:
        k1_i = float(k1[i])
        k2_i = float(k2[i])
        c_i = float(c[i])
        d1_i = float(d1[i])
        d2_i = float(d2[i])
        s_i = float(s[i])

        K = np.array(
            [
                [k1_i, c_i, d1_i],
                [c_i, k2_i, d2_i],
                [d1_i, d2_i, s_i],
            ],
            dtype=float,
        )
        ev3 = np.linalg.eigvalsh(K)
        if np.any(ev3 <= 1e-8):
            # Resample the "hidden pathway" parameters only; keep p/Δ/c fixed.
            s[i] = params.s0 * np.exp(gen.normal(0.0, params.sigma_log_s))
            d1[i] = params.d10 + gen.normal(0.0, params.sigma_d)
            d2[i] = params.d20 + gen.normal(0.0, params.sigma_d)
            continue

        # Effective 2x2 block by Schur complement (eliminate q3)
        k1_eff = k1_i - (d1_i * d1_i) / s_i
        k2_eff = k2_i - (d2_i * d2_i) / s_i
        c_eff = c_i - (d1_i * d2_i) / s_i
        K_eff = np.array([[k1_eff, c_eff], [c_eff, k2_eff]], dtype=float)

        ev2 = np.linalg.eigvalsh(K_eff)
        if np.any(ev2 <= 1e-10) or abs(c_eff) < 1e-10:
            # Extremely rare with defaults; resample hidden pathway.
            s[i] = params.s0 * np.exp(gen.normal(0.0, params.sigma_log_s))
            d1[i] = params.d10 + gen.normal(0.0, params.sigma_d)
            d2[i] = params.d20 + gen.normal(0.0, params.sigma_d)
            continue

        # Near-degeneracy indices (bare vs effective)
        z_bare[i] = abs(k1_i - k2_i) / abs(c_i)
        z_eff[i] = abs(k1_eff - k2_eff) / abs(c_eff)

        # Mixing angle from the *lower* eigenvector, mapped to [0, 90] degrees
        lam2, vec2 = np.linalg.eigh(K_eff)  # ascending
        v_low = vec2[:, 0]
        theta_deg[i] = np.degrees(np.arctan2(abs(v_low[1]), abs(v_low[0])))

        # Swap marker: canonical e1/e2 assignment to eigenvectors (identity vs swapped)
        score_identity = vec2[0, 0] ** 2 + vec2[1, 1] ** 2
        score_swap = vec2[0, 1] ** 2 + vec2[1, 0] ** 2
        swap[i] = 1.0 if score_swap > score_identity else 0.0

        # Within-subspace routing ratio under parturition-like load f = e2
        proj = vec2.T @ f_part_2d
        e_mode = (proj ** 2) / lam2
        denom = float(e_mode.sum())
        ratio[i] = float(e_mode[0] / denom) if denom > 0 else np.nan

        # Upper-tail localization proxy under parturition-like load (static response)
        u = np.linalg.solve(K_eff, f_part_2d)
        L = params.w1 * (u[0] ** 2) + params.w2 * (u[1] ** 2)
        logL[i] = np.log10(L) if L > 0 else np.nan

        # Load routing in full 3-DOF system (modal energy fractions)
        lam3, vec3 = np.linalg.eigh(K)
        # Identify "backbone" as the eigenvector most aligned with q3
        idx_backbone = int(np.argmax(np.abs(vec3[2, :])))
        idx_cluster = [j for j in range(3) if j != idx_backbone]

        for f, out_back, out_clust in (
            (f_stand_3d, (f_backbone_stand, i), (f_cluster_stand, i)),
            (f_part_3d, (f_backbone_part, i), (f_cluster_part, i)),
        ):
            proj3 = vec3.T @ f
            e3 = (proj3 ** 2) / lam3
            denom3 = float(e3.sum())
            if denom3 <= 0:
                out_back[0][out_back[1]] = np.nan
                out_clust[0][out_clust[1]] = np.nan
            else:
                out_back[0][out_back[1]] = float(e3[idx_backbone] / denom3)
                out_clust[0][out_clust[1]] = float(e3[idx_cluster].sum() / denom3)

        i += 1

    return {
        "sex": sex,
        "p": p,
        "z_bare": z_bare,
        "z_eff": z_eff,
        "theta_deg": theta_deg,
        "swap": swap,
        "ratio": ratio,
        "logL": logL,
        "f_backbone_stand": f_backbone_stand,
        "f_cluster_stand": f_cluster_stand,
        "f_backbone_part": f_backbone_part,
        "f_cluster_part": f_cluster_part,
        "tau": np.full_like(p, params.tau, dtype=float),
    }


def _plot_dashboard(data: dict[str, np.ndarray], params: ToyParams) -> plt.Figure:
    setup_plot_style()

    sex = data["sex"]
    is_m = sex == "M"
    is_f = sex == "F"

    # Subsample for dense scatters (keeps file size manageable)
    rng = _rng(123)
    idx_all = np.arange(len(sex))
    idx_scatter = rng.choice(idx_all, size=min(4000, len(idx_all)), replace=False)

    fig = plt.figure(figsize=(COL2_WIDTH, COL2_WIDTH / GOLDEN), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, wspace=0.28, hspace=0.32)

    # ------------------------------------------------------------------
    # (a) Hidden pathway shift: z_bare vs z_eff
    # ------------------------------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    panel_label(ax, "(a) Hidden pathway shifts near-degeneracy")
    ax.plot([0, 15], [0, 15], color="0.6", lw=0.8, zorder=1)
    ax.scatter(
        data["z_bare"][idx_scatter],
        data["z_eff"][idx_scatter],
        s=10,
        c="0.2",
        alpha=0.25,
        edgecolors="none",
        zorder=2,
    )
    ax.axhline(params.tau, color="0.2", lw=0.9, ls="--")
    ax.set_xlabel(r"bare index  $|\Delta|/|c|$")
    ax.set_ylabel(r"effective index  $|\Delta^{\mathrm{eff}}|/|c^{\mathrm{eff}}|$")
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 15)
    ax.text(
        0.05,
        0.92,
        rf"mixing if  $|\Delta^{{\mathrm{{eff}}}}|\leq {params.tau:.1f}|c^{{\mathrm{{eff}}}}|$",
        transform=ax.transAxes,
        fontsize=ANNOT_SIZE,
        va="top",
    )

    # ------------------------------------------------------------------
    # (b) Distribution of effective near-degeneracy index z_eff
    # ------------------------------------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    panel_label(ax, "(b) Near-degenerate prevalence")
    bins = np.linspace(0, 10, 60)
    ax.hist(data["z_eff"][is_m], bins=bins, density=True, color=MALE_COLOR, alpha=0.45, label="M")
    ax.hist(data["z_eff"][is_f], bins=bins, density=True, color=FEMALE_COLOR, alpha=0.45, label="F")
    ax.axvline(params.tau, color="0.2", lw=0.9, ls="--")
    ax.set_xlabel(r"$|\Delta^{\mathrm{eff}}|/|c^{\mathrm{eff}}|$")
    ax.set_ylabel("density")
    ax.set_xlim(0, 10)
    ax.legend(title="sex", ncol=2, loc="upper right")

    # ------------------------------------------------------------------
    # (c) Mixing angle vs morphology proxy p, with swap marker
    # ------------------------------------------------------------------
    ax = fig.add_subplot(gs[0, 2])
    panel_label(ax, "(c) Rotation + identity exchange")
    p = data["p"][idx_scatter]
    th = data["theta_deg"][idx_scatter]
    sw = data["swap"][idx_scatter]
    m_sc = sex[idx_scatter] == "M"
    f_sc = sex[idx_scatter] == "F"

    # Swap encoded by marker shape; sex by color
    ax.scatter(p[m_sc & (sw == 0)], th[m_sc & (sw == 0)], s=10, c=MALE_COLOR, alpha=SCATTER_ALPHA, marker="o", edgecolors="none")
    ax.scatter(p[m_sc & (sw == 1)], th[m_sc & (sw == 1)], s=18, c=MALE_COLOR, alpha=SCATTER_ALPHA, marker="^", edgecolors="none")
    ax.scatter(p[f_sc & (sw == 0)], th[f_sc & (sw == 0)], s=10, c=FEMALE_COLOR, alpha=SCATTER_ALPHA, marker="o", edgecolors="none")
    ax.scatter(p[f_sc & (sw == 1)], th[f_sc & (sw == 1)], s=18, c=FEMALE_COLOR, alpha=SCATTER_ALPHA, marker="^", edgecolors="none")
    ax.set_xlabel("morphology axis  $p$  (toy proxy)")
    ax.set_ylabel(r"mixing angle  $\theta$  (deg)")
    ax.set_ylim(0, 90)

    # ------------------------------------------------------------------
    # (d) Swap and mixing prevalence by sex
    # ------------------------------------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    panel_label(ax, "(d) Swap vs mixing (sex-stratified)")
    mix_m = float(np.mean(data["z_eff"][is_m] <= params.tau))
    mix_f = float(np.mean(data["z_eff"][is_f] <= params.tau))
    swap_m = float(np.mean(data["swap"][is_m] >= 0.5))
    swap_f = float(np.mean(data["swap"][is_f] >= 0.5))
    x = np.arange(2)
    w = 0.36
    ax.bar(x - w / 2, [mix_m, mix_f], width=w, color=NEUTRAL_COLOR, alpha=0.55, label="mixing  ($z\\leq\\tau$)")
    ax.bar(x + w / 2, [swap_m, swap_f], width=w, color="0.35", alpha=0.65, label="swap (label exchange)")
    ax.set_xticks(x, ["M", "F"])
    ax.set_ylabel("fraction of subjects")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right")

    # ------------------------------------------------------------------
    # (e) Task-dependent routing in the full 3-DOF system
    # ------------------------------------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    panel_label(ax, "(e) Task switching (load routing)")
    # Medians across subjects, sex-aggregated (simplifies message)
    med = lambda a: float(np.nanmedian(a))
    fB_st = med(data["f_backbone_stand"])
    fC_st = med(data["f_cluster_stand"])
    fB_pa = med(data["f_backbone_part"])
    fC_pa = med(data["f_cluster_part"])

    # Stacked bars: backbone vs cluster fractions
    ax.bar([0, 1], [fB_st, fB_pa], color="0.55", alpha=0.75, label="backbone mode")
    ax.bar([0, 1], [fC_st, fC_pa], bottom=[fB_st, fB_pa], color=NEUTRAL_COLOR, alpha=0.65, label="two-mode cluster")
    ax.set_xticks([0, 1], ["standing-like\n(load on $q_3$)", "parturition-like\n(load on $q_2$)"])
    ax.set_ylabel("median energy fraction")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right")

    # ------------------------------------------------------------------
    # (f) Upper-tail localization proxy vs within-subspace routing
    # ------------------------------------------------------------------
    ax = fig.add_subplot(gs[1, 2])
    panel_label(ax, "(f) Routing modulates extremes (toy proxy)")
    idx = idx_scatter
    ax.scatter(
        data["ratio"][idx],
        data["logL"][idx],
        s=10,
        c=np.where(sex[idx] == "M", MALE_COLOR, FEMALE_COLOR),
        alpha=0.25,
        edgecolors="none",
    )
    ax.set_xlabel(r"within-cluster routing  $E_1/(E_1{+}E_2)$  (parturition-like)")
    ax.set_ylabel(r"$\log_{10} L$  (upper-tail proxy)")

    return fig


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-per-sex", type=int, default=ToyParams.n_per_sex)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--tau", type=float, default=ToyParams.tau)
    p.add_argument("--out-name", type=str, default="toy_population_variability_dashboard")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    params = ToyParams(n_per_sex=args.n_per_sex, tau=args.tau)
    data = _sample_population(params, seed=args.seed)

    # Minimal console summary (useful for manuscript reporting if needed)
    mix = np.mean(data["z_eff"] <= params.tau)
    sw = np.mean(data["swap"] >= 0.5)
    print(f"[toy] N={len(data['sex'])}  mixing(z<=tau)={mix:.3f}  swap={sw:.3f}")

    fig = _plot_dashboard(data, params)
    out = _save_fig(fig, args.out_name)
    print(f"[toy] wrote {out}")


if __name__ == "__main__":
    main()
