#!/usr/bin/env python3
"""Parturition-routing trade-off figure (modes 9–10).

Three-panel figure for the rank-9/10 trade-off under LAB loading:

  (A)  d_G  vs  gap_in  by sex — females have wider gaps but more
       subspace rotation  →  trade-off between spectral separation and
       eigenvector identity stability.
  (B)  Canonical AP-inlet coupling per 1 mm (subspace-based) stratified
       by swap status × sex — small swap-associated shift, but no
       binary loss of the inlet-opening channel.
  (C)  Block LAB energy fraction (modes 9+10 summed) vs AP inlet
       diameter by sex — morphology-dependent throughput of the same
       near-degenerate subspace.

No CLI arguments; output goes to analysis_outputs/figures/.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import zarr
import matplotlib.pyplot as plt
from scipy import stats

from analysis.spectral_config import (
    DATA_DIR_FULL,
    FIG_DIR,
    FIG_DPI,
    COL2_WIDTH,
    EPS_IN_PERCENTILE,
)
from analysis.spectral_data import (
    load_eigenvalues,
    load_eigenvectors_obj,
    load_fe_mesh_coords,
    load_inlet_landmarks,
    load_mass_matrix,
    load_metadata,
    load_pelvic_dimensions,
    load_permutations,
    load_template_and_shapes,
)
from analysis.spectral_metrics import (
    build_idw_measurement_vector,
    build_patient_measurement_vectors,
    canonical_basis_in_subspace,
    compute_gap_in,
    compute_principal_angles,
    detect_clusters,
    grassmann_distance,
    invert_pairing_permutation,
    orthonormalize_l2,
)
from analysis.spectral_plots import _save_fig
from utils.plot_utils import (
    ANNOT_SIZE,
    FEMALE_COLOR,
    FILL_ALPHA,
    LINE_WIDTH,
    MALE_COLOR,
    SCATTER_ALPHA,
    MEDIAN_LW,
    BOX_LW,
    panel_label,
    setup_plot_style,
)

# ── Cluster of interest ──────────────────────────────────────
CLUSTER_START = 8   # 0-based → rank mode 9
CLUSTER_END = 10    # exclusive → rank mode 10
CLUSTER_DIM = CLUSTER_END - CLUSTER_START

# Load case index for parturition-motivated load in mode_energy_fraction.zarr
# Order: SP2leg(0), SP1leg(1), LAB_phase1(2), LAB_phase2(3), LAB_phase3(4)
LAB1_INDEX = 2
LAB2_INDEX = 3
LAB3_INDEX = 4


def _cluster_member_mask(
    eigenvalues: np.ndarray,
    eps_in: float,
) -> np.ndarray:
    """Boolean mask: which subjects have modes 9–10 as near-degenerate."""
    clusters_all = detect_clusters(eigenvalues, eps_in)
    return np.array(
        [(CLUSTER_START, CLUSTER_END) in clist for clist in clusters_all],
        dtype=bool,
    )


def _compute_dg(
    evecs_arr,
    eigenvalues: np.ndarray,
    member_mask: np.ndarray,
    M0_sp,
    n_subj: int,
) -> np.ndarray:
    """Grassmann distance d_G for modes 9–10 subspace."""
    member_idx = np.flatnonzero(member_mask)
    eig_anchor = eigenvalues[member_idx, CLUSTER_START]
    eig_med = float(np.nanmedian(eig_anchor))
    ref_idx = int(member_idx[np.nanargmin(np.abs(eig_anchor - eig_med))])
    ref_block = np.asarray(
        evecs_arr[ref_idx, CLUSTER_START:CLUSTER_END, :, :], dtype=float,
    )
    Q_ref = orthonormalize_l2(
        ref_block.reshape(CLUSTER_DIM, -1).T, M=M0_sp,
    )

    dG = np.full(n_subj, np.nan)
    for s in member_idx:
        s_block = np.asarray(
            evecs_arr[int(s), CLUSTER_START:CLUSTER_END, :, :], dtype=float,
        )
        if not np.isfinite(s_block).all():
            continue
        Q_s = orthonormalize_l2(
            s_block.reshape(CLUSTER_DIM, -1).T, M=M0_sp,
        )
        angles = compute_principal_angles(Q_ref, Q_s, M=M0_sp)
        dG[s] = grassmann_distance(angles)
    return dG


def _compute_ap_coupling(
    evecs_arr,
    member_mask: np.ndarray,
    M0_sp,
    n_subj: int,
) -> np.ndarray:
    """Canonical AP-inlet coupling per 1 mm max displacement."""
    fe_coords = load_fe_mesh_coords()
    landmarks = load_inlet_landmarks()
    if fe_coords is None or "AP" not in landmarks or "ML" not in landmarks:
        return np.full(n_subj, np.nan)

    n_nodes = int(evecs_arr.shape[2])
    if fe_coords.shape[0] != n_nodes:
        return np.full(n_subj, np.nan)

    from scipy.spatial import cKDTree

    tree = cKDTree(np.asarray(fe_coords, dtype=float))

    tpl_shapes = load_template_and_shapes()
    b_ap_all: np.ndarray | None = None
    b_ml_all: np.ndarray | None = None

    if tpl_shapes is not None:
        tpl_pts, ssm_shapes = tpl_shapes
        ssm_sub = ssm_shapes[:n_subj]
        for key, pts in landmarks.items():
            p1, p2 = pts
            if np.linalg.norm(p1 - p2) < 1e-10:
                continue
            b = build_patient_measurement_vectors(
                tree, n_nodes, tpl_pts, ssm_sub,
                p1, p2, k=32, power=2.0,
                smoothing=50.0, neighbors=10,
            )
            if key == "AP":
                b_ap_all = b
            else:
                b_ml_all = b
    else:
        for key, pts in landmarks.items():
            p1, p2 = pts
            direction = p1 - p2
            norm = np.linalg.norm(direction)
            if norm < 1e-10:
                continue
            e = direction / norm
            b_single = build_idw_measurement_vector(
                tree, n_nodes, e, p1, p2, k=32, power=2.0,
            )
            b_tiled = np.tile(b_single, (n_subj, 1))
            if key == "AP":
                b_ap_all = b_tiled
            else:
                b_ml_all = b_tiled

    if b_ap_all is None or b_ml_all is None:
        return np.full(n_subj, np.nan)

    ap_per_1mm = np.full(n_subj, np.nan)
    for s in np.flatnonzero(member_mask):
        s_block = np.asarray(
            evecs_arr[int(s), CLUSTER_START:CLUSTER_END, :, :], dtype=float,
        )
        if not np.isfinite(s_block).all():
            continue
        Q_s = orthonormalize_l2(
            s_block.reshape(CLUSTER_DIM, -1).T, M=M0_sp,
        )
        psi1, psi2, _ = canonical_basis_in_subspace(
            Q_s, b_ap_all[s], b_ml_all[s], M=M0_sp,
        )
        ap_abs = float(abs(b_ap_all[s] @ psi1))
        u1 = psi1.reshape(-1, 3)
        max_u1 = float(np.linalg.norm(u1, axis=1).max())
        if max_u1 > 1e-12:
            ap_per_1mm[s] = ap_abs / max_u1
    return ap_per_1mm


def main() -> None:
    """Generate the 3-panel trade-off figure."""
    setup_plot_style(dpi=FIG_DPI)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data ...")
    ev = load_eigenvalues(DATA_DIR_FULL)
    perm = load_permutations(DATA_DIR_FULL)
    sex, age = load_metadata()
    dims = load_pelvic_dimensions()
    n_subj = ev.shape[0]

    mask_m = sex == "M"
    mask_f = sex == "F"

    # Gap and cluster detection
    gaps = compute_gap_in(ev)
    gap_910 = gaps[:, CLUSTER_START]  # gap_in between modes 9 and 10
    eps_in = float(np.nanpercentile(gaps.ravel(), EPS_IN_PERCENTILE))
    member = _cluster_member_mask(ev, eps_in)

    # Swap status for modes 9–10
    perm_r2r = invert_pairing_permutation(perm)
    is_swapped = (
        (perm_r2r[:, CLUSTER_START] >= 0)
        & (perm_r2r[:, CLUSTER_START + 1] >= 0)
        & (perm_r2r[:, CLUSTER_START] != perm_r2r[:, CLUSTER_START + 1])
        & (perm_r2r[:, CLUSTER_START] > perm_r2r[:, CLUSTER_START + 1])
    )

    # ── Heavy computation: d_G and AP coupling ────────────────
    M0_sp = load_mass_matrix()
    evecs_arr = load_eigenvectors_obj(DATA_DIR_FULL)

    print("Computing Grassmann distances ...")
    dG = _compute_dg(evecs_arr, ev, member, M0_sp, n_subj)

    print("Computing canonical AP coupling ...")
    ap_coupling = _compute_ap_coupling(evecs_arr, member, M0_sp, n_subj)

    # LAB energy fraction (block sum over modes 9+10)
    frac_zarr = zarr.open(
        str(DATA_DIR_FULL / "mode_energy_fraction.zarr"), mode="r",
    )
    frac_data = frac_zarr["data"][:]  # (n_subj, n_modes, n_loadcases)
    lab1_block_frac = (
        frac_data[:, CLUSTER_START, LAB1_INDEX]
        + frac_data[:, CLUSTER_START + 1, LAB1_INDEX]
    )
    lab2_block_frac = (
        frac_data[:, CLUSTER_START, LAB2_INDEX]
        + frac_data[:, CLUSTER_START + 1, LAB2_INDEX]
    )
    lab3_block_frac = (
        frac_data[:, CLUSTER_START, LAB3_INDEX]
        + frac_data[:, CLUSTER_START + 1, LAB3_INDEX]
    )

    ap_dim = dims.get("AnteriorPosteriorInletDiameter")

    # ── Figure ────────────────────────────────────────────────
    print("Plotting ...")
    fig, axes = plt.subplots(
        1,
        5,
        figsize=(COL2_WIDTH * 1.6, COL2_WIDTH / 3.2),
        constrained_layout=False,
        gridspec_kw={"wspace": 0.20},
    )
    ax_a, ax_b, ax_c, ax_d, ax_e = axes

    # ── Panel (A): d_G vs gap_in, by sex ─────────────────────
    fin_a = member & np.isfinite(dG) & np.isfinite(gap_910)
    for mask_sex, color, label in [
        (mask_m, MALE_COLOR, "Male"),
        (mask_f, FEMALE_COLOR, "Female"),
    ]:
        sel = fin_a & mask_sex
        ax_a.scatter(
            gap_910[sel], dG[sel],
            c=color, alpha=SCATTER_ALPHA, s=14,
            edgecolors="none", label=label, zorder=3,
        )
    ax_a.set_xlabel("gap$_{\\mathrm{in}}$(9–10)")
    ax_a.set_ylabel("$d_G$ (modes 9–10)")

    # Mann–Whitney annotation for d_G
    dg_m = dG[fin_a & mask_m]
    dg_f = dG[fin_a & mask_f]
    u, p_dg = stats.mannwhitneyu(dg_m, dg_f, alternative="two-sided")
    r_dg = 1 - 2 * u / (len(dg_m) * len(dg_f))
    sig = "***" if p_dg < 0.001 else "**" if p_dg < 0.01 else "*" if p_dg < 0.05 else "n.s."
    ax_a.text(
        0.97, 0.97,
        f"$r_{{rb}}$={r_dg:.2f} {sig}",
        transform=ax_a.transAxes, ha="right", va="top",
        fontsize=ANNOT_SIZE,
        color="crimson" if p_dg < 0.05 else "0.4",
    )
    ax_a.legend(loc="lower right", fontsize=ANNOT_SIZE)
    panel_label(ax_a, "A Gap vs $d_G$")

    # ── Panel (B): AP coupling by swap × sex ─────────────────
    fin_b = member & np.isfinite(ap_coupling)
    groups = []
    group_labels = []
    group_colors = []
    for swap_state, swap_lbl in [(False, "No swap"), (True, "Swapped")]:
        for m_sex, color, sex_lbl in [
            (mask_m, MALE_COLOR, "M"),
            (mask_f, FEMALE_COLOR, "F"),
        ]:
            sel = fin_b & m_sex & (is_swapped == swap_state)
            vals = ap_coupling[sel]
            vals = vals[np.isfinite(vals)]
            groups.append(vals)
            group_labels.append(f"{swap_lbl}\n{sex_lbl}")
            group_colors.append(color)

    bp = ax_b.boxplot(
        groups,
        tick_labels=group_labels,
        patch_artist=True,
        showfliers=False,
        widths=0.55,
        medianprops=dict(color="black", linewidth=MEDIAN_LW),
        boxprops=dict(linewidth=BOX_LW),
        whiskerprops=dict(linewidth=BOX_LW),
        capprops=dict(linewidth=BOX_LW),
    )
    for patch, color in zip(bp["boxes"], group_colors):
        patch.set_facecolor(color)
        patch.set_alpha(FILL_ALPHA)
    ax_b.set_ylabel("AP coupling (mm/mm)")

    # Test: pooled no-swap vs swapped (ignoring sex)
    no_swap_all = np.concatenate([groups[0], groups[1]])
    swap_all = np.concatenate([groups[2], groups[3]])
    if len(no_swap_all) > 1 and len(swap_all) > 1:
        u_swap, p_swap = stats.mannwhitneyu(
            no_swap_all, swap_all, alternative="two-sided",
        )
        r_swap = 1 - 2 * u_swap / (len(no_swap_all) * len(swap_all))
        sig_swap = (
            "***" if p_swap < 0.001
            else "**" if p_swap < 0.01
            else "*" if p_swap < 0.05
            else "n.s."
        )
        ax_b.text(
            0.5, 0.97,
            f"swap effect: $r_{{rb}}$={r_swap:.2f} {sig_swap}",
            transform=ax_b.transAxes, ha="center", va="top",
            fontsize=ANNOT_SIZE,
            color="crimson" if p_swap < 0.05 else "0.4",
        )
    panel_label(ax_b, "B AP coupling")

    # ── Panel (C): LAB₁ block fraction vs AP diameter ───────
    for ax_lab, lab_frac, lab_tag, panel_lbl in [
        (ax_c, lab1_block_frac, "LAB\u2081", "C"),
        (ax_d, lab2_block_frac, "LAB\u2082", "D"),
        (ax_e, lab3_block_frac, "LAB\u2083", "E"),
    ]:
        if ap_dim is not None:
            for m_sex, color, label in [
                (mask_m, MALE_COLOR, "Male"),
                (mask_f, FEMALE_COLOR, "Female"),
            ]:
                sel = m_sex & np.isfinite(lab_frac) & np.isfinite(ap_dim)
                ax_lab.scatter(
                    ap_dim[sel], lab_frac[sel],
                    c=color, alpha=SCATTER_ALPHA, s=14,
                    edgecolors="none", label=label, zorder=3,
                )
                # OLS trend line
                x, y = ap_dim[sel], lab_frac[sel]
                if len(x) > 5:
                    slope, intercept, r, p, _ = stats.linregress(x, y)
                    x_fit = np.linspace(x.min(), x.max(), 50)
                    ax_lab.plot(
                        x_fit, intercept + slope * x_fit,
                        color=color, lw=LINE_WIDTH * 1.5, ls="--",
                        alpha=0.7, zorder=2,
                    )
            ax_lab.set_xlabel("AP inlet diameter (mm)")
            ax_lab.set_ylabel(f"{lab_tag} block energy fraction (modes 9+10)")

            # Overall Spearman for fraction vs. AP diameter
            fin_c = np.isfinite(lab_frac) & np.isfinite(ap_dim)
            rho, p_rho = stats.spearmanr(ap_dim[fin_c], lab_frac[fin_c])
            ax_lab.text(
                0.97, 0.97,
                f"$\\rho_s$={rho:.2f}, p={p_rho:.1e}",
                transform=ax_lab.transAxes, ha="right", va="top",
                fontsize=ANNOT_SIZE,
                color="crimson" if p_rho < 0.05 else "0.4",
            )
            ax_lab.legend(loc="lower left", fontsize=ANNOT_SIZE)
        else:
            ax_lab.text(
                0.5, 0.5, "AP dimension\nnot available",
                transform=ax_lab.transAxes, ha="center", va="center",
            )
        panel_label(ax_lab, f"{panel_lbl} {lab_tag} block vs AP dia.")

    fig.subplots_adjust(left=0.06, right=0.995, bottom=0.20, top=0.92, wspace=0.22)

    out = _save_fig(fig, "labour_routing_tradeoff")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
