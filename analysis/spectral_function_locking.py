#!/usr/bin/env python3
"""Function-locking index: canonical coupling × load-case energy fraction.

For each top near-degenerate cluster, computes Spearman correlations
between the cluster's canonical diameter coupling scores
(AP/ML inlet, BIS/BIT outlet) plus an outlet-AP scalar observable and its
block energy fraction under each standardised load case, separately for sex.

The resulting heatmap answers: "Why is the sex-dimorphic spectral
trade-off concentrated at rank modes 9–10?"  The answer is that
only the 9–10 cluster simultaneously satisfies:
  (a) function-locked to a parturition-motivated loading proxy (diameter coupling × LAB),
  (b) 2-level veering regime, and
  (c) sex dimorphism in subspace rotation (d_G).

Produces:
  - analysis_outputs/tables/function_locking_index.csv
  - analysis_outputs/figures/function_locking_heatmap.pdf
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from collections import Counter

import numpy as np
import pandas as pd
import zarr
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy import stats

from analysis.spectral_config import (
    DATA_DIR_FULL,
    EPS_IN_PERCENTILE,
    FIG_DIR,
    FIG_DPI,
    TAB_DIR,
    COL2_WIDTH,
    ROW_H,
)
from analysis.spectral_data import (
    load_eigenvalues,
    load_eigenvectors_obj,
    load_fe_mesh_coords,
    load_inlet_landmarks,
    load_mass_matrix,
    load_metadata,
    load_outlet_landmarks,
    load_template_and_shapes,
)
from analysis.spectral_metrics import (
    build_idw_measurement_vector,
    build_patient_measurement_vectors,
    canonical_pair_couplings,
    cluster_label,
    compute_gap_in,
    detect_clusters,
    orthonormalize_l2,
    single_functional_coupling,
)
from analysis.spectral_plots import _save_fig, _save_table
from analysis.publication_style import (
    apply_publication_style,
    panel_label as pub_panel_label,
    style_axis,
    style_colorbar,
    MALE_COLOR,
    FEMALE_COLOR,
    NEUTRAL_COLOR,
    save_publication_figure,
)
from utils.plot_utils import (
    ANNOT_SIZE,
    FEMALE_COLOR,
    FILL_ALPHA,
    MALE_COLOR,
    NEUTRAL_COLOR,
    SMALL_ANNOT_SIZE,
    setup_plot_style,
    panel_label,
)

TOP_N = 5
# Zarr loadcase axis order: SP2leg(0), SP1leg(1), LAB_phase1(2), LAB_phase2(3), LAB_phase3(4)
LOADCASE_INDICES = [0, 1, 2, 3, 4]  # standing + force-based parturition proxy
LOAD_LABELS = ["SP2leg", "SP1leg", "LAB\u2081", "LAB\u2082", "LAB\u2083"]
LOAD_LABELS_SHORT = ["SP2", "SP1", "LAB\u2081", "LAB\u2082", "LAB\u2083"]
# LaTeX-safe versions (no unicode subscripts) for .tex table output
LOAD_LABELS_TEX = ["SP2leg", "SP1leg", r"LAB$_1$", r"LAB$_2$", r"LAB$_3$"]
N_LOAD = len(LOADCASE_INDICES)
LAB_LOAD_INDICES = [2, 3, 4]  # indices into LOAD_LABELS for all LAB phases
AXIS_ORDER = ("AP", "ML", "BIS", "BIT", "OUTLETAP")
AXIS_DISPLAY = {
    "AP": "AP inlet",
    "ML": "ML inlet",
    "BIS": "BIS outlet",
    "BIT": "BIT outlet",
    "OUTLETAP": "Outlet AP scalar",
}
AXIS_PANEL_TITLES = {
    "AP": "A AP lock",
    "ML": "B ML lock",
    "BIS": "C BIS lock",
    "BIT": "D BIT lock",
    "OUTLETAP": "E outlet AP lock",
}


def _condense_regime(verdict: str) -> str:
    txt = str(verdict).strip()
    low = txt.lower()
    if "true degeneracy" in low:
        return "True degeneracy"
    if "veering" in low:
        return "Veering"
    if "weak exchange" in low:
        return "Weak exchange"
    if "mixing" in low:
        return "Mixing (4D)" if "multi-mode" in low or "4d" in low else "Mixing"
    if "stable" in low:
        return "Stable"
    return txt if txt else "?"


def _load_cluster_regimes() -> dict[str, str]:
    """Load per-cluster regime labels from veering_vs_mixing_classification.csv."""
    p = TAB_DIR / "veering_vs_mixing_classification.csv"
    if not p.exists():
        return {}

    df = pd.read_csv(p)
    out: dict[str, str] = {}
    for _, row in df.iterrows():
        key = str(row.get("cluster_key", "")).strip()
        verdict = row.get("final_verdict", "")
        if key:
            out[key] = _condense_regime(str(verdict))
    return out


def _load_sex_dg_stats() -> dict[str, tuple[float, float]]:
    """Load precomputed sex effects for d_G from sex_stratified_summary.csv.

    Returns mapping:
        "modes_9_10" -> (r_rb, p_value)
    """
    p = TAB_DIR / "sex_stratified_summary.csv"
    if not p.exists():
        return {}
    df = pd.read_csv(p)
    out: dict[str, tuple[float, float]] = {}
    for _, row in df.iterrows():
        metric = str(row.get("metric", ""))
        if not (metric.startswith("d_G (") and metric.endswith(")")):
            continue
        key = metric[len("d_G ("):-1].strip()
        if not key:
            continue
        try:
            r = float(row.get("rank_biserial_r"))
            pv = float(row.get("p_value"))
        except (TypeError, ValueError):
            continue
        out[key] = (r, pv)
    return out


def _build_measurement_vectors(
    n_subj: int,
    n_nodes: int,
    tree,
    landmarks: dict,
    outlet_landmarks: dict,
    tpl_shapes,
) -> dict[str, np.ndarray | None]:
    """Build per-patient IDW measurement vectors for inlet + outlet."""
    b: dict[str, np.ndarray | None] = {
        "AP": None, "ML": None, "BIS": None, "BIT": None, "OUTLETAP": None,
    }
    all_lm = {**landmarks, **outlet_landmarks}
    if tpl_shapes is not None:
        tpl_pts, ssm_shapes = tpl_shapes
        ssm_sub = ssm_shapes[:n_subj]
        for key, pts in all_lm.items():
            p1, p2 = pts
            if np.linalg.norm(p1 - p2) < 1e-10:
                continue
            b[key] = build_patient_measurement_vectors(
                tree, n_nodes, tpl_pts, ssm_sub,
                p1, p2, k=32, power=2.0,
                smoothing=50.0, neighbors=10,
            )
    else:
        for key, pts in all_lm.items():
            p1, p2 = pts
            direction = p1 - p2
            norm = np.linalg.norm(direction)
            if norm < 1e-10:
                continue
            e = direction / norm
            b_single = build_idw_measurement_vector(
                tree, n_nodes, e, p1, p2, k=32, power=2.0,
            )
            b[key] = np.tile(b_single, (n_subj, 1))
        return b
    return b


def _canonical_couplings(
    evecs_arr,
    member_mask: np.ndarray,
    cs: int,
    ce: int,
    b_ap: np.ndarray | None,
    b_ml: np.ndarray | None,
    b_bis: np.ndarray | None,
    b_bit: np.ndarray | None,
    b_outlet_ap: np.ndarray | None,
    M0_sp,
    n_subj: int,
) -> dict[str, np.ndarray]:
    """Canonical/scalar coupling per 1 mm for cluster members."""
    r = ce - cs
    couplings = {
        axis: np.full(n_subj, np.nan, dtype=float) for axis in AXIS_ORDER
    }

    for s in np.flatnonzero(member_mask):
        s_block = np.asarray(
            evecs_arr[int(s), cs:ce, :, :], dtype=float,
        )
        if not np.isfinite(s_block).all():
            continue
        Q_s = orthonormalize_l2(s_block.reshape(r, -1).T, M=M0_sp)

        # Inlet coupling
        if b_ap is not None and b_ml is not None:
            ap_per_1mm, ml_per_1mm, _, _, _ = canonical_pair_couplings(
                Q_s, b_ap[s], b_ml[s], M=M0_sp,
            )
            couplings["AP"][s] = ap_per_1mm
            couplings["ML"][s] = ml_per_1mm

        # Outlet coupling
        if b_bis is not None and b_bit is not None:
            bis_per_1mm, bit_per_1mm, _, _, _ = canonical_pair_couplings(
                Q_s, b_bis[s], b_bit[s], M=M0_sp,
            )
            couplings["BIS"][s] = bis_per_1mm
            couplings["BIT"][s] = bit_per_1mm

        if b_outlet_ap is not None:
            outlet_ap_per_1mm, _, _ = single_functional_coupling(
                Q_s, b_outlet_ap[s], M=M0_sp,
            )
            couplings["OUTLETAP"][s] = outlet_ap_per_1mm

    return couplings


def _spearmanr_safe(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    fin = np.isfinite(x) & np.isfinite(y)
    if fin.sum() > 5:
        return stats.spearmanr(x[fin], y[fin])
    return np.nan, np.nan


def _sig_str(p: float) -> str:
    if not np.isfinite(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def _regime_abbrev(regime: str) -> str:
    low = regime.lower()
    if "veering" in low:
        return "V"
    if "weak exchange" in low:
        return "W"
    if "mixing (4d)" in low or ("mixing" in low and "4d" in low):
        return "M4"
    if "mixing" in low:
        return "M"
    if "stable" in low:
        return "S"
    if "degeneracy" in low:
        return "D"
    return "?"


def _select_best_axis(metric_by_axis: dict[str, float]) -> str:
    """Select the axis with the largest finite absolute metric."""
    best_axis = AXIS_ORDER[0]
    best_abs = -np.inf
    for axis in AXIS_ORDER:
        value = metric_by_axis.get(axis, np.nan)
        if not np.isfinite(value):
            continue
        value_abs = abs(float(value))
        if value_abs > best_abs:
            best_axis = axis
            best_abs = value_abs
    return best_axis


def main() -> None:
    setup_plot_style(dpi=FIG_DPI)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────
    print("Loading data ...")
    ev = load_eigenvalues(DATA_DIR_FULL)
    sex, _ = load_metadata()
    n_subj = ev.shape[0]
    mask_m = sex == "M"
    mask_f = sex == "F"

    # Cluster detection
    gaps = compute_gap_in(ev)
    eps_in = float(np.nanpercentile(gaps.ravel(), EPS_IN_PERCENTILE))
    clusters_all = detect_clusters(ev, eps_in)
    sex_dg_stats = _load_sex_dg_stats()
    regimes = _load_cluster_regimes()

    cluster_counter: Counter[tuple[int, int]] = Counter()
    for clist in clusters_all:
        for cs, ce in clist:
            cluster_counter[(cs, ce)] += 1
    min_count = max(1, int(0.03 * n_subj))
    detected = [
        (cs, ce, cnt)
        for (cs, ce), cnt in cluster_counter.most_common()
        if cnt >= min_count
    ]
    top_clusters = detected[:TOP_N]

    # Energy fractions: (n_subj, n_modes, n_loadcases)
    frac_data = zarr.open(
        str(DATA_DIR_FULL / "mode_energy_fraction.zarr"), mode="r",
    )["data"][:]

    # Heavy I/O
    print("Loading eigenvectors, mass matrix, landmarks ...")
    M0_sp = load_mass_matrix()
    evecs_arr = load_eigenvectors_obj(DATA_DIR_FULL)
    n_nodes = int(evecs_arr.shape[2])
    fe_coords = load_fe_mesh_coords()
    landmarks = load_inlet_landmarks()
    outlet_landmarks = load_outlet_landmarks()
    tpl_shapes = load_template_and_shapes()

    from scipy.spatial import cKDTree

    if fe_coords is None:
        raise RuntimeError("FE mesh coordinates are required for function-locking analysis.")
    tree = cKDTree(np.asarray(fe_coords, dtype=float))

    print("Building measurement vectors (inlet + outlet) ...")
    b = _build_measurement_vectors(
        n_subj, n_nodes, tree, landmarks, outlet_landmarks, tpl_shapes,
    )
    # ── Per-cluster analysis ──────────────────────────────────
    print("\nComputing function-locking index ...")
    n_cl = len(top_clusters)
    rho_by_axis = {
        axis: np.full((n_cl, N_LOAD), np.nan, dtype=float) for axis in AXIS_ORDER
    }
    p_by_axis = {
        axis: np.full((n_cl, N_LOAD), np.nan, dtype=float) for axis in AXIS_ORDER
    }
    bf_medians = np.full((n_cl, N_LOAD), np.nan)
    dg_r_rb = np.full(n_cl, np.nan)
    dg_p = np.full(n_cl, np.nan)
    cl_labels: list[str] = []
    cl_keys: list[str] = []
    rows: list[dict] = []

    for ci, (cs, ce, cnt) in enumerate(top_clusters):
        cl_lbl = cluster_label(cs, ce)
        cl_key = f"modes_{cs + 1}_{ce}"
        r = ce - cs
        cl_labels.append(cl_lbl)
        cl_keys.append(cl_key)

        member_mask = np.array(
            [(cs, ce) in clist for clist in clusters_all], dtype=bool,
        )
        print(f"  {cl_lbl} (dim={r}, n={cnt}) ...")

        # Canonical couplings per subject
        c_subj = _canonical_couplings(
            evecs_arr, member_mask, cs, ce,
            b["AP"], b["ML"], b["BIS"], b["BIT"], b["OUTLETAP"],
            M0_sp, n_subj,
        )
        coupling_medians = {
            axis: float(np.nanmedian(values[member_mask]))
            for axis, values in c_subj.items()
        }
        dom_axis = _select_best_axis(coupling_medians)
        dom_type = AXIS_DISPLAY[dom_axis]

        # Sex dimorphism in subspace rotation (precomputed in sex_stratified_summary.csv)
        if cl_key in sex_dg_stats:
            dg_r_rb[ci], dg_p[ci] = sex_dg_stats[cl_key]

        # Block energy fraction per load case
        block_frac = np.zeros((n_subj, N_LOAD))
        for mode_j in range(cs, ce):
            block_frac += frac_data[:, mode_j, LOADCASE_INDICES]

        # Correlations
        for li in range(N_LOAD):
            bf = block_frac[:, li]
            bf_medians[ci, li] = float(np.nanmedian(bf[member_mask]))

            sel = member_mask & np.isfinite(bf)
            row = {
                "cluster": cl_lbl,
                "dim": r,
                "regime": regimes.get(cl_key, "?"),
                "load_case": LOAD_LABELS_TEX[li],
                "block_frac_median": bf_medians[ci, li],
                "coupling_type_dominant": dom_type,
                "dG_sex_r_rb": float(dg_r_rb[ci]),
                "dG_sex_p": float(dg_p[ci]),
            }
            for axis in AXIS_ORDER:
                c_axis_subj = c_subj[axis]
                rho_axis, p_axis = _spearmanr_safe(c_axis_subj[sel], bf[sel])
                rho_by_axis[axis][ci, li] = rho_axis
                p_by_axis[axis][ci, li] = p_axis

                rho_axis_m, _ = _spearmanr_safe(
                    c_axis_subj[sel & mask_m], bf[sel & mask_m],
                )
                rho_axis_f, _ = _spearmanr_safe(
                    c_axis_subj[sel & mask_f], bf[sel & mask_f],
                )

                row[f"{axis}_coupling_median"] = coupling_medians[axis]
                row[f"rho_{axis}"] = float(rho_axis)
                row[f"p_{axis}"] = float(p_axis)
                row[f"rho_{axis}_M"] = float(rho_axis_m)
                row[f"rho_{axis}_F"] = float(rho_axis_f)

            rows.append(row)

        # Print summary
        best_li = int(np.nanargmax(bf_medians[ci]))
        median_txt = ", ".join(
            f"{axis}={coupling_medians[axis]:.3f}" for axis in AXIS_ORDER
        )
        print(
            f"    Coupling medians: {median_txt} → {dom_type}"
        )
        print(
            f"    Max block frac: {LOAD_LABELS[best_li]} "
            f"({bf_medians[ci, best_li]:.3f})"
        )
        print(
            f"    d_G sex r_rb={dg_r_rb[ci]:.3f} (p={dg_p[ci]:.2e})"
        )

    # ── Save table ────────────────────────────────────────────
    df = pd.DataFrame(rows)
    _save_table(
        df, "function_locking_index",
        tex_columns=[
            "cluster", "regime", "load_case",
            "block_frac_median", "rho_AP", "rho_ML", "rho_BIS", "rho_BIT",
            "rho_OUTLETAP",
            "dG_sex_r_rb",
        ],
        tex_col_labels={
            "cluster": "Cluster",
            "regime": "Regime",
            "load_case": "Load case",
            "block_frac_median": "$\\bar f_C$",
            "rho_AP": "$\\rho_{\\mathrm{AP}}$",
            "rho_ML": "$\\rho_{\\mathrm{ML}}$",
            "rho_BIS": "$\\rho_{\\mathrm{BIS}}$",
            "rho_BIT": "$\\rho_{\\mathrm{BIT}}$",
            "rho_OUTLETAP": "$\\rho_{\\mathrm{OutletAP}}$",
            "dG_sex_r_rb": "$r_{rb}(d_G)$",
        },
    )
    print(f"\nTable: {TAB_DIR / 'function_locking_index.csv'}")

    # ── Synthesis summary table (one row per cluster per LAB phase) ──
    synth_rows: list[dict] = []
    for ci, (cs, ce, cnt) in enumerate(top_clusters):
        cl_key = cl_keys[ci]
        verdict = regimes.get(cl_key, "?")

        for lab_li in LAB_LOAD_INDICES:
            lab_label = LOAD_LABELS_TEX[lab_li]
            rho_lab = {
                axis: float(rho_by_axis[axis][ci, lab_li]) for axis in AXIS_ORDER
            }
            p_lab = {
                axis: float(p_by_axis[axis][ci, lab_li]) for axis in AXIS_ORDER
            }
            f_c_lab = float(bf_medians[ci, lab_li])

            best_axis = _select_best_axis(rho_lab)
            fli_type = AXIS_DISPLAY[best_axis]
            fli_rho = rho_lab[best_axis]
            fli_p = p_lab[best_axis]

            synth_rows.append({
                "cluster": cl_labels[ci],
                "dim": ce - cs,
                "regime": verdict,
                "dominant_coupling": fli_type,
                "FLI_rho": float(fli_rho),
                "FLI_p": float(fli_p),
                "FLI_load_case": lab_label,
                "fC_lab_median": f_c_lab,
                "rho_AP": rho_lab["AP"],
                "rho_ML": rho_lab["ML"],
                "rho_BIS": rho_lab["BIS"],
                "rho_BIT": rho_lab["BIT"],
                "rho_OUTLETAP": rho_lab["OUTLETAP"],
                "p_AP": p_lab["AP"],
                "p_ML": p_lab["ML"],
                "p_BIS": p_lab["BIS"],
                "p_BIT": p_lab["BIT"],
                "p_OUTLETAP": p_lab["OUTLETAP"],
                "dG_sex_r_rb": float(dg_r_rb[ci]),
                "dG_sex_p": float(dg_p[ci]),
            })
    synth_df = pd.DataFrame(synth_rows)
    _save_table(
        synth_df, "function_locking_synthesis",
        tex_columns=[
            "cluster", "regime", "dominant_coupling",
            "FLI_rho", "FLI_load_case", "fC_lab_median", "dG_sex_r_rb",
        ],
        tex_col_labels={
            "cluster": "Cluster",
            "regime": "Regime",
            "dominant_coupling": "Best-predictor coupling axis for FLI (AP/ML inlet, BIS/BIT outlet, or Outlet AP scalar)",
            "FLI_rho": "FLI $\\rho$",
            "FLI_load_case": "Load case",
            "fC_lab_median": "$\\bar f_C$",
            "dG_sex_r_rb": "$r_{rb}(d_G)$",
        },
    )
    print(f"Synthesis: {TAB_DIR / 'function_locking_synthesis.csv'}")
    print(synth_df.to_string(index=False))

    # ── Heatmap figure ────────────────────────────────────────
    _plot_heatmap(
        rho_by_axis, p_by_axis,
        cl_labels, cl_keys,
        dg_r_rb, dg_p, regimes,
    )


def _plot_heatmap(
    rho_by_axis: dict[str, np.ndarray],
    p_by_axis: dict[str, np.ndarray],
    cl_labels: list[str],
    cl_keys: list[str],
    dg_r_rb: np.ndarray,
    dg_p: np.ndarray,
    regimes: dict[str, str],
) -> None:
    """Six-panel figure: AP/ML/BIS/BIT/OutletAP ρ heatmaps plus d_G sex bar."""
    apply_publication_style()
    n_cl = len(cl_labels)
    # Short cluster labels for Y axis
    ylabels = []
    for ck in cl_keys:
        verdict = regimes.get(ck, "")
        parts = ck.split("_")
        if len(parts) >= 3:
            core = f"{parts[1]}–{parts[2]}"
        else:
            core = ck.replace("modes_", "").replace("_", "–")
        ylabels.append(f"{core} ({_regime_abbrev(verdict)})")

    fig, axes = plt.subplots(
        1,
        6,
        figsize=(COL2_WIDTH * 1.16, ROW_H * 1.75),
        gridspec_kw={"width_ratios": [4.0, 4.0, 4.0, 4.0, 4.0, 2.5], "wspace": 0.16},
        constrained_layout=False,
    )
    heat_axes = axes[:5]
    ax_bar = axes[5]

    vmax = max(
        max(np.nanmax(np.abs(rho_by_axis[axis])) for axis in AXIS_ORDER),
        0.3,
    )
    norm = TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax)
    cmap = "RdBu_r"
    panel_letters = ["A", "B", "C", "D", "E"]

    for idx, axis in enumerate(AXIS_ORDER):
        ax = heat_axes[idx]
        rho_mat = rho_by_axis[axis]
        p_mat = p_by_axis[axis]
        im = ax.imshow(
            rho_mat, aspect="auto", cmap=cmap, norm=norm,
            interpolation="nearest",
        )
        ax.set_xticks(range(N_LOAD))
        ax.set_xticklabels(LOAD_LABELS_SHORT, fontsize=6.8)
        ax.set_yticks(range(n_cl))
        if idx == 0:
            ax.set_yticklabels(ylabels, fontsize=7.2)
            ax.tick_params(axis="y", pad=3)
        else:
            # Keep cluster labels only on panel A to avoid label overlap.
            ax.set_yticklabels([])
            ax.tick_params(axis="y", left=False)

        # Subtle cell grid lines
        ax.set_xticks(np.arange(-0.5, N_LOAD, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n_cl, 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=0.6)
        ax.tick_params(which="minor", bottom=False, left=False)

        # Annotate cells
        for i in range(n_cl):
            for j in range(N_LOAD):
                r_val = rho_mat[i, j]
                p_val = p_mat[i, j]
                if not np.isfinite(r_val):
                    continue
                sig = _sig_str(p_val)
                txt = f"{r_val:.2f}{sig}"
                color = "white" if abs(r_val) > 0.42 * vmax else "#111827"
                ax.text(
                    j, i, txt, ha="center", va="center",
                    fontsize=4.8,
                    color=color,
                    fontweight="normal",
                )
        pub_panel_label(ax, panel_letters[idx], offset=(-0.14, 1.05))
        ax.set_title(AXIS_DISPLAY[axis], fontsize=7.8, fontweight="medium", pad=5)
        style_axis(ax, spines=("left", "bottom", "top", "right"), spine_color="#6b7280", spine_width=0.6)

    # Side panel F: d_G sex dimorphism
    bar_colors = []
    for i in range(n_cl):
        if np.isfinite(dg_p[i]) and dg_p[i] < 0.05:
            bar_colors.append(FEMALE_COLOR if dg_r_rb[i] > 0 else MALE_COLOR)
        else:
            bar_colors.append(NEUTRAL_COLOR)
    ax_bar.barh(
        range(n_cl), dg_r_rb,
        color=bar_colors, alpha=0.88,
        edgecolor="none", height=0.65,
    )
    ax_bar.set_yticks(range(n_cl))
    ax_bar.set_yticklabels([])
    ax_bar.set_xlabel(r"Rank-biserial $r_{\mathrm{rb}}(d_G)$", fontsize=7.0)
    ax_bar.set_xlim(-0.95, 1.15)
    ax_bar.axvline(0, color="#6b7280", lw=0.6, ls="--")
    for i in range(n_cl):
        if np.isfinite(dg_r_rb[i]):
            sig = _sig_str(dg_p[i])
            val = dg_r_rb[i]
            txt = f"{val:+.2f}{sig}"
            if val < 0:
                ax_bar.text(
                    val - 0.04, i, txt,
                    va="center", ha="right",
                    fontsize=5.8, color="#111827", fontweight="medium",
                )
            else:
                ax_bar.text(
                    val + 0.04, i, txt,
                    va="center", ha="left",
                    fontsize=5.8, color="#111827", fontweight="medium",
                )
    ax_bar.invert_yaxis()
    for ax in heat_axes:
        ax.invert_yaxis()
    pub_panel_label(ax_bar, "F", offset=(-0.18, 1.05))
    ax_bar.set_title(r"Sex effect ($d_G$)", fontsize=7.8, fontweight="medium", pad=5)
    style_axis(ax_bar, spines=("bottom",), x_grid=True, grid_color="#f3f4f6")

    # Colorbar
    cbar = style_colorbar(
        fig,
        im,
        ax=heat_axes.tolist(),
        orientation="horizontal",
        fraction=0.045,
        pad=0.11,
        shrink=0.65,
        label=r"Spearman rank correlation $\rho$",
    )

    fig.subplots_adjust(left=0.12, right=0.99, bottom=0.22, top=0.89, wspace=0.18)

    save_publication_figure(fig, FIG_DIR / "function_locking_heatmap")
    out = FIG_DIR / "function_locking_heatmap.pdf"
    plt.close(fig)
    print(f"Figure: {out}")


if __name__ == "__main__":
    main()
