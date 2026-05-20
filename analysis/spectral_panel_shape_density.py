#!/usr/bin/env python3
"""Spectral analysis panel: shape + bone density variability across pelvic cohort.

Systematic pipeline matching manuscript Secs. A-D:
  A. Eigenvalue gap computation + data-driven cluster detection (all modes)
  B. Subspace tracking via principal angles / Grassmann distance (detected clusters)
  C. Near-degenerate mixing vs. true degeneracy vs. veering classification
  D. Canonical inlet metrics for detected near-degenerate clusters

No CLI arguments; edit constants in spectral_config.py.
"""
from __future__ import annotations

import sys
import warnings
from collections import Counter
from pathlib import Path

# Project root = parent of this script's directory
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy import stats

from analysis.spectral_config import (
    DATA_DIR_FULL,
    DATA_DIR_MATERIAL,
    DATA_DIR_SHAPE,
    CLUSTER_MORPHOLOGY_PROXY,
    EPS_DEG,
    EPS_IN_PERCENTILE,
    FIG_DIR,
    FIG_DPI,
    MIN_CLUSTER_PREVALENCE,
    MORPHOLOGY_PROXY,
    N_MODES,
    SUBJECT_SUBSET,
    TAB_DIR,
    TOP_N_CLUSTERS,
)
from analysis.spectral_data import (
    load_eigenvalues,
    load_eigenvectors_obj,
    load_fe_mesh_coords,
    load_inlet_landmarks,
    load_mass_matrix,
    load_metadata,
    load_outlet_landmarks,
    load_pelvic_dimensions,
    load_permutations,
    load_template_and_shapes,
)
from analysis.spectral_metrics import (
    analyze_cluster_permutation_structure,
    analyze_swap_patterns,
    build_idw_measurement_vector,
    build_patient_measurement_vectors,
    canonical_pair_couplings,
    classify_cluster,
    classify_rank_adjacent_pair_swap,
    cluster_label,
    cohort_bootstrap_robustness,
    cohort_ordering_permutation_test,
    permutation_null_test_veering,
    compute_gap_in,
    compute_principal_angles,
    compute_sep_out,
    detect_clusters,
    format_mode_set_1based,
    gap_sex_age_effects,
    grassmann_distance,
    invert_pairing_permutation,
    ols_sex_age_per_mode,
    orthonormalize_l2,
    summarize_rank_cluster_reference_mapping,
    swap_count_per_subject,
    swap_rate_by_sex,
)
from analysis.spectral_plots import (
    _save_table,
    plot_sex_age_analysis,
    plot_sex_stratified_summary,
    plot_spectral_structure_dashboard,
    plot_subspace_analysis_dashboard,
)
from utils.plot_utils import setup_plot_style


def main() -> None:
    """Run the full spectral analysis pipeline."""
    setup_plot_style(dpi=FIG_DPI)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Spectral Panel Analysis: Shape + Bone Density Variability")
    print("=" * 60)

    # ── 1. Load data ──────────────────────────────────────────
    print("\n[1/8] Loading data ...")
    ev_full = load_eigenvalues(DATA_DIR_FULL)
    perm_full = load_permutations(DATA_DIR_FULL)
    sex, age = load_metadata()
    dims = load_pelvic_dimensions()
    landmarks = load_inlet_landmarks()
    outlet_landmarks = load_outlet_landmarks()

    print("  Assembling M0 (reference mass matrix) ...")
    M0_sp = load_mass_matrix()
    if M0_sp is not None:
        print(f"  M0: {M0_sp.shape}, nnz={M0_sp.nnz}")
    else:
        print("  M0 unavailable — subspace metrics will use L2")

    n_subj = ev_full.shape[0]
    if SUBJECT_SUBSET is not None:
        n_subj = min(SUBJECT_SUBSET, n_subj)
        ev_full = ev_full[:n_subj]
        perm_full = perm_full[:n_subj]
        sex = sex[:n_subj]
        age = age[:n_subj]
        dims = {k: v[:n_subj] for k, v in dims.items()}
    perm_rank_to_ref = invert_pairing_permutation(perm_full)

    mask_m = sex == "M"
    mask_f = sex == "F"
    print(f"  N = {n_subj}  (M={mask_m.sum()}, F={mask_f.sum()})")
    print(f"  Age: {age.mean():.1f} \u00b1 {age.std():.1f}  [{age.min()}\u2013{age.max()}]")

    if np.any(~np.isfinite(ev_full)) or np.any(ev_full <= 0):
        warnings.warn(
            "Non-finite or non-positive eigenvalues detected.",
            stacklevel=2,
        )
    n_unsorted = int(np.sum(np.any(np.diff(ev_full, axis=1) < 0, axis=1)))
    if n_unsorted:
        warnings.warn(
            f"{n_unsorted}/{n_subj} subjects have non-monotone eigenvalues.",
            stacklevel=2,
        )

    failed_any = (perm_full == -1).any(axis=1)
    print(f"  Mode-pairing failures (any -1): {int(failed_any.sum())}/{n_subj}")

    ev_shape, ev_material = None, None
    shape_zarr = DATA_DIR_SHAPE / "eigenvalues.zarr"
    material_zarr = DATA_DIR_MATERIAL / "eigenvalues.zarr"
    if shape_zarr.exists():
        ev_shape = load_eigenvalues(DATA_DIR_SHAPE)[:n_subj]
        print(f"  Shape-only eigenvalues: {ev_shape.shape}")
    else:
        print("  Shape-only: not available")
    if material_zarr.exists():
        ev_material = load_eigenvalues(DATA_DIR_MATERIAL)[:n_subj]
        print(f"  Material-only eigenvalues: {ev_material.shape}")
    else:
        print("  Material-only: not available")

    # ── 2. Sex and age analysis ───────────────────────────────
    print("\n[2/8] Sex and age analysis ...")
    gaps_full = compute_gap_in(ev_full)

    ols = ols_sex_age_per_mode(ev_full, sex, age, N_MODES)
    sr_m, sr_f, pval_swap_sex = swap_rate_by_sex(perm_full, mask_m, mask_f, N_MODES)
    n_swapped_per_subj = swap_count_per_subject(perm_full, N_MODES)
    rho_swap_age, p_swap_age = stats.spearmanr(age, n_swapped_per_subj)
    gap_fx = gap_sex_age_effects(gaps_full, mask_m, mask_f, age, N_MODES)

    plot_sex_age_analysis(
        age=age, mask_m=mask_m, mask_f=mask_f, sex=sex,
        beta_sex=ols["beta_sex"], beta_age=ols["beta_age"],
        ci_sex=ols["ci_sex"], ci_age=ols["ci_age"],
        pval_sex=ols["pval_sex"], pval_age=ols["pval_age"],
        swap_rate_m=sr_m, swap_rate_f=sr_f, pval_swap_sex=pval_swap_sex,
        n_swapped_per_subj=n_swapped_per_subj,
        rho_swap_age=rho_swap_age, p_swap_age=p_swap_age,
        gap_r_sex=gap_fx["r_sex"], gap_rho_age=gap_fx["rho_age"],
        gap_p_sex=gap_fx["p_sex"], gap_p_age=gap_fx["p_age"],
    )

    # Tables
    sex_age_rows = []
    for i in range(N_MODES):
        sex_age_rows.append({
            "mode": i + 1,
            "\u03b2_sex": ols["beta_sex"][i],
            "p_sex": ols["pval_sex"][i],
            "\u03b2_age": ols["beta_age"][i],
            "p_age": ols["pval_age"][i],
            "swap_rate_M": sr_m[i],
            "swap_rate_F": sr_f[i],
            "p_swap_sex": pval_swap_sex[i],
        })
    sex_age_df = pd.DataFrame(sex_age_rows)
    sex_age_df.to_csv(TAB_DIR / "sex_age_ols.csv", index=False, float_format="%.4g")
    print("  OLS coefficients (log \u03bb ~ sex + age):")
    print(sex_age_df.to_string(index=False))

    gap_rows = []
    for gi in range(N_MODES - 1):
        gap_rows.append({
            "pair": f"{gi+1}\u2013{gi+2}",
            "r_sex": gap_fx["r_sex"][gi],
            "p_sex": gap_fx["p_sex"][gi],
            "\u03c1_age": gap_fx["rho_age"][gi],
            "p_age": gap_fx["p_age"][gi],
        })
    gap_df = pd.DataFrame(gap_rows)
    gap_df.to_csv(TAB_DIR / "gap_sex_age.csv", index=False, float_format="%.4g")
    print(f"\n  Swap-count vs age: \u03c1_s = {rho_swap_age:.3f} (p = {p_swap_age:.2e})")
    print("  Gap sex/age effects:")
    print(gap_df.to_string(index=False))

    # ── 3. Gap distribution + cluster detection ───────────────
    print("\n[3/8] Gap distribution & cluster detection (Sec. A) ...")

    eps_in = float(np.nanpercentile(gaps_full.ravel(), EPS_IN_PERCENTILE))
    print(f"  eps_in (P{EPS_IN_PERCENTILE} of all gaps): {eps_in:.4f}")

    clusters_all = detect_clusters(ev_full, eps_in)
    cluster_counter: Counter[tuple[int, int]] = Counter()
    for clist in clusters_all:
        for cs, ce in clist:
            cluster_counter[(cs, ce)] += 1

    min_count = max(1, int(MIN_CLUSTER_PREVALENCE * n_subj))
    detected_clusters = [
        (cs, ce, cnt)
        for (cs, ce), cnt in cluster_counter.most_common()
        if cnt >= min_count
    ]

    print(f"  Detected {len(detected_clusters)} clusters "
          f"(prevalence >= {MIN_CLUSTER_PREVALENCE:.0%}):")
    cluster_rows = []
    for cs, ce, cnt in detected_clusters:
        member_mask = np.array(
            [(cs, ce) in clist for clist in clusters_all], dtype=bool,
        )
        n_members = int(member_mask.sum())

        sep = compute_sep_out(ev_full, cs, ce)[member_mask]
        cls = classify_cluster(gaps_full, cs, ce, EPS_DEG, member_mask=member_mask)
        true_deg = cls["true_degeneracy"]
        near_deg = cls["near_degenerate"]
        denom = max(1, true_deg + near_deg)

        internal = gaps_full[member_mask, cs:ce - 1]
        min_internal = np.nanmin(internal, axis=1) if internal.ndim == 2 else internal
        row = {
            "cluster": cluster_label(cs, ce),
            "dim": ce - cs,
            "n_subjects": n_members,
            "prevalence": f"{n_members / n_subj:.1%}",
            "median_gap_in_min": float(np.nanmedian(min_internal)),
            "median_sep_out": float(np.nanmedian(sep)),
            "true_deg_count": true_deg,
            "near_deg_count": near_deg,
            "unknown_count": cls["unknown"],
            "gap_type": "true degeneracy" if true_deg / denom >= 0.5 else "near-degenerate",
        }
        cluster_rows.append(row)
        print(
            f"    {row['cluster']}  dim={row['dim']}  "
            f"n={n_members}/{n_subj}  gap_min={row['median_gap_in_min']:.4f}  "
            f"sep_out={row['median_sep_out']:.4f}  [{row['gap_type']}]"
        )
    _save_table(
        pd.DataFrame(cluster_rows), "clusters",
        tex_columns=["cluster", "dim", "n_subjects", "prevalence",
                     "median_gap_in_min", "median_sep_out", "gap_type"],
        tex_col_labels={
            "cluster": "Cluster", "dim": "Dim",
            "n_subjects": "$n$", "prevalence": "Prev.",
            "median_gap_in_min": "Med. gap$_{\\mathrm{in,min}}$",
            "median_sep_out": "Med. sep$_{\\mathrm{out}}$",
            "gap_type": "Type",
        },
    )

    map_df = summarize_rank_cluster_reference_mapping(
        perm_rank_to_ref, clusters_all, detected_clusters,
    )
    _save_table(map_df, "rank_cluster_ref_mapping")

    top_clusters = detected_clusters[:TOP_N_CLUSTERS]
    print(f"\n  \u2192 Top {len(top_clusters)} clusters for detailed analysis")

    # ── 4. Mode permutation statistics ────────────────────────
    print("\n[4/8] Mode permutation statistics ...")
    identity_perm = np.arange(N_MODES)
    n_identity = sum(np.array_equal(p, identity_perm) for p in perm_full)
    print(f"  Identity permutations: {n_identity}/{n_subj} ({n_identity / n_subj:.1%})")

    fail_rate = np.mean(perm_full == -1, axis=0)
    swap_rate = np.array([
        np.mean((perm_full[:, m] != m) & (perm_full[:, m] >= 0))
        for m in range(N_MODES)
    ])

    swap_pair_counter: Counter[tuple[int, int]] = Counter()
    for p in perm_full:
        for m in range(N_MODES):
            m_to = int(p[m])
            if m_to >= 0 and m_to != m:
                swap_pair_counter[(m, m_to)] += 1

    _save_table(
        pd.DataFrame({
            "mode_1based": np.arange(1, N_MODES + 1),
            "swap_rate": swap_rate,
            "fail_rate": fail_rate,
            "nonid_rate": swap_rate + fail_rate,
            "n_swapped": (swap_rate * n_subj).astype(int),
            "n_failed": (fail_rate * n_subj).astype(int),
        }),
        "swap_rates",
        tex_col_labels={
            "mode_1based": "Mode",
            "swap_rate": "Swap rate",
            "fail_rate": "Fail rate",
            "nonid_rate": "Non-id. rate",
            "n_swapped": "$n$ swapped",
            "n_failed": "$n$ failed",
        },
    )

    top_swaps = swap_pair_counter.most_common(10)
    print("  Top swap pairs (1-based):")
    for (m_from, m_to), cnt in top_swaps:
        print(f"    mode {m_from + 1} \u2192 {m_to + 1}: {cnt}/{n_subj}")

    # ── Dashboard: spectral structure ─────────────────────────
    plot_spectral_structure_dashboard(
        gaps_full, eps_in, swap_rate, fail_rate, swap_pair_counter,
        n_subj, ev_full, ev_shape, ev_material,
    )

    # ── 5. Subspace analysis ──────────────────────────────────
    print("\n[5/8] Subspace analysis on detected clusters (Sec. B) ...")
    evecs_zarr = DATA_DIR_FULL / "eigenvectors.zarr"
    evecs_arr = None
    if evecs_zarr.exists():
        evecs_arr = load_eigenvectors_obj(DATA_DIR_FULL)
        print(f"  Eigenvectors (zarr): shape={tuple(evecs_arr.shape)} dtype={evecs_arr.dtype}")
    else:
        print(f"  Cannot open eigenvectors ({evecs_zarr}); skipping subspace analysis")

    grassmann_results: dict[str, np.ndarray] = {}

    if evecs_arr is not None and top_clusters:
        for cs, ce, cnt in top_clusters:
            cl_key = f"modes_{cs + 1}_{ce}"
            r = ce - cs
            cl_lbl = cluster_label(cs, ce)
            print(f"  {cl_lbl} (dim={r}, n={cnt}) ...")

            member_mask = np.array(
                [(cs, ce) in clist for clist in clusters_all], dtype=bool,
            )
            member_idx = np.flatnonzero(member_mask)
            if member_idx.size < 2:
                print(f"    Skipped (only {member_idx.size} members)")
                continue

            eig_anchor = ev_full[member_idx, cs]
            eig_med = float(np.nanmedian(eig_anchor))
            ref_idx_cl = int(member_idx[np.nanargmin(np.abs(eig_anchor - eig_med))])
            ref_block = np.asarray(evecs_arr[ref_idx_cl, cs:ce, :, :], dtype=float)
            if not np.isfinite(ref_block).all():
                print("    Skipped (non-finite reference eigenvectors)")
                continue
            Q_ref = orthonormalize_l2(ref_block.reshape(r, -1).T, M=M0_sp)

            dG_vals = np.full(n_subj, np.nan)
            for s in member_idx:
                s_block = np.asarray(evecs_arr[int(s), cs:ce, :, :], dtype=float)
                if not np.isfinite(s_block).all():
                    continue
                Q_s = orthonormalize_l2(s_block.reshape(r, -1).T, M=M0_sp)
                angles = compute_principal_angles(Q_ref, Q_s, M=M0_sp)
                dG_vals[s] = grassmann_distance(angles)

            grassmann_results[cl_key] = dG_vals
            dG_finite = dG_vals[np.isfinite(dG_vals)]
            print(
                f"    d_G: med={np.median(dG_finite):.4f}  "
                f"mean={np.mean(dG_finite):.4f}  max={np.max(dG_finite):.4f}  "
                f"({int(np.isfinite(dG_vals).sum())}/{n_subj} finite)"
            )

    # ── 6. Veering vs. mixing classification ──────────────────
    print("\n[6/8] Veering vs. mixing classification (Sec. C) ...")
    morph_proxy = dims.get(MORPHOLOGY_PROXY)
    class_rows: list[dict] = []
    pair_rows: list[dict] = []

    if morph_proxy is None:
        print(f"  Proxy '{MORPHOLOGY_PROXY}' not found; skipping")
    else:
        # Pre-compute default sort (used when no per-cluster override)
        sort_idx = np.argsort(morph_proxy)
        p_sorted_proxy = morph_proxy[sort_idx]
        ev_sorted = ev_full[sort_idx]
        gaps_sorted = compute_gap_in(ev_sorted)
        sex_sorted = sex[sort_idx]
        window = max(1, n_subj // 20)

        for cs, ce, cnt in top_clusters:
            if ce - cs < 2:
                continue
            cl_lbl = cluster_label(cs, ce)
            cl_key = f"modes_{cs + 1}_{ce}"
            cluster_modes = list(range(cs, ce))

            # Select cluster-specific morphology proxy if overridden
            cl_proxy_name = CLUSTER_MORPHOLOGY_PROXY.get(cl_key, MORPHOLOGY_PROXY)
            cl_proxy = dims.get(cl_proxy_name, morph_proxy)
            if cl_proxy_name != MORPHOLOGY_PROXY:
                print(f"    Using morphology proxy: {cl_proxy_name}")
                cl_sort_idx = np.argsort(cl_proxy)
                cl_p_sorted = cl_proxy[cl_sort_idx]
                cl_ev_sorted = ev_full[cl_sort_idx]
                cl_gaps_sorted = compute_gap_in(cl_ev_sorted)
                cl_sex_sorted = sex[cl_sort_idx]
            else:
                cl_sort_idx = sort_idx
                cl_p_sorted = p_sorted_proxy
                cl_ev_sorted = ev_sorted
                cl_gaps_sorted = gaps_sorted
                cl_sex_sorted = sex_sorted

            member_mask = np.array(
                [(cs, ce) in clist for clist in clusters_all], dtype=bool,
            )
            n_members = int(member_mask.sum())

            swap_info = analyze_swap_patterns(
                perm_rank_to_ref, cluster_modes, subject_mask=member_mask,
            )
            perm_struct = analyze_cluster_permutation_structure(
                perm_rank_to_ref, cluster_modes, subject_mask=member_mask,
            )
            deg = classify_cluster(gaps_full, cs, ce, EPS_DEG, member_mask=member_mask)
            true_deg = deg["true_degeneracy"]
            near_deg = deg["near_degenerate"]
            denom = max(1, true_deg + near_deg)
            gap_type = "true degeneracy" if true_deg / denom >= 0.5 else "near-degenerate"

            # Dominant reference-mode set
            ref_mode_set_str = ""
            ref_mode_set_rate = np.nan
            paired_all_rate = np.nan
            block = perm_rank_to_ref[member_mask, cs:ce]
            if block.size:
                paired_all = np.all(block >= 0, axis=1)
                paired_all_rate = float(paired_all.mean()) if paired_all.size else np.nan
                sets_1based: list[tuple[int, ...]] = []
                for row in block[paired_all]:
                    modes = tuple(sorted(int(x) + 1 for x in row.tolist()))
                    sets_1based.append(modes)
                if sets_1based:
                    c = Counter(sets_1based)
                    mode_set, mode_cnt = c.most_common(1)[0]
                    ref_mode_set_str = format_mode_set_1based(mode_set)
                    ref_mode_set_rate = float(mode_cnt / len(sets_1based))

            # Adjacent rank-pair swap diagnostics
            pair_results = [
                classify_rank_adjacent_pair_swap(
                    perm_rank_to_ref, cl_proxy, rank_i,
                    window=window, subject_mask=member_mask,
                )
                for rank_i in range(cs, ce - 1)
            ]
            sig_pairs = [r for r in pair_results if r.get("swap_rate", 0.0) > 0.02]
            n_pairs = len(sig_pairs)
            n_veering_pairs = sum(r.get("verdict") == "veering" for r in sig_pairs)
            n_weak_pairs = sum(r.get("verdict") == "weak exchange" for r in sig_pairs)

            has_swap = n_pairs > 0
            has_veering = n_veering_pairs > 0
            has_weak = n_weak_pairs > 0
            if has_veering:
                swap_verdict = "VEERING (strongest pairwise exchange (operational))"
            elif has_weak:
                swap_verdict = "WEAK EXCHANGE (sporadic rank swaps)"
            elif has_swap:
                swap_verdict = "MIXING (scattered rank-pair shuffling)"
            else:
                swap_verdict = "STABLE (no significant swaps)"

            has_cycles = perm_struct["cycle_gt2_rate"] > 0.10
            if gap_type == "true degeneracy":
                final_verdict = "TRUE DEGENERACY (eigenspace not uniquely identified)"
            elif (ce - cs) > 2:
                if has_cycles:
                    final_verdict = "MIXING (multi-mode; cycles > 2 common)"
                elif has_swap and has_veering:
                    final_verdict = "MIXING (multi-mode; localized transpositions present)"
                elif has_swap and not has_weak:
                    final_verdict = "MIXING (multi-mode; scattered shuffling)"
                elif has_swap:
                    final_verdict = "WEAK EXCHANGE (multi-mode; sporadic rank swaps)"
                else:
                    final_verdict = "STABLE (no significant swaps)"
            else:
                final_verdict = swap_verdict

            # Per-pair diagnostics rows
            for pr in sig_pairs:
                mi, mj = pr["rank_pair"]
                gi = int(mi)
                g_all = gaps_full[:, gi]
                ref_pair = pr.get("ref_pair")
                ref_pair_str = (
                    f"{ref_pair[0] + 1}\u21c4{ref_pair[1] + 1}" if ref_pair else ""
                )
                pair_rows.append({
                    "cluster": cl_lbl,
                    "cluster_key": cl_key,
                    "rank_pair": f"{mi + 1}\u21c4{mj + 1}",
                    "ref_pair_mode": ref_pair_str,
                    "swap_rate": float(pr["swap_rate"]),
                    "invalid_rate": float(pr.get("invalid_rate", np.nan)),
                    "match_rate": float(pr.get("match_rate", np.nan)),
                    "median_gap_in": float(np.nanmedian(g_all)),
                    "near_deg_rate": float(np.mean(g_all < eps_in)),
                    "true_deg_rate": float(np.mean(g_all <= EPS_DEG)),
                    "peak_rate": float(pr.get("peak_rate", np.nan)),
                    "active_frac": float(pr.get("active_frac", np.nan)),
                    "cv_swap": float(pr.get("cv_swap", np.nan)),
                    "verdict": pr["verdict"],
                })

            internal = gaps_full[member_mask, cs:ce - 1]
            min_internal = np.nanmin(internal, axis=1) if internal.ndim == 2 else internal
            class_rows.append({
                "cluster": cl_lbl,
                "cluster_key": cl_key,
                "dim": ce - cs,
                "n_subjects": n_members,
                "prevalence": f"{n_members / n_subj:.1%}",
                "median_gap_in_min": float(np.nanmedian(min_internal)),
                "gap_type": gap_type,
                "swap_verdict": swap_verdict,
                "final_verdict": final_verdict,
                "ref_modes_mode": ref_mode_set_str,
                "ref_modes_mode_rate": float(ref_mode_set_rate) if np.isfinite(ref_mode_set_rate) else np.nan,
                "paired_all_rate": float(paired_all_rate) if np.isfinite(paired_all_rate) else np.nan,
                "any_swap_rate": float(swap_info["any_swap_rate"]),
                "failed_rate": float(swap_info["failed_rate"]),
                "within_block_rate": float(perm_struct["within_block_rate"]),
                "transposition_rate": float(perm_struct["transposition_rate"]),
                "cycle_gt2_rate": float(perm_struct["cycle_gt2_rate"]),
                "n_sig_pairs": n_pairs,
                "n_veering_pairs": n_veering_pairs,
            })

            print(f"    {cl_lbl}: {final_verdict}")

        if class_rows:
            class_df = pd.DataFrame(class_rows)
            _save_table(
                class_df, "veering_vs_mixing_classification",
                tex_columns=["cluster", "dim", "prevalence",
                             "gap_type", "any_swap_rate",
                             "ref_modes_mode", "final_verdict"],
                tex_col_labels={
                    "cluster": "Cluster", "dim": "Dim",
                    "prevalence": "Prev.",
                    "gap_type": "Gap type",
                    "any_swap_rate": "Swap rate",
                    "ref_modes_mode": "Ref. modes",
                    "final_verdict": "Verdict",
                },
            )
            print("\n  Classification summary:")
            print(class_df.to_string(index=False))
        if pair_rows:
            _save_table(
                pd.DataFrame(pair_rows), "swap_pair_diagnostics",
                tex_columns=["cluster", "rank_pair", "ref_pair_mode",
                             "swap_rate", "median_gap_in",
                             "near_deg_rate", "verdict"],
                tex_col_labels={
                    "cluster": "Cluster",
                    "rank_pair": "Rank pair",
                    "ref_pair_mode": "Ref. pair",
                    "swap_rate": "Swap rate",
                    "median_gap_in": "Med. gap$_{\\mathrm{in}}$",
                    "near_deg_rate": "Near-deg. frac.",
                    "verdict": "Verdict",
                },
            )

        # ── 6b. Permutation null test for veering ────────────
        print("\n  [6b] Permutation null test (1000 shuffles) ...")
        null_rows: list[dict] = []
        for cs, ce, cnt in top_clusters:
            if ce - cs < 2:
                continue
            cl_lbl = cluster_label(cs, ce)
            cl_key = f"modes_{cs + 1}_{ce}"
            member_mask = np.array(
                [(cs, ce) in clist for clist in clusters_all], dtype=bool,
            )
            # Use per-cluster morphology proxy
            cl_proxy_name_nt = CLUSTER_MORPHOLOGY_PROXY.get(cl_key, MORPHOLOGY_PROXY)
            cl_proxy_nt = dims.get(cl_proxy_name_nt, morph_proxy)
            for rank_i in range(cs, ce - 1):
                nt = permutation_null_test_veering(
                    perm_rank_to_ref, cl_proxy_nt, rank_i,
                    window=window, subject_mask=member_mask,
                    n_perm=1000, seed=42,
                )
                mi, mj = nt["rank_pair"]
                sig_str = "YES" if nt["significant"] else "NO"
                null_rows.append({
                    "cluster": cl_lbl,
                    "rank_pair": f"{mi + 1}\u21c4{mj + 1}",
                    "obs_peak": nt["observed_peak_rate"],
                    "null_peak_mean": nt["null_mean_peak_rate"],
                    "null_peak_std": nt["null_std_peak_rate"],
                    "p_peak": nt["p_peak_rate"],
                    "obs_active": nt["observed_active_frac"],
                    "null_active_mean": nt["null_mean_active_frac"],
                    "null_active_std": nt["null_std_active_frac"],
                    "p_active": nt["p_active_frac"],
                    "obs_cv": nt["observed_cv_swap"],
                    "null_cv_mean": nt["null_mean_cv_swap"],
                    "null_cv_std": nt["null_std_cv_swap"],
                    "p_cv": nt["p_cv_swap"],
                    "morphology_linked": sig_str,
                })
                print(
                    f"    {cl_lbl} rank {mi+1}\u21c4{mj+1}: "
                    f"peak {nt['observed_peak_rate']:.2f} "
                    f"(null {nt['null_mean_peak_rate']:.2f}\u00b1{nt['null_std_peak_rate']:.2f}, "
                    f"p={nt['p_peak_rate']:.3f})  "
                    f"CV {nt['observed_cv_swap']:.2f} "
                    f"(null {nt['null_mean_cv_swap']:.2f}\u00b1{nt['null_std_cv_swap']:.2f}, "
                    f"p={nt['p_cv_swap']:.3f})  "
                    f"\u2192 {sig_str}"
                )
        if null_rows:
            _save_table(
                pd.DataFrame(null_rows), "veering_null_test",
                tex_columns=[
                    "cluster", "rank_pair",
                    "obs_peak", "p_peak",
                    "obs_active", "p_active",
                    "obs_cv", "p_cv",
                    "morphology_linked",
                ],
                tex_col_labels={
                    "cluster": "Cluster",
                    "rank_pair": "Rank pair",
                    "obs_peak": "Peak$_{\\mathrm{obs}}$",
                    "p_peak": "$p$ (peak)",
                    "obs_active": "Active$_{\\mathrm{obs}}$",
                    "p_active": "$p$ (active)",
                    "obs_cv": "CV$_{\\mathrm{obs}}$",
                    "p_cv": "$p$ (CV)",
                    "morphology_linked": "Significant",
                },
            )

        # ── 6c. Cohort bootstrap robustness ──────────────────
        print("\n  [6c] Cohort bootstrap robustness (500 × 80% subsample) ...")
        boot = cohort_bootstrap_robustness(
            ev_full, perm_rank_to_ref, eps_in,
            target_clusters=[(cs, ce) for cs, ce, _ in top_clusters],
            n_boot=500, subsample_frac=0.8, seed=42,
        )
        boot_rows: list[dict] = []
        for lbl, s in boot["cluster_prevalence"].items():
            boot_rows.append({
                "metric": "prevalence",
                "target": lbl,
                "observed": f"{s['observed']:.3f}",
                "boot_mean": f"{s['mean']:.3f}",
                "boot_std": f"{s['std']:.3f}",
                "CI_95": f"[{s['ci95_lo']:.3f}, {s['ci95_hi']:.3f}]",
            })
            print(
                f"    {lbl}: prev = {s['observed']:.3f}  "
                f"(boot {s['mean']:.3f} ± {s['std']:.3f}, "
                f"95% CI [{s['ci95_lo']:.3f}, {s['ci95_hi']:.3f}])"
            )
        for mode_1b, s in boot["swap_rates"].items():
            if s["observed"] > 0.01:
                boot_rows.append({
                    "metric": "swap_rate",
                    "target": f"Mode {mode_1b}",
                    "observed": f"{s['observed']:.3f}",
                    "boot_mean": f"{s['mean']:.3f}",
                    "boot_std": f"{s['std']:.3f}",
                    "CI_95": f"[{s['ci95_lo']:.3f}, {s['ci95_hi']:.3f}]",
                })
        for pair_lbl, s in boot["median_gaps"].items():
            boot_rows.append({
                "metric": "median_gap",
                "target": f"Pair {pair_lbl}",
                "observed": f"{s['observed']:.3f}",
                "boot_mean": f"{s['mean']:.3f}",
                "boot_std": f"{s['std']:.3f}",
                "CI_95": f"[{s['ci95_lo']:.3f}, {s['ci95_hi']:.3f}]",
            })
        if boot_rows:
            _save_table(
                pd.DataFrame(boot_rows), "cohort_bootstrap_robustness",
                tex_col_labels={
                    "metric": "Metric",
                    "target": "Target",
                    "observed": "Observed",
                    "boot_mean": "Boot. mean",
                    "boot_std": "Boot. std",
                    "CI_95": "95\\% CI",
                },
            )

        # ── 6d. Cohort ordering invariance check ─────────────
        print("\n  [6d] Ordering invariance check (100 permutations) ...")
        ord_test = cohort_ordering_permutation_test(
            ev_full, perm_rank_to_ref, eps_in,
            target_clusters=[(cs, ce) for cs, ce, _ in top_clusters],
            n_perm=100, seed=42,
        )
        status = "PASS" if ord_test["passed"] else "FAIL"
        print(
            f"    Max prevalence deviation: {ord_test['max_prevalence_deviation']:.2e}  "
            f"Max swap deviation: {ord_test['max_swap_deviation']:.2e}  → {status}"
        )

    # ── 7. Canonical inlet scores ─────────────────────────────
    print("\n[7/8] Canonical inlet scores (Sec. D) ...")
    fe_coords = None
    tree = None
    tpl_shapes = None
    canon_data: list[dict] = []
    if evecs_arr is not None and landmarks:
        fe_coords = load_fe_mesh_coords()
        n_nodes_ev = int(evecs_arr.shape[2])

        if fe_coords is not None and fe_coords.shape[0] == n_nodes_ev:
            print(f"  FE mesh: {fe_coords.shape[0]} nodes")
            from scipy.spatial import cKDTree

            tree = cKDTree(np.asarray(fe_coords, dtype=float))
            if "AnteriorPosteriorInletDiameter" in dims:
                ap_dim = dims["AnteriorPosteriorInletDiameter"]
                print(f"  AP inlet diameter: mean={np.nanmean(ap_dim):.1f} mm")
            if "PelvisInletTransverse" in dims:
                ml_dim = dims["PelvisInletTransverse"]
                print(f"  ML inlet transverse: mean={np.nanmean(ml_dim):.1f} mm")

            # Patient-specific b vectors via deformed landmark directions
            tpl_shapes = load_template_and_shapes()
            b_per_patient: dict[str, np.ndarray | None] = {"AP": None, "ML": None}
            if tpl_shapes is not None:
                tpl_pts, ssm_shapes = tpl_shapes
                ssm_shapes_sub = ssm_shapes[:n_subj]
                for key, pts in landmarks.items():
                    p1, p2 = pts
                    if np.linalg.norm(p1 - p2) < 1e-10:
                        continue
                    b_per_patient[key] = build_patient_measurement_vectors(
                        tree, n_nodes_ev, tpl_pts, ssm_shapes_sub,
                        p1, p2, k=32, power=2.0,
                        smoothing=50.0, neighbors=10,
                    )
                    print(f"    {key}: patient-specific IDW (k=32, {n_subj} subjects)")
            else:
                print("  WARNING: SSM shapes unavailable, using template-based b")
                for key, pts in landmarks.items():
                    p1, p2 = pts
                    direction = p1 - p2
                    norm = np.linalg.norm(direction)
                    if norm < 1e-10:
                        continue
                    e = direction / norm
                    b_single = build_idw_measurement_vector(
                        tree, n_nodes_ev, e, p1, p2, k=32, power=2.0,
                    )
                    # Broadcast to all subjects
                    b_per_patient[key] = np.tile(b_single, (n_subj, 1))

            if b_per_patient["AP"] is not None and b_per_patient["ML"] is not None:
                b_ap_all = b_per_patient["AP"]
                b_ml_all = b_per_patient["ML"]
                canon_clusters = [
                    (cs, ce) for cs, ce, _ in top_clusters if ce - cs >= 2
                ]
                if canon_clusters:
                    for cs, ce in canon_clusters:
                        cl_lbl = cluster_label(cs, ce)
                        r = ce - cs
                        s_ap_abs = np.full(n_subj, np.nan)
                        s_ml_abs = np.full(n_subj, np.nan)
                        s_ap_per_1mm = np.full(n_subj, np.nan)
                        s_ml_per_1mm = np.full(n_subj, np.nan)
                        member_mask = np.array(
                            [(cs, ce) in clist for clist in clusters_all], dtype=bool,
                        )
                        for s in np.flatnonzero(member_mask):
                            s_block = np.asarray(
                                evecs_arr[int(s), cs:ce, :, :], dtype=float,
                            )
                            if not np.isfinite(s_block).all():
                                continue
                            b_ap_s = b_ap_all[s]
                            b_ml_s = b_ml_all[s]
                            Q_s = orthonormalize_l2(s_block.reshape(r, -1).T, M=M0_sp)
                            ap_per_1mm, ml_per_1mm, psi1, psi2, _ = canonical_pair_couplings(
                                Q_s, b_ap_s, b_ml_s, M=M0_sp,
                            )
                            ap_abs = float(abs(b_ap_s @ psi1))
                            ml_abs = float(abs(b_ml_s @ psi2)) if np.linalg.norm(psi2) > 1e-12 else 0.0
                            s_ap_abs[s] = ap_abs
                            s_ml_abs[s] = ml_abs

                            s_ap_per_1mm[s] = ap_per_1mm
                            s_ml_per_1mm[s] = ml_per_1mm

                        canon_data.append({
                            "label": cl_lbl,
                            "member_mask": member_mask,
                            "ap": s_ap_per_1mm,
                            "ml": s_ml_per_1mm,
                        })
                        print(
                            f"    {cl_lbl}: AP per-1mm med="
                            f"{np.nanmedian(s_ap_per_1mm[member_mask]):.3f}; "
                            f"ML per-1mm med="
                            f"{np.nanmedian(s_ml_per_1mm[member_mask]):.3f}"
                        )

            else:
                print("  AP or ML landmarks missing; skipping")
        else:
            if fe_coords is not None:
                print(
                    f"  FE mesh ({fe_coords.shape[0]}) != "
                    f"eigenvector nodes ({n_nodes_ev})"
                )
            print("  Skipping canonical scores")
    else:
        print("  Eigenvectors or landmarks not available; skipping")

    # ── 7b. Canonical outlet scores (biischiadic / bituberous) ──
    print("\n[7b/8] Canonical outlet scores (BIS/BIT) ...")
    canon_outlet_data: list[dict] = []
    if evecs_arr is not None and outlet_landmarks:
        fe_coords_out = fe_coords if fe_coords is not None else load_fe_mesh_coords()
        n_nodes_ev_out = int(evecs_arr.shape[2])

        if fe_coords_out is not None and fe_coords_out.shape[0] == n_nodes_ev_out:
            from scipy.spatial import cKDTree as _cKDTree

            tree_out = tree if tree is not None else _cKDTree(np.asarray(fe_coords_out, dtype=float))

            tpl_shapes_out = tpl_shapes if tpl_shapes is not None else load_template_and_shapes()
            outlet_pairs = ("BIS", "BIT")
            b_outlet_patient: dict[str, np.ndarray | None] = {"BIS": None, "BIT": None}
            if tpl_shapes_out is not None:
                tpl_pts_o, ssm_shapes_o = tpl_shapes_out
                ssm_shapes_o_sub = ssm_shapes_o[:n_subj]
                for key in outlet_pairs:
                    if key not in outlet_landmarks:
                        continue
                    pts = outlet_landmarks[key]
                    p1, p2 = pts
                    if np.linalg.norm(p1 - p2) < 1e-10:
                        continue
                    b_outlet_patient[key] = build_patient_measurement_vectors(
                        tree_out, n_nodes_ev_out, tpl_pts_o, ssm_shapes_o_sub,
                        p1, p2, k=32, power=2.0,
                        smoothing=50.0, neighbors=10,
                    )
                    print(f"    {key}: patient-specific IDW (k=32, {n_subj} subjects)")
            else:
                print("  WARNING: SSM shapes unavailable, using template-based b")
                for key in outlet_pairs:
                    if key not in outlet_landmarks:
                        continue
                    pts = outlet_landmarks[key]
                    p1, p2 = pts
                    direction = p1 - p2
                    norm = np.linalg.norm(direction)
                    if norm < 1e-10:
                        continue
                    e = direction / norm
                    b_single = build_idw_measurement_vector(
                        tree_out, n_nodes_ev_out, e, p1, p2, k=32, power=2.0,
                    )
                    b_outlet_patient[key] = np.tile(b_single, (n_subj, 1))

            if b_outlet_patient["BIS"] is not None and b_outlet_patient["BIT"] is not None:
                b_bis_all = b_outlet_patient["BIS"]
                b_bit_all = b_outlet_patient["BIT"]
                outlet_canon_clusters = [
                    (cs, ce) for cs, ce, _ in top_clusters if ce - cs >= 2
                ]
                if outlet_canon_clusters:
                    for cs, ce in outlet_canon_clusters:
                        cl_lbl = cluster_label(cs, ce)
                        r = ce - cs
                        s_bis_per_1mm = np.full(n_subj, np.nan)
                        s_bit_per_1mm = np.full(n_subj, np.nan)
                        member_mask = np.array(
                            [(cs, ce) in clist for clist in clusters_all], dtype=bool,
                        )
                        for s in np.flatnonzero(member_mask):
                            s_block = np.asarray(
                                evecs_arr[int(s), cs:ce, :, :], dtype=float,
                            )
                            if not np.isfinite(s_block).all():
                                continue
                            b_bis_s = b_bis_all[s]
                            b_bit_s = b_bit_all[s]
                            Q_s = orthonormalize_l2(s_block.reshape(r, -1).T, M=M0_sp)
                            bis_per_1mm, bit_per_1mm, _, psi2, _ = canonical_pair_couplings(
                                Q_s, b_bis_s, b_bit_s, M=M0_sp,
                            )
                            s_bis_per_1mm[s] = bis_per_1mm
                            s_bit_per_1mm[s] = bit_per_1mm

                        canon_outlet_data.append({
                            "label": cl_lbl,
                            "member_mask": member_mask,
                            "bis": s_bis_per_1mm,
                            "bit": s_bit_per_1mm,
                        })
                        print(
                            f"    {cl_lbl}: BIS per-1mm med="
                            f"{np.nanmedian(s_bis_per_1mm[member_mask]):.3f}; "
                            f"BIT per-1mm med="
                            f"{np.nanmedian(s_bit_per_1mm[member_mask]):.3f}"
                        )

            else:
                print("  BIS or BIT landmarks missing; skipping")
        else:
            print("  Skipping outlet canonical scores (FE mesh mismatch)")
    else:
        print("  Eigenvectors or outlet landmarks not available; skipping")

    # ── Dashboard: subspace analysis ──────────────────────────
    if grassmann_results and top_clusters:
        plot_subspace_analysis_dashboard(
            top_clusters, grassmann_results, clusters_all,
            canon_data, canon_outlet_data,
            mask_m, mask_f, cluster_label,
        )

    # ── 8. Sex-stratified summary ─────────────────────────────
    print("\n[8/8] Sex-stratified summary ...")
    summary_rows: list[dict] = []

    # (a) gap_in for each gap index inside any detected cluster
    gap_done: set[int] = set()
    for cs, ce, _ in detected_clusters:
        for gi in range(cs, ce - 1):
            if gi in gap_done:
                continue
            gap_done.add(gi)
            g_m = gaps_full[mask_m, gi]
            g_f = gaps_full[mask_f, gi]
            g_m = g_m[np.isfinite(g_m)]
            g_f = g_f[np.isfinite(g_f)]
            if len(g_m) < 2 or len(g_f) < 2:
                continue
            u_stat, p_val = stats.mannwhitneyu(g_m, g_f, alternative="two-sided")
            n_m, n_f = len(g_m), len(g_f)
            r_eff = 1 - 2 * u_stat / (n_m * n_f)
            summary_rows.append({
                "metric": f"gap_in ({gi + 1}\u2013{gi + 2})",
                "male_median": float(np.median(g_m)),
                "female_median": float(np.median(g_f)),
                "U_statistic": float(u_stat),
                "p_value": float(p_val),
                "rank_biserial_r": float(r_eff),
            })

    # (b) Swap count per subject (Mann-Whitney)
    n_swaps_m = np.array([
        np.sum((perm_full[i] != identity_perm) & (perm_full[i] >= 0))
        for i in range(n_subj) if mask_m[i]
    ], dtype=float)
    n_swaps_f = np.array([
        np.sum((perm_full[i] != identity_perm) & (perm_full[i] >= 0))
        for i in range(n_subj) if mask_f[i]
    ], dtype=float)
    u_stat_perm, p_perm = stats.mannwhitneyu(
        n_swaps_m, n_swaps_f, alternative="two-sided",
    )
    r_perm = 1 - 2 * u_stat_perm / (len(n_swaps_m) * len(n_swaps_f))
    summary_rows.append({
        "metric": "swap_count_per_subject",
        "male_median": float(np.median(n_swaps_m)),
        "female_median": float(np.median(n_swaps_f)),
        "U_statistic": float(u_stat_perm),
        "p_value": float(p_perm),
        "rank_biserial_r": float(r_perm),
    })

    # (b2) Permutation rate (binary, Fisher exact)
    n_nonid_m = sum(
        not np.array_equal(perm_full[i], identity_perm)
        for i in range(n_subj) if mask_m[i]
    )
    n_nonid_f = sum(
        not np.array_equal(perm_full[i], identity_perm)
        for i in range(n_subj) if mask_f[i]
    )
    rate_m = n_nonid_m / mask_m.sum()
    rate_f = n_nonid_f / mask_f.sum()
    table_fisher = [
        [n_nonid_m, int(mask_m.sum()) - n_nonid_m],
        [n_nonid_f, int(mask_f.sum()) - n_nonid_f],
    ]
    _, p_fisher = stats.fisher_exact(table_fisher)
    odds_m = n_nonid_m / max(1, int(mask_m.sum()) - n_nonid_m)
    odds_f = n_nonid_f / max(1, int(mask_f.sum()) - n_nonid_f)
    odds_ratio = odds_m / odds_f if odds_f > 0 else float("inf")
    summary_rows.append({
        "metric": "permutation_rate",
        "male_median": rate_m,
        "female_median": rate_f,
        "U_statistic": odds_ratio,
        "p_value": float(p_fisher),
        "rank_biserial_r": (
            float(np.log(odds_ratio))
            if np.isfinite(odds_ratio) else float("inf")
        ),
    })

    # (c) Grassmann distances
    for cl_key, dG in grassmann_results.items():
        dg_m = dG[mask_m]
        dg_f = dG[mask_f]
        dg_m = dg_m[np.isfinite(dg_m)]
        dg_f = dg_f[np.isfinite(dg_f)]
        if len(dg_m) < 2 or len(dg_f) < 2:
            continue
        u_stat, p_val = stats.mannwhitneyu(dg_m, dg_f, alternative="two-sided")
        n_m, n_f = len(dg_m), len(dg_f)
        r_eff = 1 - 2 * u_stat / (n_m * n_f)
        summary_rows.append({
            "metric": f"d_G ({cl_key})",
            "male_median": float(np.median(dg_m)),
            "female_median": float(np.median(dg_f)),
            "U_statistic": float(u_stat),
            "p_value": float(p_val),
            "rank_biserial_r": float(r_eff),
        })

    summary_df = pd.DataFrame(summary_rows)
    _save_table(
        summary_df, "sex_stratified_summary",
        tex_col_labels={
            "metric": "Metric",
            "male_median": "Male med.",
            "female_median": "Female med.",
            "U_statistic": "$U$",
            "p_value": "$p$",
            "rank_biserial_r": "$r_{\\mathrm{rb}}$",
        },
    )
    print(summary_df.to_string(index=False))

    plot_sex_stratified_summary(
        summary_rows, gaps_full, grassmann_results,
        n_swaps_m, n_swaps_f, rate_m, rate_f,
        mask_m, mask_f,
    )

    # ── Report ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"Figures: {FIG_DIR}")
    print(f"Tables:  {TAB_DIR}")
    generated = []
    if FIG_DIR.exists():
        generated.extend(sorted(FIG_DIR.iterdir()))
    if TAB_DIR.exists():
        generated.extend(sorted(TAB_DIR.iterdir()))
    for f in generated:
        from analysis.spectral_config import OUT_DIR
        print(f"  {f.relative_to(OUT_DIR)}")


if __name__ == "__main__":
    main()
