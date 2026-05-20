#!/usr/bin/env python3
"""Publication-quality tail-risk / variability pipeline.

Reads pre-computed DG0 strain & stress tensors from Zarr and computes
volume-weighted tail/risk metrics per tissue group using a **two-pass
streaming** design (O(num_cells) peak memory, never caches full cohort).

**Pass 1** — per-subject tail metrics + per-subject P95 for threshold τ.
**Pass 2** — exceedance volume-fraction with τ (streaming, no full-field cache).

Spectral operating-point indicators (dG, gap_in, swap) are computed via the
validated ``analysis.spectral_metrics`` module with **permutation-aware block
tracking** — NOT reimplemented here.

Statistical comparisons aggregate per subject (mean or max over loadcases)
before Mann–Whitney U to avoid repeated-measures inflation.  Dispersion
tests use Brown–Forsythe (Levene with median) and are annotated as
exploratory.

Outputs
-------
metrics_tail.csv / .parquet  — per (subject, loadcase, domain) raw metrics
metrics_tail_aggregated.csv  — per (subject, standing|labour, domain)
sex_comparison_stats.csv     — Mann–Whitney U + Brown–Forsythe per group
thresholds.json              — τ per (loadcase, domain)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from analysis.tail_risk_metrics import (
    compute_tail_metrics,
    element_scalars,
    exceedance_volfrac,
    sed_only,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEB_FILE = str(PROJECT_ROOT / "anatomy_data" / "full_model.feb")

DEFAULT_DATA_DIR = PROJECT_ROOT / "results" / "ref_S1P_fixed_new2" / "data"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "analysis_outputs" / "tail_risk"

LOAD_CASES: tuple[str, ...] = (
    "SP2leg", "SP1leg", "LAB_phase1", "LAB_phase2", "LAB_phase3",
)
LOAD_CASE_LABELS: dict[str, str] = {
    "SP2leg": "SP 2-leg",
    "SP1leg": "SP 1-leg",
    "LAB_phase1": "LAB\u2081",
    "LAB_phase2": "LAB\u2082",
    "LAB_phase3": "LAB\u2083",
}
LC_STANDING = frozenset({"SP2leg", "SP1leg"})
LC_LABOUR = frozenset({"LAB_phase1", "LAB_phase2", "LAB_phase3"})

logger = logging.getLogger(__name__)


# ===========================================================================
# CLI
# ===========================================================================

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="Tail-risk / variability analysis pipeline",
    )
    p.add_argument(
        "--data-dir", type=Path, default=DEFAULT_DATA_DIR,
        help="Dir containing strain_*.zarr and stress_*.zarr",
    )
    p.add_argument(
        "--spectral-dir", type=Path, default=None,
        help="Dir for eigenvalues/eigenvectors/permutations (default: same as --data-dir)",
    )
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument(
        "--domains", nargs="+", default=["all", "bone", "cartilage"],
        help="Tissue domains to analyse",
    )
    p.add_argument(
        "--loadcases", nargs="+", default=list(LOAD_CASES),
    )
    p.add_argument(
        "--nu", type=float, default=0.3,
        help="Poisson ratio (annotation only — FE data is pre-computed)",
    )
    p.add_argument(
        "--standing-agg", choices=["mean", "max"], default="mean",
        help="Aggregation across standing loadcases per subject",
    )
    p.add_argument(
        "--labour-agg", choices=["mean", "max"], default="mean",
        help="Aggregation across labour loadcases per subject",
    )
    p.add_argument(
        "--threshold-quantile", type=float, default=0.95,
        help="Quantile of per-subject P95 distribution used as exceedance τ",
    )
    args = p.parse_args(argv)
    if not (0.0 < float(args.threshold_quantile) < 1.0):
        raise ValueError("--threshold-quantile must be in (0, 1)")
    if args.spectral_dir is None:
        args.spectral_dir = args.data_dir
    return args


# ===========================================================================
# DOLFINx mesh setup
# ===========================================================================

def _setup_dolfinx_mesh(
    domains: list[str],
) -> tuple[int, np.ndarray, dict[str, np.ndarray]]:
    """Build DOLFINx mesh, element volumes, and material masks.

    Returns
    -------
    num_cells : int
    volumes : (num_cells,)
    group_masks : {domain_name: bool_mask}
    """
    from dolfinx import fem
    import ufl
    from simulation.febio_parser import FEBio2Dolfinx

    logger.info("Building DOLFINx mesh via FEBio2Dolfinx …")
    f2x = FEBio2Dolfinx(FEB_FILE)
    domain_mesh = f2x.mesh_dolfinx
    material_labels = np.asarray(f2x.material_labels, dtype=str)

    tdim = domain_mesh.topology.dim
    num_cells = domain_mesh.topology.index_map(tdim).size_local

    # Element volumes via DG0 assembly
    W = fem.functionspace(domain_mesh, ("DG", 0))
    v_test = ufl.TestFunction(W)
    dx = ufl.Measure("dx", domain=domain_mesh)
    vol_vec = fem.assemble_vector(fem.form(v_test * dx))
    volumes = np.copy(vol_vec.array)

    # Material masks (same logic as MaterialMapper._material_masks)
    labels_lower = np.char.lower(material_labels)
    _has = lambda needle: np.char.find(labels_lower, needle) >= 0
    bone_mask = _has("bone") & ~_has("cartilage") & ~_has("symph")
    cart_mask = _has("cartilage") | _has("symph")

    available_masks: dict[str, np.ndarray] = {
        "bone": bone_mask,
        "cartilage": cart_mask,
        "all": np.ones(num_cells, dtype=bool),
    }
    group_masks = {d: available_masks[d] for d in domains if d in available_masks}

    logger.info(
        "  %d cells, vol=%.1f mm³, bone=%d, cart=%d",
        num_cells, volumes.sum(), bone_mask.sum(), cart_mask.sum(),
    )
    return num_cells, volumes, group_masks


# ===========================================================================
# Spectral metrics — delegates to validated spectral_metrics module
# ===========================================================================

def _compute_spectral_metrics(spectral_dir: Path) -> pd.DataFrame:
    """Permutation-aware Grassmann distance + gap_in + swap detection.

    Uses ``analysis.spectral_metrics`` for all subspace computations.
    The key difference from the old code: eigenvector slicing uses the
    MAC-based permutation array so that reference modes {9, 10} are
    correctly tracked even when their rank ordering swaps.
    """
    from analysis.spectral_data import (
        load_eigenvalues,
        load_eigenvectors_obj,
        load_mass_matrix,
        load_permutations,
    )
    from analysis.spectral_metrics import (
        compute_gap_in,
        compute_principal_angles,
        grassmann_distance,
        orthonormalize_l2,
        ref_pair_swap_indicator,
        swap_count_per_subject,
    )

    eigvals = load_eigenvalues(spectral_dir)   # (N, M)
    perms = load_permutations(spectral_dir)     # perm_ref_to_rank[s, i_ref] = j_rank
    eigvecs_z = load_eigenvectors_obj(spectral_dir)  # lazy Zarr (N, M, n_nodes, 3)
    N, M = eigvals.shape
    if M < 10:
        raise ValueError(f"Need at least 10 modes for 9–10 metrics; got M={M}")
    if perms.shape != (N, M):
        raise ValueError(
            f"Permutation shape mismatch: eigenvalues {(N, M)} vs perms {perms.shape}"
        )
    if getattr(eigvecs_z, "shape", None) is not None:
        if eigvecs_z.shape[0] != N or eigvecs_z.shape[1] != M:
            raise ValueError(
                "Eigenvector array shape mismatch: expected "
                f"({N}, {M}, ..., ...), got {eigvecs_z.shape}"
            )

    # gap_in → (N, M-1); index 8 = gap between modes 9 and 10 (1-based)
    gaps = compute_gap_in(eigvals)
    gap_in_9_10 = gaps[:, 8]

    # Permutation-aware: rank positions of ref modes 9, 10 (0-based indices 8, 9)
    r8_all = perms[:, 8].astype(int, copy=False)
    r9_all = perms[:, 9].astype(int, copy=False)
    swap_9_10 = ref_pair_swap_indicator(perms, 8, 9)
    valid_pair = np.isfinite(swap_9_10)

    # ── Mass-matrix inner product for subspace angles (falls back to L2 if unavailable) ──
    M0_sp = load_mass_matrix()
    if M0_sp is None:
        logger.info("  Spectral: using L2 inner product (M0 unavailable)")
    else:
        logger.info("  Spectral: using reference mass matrix inner product (M0)")

    # ── Reference subject (closest to median eigenvalue of ref mode 9) ──
    ref_mode9_eig = np.full(N, np.nan)
    idx_valid = np.flatnonzero(valid_pair)
    ref_mode9_eig[idx_valid] = eigvals[idx_valid, r8_all[idx_valid]]
    med = float(np.nanmedian(ref_mode9_eig))
    if not np.isfinite(med):
        raise RuntimeError("Cannot select reference subject: ref-mode-9 eigenvalues are all non-finite")
    deltas = np.abs(ref_mode9_eig - med)
    ref_idx = int(np.nanargmin(deltas))

    ref_r8 = int(r8_all[ref_idx])
    ref_r9 = int(r9_all[ref_idx])
    ref_block = np.stack(
        [
            np.asarray(eigvecs_z[ref_idx, ref_r8], dtype=float),
            np.asarray(eigvecs_z[ref_idx, ref_r9], dtype=float),
        ],
        axis=0,
    )  # (2, n_nodes, 3)
    if not np.isfinite(ref_block).all():
        raise RuntimeError(f"Non-finite eigenvectors for reference subject {ref_idx}")
    ref_sub = orthonormalize_l2(ref_block.reshape(2, -1).T, M=M0_sp)

    # ── Per-subject dG for ref modes 9–10 ──
    dG_9_10 = np.full(N, np.nan)
    for s in idx_valid.tolist():
        r8 = int(r8_all[s])
        r9 = int(r9_all[s])
        block = np.stack(
            [
                np.asarray(eigvecs_z[s, r8], dtype=float),
                np.asarray(eigvecs_z[s, r9], dtype=float),
            ],
            axis=0,
        )
        if not np.isfinite(block).all():
            continue
        s_sub = orthonormalize_l2(block.reshape(2, -1).T, M=M0_sp)
        angles = compute_principal_angles(ref_sub, s_sub, M=M0_sp)
        dG_9_10[s] = grassmann_distance(angles)

    swap_count = swap_count_per_subject(perms, M)

    logger.info(
        "  Spectral: gap_in median=%.4f, dG median=%.4f, swap_rate=%.1f%%",
        np.nanmedian(gap_in_9_10),
        np.nanmedian(dG_9_10),
        100 * np.nanmean(swap_9_10),
    )
    return pd.DataFrame({
        "subject_idx": np.arange(N),
        "gap_in_9_10": gap_in_9_10,
        "dG_9_10": dG_9_10,
        "swap_9_10": swap_9_10,
        "swap_count": swap_count,
    })


# ===========================================================================
# Zarr I/O
# ===========================================================================

def _open_zarr_lazy(path: Path, key: str = "data"):
    """Open a Zarr array lazily (no full load into memory)."""
    import zarr

    store = zarr.open_group(str(path), mode="r")
    if key in store:
        return store[key]
    return zarr.open_array(str(path), mode="r")


# ===========================================================================
# Aggregation + statistical testing
# ===========================================================================

METRIC_COLS: list[str] = [
    "P99_SED", "P95_SED", "top1pct_mean_SED", "exceedance_volfrac_SED",
    "entropy_SED", "gini_SED", "P99_eps1", "P99_abs_eps3",
    "median_SED", "IQR_SED", "MAD_SED",
]


def _aggregate_per_subject(
    df: pd.DataFrame,
    metric_cols: list[str],
    standing_agg: str,
    labour_agg: str,
) -> pd.DataFrame:
    """Aggregate per subject across loadcases within standing / labour.

    Assumption (documented):
    * **standing_agg** / **labour_agg** = "mean" (default) averages across
      SP-loadcases / LAB-phases per subject to avoid repeated-measures
      inflation in downstream group comparisons.  "max" keeps the worst-case
      loadcase value.
    """
    rows: list[dict] = []
    for (subj, group), sub in df.groupby(["subject_idx", "group"]):
        for category, lc_set, agg_fn in [
            ("standing", LC_STANDING, standing_agg),
            ("labour", LC_LABOUR, labour_agg),
        ]:
            lc_sub = sub[sub["loadcase"].isin(lc_set)]
            if lc_sub.empty:
                continue
            row: dict = {"subject_idx": subj, "group": group, "category": category}
            for col in metric_cols:
                vals = lc_sub[col].dropna().values
                if len(vals) == 0:
                    row[col] = np.nan
                elif agg_fn == "mean":
                    row[col] = float(np.mean(vals))
                else:
                    row[col] = float(np.max(vals))
            rows.append(row)
    return pd.DataFrame(rows)


def _sex_comparison(
    df_agg: pd.DataFrame,
    metric_cols: list[str],
) -> pd.DataFrame:
    """Mann–Whitney U (location) + Brown–Forsythe (dispersion) on aggregated data.

    Brown–Forsythe is annotated as *exploratory* (no multiplicity correction).
    """
    from scipy import stats as sp_stats

    rows: list[dict] = []
    for (group, cat), sub in df_agg.groupby(["group", "category"]):
        m_sub = sub[sub["sex"] == "M"]
        f_sub = sub[sub["sex"] == "F"]
        for col in metric_cols:
            mv = m_sub[col].dropna().values
            fv = f_sub[col].dropna().values
            row: dict = {
                "group": group, "category": cat, "metric": col,
                "n_M": len(mv), "n_F": len(fv),
                "median_M": np.nan, "median_F": np.nan,
                "IQR_M": np.nan, "IQR_F": np.nan,
                "MW_U": np.nan, "MW_p": np.nan,
                "BF_stat": np.nan, "BF_p": np.nan,
            }
            if len(mv) >= 3:
                row["median_M"] = float(np.median(mv))
                row["IQR_M"] = float(np.percentile(mv, 75) - np.percentile(mv, 25))
            if len(fv) >= 3:
                row["median_F"] = float(np.median(fv))
                row["IQR_F"] = float(np.percentile(fv, 75) - np.percentile(fv, 25))
            if len(mv) >= 5 and len(fv) >= 5:
                U, p = sp_stats.mannwhitneyu(mv, fv, alternative="two-sided")
                row["MW_U"] = float(U)
                row["MW_p"] = float(p)
                # Brown–Forsythe (exploratory dispersion test)
                bf_stat, bf_p = sp_stats.levene(mv, fv, center="median")
                row["BF_stat"] = float(bf_stat)
                row["BF_p"] = float(bf_p)
            rows.append(row)
    return pd.DataFrame(rows)


# ===========================================================================
# Self-check
# ===========================================================================

def _self_check(
    num_cells: int,
    volumes: np.ndarray,
    group_masks: dict[str, np.ndarray],
    spec_df: pd.DataFrame,
) -> None:
    """Validate critical invariants before proceeding."""
    # (1) domain="all" volume equals total mesh volume
    if "all" in group_masks:
        all_vol = float(volumes[group_masks["all"]].sum())
        total_vol = float(volumes.sum())
        rel_err = abs(all_vol - total_vol) / max(total_vol, 1e-30)
        logger.info(
            "Self-check: domain='all' vol=%.2f, mesh vol=%.2f, rel_err=%.2e",
            all_vol, total_vol, rel_err,
        )
        assert rel_err < 1e-10, f"Volume mismatch: {rel_err}"

    # (2) dG non-negative and finite where valid
    dg = spec_df["dG_9_10"].values
    valid_dg = dg[np.isfinite(dg)]
    if len(valid_dg) > 0:
        assert np.all(valid_dg >= -1e-12), "Negative Grassmann distances"
        logger.info(
            "Self-check: dG_9_10 range [%.4f, %.4f], %d/%d finite",
            valid_dg.min(), valid_dg.max(), len(valid_dg), len(dg),
        )

    # (3) bone + cartilage ≤ all
    if "bone" in group_masks and "cartilage" in group_masks:
        overlap = int((group_masks["bone"] & group_masks["cartilage"]).sum())
        assert overlap == 0, f"Bone/cartilage masks overlap ({overlap} cells)"
        bc = int(group_masks["bone"].sum() + group_masks["cartilage"].sum())
        a = int(group_masks.get("all", np.ones(num_cells, dtype=bool)).sum())
        assert bc <= a, f"Bone+cart ({bc}) > all ({a})"
        logger.info("Self-check: bone+cart=%d, all=%d — OK", bc, a)

    logger.info("Self-check passed.")


# ===========================================================================
# Main pipeline
# ===========================================================================

def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== Tail-risk analysis pipeline ===")
    logger.info("Data dir   : %s", args.data_dir)
    logger.info("Spectral   : %s", args.spectral_dir)
    logger.info("Domains    : %s", args.domains)
    logger.info("Loadcases  : %s", args.loadcases)
    logger.info("Aggregation: standing=%s, labour=%s", args.standing_agg, args.labour_agg)

    # ── 1. Mesh ──
    num_cells, volumes, group_masks = _setup_dolfinx_mesh(args.domains)

    # ── 2. Metadata ──
    from analysis.spectral_data import load_metadata
    sex, age = load_metadata()
    if sex.shape != age.shape:
        raise ValueError(f"Metadata shape mismatch: sex{sex.shape} vs age{age.shape}")
    N_subjects = sex.shape[0]
    logger.info(
        "Subjects: %d (M=%d, F=%d)",
        N_subjects, int((sex == "M").sum()), int((sex == "F").sum()),
    )

    # ── 3. Spectral metrics (permutation-aware) ──
    logger.info("Computing spectral metrics (permutation-aware dG) …")
    spec_df = _compute_spectral_metrics(args.spectral_dir)
    if len(spec_df) != N_subjects:
        raise ValueError(
            f"Subject count mismatch: spectral N={len(spec_df)} vs metadata N={N_subjects}. "
            "Ensure results/*/data and data/sex.npy refer to the same cohort ordering."
        )

    # ── 4. Self-check ──
    _self_check(num_cells, volumes, group_masks, spec_df)

    # ── 5. PASS 1: per-subject tail metrics + collect P95 for τ ──
    logger.info("Pass 1: per-subject tail metrics …")
    records: list[dict] = []
    # Collect per-subject P95 for threshold computation (N floats per (lc, dom))
    p95_collector: dict[tuple[str, str], list[float]] = {
        (lc, dom): [] for lc in args.loadcases for dom in group_masks
    }

    for lc_idx, lc in enumerate(args.loadcases):
        strain_path = args.data_dir / f"strain_{lc}.zarr"
        stress_path = args.data_dir / f"stress_{lc}.zarr"
        if not strain_path.exists() or not stress_path.exists():
            logger.warning("Missing strain/stress for %s — skipping.", lc)
            continue

        strain_z = _open_zarr_lazy(strain_path)
        stress_z = _open_zarr_lazy(stress_path)
        if strain_z.shape[0] != N_subjects or stress_z.shape[0] != N_subjects:
            raise ValueError(
                f"Subject count mismatch for {lc}: metadata N={N_subjects}, "
                f"strain N={strain_z.shape[0]}, stress N={stress_z.shape[0]}"
            )
        assert strain_z.shape[1] == num_cells, (
            f"Cell count mismatch: Zarr {strain_z.shape[1]} vs mesh {num_cells}"
        )
        logger.info("  %s (%d/%d) …", lc, lc_idx + 1, len(args.loadcases))

        for s in range(N_subjects):
            strain_3x3 = np.asarray(strain_z[s])  # (num_cells, 3, 3)
            stress_3x3 = np.asarray(stress_z[s])
            sed, eps1, eps3, _vms = element_scalars(strain_3x3, stress_3x3)

            for dom, mask in group_masks.items():
                m = compute_tail_metrics(sed, eps1, eps3, volumes, mask)
                m["subject_idx"] = s
                m["loadcase"] = lc
                m["group"] = dom
                records.append(m)
                # P95 for threshold
                p95_val = m["P95_SED"]
                if np.isfinite(p95_val):
                    p95_collector[(lc, dom)].append(p95_val)

            if (s + 1) % 50 == 0:
                logger.info("    … %d / %d", s + 1, N_subjects)

    df = pd.DataFrame(records)

    # ── 6. Compute thresholds τ ──
    logger.info("Computing exceedance thresholds τ …")
    tau_dict: dict[tuple[str, str], float] = {}
    for (lc, dom), p95s in p95_collector.items():
        if len(p95s) < 3:
            continue
        # τ = threshold_quantile-th percentile of per-subject P95 distribution
        tau = float(np.percentile(p95s, 100 * args.threshold_quantile))
        tau_dict[(lc, dom)] = tau
        logger.info("  τ(%s, %s) = %.4e", lc, dom, tau)

    # ── 7. PASS 2: exceedance vol-fracs (streaming — O(num_cells) memory) ──
    logger.info("Pass 2: exceedance vol-fracs (streaming) …")
    exceed_map: dict[tuple[str, int, str], float] = {}

    for lc_idx, lc in enumerate(args.loadcases):
        strain_path = args.data_dir / f"strain_{lc}.zarr"
        stress_path = args.data_dir / f"stress_{lc}.zarr"
        if not strain_path.exists() or not stress_path.exists():
            continue

        # Check if this LC has any thresholds at all
        has_tau = any((lc, dom) in tau_dict for dom in group_masks)
        if not has_tau:
            for s in range(N_subjects):
                for dom in group_masks:
                    exceed_map[(lc, s, dom)] = np.nan
            continue

        strain_z = _open_zarr_lazy(strain_path)
        stress_z = _open_zarr_lazy(stress_path)
        if strain_z.shape[0] != N_subjects or stress_z.shape[0] != N_subjects:
            raise ValueError(
                f"Subject count mismatch for {lc}: metadata N={N_subjects}, "
                f"strain N={strain_z.shape[0]}, stress N={stress_z.shape[0]}"
            )
        logger.info("  %s (exceedance) …", lc)

        for s in range(N_subjects):
            strain_3x3 = np.asarray(strain_z[s])
            stress_3x3 = np.asarray(stress_z[s])
            sed = sed_only(strain_3x3, stress_3x3)  # fast path

            for dom, mask in group_masks.items():
                tau = tau_dict.get((lc, dom))
                if tau is None or tau <= 0:
                    exceed_map[(lc, s, dom)] = np.nan
                else:
                    exceed_map[(lc, s, dom)] = exceedance_volfrac(
                        sed, volumes, mask, tau,
                    )

            if (s + 1) % 100 == 0:
                logger.info("    … %d / %d", s + 1, N_subjects)

    # Fill exceedance column into df
    df["exceedance_volfrac_SED"] = [
        exceed_map.get(
            (row["loadcase"], int(row["subject_idx"]), row["group"]),
            np.nan,
        )
        for _, row in df.iterrows()
    ]

    # ── 8. Merge spectral + metadata ──
    df = df.merge(spec_df, on="subject_idx", how="left")
    df["sex"] = sex[df["subject_idx"].values.astype(int)]
    df["age"] = age[df["subject_idx"].values.astype(int)]

    # ── 9. Export raw metrics ──
    out_csv = args.output_dir / "metrics_tail.csv"
    df.to_csv(out_csv, index=False, float_format="%.6e")
    logger.info("Saved %s (%d rows)", out_csv, len(df))

    out_parquet = args.output_dir / "metrics_tail.parquet"
    df.to_parquet(out_parquet, index=False)
    logger.info("Saved %s", out_parquet)

    tau_json = {f"{lc}|{dom}": v for (lc, dom), v in tau_dict.items()}
    with open(args.output_dir / "thresholds.json", "w") as f:
        json.dump(tau_json, f, indent=2)

    # ── 10. Per-subject aggregation + statistical tests ──
    logger.info("Aggregating per subject (standing=%s, labour=%s) …",
                args.standing_agg, args.labour_agg)
    df_agg = _aggregate_per_subject(
        df, METRIC_COLS, args.standing_agg, args.labour_agg,
    )
    df_agg = df_agg.merge(spec_df, on="subject_idx", how="left")
    df_agg["sex"] = sex[df_agg["subject_idx"].values.astype(int)]
    df_agg["age"] = age[df_agg["subject_idx"].values.astype(int)]
    df_agg.to_csv(
        args.output_dir / "metrics_tail_aggregated.csv",
        index=False, float_format="%.6e",
    )

    stats_df = _sex_comparison(df_agg, METRIC_COLS)
    stats_df.to_csv(
        args.output_dir / "sex_comparison_stats.csv",
        index=False, float_format="%.4e",
    )
    logger.info("Statistical comparisons: %d rows", len(stats_df))

    logger.info("=== Pipeline complete ===")


if __name__ == "__main__":
    main()
