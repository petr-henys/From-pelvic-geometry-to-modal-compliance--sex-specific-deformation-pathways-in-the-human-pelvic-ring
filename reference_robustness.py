#!/usr/bin/env python3
"""Reference-subject and bootstrap robustness analysis for mode pairing.

Reuses mode_pairing.pair_modes (Hungarian on mass-weighted MAC) to assess
how sensitive the spectral analysis is to:
  A) choice of reference subject,
  B) metric used for pairing (M0 vs simple Euclidean),
  C) cohort subsampling (stratified 80 % bootstrap).

Requires DOLFINx + PETSc to assemble M0 for mass-weighted MAC.
Falls back to Euclidean MAC (pair_modes_simple) if DOLFINx unavailable.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_FEB = PROJECT_ROOT / "anatomy_data" / "full_model.feb"
DEFAULT_ROOT = PROJECT_ROOT / "results" / "ref_S1P_fixed_new2" / "data"
DEFAULT_OUT = PROJECT_ROOT / "results" / "reference_robustness"

# Target near-degenerate pairs / blocks (0-based)
PAIR_9_10 = (8, 9)
BLOCK_12_15 = (11, 12, 13, 14)


# ===================================================================
# Data loading (Zarr + NumPy)
# ===================================================================
def _open_zarr(path: Path) -> np.ndarray:
    """Load a Zarr group's 'data' array."""
    import zarr

    store = zarr.open_group(str(path), mode="r")
    if "data" in store:
        return np.asarray(store["data"][:])
    return np.asarray(zarr.open_array(str(path), mode="r")[:])


def discover_data(root: Path) -> dict:
    """Scan *root* for eigenvalues, eigenvectors, permutations, and metadata."""
    report: dict = {"root": str(root), "found": {}, "missing": []}

    for name in ("eigenvalues", "eigenvectors", "eig_permutations"):
        p = root / f"{name}.zarr"
        if p.exists():
            report["found"][name] = str(p)
        else:
            report["missing"].append(name)

    # Metadata: sex, age, pelvic dimensions
    raw = PROJECT_ROOT / "data"
    for npy_name in ("sex", "age"):
        p = raw / f"{npy_name}.npy"
        if p.exists():
            report["found"][npy_name] = str(p)
        else:
            report["missing"].append(npy_name)

    dims_zarr = raw / "dimensions_ref.zarr"
    if dims_zarr.exists():
        report["found"]["dimensions"] = str(dims_zarr)
    else:
        report["missing"].append("dimensions")

    return report


def load_registry(root: Path, K: int) -> dict:
    """Build subject registry from discovered Zarr arrays.

    Returns dict with keys: eigenvalues (N,K), eigenvectors_obj (lazy Zarr),
    permutations (N,K), sex (N,), age (N,), dims (dict[str, (N,)]), N, K.
    """
    eigenvalues = _open_zarr(root / "eigenvalues.zarr")[:, :K]
    permutations = _open_zarr(root / "eig_permutations.zarr")[:, :K]

    import zarr

    evec_store = zarr.open_group(str(root / "eigenvectors.zarr"), mode="r")
    eigenvectors_obj = evec_store["data"] if "data" in evec_store else zarr.open_array(
        str(root / "eigenvectors.zarr"), mode="r"
    )

    N = eigenvalues.shape[0]
    raw = PROJECT_ROOT / "data"

    sex = np.load(raw / "sex.npy", allow_pickle=True) if (raw / "sex.npy").exists() else None
    age = np.load(raw / "age.npy") if (raw / "age.npy").exists() else None

    dims: dict[str, np.ndarray] = {}
    dims_path = raw / "dimensions_ref.zarr"
    if dims_path.exists():
        ds = zarr.open_group(str(dims_path), mode="r")
        for k in ds:
            dims[k] = np.asarray(ds[k][:])

    return {
        "eigenvalues": eigenvalues,
        "eigenvectors_obj": eigenvectors_obj,
        "permutations": permutations,
        "sex": sex,
        "age": age,
        "dims": dims,
        "N": N,
        "K": K,
    }


# ===================================================================
# M0 construction (DOLFINx)
# ===================================================================
def build_M0(feb_path: Path):
    """Assemble unmapped reference mass matrix M0 from FEBio mesh.

    Returns (M0: PETSc.Mat, V: FunctionSpace) or (None, None) on failure.
    """
    try:
        from dolfinx import fem
        from dolfinx.fem import petsc
        import ufl
        from simulation.febio_parser import FEBio2Dolfinx
    except ImportError:
        logger.warning("DOLFINx/PETSc not available — will use Euclidean MAC")
        return None, None

    logger.info("Building M0 from %s ...", feb_path)
    t0 = time.perf_counter()
    f2x = FEBio2Dolfinx(str(feb_path))
    domain = f2x.mesh_dolfinx
    gdim = domain.geometry.dim
    V = fem.functionspace(domain, ("P", 1, (gdim,)))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    dx = ufl.Measure("dx", domain=domain)
    form = fem.form(ufl.inner(u, v) * dx)
    M0 = petsc.create_matrix(form)
    petsc.assemble_matrix(M0, form)
    M0.assemble()
    logger.info("M0 assembled in %.1f s  (nnz=%d)", time.perf_counter() - t0, M0.getInfo()["nz_used"])
    return M0, V


# ===================================================================
# Pairing adapter
# ===================================================================
def run_pairing(
    ref_vecs: np.ndarray,
    ref_vals: np.ndarray,
    samp_vecs: np.ndarray,
    samp_vals: np.ndarray,
    M0,
    *,
    mac_cut: float = 0.05,
) -> dict:
    """Pair sample modes to reference; return dict(perm, failures).

    Uses mass-weighted MAC if M0 is provided, else falls back to Euclidean.
    """
    if M0 is not None:
        from mode_pairing import pair_modes

        perm = pair_modes(ref_vecs, ref_vals, samp_vecs, samp_vals, M0, mac_cut=mac_cut)
    else:
        from ligament_effects import pair_modes_simple

        perm = pair_modes_simple(ref_vecs, ref_vals, samp_vecs, samp_vals, mac_cut=mac_cut)

    failures: list[tuple[int, str]] = []
    unpaired = np.where(perm < 0)[0]
    for idx in unpaired:
        failures.append((int(idx), f"MAC < {mac_cut}"))

    return {"perm": perm, "failures": failures}


# ===================================================================
# Summary statistics helpers
# ===================================================================
def swap_count(perm: np.ndarray) -> int:
    """Number of modes where pi(k) != k (excluding -1 = unpaired)."""
    valid = perm >= 0
    return int(np.sum(perm[valid] != np.arange(len(perm))[valid]))


def swap_910(perm: np.ndarray) -> bool:
    """True if modes 9↔10 are swapped (0-based indices 8,9)."""
    a, b = PAIR_9_10
    if perm[a] < 0 or perm[b] < 0:
        return False
    return bool(perm[a] == b and perm[b] == a)


def any_swap_in_block(perm: np.ndarray, block: tuple[int, ...] = BLOCK_12_15) -> bool:
    """True if any mode within *block* is permuted to another position in the block."""
    identity = np.arange(len(perm))
    for idx in block:
        if perm[idx] >= 0 and perm[idx] != identity[idx] and perm[idx] in block:
            return True
    return False


def swap_matrix(perms: np.ndarray, K: int) -> np.ndarray:
    """Count how often pi(k) = j across cohort. Shape (K, K)."""
    mat = np.zeros((K, K), dtype=int)
    for perm in perms:
        for k in range(min(K, len(perm))):
            j = perm[k]
            if 0 <= j < K:
                mat[k, j] += 1
    return mat


def gap_in(eigenvalues: np.ndarray) -> np.ndarray:
    """Intra-pair gap: (lam_{i+1} - lam_i) / lam_i.  Shape (N, K-1)."""
    return np.diff(eigenvalues, axis=1) / eigenvalues[:, :-1]


def near_degenerate_prevalence(
    eigenvalues: np.ndarray, pair: tuple[int, int], eps_in: float
) -> float:
    """Fraction of subjects with gap_in < eps_in for a given mode pair."""
    g = gap_in(eigenvalues)
    gi = min(pair)  # gap index = lower of the two 0-based mode indices
    return float(np.mean(g[:, gi] < eps_in))


# ===================================================================
# A) Reference-subject sweep
# ===================================================================
def select_reference_subjects(
    registry: dict, R_random: int, seed: int
) -> list[dict]:
    """Choose reference subjects: 1 median-AP + R_random random ones.

    Returns list of dicts with keys: idx (int), label (str).
    """
    refs: list[dict] = []

    # 1) Median by AP inlet diameter (or lam_1 fallback)
    dims = registry["dims"]
    if "AnteriorPosteriorInletDiameter" in dims:
        ap = dims["AnteriorPosteriorInletDiameter"]
        median_idx = int(np.argmin(np.abs(ap - np.median(ap))))
        refs.append({"idx": median_idx, "label": f"median_AP (subj {median_idx})"})
    else:
        lam1 = registry["eigenvalues"][:, 0]
        median_idx = int(np.argmin(np.abs(lam1 - np.median(lam1))))
        refs.append({"idx": median_idx, "label": f"median_lam1 (subj {median_idx})"})

    # 2) Random subjects
    rng = np.random.default_rng(seed)
    all_idx = np.arange(registry["N"])
    pool = np.setdiff1d(all_idx, [refs[0]["idx"]])
    chosen = rng.choice(pool, size=min(R_random, len(pool)), replace=False)
    for ci in chosen:
        refs.append({"idx": int(ci), "label": f"random (subj {ci})"})

    return refs


def reference_sweep(
    registry: dict,
    M0,
    refs: list[dict],
    mac_cut: float = 0.05,
) -> list[dict]:
    """Re-pair entire cohort against each reference subject.

    Returns list of per-reference summary dicts.
    """
    N, K = registry["N"], registry["K"]
    evec_obj = registry["eigenvectors_obj"]
    evals = registry["eigenvalues"]
    results = []

    for ri, ref_info in enumerate(refs):
        ref_idx = ref_info["idx"]
        logger.info(
            "[Ref %d/%d] idx=%d (%s)", ri + 1, len(refs), ref_idx, ref_info["label"]
        )
        ref_vecs = np.asarray(evec_obj[ref_idx, :K])  # (K, n_nodes, 3)
        ref_vals = evals[ref_idx]

        perms = np.full((N, K), -1, dtype=np.int32)
        n_failures = 0

        for si in range(N):
            if si == ref_idx:
                perms[si] = np.arange(K, dtype=np.int32)
                continue
            samp_vecs = np.asarray(evec_obj[si, :K])
            samp_vals = evals[si]
            res = run_pairing(ref_vecs, ref_vals, samp_vecs, samp_vals, M0, mac_cut=mac_cut)
            perms[si] = res["perm"]
            n_failures += len(res["failures"])
            if (si + 1) % 50 == 0:
                logger.info("  paired %d/%d subjects", si + 1, N)

        swap_counts = np.array([swap_count(perms[s]) for s in range(N)])
        swap_910_flags = np.array([swap_910(perms[s]) for s in range(N)])
        block_flags = np.array([any_swap_in_block(perms[s]) for s in range(N)])
        sm = swap_matrix(perms, K)

        results.append({
            "ref_idx": ref_idx,
            "ref_label": ref_info["label"],
            "swap_count_median": float(np.median(swap_counts)),
            "swap_count_mean": float(np.mean(swap_counts)),
            "swap_count_std": float(np.std(swap_counts)),
            "swap_910_rate": float(np.mean(swap_910_flags)),
            "block_12_15_rate": float(np.mean(block_flags)),
            "total_failures": n_failures,
            "swap_matrix": sm.tolist(),
            "perms": perms,
        })
        logger.info(
            "  swap_count median=%.1f  9<>10=%.1f%%  12-15=%.1f%%  failures=%d",
            results[-1]["swap_count_median"],
            100 * results[-1]["swap_910_rate"],
            100 * results[-1]["block_12_15_rate"],
            n_failures,
        )

    return results


# ===================================================================
# B) Metric sensitivity (M0 vs Euclidean)
# ===================================================================
def metric_sensitivity(
    registry: dict,
    M0,
    ref_idx: int,
    mac_cut: float = 0.05,
) -> dict | None:
    """Compare pairing with M0-weighted MAC vs Euclidean MAC.

    Returns None if M0 is not available (only one metric possible).
    """
    if M0 is None:
        logger.info("Metric sensitivity: skipped (M0 unavailable)")
        return None

    N, K = registry["N"], registry["K"]
    evec_obj = registry["eigenvectors_obj"]
    evals = registry["eigenvalues"]
    ref_vecs = np.asarray(evec_obj[ref_idx, :K])
    ref_vals = evals[ref_idx]

    metrics_result: dict[str, dict] = {}

    for metric_name, use_M0 in [("M0_weighted", M0), ("euclidean", None)]:
        logger.info("Metric sensitivity: %s ...", metric_name)
        perms = np.full((N, K), -1, dtype=np.int32)
        for si in range(N):
            if si == ref_idx:
                perms[si] = np.arange(K, dtype=np.int32)
                continue
            samp_vecs = np.asarray(evec_obj[si, :K])
            samp_vals = evals[si]
            res = run_pairing(ref_vecs, ref_vals, samp_vecs, samp_vals, use_M0, mac_cut=mac_cut)
            perms[si] = res["perm"]

        swap_counts = np.array([swap_count(perms[s]) for s in range(N)])
        swap_910_flags = np.array([swap_910(perms[s]) for s in range(N)])
        block_flags = np.array([any_swap_in_block(perms[s]) for s in range(N)])

        agreement_with_baseline = None
        if metric_name == "euclidean" and "M0_weighted" in metrics_result:
            baseline = metrics_result["M0_weighted"]["perms"]
            agreement = np.mean([
                np.array_equal(perms[s], baseline[s]) for s in range(N)
            ])
            agreement_with_baseline = float(agreement)

        metrics_result[metric_name] = {
            "swap_count_median": float(np.median(swap_counts)),
            "swap_910_rate": float(np.mean(swap_910_flags)),
            "block_12_15_rate": float(np.mean(block_flags)),
            "perms": perms,
            "agreement_with_baseline": agreement_with_baseline,
        }
        logger.info(
            "  %s: swap_median=%.1f  9<>10=%.1f%%  12-15=%.1f%%",
            metric_name,
            metrics_result[metric_name]["swap_count_median"],
            100 * metrics_result[metric_name]["swap_910_rate"],
            100 * metrics_result[metric_name]["block_12_15_rate"],
        )

    return {k: {kk: vv for kk, vv in v.items() if kk != "perms"} for k, v in metrics_result.items()}


# ===================================================================
# C) Stratified m-out-of-n subsampling
# ===================================================================
def stratified_subsample_indices(
    N: int, frac: float, sex: np.ndarray | None, rng: np.random.Generator
) -> np.ndarray:
    """Draw a stratified subsample (by sex if available)."""
    if sex is not None:
        mask_m = sex == "M"
        mask_f = sex == "F"
        idx_m = np.where(mask_m)[0]
        idx_f = np.where(mask_f)[0]
        n_m = max(1, int(round(len(idx_m) * frac)))
        n_f = max(1, int(round(len(idx_f) * frac)))
        sel_m = rng.choice(idx_m, size=n_m, replace=False)
        sel_f = rng.choice(idx_f, size=n_f, replace=False)
        return np.sort(np.concatenate([sel_m, sel_f]))
    else:
        logger.warning("Sex metadata unavailable — unstratified subsampling")
        n = max(1, int(round(N * frac)))
        return np.sort(rng.choice(N, size=n, replace=False))


def bootstrap_analysis(
    registry: dict,
    M0,
    ref_idx: int,
    B: int,
    seed: int,
    frac: float = 0.80,
    eps_in: float | None = None,
    mac_cut: float = 0.05,
) -> dict:
    """Stratified 80 % subsampling with B replicates.

    For each replicate computes:
      - near-degenerate prevalence (eigenvalue gaps only, no re-pairing)
      - pairing-based swap rates (re-pair subsample, one reference)

    Returns percentile CIs (2.5/97.5) + median for each statistic.
    """
    N, K = registry["N"], registry["K"]
    evals = registry["eigenvalues"]
    evec_obj = registry["eigenvectors_obj"]
    sex = registry["sex"]

    # Determine eps_in threshold from full cohort if not provided
    if eps_in is None:
        full_gaps = gap_in(evals)
        eps_in = float(np.percentile(full_gaps, 25))
        logger.info("eps_in threshold (25th pctl of full cohort): %.6f", eps_in)

    ref_vecs = np.asarray(evec_obj[ref_idx, :K])
    ref_vals = evals[ref_idx]

    rng = np.random.default_rng(seed)

    # Pre-allocate collectors
    stat_names = [
        "prevalence_9_10",
        "prevalence_12_15",
        "swap_910_rate",
        "block_12_15_rate",
        "swap_count_median",
    ]
    collectors: dict[str, list[float]] = {s: [] for s in stat_names}

    t0 = time.perf_counter()
    for b in range(B):
        idx = stratified_subsample_indices(N, frac, sex, rng)
        sub_evals = evals[idx]

        # (1) Reference-free: eigenvalue-gap prevalence
        collectors["prevalence_9_10"].append(
            near_degenerate_prevalence(sub_evals, PAIR_9_10, eps_in)
        )
        # For block 12-15, use min gap across consecutive pairs in the block
        block_gaps = gap_in(sub_evals)
        block_min_gap = np.min(
            block_gaps[:, [p for p in range(min(BLOCK_12_15), max(BLOCK_12_15))]],
            axis=1,
        )
        collectors["prevalence_12_15"].append(float(np.mean(block_min_gap < eps_in)))

        # (2) Pairing-based: re-pair subsample
        sub_perms = np.full((len(idx), K), -1, dtype=np.int32)
        for si_local, si_global in enumerate(idx):
            if si_global == ref_idx:
                sub_perms[si_local] = np.arange(K, dtype=np.int32)
                continue
            samp_vecs = np.asarray(evec_obj[si_global, :K])
            samp_vals = evals[si_global]
            res = run_pairing(ref_vecs, ref_vals, samp_vecs, samp_vals, M0, mac_cut=mac_cut)
            sub_perms[si_local] = res["perm"]

        swap_910_flags = np.array([swap_910(sub_perms[s]) for s in range(len(idx))])
        block_flags = np.array([any_swap_in_block(sub_perms[s]) for s in range(len(idx))])
        swap_counts = np.array([swap_count(sub_perms[s]) for s in range(len(idx))])

        collectors["swap_910_rate"].append(float(np.mean(swap_910_flags)))
        collectors["block_12_15_rate"].append(float(np.mean(block_flags)))
        collectors["swap_count_median"].append(float(np.median(swap_counts)))

        elapsed = time.perf_counter() - t0
        if (b + 1) % 10 == 0 or b == 0:
            logger.info(
                "  bootstrap %d/%d  (%.1f s elapsed, ~%.0f s remaining)",
                b + 1, B, elapsed, elapsed / (b + 1) * (B - b - 1),
            )

    # Build CI summary
    ci_summary: dict[str, dict[str, float]] = {}
    for stat in stat_names:
        arr = np.array(collectors[stat])
        ci_summary[stat] = {
            "median": float(np.median(arr)),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "ci_2.5": float(np.percentile(arr, 2.5)),
            "ci_97.5": float(np.percentile(arr, 97.5)),
        }

    logger.info("Bootstrap complete (%.1f s total)", time.perf_counter() - t0)
    return {"eps_in": eps_in, "B": B, "frac": frac, "seed": seed, "stats": ci_summary}


# ===================================================================
# Output
# ===================================================================
def save_results(
    out_dir: Path,
    discovery: dict,
    ref_results: list[dict],
    metric_results: dict | None,
    bootstrap_results: dict,
) -> None:
    """Write all results as JSON files to *out_dir*."""
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "discovery_report.json").write_text(
        json.dumps(discovery, indent=2), encoding="utf-8"
    )

    # Reference sweep (strip numpy arrays for JSON)
    ref_json = []
    for r in ref_results:
        rj = {k: v for k, v in r.items() if k != "perms"}
        ref_json.append(rj)
    (out_dir / "reference_sweep.json").write_text(
        json.dumps(ref_json, indent=2), encoding="utf-8"
    )

    # Save per-reference permutation arrays as .npy
    perm_dir = out_dir / "permutations"
    perm_dir.mkdir(exist_ok=True)
    for r in ref_results:
        np.save(perm_dir / f"perms_ref{r['ref_idx']}.npy", r["perms"])

    if metric_results is not None:
        (out_dir / "metric_sensitivity.json").write_text(
            json.dumps(metric_results, indent=2), encoding="utf-8"
        )

    (out_dir / "bootstrap_CI.json").write_text(
        json.dumps(bootstrap_results, indent=2), encoding="utf-8"
    )

    # Summary table (human-readable)
    lines = ["=" * 72, "REFERENCE SWEEP SUMMARY", "=" * 72]
    for r in ref_json:
        lines.append(
            f"  ref {r['ref_idx']:>3d}  ({r['ref_label']:<30s})  "
            f"swap_med={r['swap_count_median']:.1f}  "
            f"9<>10={100 * r['swap_910_rate']:.1f}%  "
            f"12-15={100 * r['block_12_15_rate']:.1f}%  "
            f"fail={r['total_failures']}"
        )
    lines += [
        "",
        "=" * 72,
        f"BOOTSTRAP CI (B={bootstrap_results['B']})",
        "=" * 72,
    ]
    for stat, vals in bootstrap_results["stats"].items():
        lines.append(
            f"  {stat:<25s}  median={vals['median']:.4f}  "
            f"95% CI=[{vals['ci_2.5']:.4f}, {vals['ci_97.5']:.4f}]"
        )
    summary_text = "\n".join(lines)
    (out_dir / "summary.txt").write_text(summary_text, encoding="utf-8")
    print(summary_text)

    logger.info("Results written to %s", out_dir)


def main(
    root: Path = DEFAULT_ROOT,
    feb: Path = DEFAULT_FEB,
    K: int = 15,
    R: int = 4,
    B: int = 200,
    seed: int = 123,
    out: Path = DEFAULT_OUT,
    mac_cut: float = 0.05,
    use_m0: bool = True,
) -> None:
    """Run full robustness analysis."""
    # Step 1: Discovery
    logger.info("Scanning %s ...", root)
    discovery = discover_data(root)

    if discovery["missing"]:
        logger.warning("Missing data: %s", discovery["missing"])
    logger.info(
        "Found: %s",
        ", ".join(f"{k}={v}" for k, v in discovery["found"].items()),
    )

    if "eigenvalues" not in discovery["found"] or "eigenvectors" not in discovery["found"]:
        logger.error("Cannot proceed without eigenvalues and eigenvectors")
        sys.exit(1)

    # Load data
    registry = load_registry(root, K)
    logger.info("Registry: N=%d, K=%d", registry["N"], registry["K"])
    if registry["sex"] is not None:
        n_f = int(np.sum(registry["sex"] == "F"))
        n_m = int(np.sum(registry["sex"] == "M"))
        logger.info("Sex: F=%d, M=%d", n_f, n_m)

    # Build M0
    M0 = None
    if use_m0 and feb.exists():
        M0, _ = build_M0(feb)
    elif use_m0:
        logger.warning("FEBio file not found (%s) — using Euclidean MAC", feb)

    # A) Reference sweep
    refs = select_reference_subjects(registry, R, seed)
    logger.info("Reference subjects: %s", [r["label"] for r in refs])
    ref_results = reference_sweep(registry, M0, refs, mac_cut=mac_cut)

    # B) Metric sensitivity (M0 vs Euclidean)
    metric_results = metric_sensitivity(
        registry, M0, refs[0]["idx"], mac_cut=mac_cut
    )

    # C) Bootstrap
    bootstrap_results = bootstrap_analysis(
        registry, M0, refs[0]["idx"], B, seed, mac_cut=mac_cut
    )

    # Save
    save_results(out, discovery, ref_results, metric_results, bootstrap_results)

    # Cleanup PETSc
    if M0 is not None:
        M0.destroy()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    main()
