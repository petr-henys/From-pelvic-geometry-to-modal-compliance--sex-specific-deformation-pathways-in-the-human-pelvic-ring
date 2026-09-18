"""Spectral analysis metrics: gaps, clusters, subspace distances, permutation
diagnostics, and canonical inlet coupling.

Pure computation — no plotting, no I/O.
"""
from __future__ import annotations

import warnings
from collections import Counter

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


# ============================================================
# Eigenvalue gap & cluster detection  (Sec. A)
# ============================================================
def compute_gap_in(eigenvalues: np.ndarray) -> np.ndarray:
    """gap_in(i) = (λ_{i+1} − λ_i) / λ_i  per subject.

    Parameters
    ----------
    eigenvalues : (N_subjects, N_modes)

    Returns
    -------
    (N_subjects, N_modes − 1)
    """
    ev = np.asarray(eigenvalues, dtype=float)
    diffs = np.diff(ev, axis=1)
    if np.isfinite(diffs).any() and np.nanmin(diffs) < -1e-12:
        warnings.warn(
            "Negative eigenvalue gaps detected. "
            "gap_in assumes solver-ordered eigenvalues (nondecreasing).",
            stacklevel=2,
        )
    denom = ev[:, :-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        gaps = diffs / denom
    gaps[~np.isfinite(gaps)] = np.nan
    return gaps


def detect_clusters(
    eigenvalues: np.ndarray,
    eps_in: float,
) -> list[list[tuple[int, int]]]:
    """Detect near-degenerate clusters per subject over ALL modes.

    A cluster [i, j) is a maximal contiguous index set where all
    internal gaps satisfy gap_in(k) < eps_in  for k = i, ..., j-2.

    Returns list (per subject) of list of (start, end) tuples [start, end).
    Indices are 0-based mode numbers.
    """
    n_subj, n_modes = eigenvalues.shape
    gaps = compute_gap_in(eigenvalues)
    all_clusters: list[list[tuple[int, int]]] = []
    for s in range(n_subj):
        clusters_s: list[tuple[int, int]] = []
        i = 0
        while i < n_modes - 1:
            if gaps[s, i] < eps_in:
                j = i
                while j < n_modes - 1 and gaps[s, j] < eps_in:
                    j += 1
                clusters_s.append((i, j + 1))
                i = j + 1
            else:
                i += 1
        all_clusters.append(clusters_s)
    return all_clusters


def compute_sep_out(
    eigenvalues: np.ndarray,
    cluster_start: int,
    cluster_end: int,
) -> np.ndarray:
    """Two-sided separation for [start, end); NaN at truncated boundaries."""
    n_subj, n_modes = eigenvalues.shape
    sep = np.full(n_subj, np.nan)
    if cluster_start == 0 or cluster_end == n_modes:
        return sep
    for s in range(n_subj):
        vals = []
        if cluster_start > 0:
            num = eigenvalues[s, cluster_start] - eigenvalues[s, cluster_start - 1]
            den = eigenvalues[s, cluster_start]
            if np.isfinite(num) and np.isfinite(den) and den > 0:
                vals.append(num / den)
        if cluster_end < n_modes:
            num = eigenvalues[s, cluster_end] - eigenvalues[s, cluster_end - 1]
            den = eigenvalues[s, cluster_end - 1]
            if np.isfinite(num) and np.isfinite(den) and den > 0:
                vals.append(num / den)
        if vals:
            sep[s] = min(vals)
    return sep


def classify_cluster(
    gaps_in: np.ndarray,
    cluster_start: int,
    cluster_end: int,
    eps_deg: float,
    member_mask: np.ndarray | None = None,
) -> dict[str, int]:
    """Classify each subject: true degeneracy vs. near-degenerate."""
    internal = gaps_in[:, cluster_start : cluster_end - 1]
    if member_mask is not None:
        internal = internal[member_mask]

    if internal.size == 0:
        return {"true_degeneracy": 0, "near_degenerate": 0, "unknown": 0}

    finite_rows = np.isfinite(internal).all(axis=1)
    true_rows = finite_rows & np.all(internal <= eps_deg, axis=1)
    near_rows = finite_rows & ~true_rows
    unknown = int((~finite_rows).sum())

    return {
        "true_degeneracy": int(true_rows.sum()),
        "near_degenerate": int(near_rows.sum()),
        "unknown": unknown,
    }


# ============================================================
# Subspace metrics  (Sec. B: Grassmann distance)
# ============================================================
def orthonormalize_l2(
    vecs: np.ndarray,
    M: np.ndarray | None = None,
) -> np.ndarray:
    """Orthonormalize columns of *vecs* w.r.t. inner product defined by *M*.

    When *M* is None the standard L2 inner product is used (plain QR).
    When *M* is a scipy sparse or dense matrix, M-weighted Gram-Schmidt via
    Cholesky of the Gram matrix G = V^T M V is applied.

    Parameters
    ----------
    vecs : (n_dof, k)
    M : (n_dof, n_dof) sparse/dense or None

    Returns
    -------
    Q : (n_dof, k)  columns M-orthonormal: Q^T M Q = I_k
    """
    if M is None:
        Q, _ = np.linalg.qr(vecs)
        return Q[:, : vecs.shape[1]]

    # M-weighted: compute Gram matrix G = V^T M V, Cholesky factor, invert
    MV = M @ vecs  # sparse @ dense → dense
    G = vecs.T @ MV  # (k, k)
    L = np.linalg.cholesky(G)  # G = L L^T
    # Q = V L^{-T}  so that Q^T M Q = L^{-1} V^T M V L^{-T} = L^{-1} G L^{-T} = I
    Q = np.linalg.solve(L, vecs.T).T  # solve L X^T = V^T → X = V L^{-T}
    return Q[:, : vecs.shape[1]]


def compute_principal_angles(
    q_ref: np.ndarray,
    q_s: np.ndarray,
    M: np.ndarray | None = None,
) -> np.ndarray:
    """Principal angles between two subspaces (SVD of overlap matrix).

    When *M* is provided the overlap is S = Q_ref^T M Q_s (mass-weighted).
    Both Q_ref and Q_s must already be M-orthonormal.
    """
    if M is not None:
        S = q_ref.T @ (M @ q_s)
    else:
        S = q_ref.T @ q_s
    _, sigmas, _ = np.linalg.svd(S)
    return np.arccos(np.clip(sigmas, -1.0, 1.0))


def grassmann_distance(angles: np.ndarray) -> float:
    """d_G = ||θ||_2."""
    return float(np.linalg.norm(angles))


# ============================================================
# Permutation-based swap analysis  (veering vs. mixing)
# ============================================================
def invert_pairing_permutation(perm_ref_to_rank: np.ndarray) -> np.ndarray:
    """Invert ref→rank pairing to rank→ref.

    perm_ref_to_rank[s, i_ref] = j_rank, or -1 if unpaired.
    Returns perm_rank_to_ref[s, j_rank] = i_ref, or -1.
    """
    perm = np.asarray(perm_ref_to_rank, dtype=int)
    if perm.ndim != 2:
        raise ValueError(f"Expected 2D, got shape {perm.shape}")
    n_subj, n_modes = perm.shape
    inv = np.full((n_subj, n_modes), -1, dtype=int)
    for s in range(n_subj):
        for i_ref in range(n_modes):
            j_rank = int(perm[s, i_ref])
            if j_rank < 0:
                continue
            if j_rank >= n_modes:
                warnings.warn(
                    f"Permutation value out of range: perm[{s},{i_ref}]={j_rank}",
                    stacklevel=2,
                )
                continue
            prev = inv[s, j_rank]
            if prev != -1 and prev != i_ref:
                warnings.warn(
                    f"Non-invertible pairing at subject {s}: rank {j_rank} "
                    f"assigned to both ref {prev} and ref {i_ref}.",
                    stacklevel=2,
                )
                inv[s, j_rank] = -1
            else:
                inv[s, j_rank] = i_ref
    return inv


def ref_pair_swap_indicator(
    perm_ref_to_rank: np.ndarray,
    ref_i: int,
    ref_j: int,
) -> np.ndarray:
    """Swap/inversion indicator for a *reference* mode pair (ref_i, ref_j).

    Parameters
    ----------
    perm_ref_to_rank
        Array shaped (n_subjects, n_modes) with entries ``perm[s, i_ref] = j_rank``
        (or -1 if unpaired).
    ref_i, ref_j
        Reference-mode indices (0-based).

    Returns
    -------
    np.ndarray
        Float array shaped (n_subjects,) with values:
          - 1.0 if rank(ref_i) > rank(ref_j) (inversion / exchange),
          - 0.0 if rank(ref_i) < rank(ref_j),
          - NaN if either mode is unpaired/invalid or maps to the same rank.
    """
    perm = np.asarray(perm_ref_to_rank, dtype=int)
    if perm.ndim != 2:
        raise ValueError(f"Expected 2D, got shape {perm.shape}")
    n_subj, n_modes = perm.shape
    if not (0 <= ref_i < n_modes) or not (0 <= ref_j < n_modes):
        raise ValueError(f"ref_i/ref_j out of range: {(ref_i, ref_j)} (n_modes={n_modes})")

    r_i = perm[:, int(ref_i)]
    r_j = perm[:, int(ref_j)]
    valid = (r_i >= 0) & (r_j >= 0) & (r_i < n_modes) & (r_j < n_modes) & (r_i != r_j)

    out = np.full(n_subj, np.nan, dtype=float)
    out[valid] = (r_i[valid] > r_j[valid]).astype(float)
    return out


def summarize_rank_cluster_reference_mapping(
    perm_rank_to_ref: np.ndarray,
    clusters_all: list[list[tuple[int, int]]],
    detected_clusters: list[tuple[int, int, int]],
) -> pd.DataFrame:
    """Summarize which reference-mode sets occupy each detected rank cluster."""
    n_subj, _ = perm_rank_to_ref.shape
    rows: list[dict] = []
    for cs, ce, _cnt in detected_clusters:
        cl_label = cluster_label(cs, ce)
        member_mask = np.array(
            [(cs, ce) in clist for clist in clusters_all], dtype=bool,
        )
        n_members = int(member_mask.sum())
        if n_members == 0:
            continue

        block = perm_rank_to_ref[member_mask, cs:ce]
        paired_all = np.all(block >= 0, axis=1)
        n_paired_all = int(paired_all.sum())

        sets_1based: list[tuple[int, ...]] = []
        for row in block[paired_all]:
            modes = tuple(sorted(int(x) + 1 for x in row.tolist()))
            sets_1based.append(modes)

        if sets_1based:
            c = Counter(sets_1based)
            mode_set, mode_cnt = c.most_common(1)[0]
            top3 = c.most_common(3)
            top3_str = "; ".join(
                f"{format_mode_set_1based(s)} ({cnt / len(sets_1based):.0%})"
                for s, cnt in top3
            )
            mode_set_str = format_mode_set_1based(mode_set)
            mode_rate = float(mode_cnt / len(sets_1based))
        else:
            mode_set_str = ""
            mode_rate = np.nan
            top3_str = ""

        rows.append({
            "rank_cluster": cl_label,
            "dim": ce - cs,
            "n_subjects": n_members,
            "prevalence": f"{n_members / n_subj:.1%}",
            "paired_all_rate": float(n_paired_all / n_members),
            "ref_modes_mode": mode_set_str,
            "ref_modes_mode_rate": mode_rate,
            "ref_modes_top3": top3_str,
        })

    return pd.DataFrame(rows)


def classify_rank_adjacent_pair_swap(
    perm_rank_to_ref: np.ndarray,
    morph_proxy: np.ndarray,
    rank_i: int,
    *,
    window: int,
    subject_mask: np.ndarray | None = None,
    swap_rate_min: float = 0.02,
    swap_rate_veering_min: float = 0.15,
) -> dict:
    """Classify swapping/veering for a rank-adjacent pair (rank_i, rank_i+1)."""
    n_subj, n_modes = perm_rank_to_ref.shape
    if rank_i < 0 or rank_i >= n_modes - 1:
        raise ValueError(f"rank_i out of range: {rank_i} (n_modes={n_modes})")

    rank_j = rank_i + 1
    ids_i = perm_rank_to_ref[:, rank_i]
    ids_j = perm_rank_to_ref[:, rank_j]

    if subject_mask is None:
        subject_mask = np.ones(n_subj, dtype=bool)
    subject_mask = np.asarray(subject_mask, dtype=bool)
    if subject_mask.shape != (n_subj,):
        raise ValueError(
            f"subject_mask shape mismatch: expected ({n_subj},), got {subject_mask.shape}"
        )
    n_mask = int(subject_mask.sum())
    sort_idx = np.argsort(morph_proxy)
    _empty = {
        "rank_pair": (rank_i, rank_j), "ref_pair": None, "swap_rate": 0.0,
        "invalid_rate": 1.0, "match_rate": 0.0, "verdict": "empty mask",
        "is_swap": np.zeros(n_subj)[sort_idx].astype(float),
        "valid": np.zeros(n_subj)[sort_idx].astype(float),
        "smoothed": np.full(n_subj, np.nan),
        "sort_idx": sort_idx,
        "peak_rate": 0.0, "active_frac": 0.0, "cv_swap": 0.0,
    }
    if n_mask == 0:
        return _empty

    valid = subject_mask & (ids_i >= 0) & (ids_j >= 0)
    n_valid = int(valid.sum())
    invalid_rate = float(1.0 - n_valid / n_mask)

    if n_valid == 0:
        _empty.update(invalid_rate=invalid_rate, verdict="unpaired")
        return _empty

    unordered_pairs = np.sort(
        np.column_stack([ids_i[valid], ids_j[valid]]), axis=1,
    )
    pair_tuples = [tuple(map(int, p)) for p in unordered_pairs]
    (a, b), cnt = Counter(pair_tuples).most_common(1)[0]
    a, b = int(a), int(b)
    if a == b:
        return {
            "rank_pair": (rank_i, rank_j), "ref_pair": (a, b), "swap_rate": 0.0,
            "invalid_rate": invalid_rate, "match_rate": float(cnt / n_mask),
            "verdict": "invalid labels",
            "is_swap": np.zeros(n_subj)[sort_idx].astype(float),
            "valid": valid[sort_idx].astype(float),
            "smoothed": np.full(n_subj, np.nan),
            "sort_idx": sort_idx,
            "peak_rate": 0.0, "active_frac": 0.0, "cv_swap": 0.0,
        }

    match = valid & (np.minimum(ids_i, ids_j) == a) & (np.maximum(ids_i, ids_j) == b)
    n_match = int(match.sum())
    match_rate = float(n_match / n_mask)

    is_swap = match & (ids_i == b) & (ids_j == a)
    swap_rate = float(is_swap.sum() / n_match) if n_match > 0 else 0.0

    valid_sorted = match[sort_idx].astype(float)
    swap_sorted = is_swap[sort_idx].astype(float)

    sm_num = pd.Series(swap_sorted).rolling(window, center=True, min_periods=1).sum().values
    sm_den = pd.Series(valid_sorted).rolling(window, center=True, min_periods=1).sum().values

    min_valid_in_window = max(3, int(np.ceil(0.25 * window)))
    smoothed = np.full(sm_num.shape, np.nan, dtype=float)
    good = sm_den >= min_valid_in_window
    smoothed[good] = sm_num[good] / sm_den[good]

    finite_sm = smoothed[np.isfinite(smoothed)]
    if finite_sm.size:
        peak_rate = float(np.max(finite_sm))
        mean_sm = float(np.mean(finite_sm))
        cv_swap = float(np.std(finite_sm) / mean_sm) if mean_sm > 1e-6 else 0.0
        active_frac = float(np.mean(finite_sm > 0.5 * peak_rate)) if peak_rate > 0 else 0.0
    else:
        peak_rate = 0.0
        mean_sm = 0.0
        cv_swap = 0.0
        active_frac = 0.0

    if swap_rate < swap_rate_min:
        verdict = "no swap"
    elif swap_rate < swap_rate_veering_min:
        verdict = "weak exchange"
    else:
        is_veering = (active_frac < 0.50) and (cv_swap > 0.3)
        verdict = "veering" if is_veering else "mixing"

    return {
        "rank_pair": (rank_i, rank_j),
        "ref_pair": (a, b),
        "swap_rate": swap_rate,
        "invalid_rate": invalid_rate,
        "match_rate": match_rate,
        "peak_rate": peak_rate,
        "active_frac": active_frac,
        "cv_swap": cv_swap,
        "verdict": verdict,
        "is_swap": swap_sorted,
        "valid": valid_sorted,
        "smoothed": smoothed,
        "sort_idx": sort_idx,
    }


def permutation_null_test_veering(
    perm_rank_to_ref: np.ndarray,
    morph_proxy: np.ndarray,
    rank_i: int,
    *,
    window: int,
    subject_mask: np.ndarray | None = None,
    n_perm: int = 1000,
    seed: int = 42,
) -> dict:
    """Permutation null test for veering: shuffle morphology axis and recompute diagnostics.

    Parameters
    ----------
    perm_rank_to_ref : (n_subj, n_modes)
    morph_proxy : (n_subj,) morphology values to shuffle
    rank_i : rank index of the lower mode in the pair
    window : rolling-window size
    subject_mask : boolean mask for cluster members
    n_perm : number of random permutations
    seed : RNG seed

    Returns
    -------
    dict with observed and null-distribution statistics:
      observed_{peak_rate, active_frac, cv_swap},
      null_mean_{peak_rate, active_frac, cv_swap},
      null_std_{peak_rate, active_frac, cv_swap},
      p_value_{peak_rate, active_frac, cv_swap},
      significant : bool (all three p < 0.05)
    """
    obs = classify_rank_adjacent_pair_swap(
        perm_rank_to_ref, morph_proxy, rank_i,
        window=window, subject_mask=subject_mask,
    )
    obs_peak = obs.get("peak_rate", 0.0)
    obs_active = obs.get("active_frac", 0.0)
    obs_cv = obs.get("cv_swap", 0.0)

    rng = np.random.default_rng(seed)
    null_peak = np.empty(n_perm)
    null_active = np.empty(n_perm)
    null_cv = np.empty(n_perm)

    for k in range(n_perm):
        shuffled_proxy = morph_proxy.copy()
        rng.shuffle(shuffled_proxy)
        r = classify_rank_adjacent_pair_swap(
            perm_rank_to_ref, shuffled_proxy, rank_i,
            window=window, subject_mask=subject_mask,
        )
        null_peak[k] = r.get("peak_rate", 0.0)
        null_active[k] = r.get("active_frac", 0.0)
        null_cv[k] = r.get("cv_swap", 0.0)

    def _p(obs_val: float, null_arr: np.ndarray) -> float:
        return float((np.sum(null_arr >= obs_val) + 1) / (n_perm + 1))

    p_peak = _p(obs_peak, null_peak)
    p_active = _p(obs_active, null_active)
    p_cv = _p(obs_cv, null_cv)

    return {
        "rank_pair": obs["rank_pair"],
        "ref_pair": obs.get("ref_pair"),
        "observed_peak_rate": obs_peak,
        "observed_active_frac": obs_active,
        "observed_cv_swap": obs_cv,
        "null_mean_peak_rate": float(null_peak.mean()),
        "null_std_peak_rate": float(null_peak.std()),
        "null_mean_active_frac": float(null_active.mean()),
        "null_std_active_frac": float(null_active.std()),
        "null_mean_cv_swap": float(null_cv.mean()),
        "null_std_cv_swap": float(null_cv.std()),
        "p_peak_rate": p_peak,
        "p_active_frac": p_active,
        "p_cv_swap": p_cv,
        "n_perm": n_perm,
        "significant": (p_peak < 0.05) and (p_active < 0.05) and (p_cv < 0.05),
    }


def analyze_swap_patterns(
    perm_rank_to_ref: np.ndarray,
    modes: list[int],
    *,
    subject_mask: np.ndarray | None = None,
) -> dict:
    """Analyze reference-label shuffling inside a *rank* mode block."""
    perm = np.asarray(perm_rank_to_ref, dtype=int)
    n_subj, n_modes = perm.shape

    modes_arr = np.asarray(modes, dtype=int)
    _empty = {
        "identity_rate": 0.0, "any_swap_rate": 0.0,
        "pairwise_swap_rates": {},
        "per_mode_swap_rate": {int(m): 0.0 for m in modes_arr.tolist()},
        "failed_rate": 0.0,
    }
    if modes_arr.size == 0:
        return _empty
    if np.any(modes_arr < 0) or np.any(modes_arr >= n_modes):
        raise ValueError(f"Rank index out of range in modes={modes}")

    if subject_mask is None:
        subject_mask = np.ones(n_subj, dtype=bool)
    subject_mask = np.asarray(subject_mask, dtype=bool)
    if subject_mask.shape != (n_subj,):
        raise ValueError("subject_mask shape mismatch")
    n_mask = int(subject_mask.sum())
    if n_mask == 0:
        return _empty

    block = perm[subject_mask][:, modes_arr]
    failed = np.any(block < 0, axis=1)
    failed_rate = float(np.mean(failed))

    paired = ~failed
    if not np.any(paired):
        _empty["failed_rate"] = failed_rate
        return _empty

    block_p = block[paired]
    sets = [tuple(sorted(map(int, row.tolist()))) for row in block_p]
    (dom_set, _) = Counter(sets).most_common(1)[0]

    in_dom_set = np.array(
        [tuple(sorted(map(int, row.tolist()))) == dom_set for row in block_p],
        dtype=bool,
    )
    if not np.any(in_dom_set):
        _empty["failed_rate"] = failed_rate
        return _empty

    block_dom = block_p[in_dom_set]
    orders = [tuple(map(int, row.tolist())) for row in block_dom]
    (dom_order, _) = Counter(orders).most_common(1)[0]

    identity = np.array(
        [tuple(map(int, row.tolist())) == dom_order for row in block_dom],
        dtype=bool,
    )
    n_dom = int(block_dom.shape[0])
    identity_rate = float(identity.sum() / n_mask)
    any_swap_rate = float((n_dom - int(identity.sum())) / n_mask)

    dom_order_arr = np.asarray(dom_order, dtype=int)
    per_mode_swap_rate: dict[int, float] = {}
    for local_k, rank_idx in enumerate(modes_arr.tolist()):
        moved = np.zeros(n_mask, dtype=bool)
        moved_paired = paired.copy()
        moved_paired[paired] = block_p[:, local_k] != dom_order_arr[local_k]
        moved[paired] = moved_paired[paired]
        per_mode_swap_rate[int(rank_idx)] = float(moved.mean())

    pair_swap_counts: dict[tuple[int, int], int] = {}
    if n_dom >= 1 and modes_arr.size >= 2:
        label_to_pos0 = {
            int(lbl): int(pos) for pos, lbl in enumerate(dom_order_arr.tolist())
        }
        for row in block_dom:
            pos = np.array(
                [label_to_pos0[int(lbl)] for lbl in row.tolist()], dtype=int,
            )
            pi = np.empty_like(pos)
            pi[pos] = np.arange(pos.size, dtype=int)
            moved = np.flatnonzero(pi != np.arange(pi.size))
            if moved.size == 2:
                i0, j0 = int(moved[0]), int(moved[1])
                if pi[i0] == j0 and pi[j0] == i0:
                    key = (int(modes_arr[i0]), int(modes_arr[j0]))
                    pair_swap_counts[key] = pair_swap_counts.get(key, 0) + 1

    pairwise_swap_rates = (
        {k: v / n_dom for k, v in pair_swap_counts.items()} if n_dom > 0 else {}
    )

    return {
        "identity_rate": identity_rate,
        "any_swap_rate": any_swap_rate,
        "pairwise_swap_rates": pairwise_swap_rates,
        "per_mode_swap_rate": per_mode_swap_rate,
        "failed_rate": failed_rate,
    }


def analyze_cluster_permutation_structure(
    perm_rank_to_ref: np.ndarray,
    cluster_modes: list[int],
    *,
    subject_mask: np.ndarray | None = None,
) -> dict[str, float]:
    """Summarize reference-label permutation structure inside a rank mode block.

    Returns rates: valid_all_rate, within_block_rate, identity_rate,
    transposition_rate, cycle_gt2_rate.
    """
    perm = np.asarray(perm_rank_to_ref, dtype=int)
    n_subj, n_modes = perm.shape

    modes_arr = np.asarray(cluster_modes, dtype=int)
    _zero = {
        "valid_all_rate": 0.0, "within_block_rate": 0.0,
        "identity_rate": 0.0, "transposition_rate": 0.0, "cycle_gt2_rate": 0.0,
    }
    if modes_arr.size == 0:
        return _zero
    if np.any(modes_arr < 0) or np.any(modes_arr >= n_modes):
        raise ValueError(f"Rank index out of range in cluster_modes={cluster_modes}")

    if subject_mask is None:
        subject_mask = np.ones(n_subj, dtype=bool)
    subject_mask = np.asarray(subject_mask, dtype=bool)
    if subject_mask.shape != (n_subj,):
        raise ValueError("subject_mask shape mismatch")
    if not subject_mask.any():
        return _zero

    block = perm[subject_mask][:, modes_arr]

    valid_all = np.all(block >= 0, axis=1)
    valid_all_rate = float(np.mean(valid_all))
    if not np.any(valid_all):
        return {**_zero, "valid_all_rate": valid_all_rate}

    block_v = block[valid_all]
    sets = [tuple(sorted(map(int, row.tolist()))) for row in block_v]
    (dom_set, _) = Counter(sets).most_common(1)[0]
    in_dom_set = np.array(
        [tuple(sorted(map(int, row.tolist()))) == dom_set for row in block_v],
        dtype=bool,
    )
    within_block_rate = (
        float(np.mean(valid_all)) * float(np.mean(in_dom_set))
        if block_v.shape[0] else 0.0
    )

    if not np.any(in_dom_set):
        return {**_zero, "valid_all_rate": valid_all_rate, "within_block_rate": within_block_rate}

    block_dom = block_v[in_dom_set]
    orders = [tuple(map(int, row.tolist())) for row in block_dom]
    (dom_order, _) = Counter(orders).most_common(1)[0]
    dom_order_arr = np.asarray(dom_order, dtype=int)

    n_mask = int(subject_mask.sum())
    identity = np.array(
        [tuple(map(int, row.tolist())) == dom_order for row in block_dom],
        dtype=bool,
    )
    identity_rate = float(identity.sum() / n_mask)

    within_n = int(block_dom.shape[0])
    if within_n == 0:
        return {
            "valid_all_rate": valid_all_rate,
            "within_block_rate": within_block_rate,
            "identity_rate": identity_rate,
            "transposition_rate": 0.0,
            "cycle_gt2_rate": 0.0,
        }

    label_to_pos0 = {
        int(lbl): int(pos) for pos, lbl in enumerate(dom_order_arr.tolist())
    }
    transposition = 0
    cycle_gt2 = 0

    for row in block_dom:
        pos = np.array(
            [label_to_pos0[int(lbl)] for lbl in row.tolist()], dtype=int,
        )
        pi = np.empty_like(pos)
        pi[pos] = np.arange(pos.size, dtype=int)

        moved = np.flatnonzero(pi != np.arange(pi.size))
        if moved.size == 0:
            continue
        if moved.size == 2:
            i0, j0 = int(moved[0]), int(moved[1])
            if pi[i0] == j0 and pi[j0] == i0:
                transposition += 1
                continue

        visited = np.zeros(pi.size, dtype=bool)
        has_gt2 = False
        for start in range(pi.size):
            if visited[start]:
                continue
            cur = start
            cyc_len = 0
            while not visited[cur]:
                visited[cur] = True
                cyc_len += 1
                cur = int(pi[cur])
            if cyc_len > 2 and cur == start:
                has_gt2 = True
                break
        if has_gt2:
            cycle_gt2 += 1

    return {
        "valid_all_rate": valid_all_rate,
        "within_block_rate": within_block_rate,
        "identity_rate": identity_rate,
        "transposition_rate": float(transposition / within_n),
        "cycle_gt2_rate": float(cycle_gt2 / within_n),
    }


# ============================================================
# Canonical inlet metrics  (Sec. D)
# ============================================================
def _idw_weights(
    tree,
    point: np.ndarray,
    k: int,
    *,
    power: float = 2.0,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Inverse-distance weights for k nearest neighbors of point."""
    dist, idx = tree.query(np.asarray(point, dtype=float), k=int(k))
    dist = np.atleast_1d(dist).astype(float, copy=False)
    idx = np.atleast_1d(idx).astype(int, copy=False)
    w = 1.0 / np.maximum(dist, float(eps)) ** float(power)
    w_sum = float(np.sum(w))
    if not np.isfinite(w_sum) or w_sum <= 0.0:
        raise ValueError("Invalid IDW weight normalization")
    w /= w_sum
    return idx, w


def build_idw_measurement_vector(
    tree,
    n_nodes: int,
    e_dir: np.ndarray,
    point_pos: np.ndarray,
    point_neg: np.ndarray,
    *,
    k: int = 32,
    power: float = 2.0,
    eps: float = 1e-12,
) -> np.ndarray:
    """Dense measurement vector using IDW interpolation.

    Returns ``b`` such that ``b^T u`` approximates  e_dir · (u(pos) − u(neg)).
    """
    e = np.asarray(e_dir, dtype=float).reshape(3)
    n = int(n_nodes)
    if n <= 0:
        raise ValueError("n_nodes must be positive")
    k_eff = max(1, min(int(k), n))

    idx_pos, w_pos = _idw_weights(tree, point_pos, k_eff, power=power, eps=eps)
    idx_neg, w_neg = _idw_weights(tree, point_neg, k_eff, power=power, eps=eps)

    b = np.zeros(n * 3, dtype=float)
    for node_idx, w in zip(idx_pos, w_pos):
        base = int(node_idx) * 3
        b[base : base + 3] += e * float(w)
    for node_idx, w in zip(idx_neg, w_neg):
        base = int(node_idx) * 3
        b[base : base + 3] -= e * float(w)
    return b


def canonical_basis_in_subspace(
    Q: np.ndarray,
    b_ap: np.ndarray,
    b_ml: np.ndarray,
    M: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Canonical inlet basis inside subspace Q.

    Measurement vectors are covectors: the observable is b.T @ u.
    Thus its restriction to the M-orthonormal basis Q is Q.T @ b,
    not Q.T @ M @ b. M is retained for API compatibility; it enters
    only the construction of Q, never the covector restriction.

    Returns (ψ1, ψ2, singular_values).
    """
    c_ap = Q.T @ b_ap
    c_ml = Q.T @ b_ml
    C = np.column_stack([c_ap, c_ml])
    U, sigmas, _ = np.linalg.svd(C, full_matrices=False)
    psi1 = Q @ U[:, 0]
    psi2 = Q @ U[:, 1] if U.shape[1] > 1 else np.zeros_like(psi1)
    # Fix sign conventions: ψ₁ aligned with b_ap, ψ₂ aligned with b_ml
    if b_ap @ psi1 < 0:
        psi1 = -psi1
    if b_ml @ psi2 < 0:
        psi2 = -psi2
    return psi1, psi2, sigmas


def coupling_per_1mm_max(
    b_vec: np.ndarray,
    psi: np.ndarray,
) -> float:
    """Canonical coupling in mm/mm, normalized by max nodal displacement."""
    b_arr = np.asarray(b_vec, dtype=float)
    psi_arr = np.asarray(psi, dtype=float)
    if b_arr.ndim != 1 or psi_arr.ndim != 1:
        raise ValueError("Expected 1D measurement and mode vectors")
    if b_arr.shape != psi_arr.shape:
        raise ValueError(
            f"Shape mismatch: measurement {b_arr.shape} vs mode {psi_arr.shape}"
        )
    if psi_arr.size % 3 != 0:
        raise ValueError("Mode vector size must be divisible by 3")

    u = psi_arr.reshape(-1, 3)
    max_u = float(np.linalg.norm(u, axis=1).max())
    if max_u <= 1e-12:
        return np.nan
    return float(abs(b_arr @ psi_arr)) / max_u


def canonical_pair_couplings(
    Q: np.ndarray,
    b_primary: np.ndarray,
    b_secondary: np.ndarray,
    M: np.ndarray | None = None,
) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    """Canonical pair coupling scores for the first and second paired axes.

    Returns
    -------
    primary_per_1mm, secondary_per_1mm, psi1, psi2, sigmas
        ``psi1`` is the leading canonical direction for ``b_primary`` and
        ``psi2`` is the companion direction for ``b_secondary``.
    """
    psi1, psi2, sigmas = canonical_basis_in_subspace(
        Q, b_primary, b_secondary, M=M,
    )
    primary = coupling_per_1mm_max(b_primary, psi1)
    secondary = (
        coupling_per_1mm_max(b_secondary, psi2)
        if np.linalg.norm(psi2) > 1e-12 else np.nan
    )
    return primary, secondary, psi1, psi2, sigmas


def single_functional_coupling(
    Q: np.ndarray,
    b_vec: np.ndarray,
    M: np.ndarray | None = None,
) -> tuple[float, np.ndarray, float]:
    """Maximally coupled direction in subspace Q for a single observable.

    This is the rank-1 analogue of the paired canonical basis:
    the observable is projected into the subspace and normalized to obtain the
    unique direction in ``Q`` that maximizes that scalar functional.
    """
    c = Q.T @ b_vec
    sigma = float(np.linalg.norm(c))
    if sigma <= 1e-12:
        return np.nan, np.zeros(Q.shape[0], dtype=float), 0.0
    psi = Q @ (c / sigma)
    if b_vec @ psi < 0:
        psi = -psi
    return coupling_per_1mm_max(b_vec, psi), psi, sigma


def compute_patient_directions(
    template_points: np.ndarray,
    shapes: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    *,
    smoothing: float = 50.0,
    neighbors: int = 10,
) -> np.ndarray:
    """Compute per-patient measurement direction from deformed landmark positions.

    For each subject *s*, RBF-interpolates the displacement field
    ``shapes[s] - template_points`` at template landmark positions *p1*, *p2*
    to obtain deformed positions, then returns the unit direction vector.

    Parameters
    ----------
    template_points : (N_template, 3)
    shapes : (N_subjects, N_template, 3)
    p1, p2 : (3,) template landmark coordinates (positive / negative end)
    smoothing, neighbors : RBF parameters matching simulation config

    Returns
    -------
    directions : (N_subjects, 3)  unit vectors  e_s = (p1_s − p2_s) / ‖…‖
    """
    from scipy.interpolate import RBFInterpolator

    p1 = np.asarray(p1, dtype=float).reshape(1, 3)
    p2 = np.asarray(p2, dtype=float).reshape(1, 3)
    query = np.vstack([p1, p2])  # (2, 3)

    n_subj = shapes.shape[0]
    directions = np.empty((n_subj, 3), dtype=float)

    for s in range(n_subj):
        disp = shapes[s] - template_points  # (N_template, 3)
        rbf = RBFInterpolator(
            template_points, disp,
            smoothing=smoothing, neighbors=neighbors,
        )
        phi_at_lm = rbf(query)  # (2, 3)
        p1_s = query[0] + phi_at_lm[0]
        p2_s = query[1] + phi_at_lm[1]
        d = p1_s - p2_s
        norm = np.linalg.norm(d)
        directions[s] = d / norm if norm > 1e-12 else 0.0

    return directions


def build_patient_measurement_vectors(
    tree,
    n_nodes: int,
    template_points: np.ndarray,
    shapes: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    *,
    k: int = 32,
    power: float = 2.0,
    smoothing: float = 50.0,
    neighbors: int = 10,
) -> np.ndarray:
    """Build per-patient IDW measurement vectors with patient-specific directions.

    Uses the same RBF approach as ``ShapeMapper.apply_sample`` to deform
    landmark positions, then rebuilds the measurement vector with the
    patient-specific direction while keeping IDW weights on the template mesh.

    Parameters
    ----------
    tree : cKDTree of template FE mesh coordinates
    n_nodes : number of FE mesh nodes
    template_points : (N_template, 3) SSM template point cloud
    shapes : (N_subjects, N_template, 3) SSM shapes
    p1, p2 : (3,) template landmark positions
    k, power : IDW parameters
    smoothing, neighbors : RBF parameters matching simulation config

    Returns
    -------
    b_all : (N_subjects, n_nodes * 3)  per-patient measurement vectors
    """
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    dirs = compute_patient_directions(
        template_points, shapes, p1, p2,
        smoothing=smoothing, neighbors=neighbors,
    )
    n_subj = shapes.shape[0]
    b_all = np.empty((n_subj, n_nodes * 3), dtype=float)
    for s in range(n_subj):
        b_all[s] = build_idw_measurement_vector(
            tree, n_nodes, dirs[s], p1, p2, k=k, power=power,
        )
    return b_all


# ============================================================
# Sex / age statistical analysis
# ============================================================
def ols_sex_age_per_mode(
    eigenvalues: np.ndarray,
    sex: np.ndarray,
    age: np.ndarray,
    n_modes: int,
) -> dict[str, np.ndarray]:
    """OLS: log(λ_i) ~ sex + age  for each mode.

    Returns dict with arrays: beta_sex, beta_age, ci_sex, ci_age,
    pval_sex, pval_age.
    """
    sex_binary = (sex == "F").astype(float)
    beta_sex = np.empty(n_modes)
    beta_age = np.empty(n_modes)
    ci_sex = np.empty((n_modes, 2))
    ci_age = np.empty((n_modes, 2))
    pval_sex = np.empty(n_modes)
    pval_age = np.empty(n_modes)

    for i in range(n_modes):
        y = np.log(eigenvalues[:, i])
        X = sm.add_constant(np.column_stack([sex_binary, age]))
        res = sm.OLS(y, X).fit()
        beta_sex[i] = res.params[1]
        beta_age[i] = res.params[2]
        ci_sex[i] = res.conf_int(alpha=0.05)[1]
        ci_age[i] = res.conf_int(alpha=0.05)[2]
        pval_sex[i] = res.pvalues[1]
        pval_age[i] = res.pvalues[2]

    return {
        "beta_sex": beta_sex, "beta_age": beta_age,
        "ci_sex": ci_sex, "ci_age": ci_age,
        "pval_sex": pval_sex, "pval_age": pval_age,
    }


def swap_rate_by_sex(
    perm_full: np.ndarray,
    mask_m: np.ndarray,
    mask_f: np.ndarray,
    n_modes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-mode swap rate by sex + Fisher exact p-value."""
    swap_rate_m = np.empty(n_modes)
    swap_rate_f = np.empty(n_modes)
    pval = np.empty(n_modes)
    for m in range(n_modes):
        swapped_m = (perm_full[mask_m, m] != m) & (perm_full[mask_m, m] >= 0)
        swapped_f = (perm_full[mask_f, m] != m) & (perm_full[mask_f, m] >= 0)
        swap_rate_m[m] = swapped_m.mean()
        swap_rate_f[m] = swapped_f.mean()
        table_2x2 = [
            [int(swapped_m.sum()), int((~swapped_m).sum())],
            [int(swapped_f.sum()), int((~swapped_f).sum())],
        ]
        _, pval[m] = stats.fisher_exact(table_2x2)
    return swap_rate_m, swap_rate_f, pval


def swap_count_per_subject(
    perm_full: np.ndarray,
    n_modes: int,
) -> np.ndarray:
    """Number of swapped modes per subject."""
    identity = np.arange(n_modes)
    return np.array([
        np.sum((perm_full[s] != identity) & (perm_full[s] >= 0))
        for s in range(perm_full.shape[0])
    ], dtype=float)


def gap_sex_age_effects(
    gaps: np.ndarray,
    mask_m: np.ndarray,
    mask_f: np.ndarray,
    age: np.ndarray,
    n_modes: int,
) -> dict[str, np.ndarray]:
    """Sex (Mann–Whitney r_eff) and age (Spearman ρ) effects per gap pair."""
    n_pairs = n_modes - 1
    r_sex = np.full(n_pairs, np.nan)
    p_sex = np.full(n_pairs, np.nan)
    rho_age = np.full(n_pairs, np.nan)
    p_age = np.full(n_pairs, np.nan)

    for gi in range(n_pairs):
        g_m = gaps[mask_m, gi]
        g_f = gaps[mask_f, gi]
        g_m_fin = g_m[np.isfinite(g_m)]
        g_f_fin = g_f[np.isfinite(g_f)]
        if len(g_m_fin) > 1 and len(g_f_fin) > 1:
            u, p = stats.mannwhitneyu(g_m_fin, g_f_fin, alternative="two-sided")
            r_sex[gi] = 1 - 2 * u / (len(g_m_fin) * len(g_f_fin))
            p_sex[gi] = p
        g_all = gaps[:, gi]
        finite = np.isfinite(g_all)
        if finite.sum() > 2:
            rho_age[gi], p_age[gi] = stats.spearmanr(age[finite], g_all[finite])

    return {"r_sex": r_sex, "p_sex": p_sex, "rho_age": rho_age, "p_age": p_age}


# ============================================================
# Formatting helpers
# ============================================================
def cluster_label(cs: int, ce: int) -> str:
    """1-based label, e.g. 'Rank modes 12–13'."""
    return f"Rank modes {cs + 1}\u2013{ce}"


def format_mode_set_1based(modes_1based: tuple[int, ...]) -> str:
    """Format a sorted tuple of 1-based mode indices into a compact label."""
    if not modes_1based:
        return ""
    arr = sorted(int(x) for x in modes_1based)
    if arr == list(range(arr[0], arr[-1] + 1)):
        return f"{arr[0]}\u2013{arr[-1]}" if len(arr) > 1 else f"{arr[0]}"
    return ",".join(map(str, arr))


# ============================================================
# Cohort bootstrap robustness test
# ============================================================
def cohort_bootstrap_robustness(
    eigenvalues: np.ndarray,
    perm_rank_to_ref: np.ndarray,
    eps_in: float,
    target_clusters: list[tuple[int, int]],
    *,
    n_boot: int = 500,
    subsample_frac: float = 1.0,
    seed: int = 42,
    threshold_percentile: float | None = None,
) -> dict:
    """Subject bootstrap: verify key metrics are stable under cohort subsampling.

    Repeatedly draws ``subsample_frac`` of subjects with replacement and
    recomputes cluster prevalence, per-mode swap rates, and median gap_in for
    each target cluster.

    Parameters
    ----------
    eigenvalues : (n_subj, n_modes) sorted eigenvalues.
    perm_rank_to_ref : (n_subj, n_modes) permutation array.
    eps_in : cluster detection threshold.
    target_clusters : list of (start, end) 0-based cluster boundaries to track.
    n_boot : number of bootstrap iterations.
    subsample_frac : fraction of subjects to draw per iteration.
    seed : RNG seed.

    Returns
    -------
    dict with per-cluster and per-mode-swap bootstrap distributions:
      - ``cluster_prevalence``: {label: {mean, std, ci95_lo, ci95_hi, observed}}
      - ``swap_rates``: {mode_1b: {mean, std, ci95_lo, ci95_hi, observed}}
      - ``median_gaps``: {pair_label: {mean, std, ci95_lo, ci95_hi, observed}}
    """
    n_subj, n_modes = eigenvalues.shape
    n_draw = max(2, int(n_subj * subsample_frac))
    rng = np.random.default_rng(seed)

    # observed values
    gaps_full = compute_gap_in(eigenvalues)
    clusters_all = detect_clusters(eigenvalues, eps_in)

    obs_prev: dict[str, float] = {}
    for cs, ce in target_clusters:
        lbl = cluster_label(cs, ce)
        obs_prev[lbl] = float(np.mean([(cs, ce) in cl for cl in clusters_all]))

    obs_swap = np.zeros(n_modes)
    for s in range(n_subj):
        for m in range(n_modes):
            if perm_rank_to_ref[s, m] != m and perm_rank_to_ref[s, m] >= 0:
                obs_swap[m] += 1
    obs_swap /= n_subj

    obs_med_gaps: dict[str, float] = {}
    for gi in range(n_modes - 1):
        plbl = f"{gi + 1}\u2013{gi + 2}"
        obs_med_gaps[plbl] = float(np.nanmedian(gaps_full[:, gi]))

    # bootstrap
    boot_prev = {cluster_label(cs, ce): np.empty(n_boot) for cs, ce in target_clusters}
    boot_swap = np.empty((n_boot, n_modes))
    boot_med_gaps = np.empty((n_boot, n_modes - 1))

    for b in range(n_boot):
        idx = rng.choice(n_subj, size=n_draw, replace=True)
        ev_b = eigenvalues[idx]
        perm_b = perm_rank_to_ref[idx]

        threshold_b = (float(np.nanpercentile(compute_gap_in(ev_b), threshold_percentile))
                       if threshold_percentile is not None else eps_in)
        cl_b = detect_clusters(ev_b, threshold_b)
        for cs, ce in target_clusters:
            lbl = cluster_label(cs, ce)
            boot_prev[lbl][b] = float(np.mean([(cs, ce) in cl for cl in cl_b]))

        for m in range(n_modes):
            boot_swap[b, m] = float(np.mean((perm_b[:, m] != m) & (perm_b[:, m] >= 0)))

        g_b = compute_gap_in(ev_b)
        for gi in range(n_modes - 1):
            boot_med_gaps[b, gi] = float(np.nanmedian(g_b[:, gi]))

    def _summary(arr: np.ndarray, obs: float) -> dict:
        return {
            "observed": obs,
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "ci95_lo": float(np.percentile(arr, 2.5)),
            "ci95_hi": float(np.percentile(arr, 97.5)),
        }

    result_prev = {lbl: _summary(boot_prev[lbl], obs_prev[lbl]) for lbl in boot_prev}
    result_swap = {
        m + 1: _summary(boot_swap[:, m], obs_swap[m]) for m in range(n_modes)
    }
    result_gaps = {
        f"{gi + 1}\u2013{gi + 2}": _summary(boot_med_gaps[:, gi], obs_med_gaps[f"{gi + 1}\u2013{gi + 2}"])
        for gi in range(n_modes - 1)
    }

    return {
        "cluster_prevalence": result_prev,
        "swap_rates": result_swap,
        "median_gaps": result_gaps,
        "n_boot": n_boot,
        "subsample_frac": subsample_frac,
        "n_subjects": n_subj,
        "n_draw": n_draw,
    }


def cohort_ordering_permutation_test(
    eigenvalues: np.ndarray,
    perm_rank_to_ref: np.ndarray,
    eps_in: float,
    target_clusters: list[tuple[int, int]],
    *,
    n_perm: int = 1000,
    seed: int = 42,
) -> dict:
    """Test whether results depend on dataset ordering.

    Randomly permutes the subject axis and recomputes cluster prevalence and
    swap rates. Since these metrics are permutation-invariant by construction,
    we expect zero variance. This test formally verifies that invariance and
    serves as a sanity check against ordering-dependent bugs.

    Parameters
    ----------
    eigenvalues : (n_subj, n_modes)
    perm_rank_to_ref : (n_subj, n_modes)
    eps_in : cluster threshold
    target_clusters : clusters to track
    n_perm : number of permutations
    seed : RNG seed

    Returns
    -------
    dict with ``max_prevalence_deviation``, ``max_swap_deviation``, ``passed`` (bool)
    """
    n_subj, n_modes = eigenvalues.shape
    rng = np.random.default_rng(seed)

    # reference values
    clusters_ref = detect_clusters(eigenvalues, eps_in)
    ref_prev = {}
    for cs, ce in target_clusters:
        lbl = cluster_label(cs, ce)
        ref_prev[lbl] = float(np.mean([(cs, ce) in cl for cl in clusters_ref]))

    ref_swap = np.array([float(np.mean((perm_rank_to_ref[:, m] != m) & (perm_rank_to_ref[:, m] >= 0))) for m in range(n_modes)])

    max_prev_dev = 0.0
    max_swap_dev = 0.0

    for _ in range(n_perm):
        idx = rng.permutation(n_subj)
        ev_p = eigenvalues[idx]
        perm_p = perm_rank_to_ref[idx]

        cl_p = detect_clusters(ev_p, eps_in)
        for cs, ce in target_clusters:
            lbl = cluster_label(cs, ce)
            val = float(np.mean([(cs, ce) in cl for cl in cl_p]))
            max_prev_dev = max(max_prev_dev, abs(val - ref_prev[lbl]))

        swap_p = np.array([float(np.mean((perm_p[:, m] != m) & (perm_p[:, m] >= 0))) for m in range(n_modes)])
        max_swap_dev = max(max_swap_dev, float(np.max(np.abs(swap_p - ref_swap))))

    return {
        "max_prevalence_deviation": max_prev_dev,
        "max_swap_deviation": max_swap_dev,
        "n_perm": n_perm,
        "passed": max_prev_dev < 1e-12 and max_swap_dev < 1e-12,
    }
