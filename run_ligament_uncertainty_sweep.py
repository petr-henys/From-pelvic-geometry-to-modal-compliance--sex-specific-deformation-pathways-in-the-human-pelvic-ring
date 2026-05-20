#!/usr/bin/env python3
"""Ligament uncertainty propagation via hierarchical Latin Hypercube sampling.

Models correlated uncertainty in ligament stiffness and pretension using a
three-level hierarchical factor model in log-space:

    log(p_ij) = log(p_ij^baseline)
              + σ_global · z_global        (shared laxity across all bundles)
              + σ_type   · z_type_i        (per-ligament-type residual)
              + σ_asym   · δ_j · z_asym_i  (left–right asymmetry)

Levels
------
0 — Global laxity  : single shared factor for ALL bundles (1 latent var each
    for stiffness and pretension, cross-correlated via ρ_kT).
1 — Per-type       : independent residual per ligament type (6 types).
2 — Asymmetry      : left–right ratio per bilateral pair (5 bilateral types).

Total latent dimension:  1 + 1 + 6 + 5 + 6 + 5 = 24
                        (global_k, global_T, type_k×6, asym_k×5, type_T×6, asym_T×5)

Implied correlations (stiffness, with default σ values):
    Same type, opposite sides:  ρ ≈ 0.82
    Different types:            ρ ≈ 0.55
    Stiffness ↔ Pretension:    ρ ≈ 0.50  (global level)

Sampling uses a Latin Hypercube in the 24-D standard-normal space, giving
N = 300 representative runs instead of the infeasible full-factorial grid.

Usage:
    python run_ligament_uncertainty_sweep.py
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, qmc

sys.path.insert(0, str(Path(__file__).parent))

from config import SimulationConfig
from logging_config import get_logger
from parametrizer import Parametrizer, ParameterSweep

logger = get_logger("UncertaintySweep")

# ---------------------------------------------------------------------------
# Ligament topology
# ---------------------------------------------------------------------------

LIGAMENT_TYPES: list[str] = [
    "symphisys",
    "anterior_SIJ",
    "posterior_SIJ",
    "INL",
    "SS",
    "ST",
]

TYPE_TO_BUNDLES: dict[str, list[str]] = {
    "symphisys":     ["symphisys"],
    "anterior_SIJ":  ["right_anterior_SIJ_ligaments", "left_anterior_SIJ_ligaments"],
    "posterior_SIJ": ["right_posterior_SIJ_ligaments", "left_posterior_SIJ_ligaments"],
    "INL":           ["right_INL", "left_INL"],
    "SS":            ["right_SS", "left_SS"],
    "ST":            ["right_ST", "left_ST"],
}

BILATERAL_TYPES: list[str] = [t for t in LIGAMENT_TYPES if t != "symphisys"]

# Side sign: +1 right, −1 left, 0 midline
BUNDLE_SIDE_SIGN: dict[str, float] = {}
for _type, _bundles in TYPE_TO_BUNDLES.items():
    if len(_bundles) == 1:
        BUNDLE_SIDE_SIGN[_bundles[0]] = 0.0
    else:
        BUNDLE_SIDE_SIGN[_bundles[0]] = +1.0   # right
        BUNDLE_SIDE_SIGN[_bundles[1]] = -1.0   # left

N_TYPES = len(LIGAMENT_TYPES)         # 6
N_BILATERAL = len(BILATERAL_TYPES)    # 5

# All 11 bundle names (ordered by type, then right-before-left)
ALL_BUNDLES: list[str] = [b for bundles in TYPE_TO_BUNDLES.values() for b in bundles]

# ---------------------------------------------------------------------------
# Latent space layout  (total = 24)
# ---------------------------------------------------------------------------
#  Index  Meaning
#  -----  --------------------------------------------------------
#  0      global stiffness factor  z_gk
#  1      global pretension factor z_gT  (independent; mixed with
#                                         z_gk via ρ_kT post-hoc)
#  2..7   per-type stiffness residuals    (6 types)
#  8..12  bilateral stiffness asymmetry   (5 bilateral types)
#  13..18 per-type pretension residuals   (6 types)
#  19..23 bilateral pretension asymmetry  (5 bilateral types)
# ---------------------------------------------------------------------------
N_LATENT = 24

IDX_GLOBAL_K = 0
IDX_GLOBAL_T = 1
IDX_TYPE_K = slice(2, 8)
IDX_ASYM_K = slice(8, 13)
IDX_TYPE_T = slice(13, 19)
IDX_ASYM_T = slice(19, 24)

_MAX_LOGVAR_FOR_EXP = 700.0


# ---------------------------------------------------------------------------
# Hierarchical uncertainty specification
# ---------------------------------------------------------------------------

@dataclass
class UncertaintySpec:
    """Hyperparameters for hierarchical ligament uncertainty model.

    All σ values are log-space standard deviations.  For small σ the
    coefficient of variation in physical space is approximately σ.

    Default values are consistent with the literature on pelvic-ligament
    property scatter (~30–50 % interindividual CoV, ~10–15 % L/R asymmetry;
    Hammer et al. 2009, Kibsgård et al. 2014).

    Implied total CoV per bundle (stiffness):
        √(σ_g² + σ_t² + σ_a²) ≈ 0.34  →  CoV ≈ 34 %

    Implied correlations (stiffness):
        Same type, opposite sides:  ρ = (σ_g² + σ_t² − σ_a²) / σ_total² ≈ 0.82
        Different types:            ρ = σ_g² / σ_total²                  ≈ 0.55
    """

    # — Stiffness σ (log-space) —
    sigma_global_k: float = 0.25   # shared laxity
    sigma_type_k: float = 0.20     # per-type residual
    sigma_asym_k: float = 0.10     # L/R asymmetry

    # — Pretension σ (log-space) —
    sigma_global_T: float = 0.35   # shared pretension shift
    sigma_type_T: float = 0.25     # per-type residual
    sigma_asym_T: float = 0.12     # L/R asymmetry

    # — Cross-correlation (global level) —
    rho_kT: float = 0.50           # fraction of global factor shared k ↔ T

    # — Sampling —
    n_samples: int = 300
    seed: int = 42

    def total_sigma_k(self) -> float:
        """Total log-space standard deviation for stiffness."""
        return float(np.sqrt(
            self.sigma_global_k ** 2
            + self.sigma_type_k ** 2
            + self.sigma_asym_k ** 2
        ))

    def total_sigma_T(self) -> float:
        """Total log-space standard deviation for pretension."""
        return float(np.sqrt(
            self.sigma_global_T ** 2
            + self.sigma_type_T ** 2
            + self.sigma_asym_T ** 2
        ))

    def correlation_same_type_opposite_side(self) -> float:
        """Correlation between L/R bundles of same type (stiffness).

        Cov(R,L) = σ_g² + σ_t² − σ_a²  (asymmetry contributes negatively
        because side signs are +1 and −1).
        """
        s2 = self.total_sigma_k() ** 2
        return (self.sigma_global_k ** 2 + self.sigma_type_k ** 2 - self.sigma_asym_k ** 2) / s2

    def correlation_different_type(self) -> float:
        """Correlation between bundles of different types (stiffness)."""
        s2 = self.total_sigma_k() ** 2
        return self.sigma_global_k ** 2 / s2


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def generate_lhs_samples(spec: UncertaintySpec) -> np.ndarray:
    """Generate optimized LHS samples in the 24-D standard-normal space.

    Returns array of shape (n_samples, N_LATENT).
    """
    sampler = qmc.LatinHypercube(d=N_LATENT, seed=spec.seed, optimization="random-cd")
    unit_samples = sampler.random(n=spec.n_samples)    # (N, 24) in [0, 1]
    return norm.ppf(unit_samples)                       # → standard normal


def latent_to_ligaments(
    z: np.ndarray,
    baseline: dict[str, dict[str, float]],
    spec: UncertaintySpec,
) -> dict[str, dict[str, float]]:
    """Map a single latent vector (length 24) to a LIGAMENTS config dict.

    Parameters
    ----------
    z : 1-D array of length N_LATENT (standard-normal draws)
    baseline : LIGAMENTS dict ``{bundle_name: {stiffness, pretension}}``
    spec : uncertainty hyperparameters

    Returns
    -------
    Perturbed LIGAMENTS dict with same structure as *baseline*.
    """
    # Unpack latent factors
    z_gk      = z[IDX_GLOBAL_K]        # scalar
    z_gT_raw  = z[IDX_GLOBAL_T]        # scalar (independent)
    z_type_k  = z[IDX_TYPE_K]          # (6,)
    z_asym_k  = z[IDX_ASYM_K]          # (5,)
    z_type_T  = z[IDX_TYPE_T]          # (6,)
    z_asym_T  = z[IDX_ASYM_T]          # (5,)

    # Mix global pretension with global stiffness to induce cross-correlation
    rho = spec.rho_kT
    z_gT = rho * z_gk + np.sqrt(1.0 - rho ** 2) * z_gT_raw

    result: dict[str, dict[str, float]] = {}

    for type_idx, lig_type in enumerate(LIGAMENT_TYPES):
        bilateral_idx = (
            BILATERAL_TYPES.index(lig_type)
            if lig_type in BILATERAL_TYPES
            else -1
        )

        for bundle in TYPE_TO_BUNDLES[lig_type]:
            base_k = baseline[bundle]["stiffness"]
            base_T = baseline[bundle]["pretension"]
            side = BUNDLE_SIDE_SIGN[bundle]

            # Log-space stiffness perturbation
            log_dk = spec.sigma_global_k * z_gk + spec.sigma_type_k * z_type_k[type_idx]
            if bilateral_idx >= 0:
                log_dk += spec.sigma_asym_k * side * z_asym_k[bilateral_idx]

            # Log-space pretension perturbation
            log_dT = spec.sigma_global_T * z_gT + spec.sigma_type_T * z_type_T[type_idx]
            if bilateral_idx >= 0:
                log_dT += spec.sigma_asym_T * side * z_asym_T[bilateral_idx]

            result[bundle] = {
                "stiffness": float(base_k * np.exp(log_dk)),
                "pretension": float(base_T * np.exp(log_dT)),
            }

    return result


def build_sample_configs(
    spec: UncertaintySpec,
    baseline_ligaments: dict[str, dict[str, float]],
) -> tuple[list[dict[str, dict[str, float]]], np.ndarray]:
    """Generate all LHS samples as LIGAMENTS config dicts.

    Returns
    -------
    configs : list[dict]
        List of LIGAMENTS dicts (length n_samples).
    latent_samples : ndarray, shape (n_samples, 24)
        Latent draws for post-hoc variance decomposition.
    """
    Z = generate_lhs_samples(spec)
    configs = [latent_to_ligaments(Z[i], baseline_ligaments, spec) for i in range(spec.n_samples)]
    return configs, Z


# ---------------------------------------------------------------------------
# First-order analytical propagation (free — no new simulations)
# ---------------------------------------------------------------------------

def build_input_covariance(spec: UncertaintySpec, bundle_names: list[str]) -> np.ndarray:
    """Build the 22×22 covariance matrix of [log k₁…k₁₁, log T₁…T₁₁].

    The matrix encodes all three hierarchical levels plus the stiffness–
    pretension cross-correlation.

    Parameters
    ----------
    spec : uncertainty hyperparameters
    bundle_names : ordered list of 11 bundle names (must match sensitivity columns)

    Returns
    -------
    Σ : ndarray, shape (22, 22)
    """
    n = len(bundle_names)
    Sigma = np.zeros((2 * n, 2 * n))

    # Helper: determine type index and bilateral flag for a bundle
    bundle_info: list[tuple[int, int, float]] = []  # (type_idx, bilateral_idx, side)
    for bname in bundle_names:
        for type_idx, lig_type in enumerate(LIGAMENT_TYPES):
            if bname in TYPE_TO_BUNDLES[lig_type]:
                bilateral_idx = (
                    BILATERAL_TYPES.index(lig_type)
                    if lig_type in BILATERAL_TYPES
                    else -1
                )
                side = BUNDLE_SIDE_SIGN[bname]
                bundle_info.append((type_idx, bilateral_idx, side))
                break

    # Fill blocks: Σ_kk (top-left), Σ_TT (bottom-right), Σ_kT (off-diagonal)
    for a in range(n):
        ta, ba, sa = bundle_info[a]
        for b in range(n):
            tb, bb, sb = bundle_info[b]

            same_type = ta == tb
            # — Σ_kk block —
            cov_kk = spec.sigma_global_k ** 2
            if same_type:
                cov_kk += spec.sigma_type_k ** 2
                if ba >= 0 and bb >= 0 and ba == bb:
                    cov_kk += spec.sigma_asym_k ** 2 * sa * sb
            Sigma[a, b] = cov_kk

            # — Σ_TT block —
            # Global pretension variance includes the mixed component:
            # Var(z_gT) = ρ²·Var(z_gk) + (1−ρ²)·Var(z_gT_raw) = 1
            # Cov(z_gT_a, z_gT_b) = 1 (same factor) → σ_gT² contribution
            cov_TT = spec.sigma_global_T ** 2
            if same_type:
                cov_TT += spec.sigma_type_T ** 2
                if ba >= 0 and bb >= 0 and ba == bb:
                    cov_TT += spec.sigma_asym_T ** 2 * sa * sb
            Sigma[n + a, n + b] = cov_TT

            # — Σ_kT cross-block —
            # Cov(log k_a, log T_b) = σ_gk · σ_gT · Cov(z_gk, z_gT)
            # where Cov(z_gk, z_gT) = ρ_kT (from the mixing formula)
            cov_kT = spec.sigma_global_k * spec.sigma_global_T * spec.rho_kT
            # Per-type and asymmetry are independent between k and T
            Sigma[a, n + b] = cov_kT
            Sigma[n + b, a] = cov_kT

    return Sigma


def first_order_propagation(
    stiffness_sens: np.ndarray,
    pretension_sens: np.ndarray,
    eigenvalues: np.ndarray,
    stiffness_baseline: np.ndarray,
    pretension_baseline: np.ndarray,
    bundle_names: list[str],
    spec: UncertaintySpec,
) -> pd.DataFrame:
    """First-order (Taylor) propagation of ligament uncertainty to eigenvalues.

    Uses the dimensionless sensitivity J = (p/λ) · ∂λ/∂p and returns
    Var(log λ_m) = J_m^T Σ J_m for each mode, with variance decomposition
    by hierarchical level.

    Parameters
    ----------
    stiffness_sens : (n_modes, n_lig) raw sensitivities ∂λ/∂k
    pretension_sens : (n_modes, n_lig) raw sensitivities ∂λ/∂T
    eigenvalues, stiffness_baseline, pretension_baseline : reference values
    bundle_names : ordered ligament names matching sensitivity columns
    spec : uncertainty hyperparameters

    Returns
    -------
    DataFrame with columns: mode, eigenvalue, var_log_lambda,
        std_log_lambda, cov_total, var_global, var_type, var_asym,
        frac_global, frac_type, frac_asym
    """
    n_modes = eigenvalues.size

    def _cov_from_log_variance(var_log: float) -> float:
        """Convert Var(log λ) -> CoV(λ) robustly without overflow warnings."""
        if not np.isfinite(var_log):
            return float("nan")
        clipped = max(var_log, 0.0)
        if clipped >= _MAX_LOGVAR_FOR_EXP:
            return float("inf")
        return float(np.sqrt(np.expm1(clipped)))

    # Dimensionless Jacobian: J_k = (k/λ)·∂λ/∂k, J_T = (T/λ)·∂λ/∂T
    MIN_EIG = 1e-12
    safe_eig = np.where(np.abs(eigenvalues) < MIN_EIG, np.nan, eigenvalues)
    J_k = stiffness_sens * (stiffness_baseline[np.newaxis, :] / safe_eig[:, np.newaxis])
    J_T = pretension_sens * (pretension_baseline[np.newaxis, :] / safe_eig[:, np.newaxis])

    # Full Jacobian per mode: (22,) vector = [J_k_1,..,J_k_11, J_T_1,..,J_T_11]
    Sigma = build_input_covariance(spec, bundle_names)

    # Build sub-covariance matrices for variance decomposition
    # Level 0: global only
    spec_global = UncertaintySpec(
        sigma_global_k=spec.sigma_global_k, sigma_type_k=0.0, sigma_asym_k=0.0,
        sigma_global_T=spec.sigma_global_T, sigma_type_T=0.0, sigma_asym_T=0.0,
        rho_kT=spec.rho_kT,
    )
    Sigma_global = build_input_covariance(spec_global, bundle_names)

    # Level 1: global + type (no asymmetry)
    spec_no_asym = UncertaintySpec(
        sigma_global_k=spec.sigma_global_k, sigma_type_k=spec.sigma_type_k, sigma_asym_k=0.0,
        sigma_global_T=spec.sigma_global_T, sigma_type_T=spec.sigma_type_T, sigma_asym_T=0.0,
        rho_kT=spec.rho_kT,
    )
    Sigma_no_asym = build_input_covariance(spec_no_asym, bundle_names)

    records: list[dict] = []
    for m in range(n_modes):
        J_m = np.concatenate([J_k[m], J_T[m]])   # (22,)
        if np.any(np.isnan(J_m)):
            continue

        var_total = float(J_m @ Sigma @ J_m)
        var_global = float(J_m @ Sigma_global @ J_m)
        var_no_asym = float(J_m @ Sigma_no_asym @ J_m)
        var_type = var_no_asym - var_global
        var_asym = var_total - var_no_asym

        records.append({
            "mode": m + 1,
            "eigenvalue": float(eigenvalues[m]),
            "var_log_lambda": var_total,
            "std_log_lambda": float(np.sqrt(max(var_total, 0.0))),
            "CoV_lambda": _cov_from_log_variance(var_total),
            "var_global": var_global,
            "var_type": var_type,
            "var_asym": var_asym,
            "frac_global": var_global / var_total if var_total > 0 else np.nan,
            "frac_type": var_type / var_total if var_total > 0 else np.nan,
            "frac_asym": var_asym / var_total if var_total > 0 else np.nan,
        })

    return pd.DataFrame.from_records(records)


# ---------------------------------------------------------------------------
# Simulation runner
# ---------------------------------------------------------------------------

def _run_simulation(param_point: dict, output_path: Path, base_config: dict) -> None:
    """Run single simulation with given parameters."""
    from utils.sweep_runner import run_sweep_simulation
    run_sweep_simulation(param_point, output_path, base_config, compute_sensitivities=False)


# ---------------------------------------------------------------------------
# Physical-parameter export
# ---------------------------------------------------------------------------

def export_physical_samples(
    configs: list[dict[str, dict[str, float]]],
    output_path: Path,
) -> None:
    """Write per-sample physical parameter values to CSV for post-processing."""
    records = []
    for i, lig_cfg in enumerate(configs):
        row: dict[str, float | int] = {"sample": i}
        for bundle, props in lig_cfg.items():
            row[f"{bundle}_stiffness"] = props["stiffness"]
            row[f"{bundle}_pretension"] = props["pretension"]
        records.append(row)
    pd.DataFrame(records).to_csv(output_path, index=False)


# ---------------------------------------------------------------------------
# First-order analysis from existing reference solution
# ---------------------------------------------------------------------------

def run_first_order_analysis(
    reference_path: Path,
    metadata_path: Path,
    spec: UncertaintySpec,
    output_dir: Path,
) -> pd.DataFrame | None:
    """Run analytical first-order propagation using saved reference sensitivities.

    Returns the variance-decomposition DataFrame, or None if input files are
    missing.
    """
    if not reference_path.exists() or not metadata_path.exists():
        logger.warning("Reference solution not found — skipping first-order analysis")
        return None

    with np.load(reference_path) as archive:
        eigenvalues = np.asarray(archive["eigenvalues"], dtype=float)
        stiffness_sens = np.asarray(archive["ligament_sensitivities"], dtype=float)
        pretension_sens = np.asarray(archive["ligament_pretension_sensitivities"], dtype=float)
        bundle_names = [str(n) for n in archive["ligament_names"]]
        pretension_baseline = np.asarray(archive["ligament_pretensions"], dtype=float)

    with metadata_path.open() as f:
        metadata = json.load(f)
    config = metadata.get("config", {})
    ligaments_dict = config.get("LIGAMENTS", {})
    stiffness_baseline = np.array(
        [ligaments_dict[name]["stiffness"] for name in bundle_names], dtype=float
    )

    df = first_order_propagation(
        stiffness_sens, pretension_sens,
        eigenvalues, stiffness_baseline, pretension_baseline,
        bundle_names, spec,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "first_order_variance_decomposition.csv", index=False)
    logger.info(f"First-order variance decomposition → {output_dir}")
    logger.info("Mode-level CoV (eigenvalue):")
    for _, row in df.iterrows():
        logger.info(
            f"  Mode {int(row['mode']):2d}: CoV = {row['CoV_lambda']:.1%}"
            f"  (global {row['frac_global']:.0%} | type {row['frac_type']:.0%}"
            f" | asym {row['frac_asym']:.0%})"
        )
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    BASE_CONFIG = Path("results/ref_S1P_fixed_new2/simulation_metadata.json")
    RESULTS_DIR = Path("results/ref_S1P_fixed_new2")
    OUTPUT_DIR = Path("results/ligament_uncertainty_sweep")

    # Load config from metadata since simulation_config.json is missing
    with BASE_CONFIG.open() as f:
        metadata = json.load(f)
    base_config_dict = metadata.get("config", {})
    
    cfg = SimulationConfig(base_config_dict)
    baseline_ligaments: dict[str, dict[str, float]] = cfg["LIGAMENTS"]

    spec = UncertaintySpec(n_samples=300, seed=42)

    # --- Diagnostics ---
    logger.info(f"Latent dimension: {N_LATENT}")
    logger.info(f"LHS samples: {spec.n_samples}")
    logger.info(
        f"Stiffness σ  — global={spec.sigma_global_k:.2f}  "
        f"type={spec.sigma_type_k:.2f}  asym={spec.sigma_asym_k:.2f}  "
        f"total={spec.total_sigma_k():.2f}"
    )
    logger.info(
        f"Pretension σ — global={spec.sigma_global_T:.2f}  "
        f"type={spec.sigma_type_T:.2f}  asym={spec.sigma_asym_T:.2f}  "
        f"total={spec.total_sigma_T():.2f}"
    )
    logger.info(f"ρ(k,T) = {spec.rho_kT:.2f}")
    logger.info(
        f"ρ(same type, L–R) = {spec.correlation_same_type_opposite_side():.2f}"
    )
    logger.info(f"ρ(different types)  = {spec.correlation_different_type():.2f}")

    # --- First-order analytical propagation (free) ---
    reference_path = RESULTS_DIR / "reference_solution.npz"
    metadata_path = RESULTS_DIR / "simulation_metadata.json"
    fo_df = run_first_order_analysis(reference_path, metadata_path, spec, OUTPUT_DIR)

    # --- Generate LHS samples ---
    ligament_configs, latent_samples = build_sample_configs(spec, baseline_ligaments)

    # --- Save sampling artefacts ---
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT_DIR / "latent_samples.npy", latent_samples)
    with open(OUTPUT_DIR / "uncertainty_spec.json", "w") as f:
        json.dump(asdict(spec), f, indent=2)
    export_physical_samples(ligament_configs, OUTPUT_DIR / "sample_parameters.csv")

    logger.info("Saved: latent_samples.npy, uncertainty_spec.json, sample_parameters.csv")

    # --- Run Monte Carlo sweep ---
    params: dict[str, list] = {"LIGAMENTS": ligament_configs}

    sweep = ParameterSweep(params, OUTPUT_DIR)
    parametrizer = Parametrizer(
        sweep,
        lambda p, path: _run_simulation(p, path, base_config_dict),
        verbose=True,
    )

    result = parametrizer.run()

    logger.info(f"Complete: {len(result.param_points)} runs")
    logger.info(f"Results: {OUTPUT_DIR}/sweep_results.csv")
