#!/usr/bin/env python3
"""First-order uncertainty propagation for cartilage and bone density-law parameters.

Mirrors ``run_ligament_uncertainty_sweep.py`` but for the two remaining
material-property groups:

1. **Cartilage modulus** — three parameters (SIJ-L, SIJ-R, Pubic Symphysis)
   with a hierarchical correlation structure (shared global factor, L/R
   correlation for SIJ).
2. **Bone density law** — two parameters α and β in E = α ρ_ash^β.

All propagation is purely analytical (first-order Taylor; no new FEM runs).
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Cartilage parameter ordering (must match Zarr axis-2) ────────────────
CARTILAGE_PARAMS: list[str] = ["SIJCartilageLeft", "SIJCartilageRight", "PubicSymphysis"]
N_CART = len(CARTILAGE_PARAMS)

# Which cartilage params share the SIJ "type" factor (indices into CARTILAGE_PARAMS)
SIJ_INDICES = [0, 1]
SYM_INDEX = 2

# Side sign for L/R asymmetry: Left = +1, Right = −1 (arbitrary sign convention)
CART_SIDE_SIGN = {
    "SIJCartilageLeft": +1.0,
    "SIJCartilageRight": -1.0,
    "PubicSymphysis": 0.0,
}

# ── Bone parameter ordering ─────────────────────────────────────────────
BONE_PARAMS: list[str] = ["alpha", "beta"]
N_BONE = len(BONE_PARAMS)
_MAX_LOGVAR_FOR_EXP = 700.0


def cov_from_log_variance(var_log: float) -> float:
    """Convert Var(log λ) to CoV(λ) robustly, avoiding overflow warnings."""
    if not np.isfinite(var_log):
        return float("nan")
    clipped = max(float(var_log), 0.0)
    if clipped >= _MAX_LOGVAR_FOR_EXP:
        return float("inf")
    return float(np.sqrt(np.expm1(clipped)))


# ---------------------------------------------------------------------------
# Uncertainty specifications
# ---------------------------------------------------------------------------

@dataclass
class CartilageUncertaintySpec:
    """Log-space uncertainty for the three cartilage moduli.

    Hierarchical structure:
    - ``sigma_global``: shared overall cartilage-stiffness factor
    - ``sigma_type``:   per-joint-type residual (SIJ vs symphysis)
    - ``sigma_asym``:   L/R asymmetry within SIJ (does not apply to symphysis)

    Literature basis:
    - SIJ cartilage modulus ranges from ~0.2 to 10 MPa (Zheng et al. 2006,
      Eichenseer et al. 2011, Becker & Willburger 2016).
    - Pubic symphysis modulus ranges from ~0.5 to 5 MPa (Li et al. 2007).
    - Total implied CoV ≈ √(σ_g² + σ_t² + σ_a²) → ~53 % for SIJ,
      ~47 % for symphysis (no asymmetry term).

    Implied correlations (SIJ):
        ρ(L, R) = (σ_g² + σ_t²) / (σ_g² + σ_t² + σ_a²)  ≈ 0.96
        ρ(SIJ, Sym) = σ_g² / σ_total²_SIJ                 ≈ 0.56
    """

    sigma_global: float = 0.35     # shared cartilage laxity
    sigma_type: float = 0.25       # SIJ vs symphysis residual
    sigma_asym: float = 0.10       # L/R asymmetry (SIJ only)

    def total_sigma_sij(self) -> float:
        """Total log-space σ for an SIJ cartilage parameter."""
        return float(np.sqrt(
            self.sigma_global ** 2
            + self.sigma_type ** 2
            + self.sigma_asym ** 2
        ))

    def total_sigma_sym(self) -> float:
        """Total log-space σ for the symphysis parameter."""
        return float(np.sqrt(
            self.sigma_global ** 2
            + self.sigma_type ** 2
        ))


@dataclass
class BoneUncertaintySpec:
    r"""Log-space uncertainty for the bone density-law parameters.

    E = α · ρ_ash^β,  defaults α = 10200, β = 2.0.

    Literature-informed ranges:
    - α: 3790 (Carter & Hayes 1977) – 10500 (Rho et al. 1995) →
      factor ~2.7×; σ_α ≈ 0.30 gives 95 % CI ≈ [5 500, 19 000].
    - β: 1.49 (Morgan et al. 2003) – 3.0 (Carter & Hayes 1977) →
      in log-space, σ_β ≈ 0.20 gives 95 % CI of exponent ≈ [1.35, 2.97].

    Cross-correlation: α and β in published fits are usually derived from the
    *same* regression, so they covary.  A negative ρ is expected (higher
    exponent compensates lower prefactor).  Default ρ = −0.40.
    """

    sigma_alpha: float = 0.30      # prefactor σ (log-space)
    sigma_beta: float = 0.20       # exponent σ (log-space)
    rho_alpha_beta: float = -0.40  # cross-correlation α ↔ β


# ---------------------------------------------------------------------------
# Covariance builders
# ---------------------------------------------------------------------------

def build_cartilage_covariance(spec: CartilageUncertaintySpec) -> np.ndarray:
    r"""Build the 3×3 covariance matrix of [log E_SIJ_L, log E_SIJ_R, log E_Sym].

    Σ_ij = σ_g² + δ(type_i = type_j) · σ_t² + δ(bilat_i = bilat_j) · σ_a² · s_i · s_j

    where s_i is the side sign (+1 left, −1 right, 0 symphysis).
    """
    Sigma = np.zeros((N_CART, N_CART))

    for i in range(N_CART):
        for j in range(N_CART):
            cov = spec.sigma_global ** 2

            # Same type contribution (SIJ ↔ SIJ or Sym ↔ Sym)
            same_type = (
                (i in SIJ_INDICES and j in SIJ_INDICES)
                or (i == SYM_INDEX and j == SYM_INDEX)
            )
            if same_type:
                cov += spec.sigma_type ** 2

            # Asymmetry contribution (SIJ only, side_i × side_j)
            if i in SIJ_INDICES and j in SIJ_INDICES:
                si = CART_SIDE_SIGN[CARTILAGE_PARAMS[i]]
                sj = CART_SIDE_SIGN[CARTILAGE_PARAMS[j]]
                cov += spec.sigma_asym ** 2 * si * sj

            Sigma[i, j] = cov

    return Sigma


def build_bone_covariance(spec: BoneUncertaintySpec) -> np.ndarray:
    r"""Build the 2×2 covariance matrix of [log α, log β].

    .. math::
        \Sigma = \begin{pmatrix}
            \sigma_\alpha^2 & \rho \sigma_\alpha \sigma_\beta \\
            \rho \sigma_\alpha \sigma_\beta & \sigma_\beta^2
        \end{pmatrix}
    """
    off = spec.rho_alpha_beta * spec.sigma_alpha * spec.sigma_beta
    return np.array([
        [spec.sigma_alpha ** 2, off],
        [off, spec.sigma_beta ** 2],
    ])


# ---------------------------------------------------------------------------
# Dimensionless Jacobians
# ---------------------------------------------------------------------------
MIN_EIG = 1e-12


def _safe_eigenvalues(eigenvalues: np.ndarray) -> np.ndarray:
    return np.where(np.abs(eigenvalues) < MIN_EIG, np.nan, eigenvalues)


def cartilage_dimensionless_jacobian(
    cart_sens: np.ndarray,
    eigenvalues: np.ndarray,
    moduli_baseline: np.ndarray,
) -> np.ndarray:
    """Compute J_cart[m, j] = (E_j / λ_m) · ∂λ_m / ∂E_j.

    Parameters
    ----------
    cart_sens : (n_modes, 3) raw ∂λ/∂E_j
    eigenvalues : (n_modes,)
    moduli_baseline : (3,)  baseline modulus per cartilage parameter

    Returns
    -------
    J : (n_modes, 3) dimensionless Jacobian
    """
    safe_eig = _safe_eigenvalues(eigenvalues)
    return cart_sens * (moduli_baseline[np.newaxis, :] / safe_eig[:, np.newaxis])


def bone_dimensionless_jacobian(
    alpha_sens: np.ndarray,
    beta_sens: np.ndarray,
    eigenvalues: np.ndarray,
    alpha_baseline: float,
    beta_baseline: float,
) -> np.ndarray:
    """Compute J_bone[m, :] = [(α/λ)·∂λ/∂α, (β/λ)·∂λ/∂β].

    Parameters
    ----------
    alpha_sens, beta_sens : (n_modes,)
    eigenvalues : (n_modes,)
    alpha_baseline, beta_baseline : scalar baselines

    Returns
    -------
    J : (n_modes, 2) dimensionless Jacobian
    """
    safe_eig = _safe_eigenvalues(eigenvalues)
    J_a = alpha_sens * (alpha_baseline / safe_eig)
    J_b = beta_sens * (beta_baseline / safe_eig)
    return np.column_stack([J_a, J_b])


# ---------------------------------------------------------------------------
# First-order propagation
# ---------------------------------------------------------------------------

def cartilage_first_order_propagation(
    cart_sens: np.ndarray,
    eigenvalues: np.ndarray,
    moduli_baseline: np.ndarray,
    spec: CartilageUncertaintySpec,
) -> pd.DataFrame:
    """First-order propagation of cartilage modulus uncertainty.

    Returns per-mode variance in log λ with decomposition into
    global / type / asymmetry levels plus per-region marginals.
    """
    n_modes = eigenvalues.size
    J = cartilage_dimensionless_jacobian(cart_sens, eigenvalues, moduli_baseline)
    Sigma = build_cartilage_covariance(spec)

    # Sub-covariances for decomposition
    spec_global = CartilageUncertaintySpec(
        sigma_global=spec.sigma_global, sigma_type=0.0, sigma_asym=0.0,
    )
    Sigma_global = build_cartilage_covariance(spec_global)

    spec_no_asym = CartilageUncertaintySpec(
        sigma_global=spec.sigma_global, sigma_type=spec.sigma_type, sigma_asym=0.0,
    )
    Sigma_no_asym = build_cartilage_covariance(spec_no_asym)

    records: list[dict] = []
    for m in range(n_modes):
        Jm = J[m]
        if np.any(np.isnan(Jm)):
            continue

        var_total = float(Jm @ Sigma @ Jm)
        var_global = float(Jm @ Sigma_global @ Jm)
        var_no_asym = float(Jm @ Sigma_no_asym @ Jm)
        var_type = var_no_asym - var_global
        var_asym = var_total - var_no_asym

        # Per-region marginal variance: var_j = J_j² · Σ_jj
        marginals = {}
        for j, name in enumerate(CARTILAGE_PARAMS):
            marginals[f"marg_var_{name}"] = float(Jm[j] ** 2 * Sigma[j, j])

        records.append({
            "mode": m + 1,
            "eigenvalue": float(eigenvalues[m]),
            "var_log_lambda_cart": var_total,
            "std_log_lambda_cart": float(np.sqrt(max(var_total, 0.0))),
            "CoV_lambda_cart": cov_from_log_variance(var_total),
            "cart_var_global": var_global,
            "cart_var_type": var_type,
            "cart_var_asym": var_asym,
            "cart_frac_global": var_global / var_total if var_total > 0 else np.nan,
            "cart_frac_type": var_type / var_total if var_total > 0 else np.nan,
            "cart_frac_asym": var_asym / var_total if var_total > 0 else np.nan,
            **marginals,
        })

    return pd.DataFrame.from_records(records)


def bone_first_order_propagation(
    alpha_sens: np.ndarray,
    beta_sens: np.ndarray,
    eigenvalues: np.ndarray,
    alpha_baseline: float,
    beta_baseline: float,
    spec: BoneUncertaintySpec,
) -> pd.DataFrame:
    """First-order propagation of bone density-law uncertainty.

    Returns per-mode variance in log λ with decomposition into
    α-only, β-only, and cross-term contributions.
    """
    n_modes = eigenvalues.size
    J = bone_dimensionless_jacobian(
        alpha_sens, beta_sens, eigenvalues, alpha_baseline, beta_baseline,
    )
    Sigma = build_bone_covariance(spec)

    records: list[dict] = []
    for m in range(n_modes):
        Jm = J[m]
        if np.any(np.isnan(Jm)):
            continue

        var_total = float(Jm @ Sigma @ Jm)

        # Decompose: var_alpha + var_beta + 2 * cov_cross
        var_alpha = float(Jm[0] ** 2 * Sigma[0, 0])
        var_beta = float(Jm[1] ** 2 * Sigma[1, 1])
        var_cross = float(2 * Jm[0] * Jm[1] * Sigma[0, 1])

        records.append({
            "mode": m + 1,
            "eigenvalue": float(eigenvalues[m]),
            "var_log_lambda_bone": var_total,
            "std_log_lambda_bone": float(np.sqrt(max(var_total, 0.0))),
            "CoV_lambda_bone": cov_from_log_variance(var_total),
            "bone_var_alpha": var_alpha,
            "bone_var_beta": var_beta,
            "bone_var_cross": var_cross,
            "bone_frac_alpha": var_alpha / var_total if var_total > 0 else np.nan,
            "bone_frac_beta": var_beta / var_total if var_total > 0 else np.nan,
            "bone_frac_cross": var_cross / var_total if var_total > 0 else np.nan,
        })

    return pd.DataFrame.from_records(records)


# ---------------------------------------------------------------------------
# Combined total variance across all three parameter groups
# ---------------------------------------------------------------------------

def combine_variances(
    df_lig: pd.DataFrame,
    df_cart: pd.DataFrame,
    df_bone: pd.DataFrame,
) -> pd.DataFrame:
    """Merge per-mode variances from ligament, cartilage and bone groups.

    The three groups are assumed independent → total variance is additive.
    Returns a DataFrame with total variance and group fractions.
    """
    merged = df_lig[["mode", "eigenvalue", "var_log_lambda", "CoV_lambda"]].rename(
        columns={"var_log_lambda": "var_lig", "CoV_lambda": "CoV_lig"}
    )
    merged = merged.merge(
        df_cart[["mode", "var_log_lambda_cart", "CoV_lambda_cart"]].rename(
            columns={"var_log_lambda_cart": "var_cart", "CoV_lambda_cart": "CoV_cart"}
        ),
        on="mode",
    )
    merged = merged.merge(
        df_bone[["mode", "var_log_lambda_bone", "CoV_lambda_bone"]].rename(
            columns={"var_log_lambda_bone": "var_bone", "CoV_lambda_bone": "CoV_bone"}
        ),
        on="mode",
    )

    merged["var_total"] = merged["var_lig"] + merged["var_cart"] + merged["var_bone"]
    merged["CoV_total"] = merged["var_total"].map(cov_from_log_variance)

    for group in ("lig", "cart", "bone"):
        merged[f"frac_{group}"] = merged[f"var_{group}"] / merged["var_total"]

    return merged


# ---------------------------------------------------------------------------
# Reference-only analysis (quick single-subject run)
# ---------------------------------------------------------------------------

def run_reference_analysis(
    reference_path: Path,
    metadata_path: Path,
    cart_spec: CartilageUncertaintySpec | None = None,
    bone_spec: BoneUncertaintySpec | None = None,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Analyse the reference solution for cartilage and bone uncertainty.

    Returns (df_cart, df_bone).  Either can be None if data is missing.
    """
    if not reference_path.exists():
        logger.warning("Reference solution not found — skipping")
        return None, None

    with np.load(reference_path, allow_pickle=True) as archive:
        eigenvalues = np.asarray(archive["eigenvalues"], dtype=float)
        cart_sens = np.asarray(archive["cartilage_sensitivities"], dtype=float)
        cart_names = [str(n) for n in archive["cartilage_parameter_names"]]

    with metadata_path.open() as f:
        metadata = json.load(f)
    cfg = metadata.get("config", {})

    # ── Cartilage ────
    df_cart = None
    if cart_sens.shape[1] > 0:
        moduli_baseline = np.array(
            [cfg["SIJ_CARTILAGE_MODULUS"]] * 2 + [cfg["SYMPHYSIS_MODULUS"]],
            dtype=float,
        )
        if cart_spec is None:
            cart_spec = CartilageUncertaintySpec()
        df_cart = cartilage_first_order_propagation(
            cart_sens, eigenvalues, moduli_baseline, cart_spec,
        )
        logger.info("Cartilage CoV per mode:")
        for _, row in df_cart.iterrows():
            logger.info(
                "  Mode %2d: CoV = %.1f%%  (global %.0f%% | type %.0f%% | asym %.0f%%)",
                int(row["mode"]),
                row["CoV_lambda_cart"] * 100,
                row["cart_frac_global"] * 100,
                row["cart_frac_type"] * 100,
                row["cart_frac_asym"] * 100,
            )

    # ── Bone (not in reference_solution.npz — only in Zarr) ────
    df_bone = None

    return df_cart, df_bone


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    RESULTS_DIR = Path("results/ref_S1P_fixed_new2")
    reference_path = RESULTS_DIR / "reference_solution.npz"
    metadata_path = RESULTS_DIR / "simulation_metadata.json"

    cart_spec = CartilageUncertaintySpec()
    bone_spec = BoneUncertaintySpec()

    logger.info("=== Cartilage Uncertainty Spec ===")
    logger.info(
        "σ — global=%.2f  type=%.2f  asym=%.2f  total_SIJ=%.2f  total_Sym=%.2f",
        cart_spec.sigma_global, cart_spec.sigma_type, cart_spec.sigma_asym,
        cart_spec.total_sigma_sij(), cart_spec.total_sigma_sym(),
    )
    logger.info("Σ_cart:\n%s", build_cartilage_covariance(cart_spec))

    logger.info("=== Bone Uncertainty Spec ===")
    logger.info(
        "σ_α=%.2f  σ_β=%.2f  ρ=%.2f",
        bone_spec.sigma_alpha, bone_spec.sigma_beta, bone_spec.rho_alpha_beta,
    )
    logger.info("Σ_bone:\n%s", build_bone_covariance(bone_spec))

    df_cart, _ = run_reference_analysis(
        reference_path, metadata_path, cart_spec, bone_spec,
    )
