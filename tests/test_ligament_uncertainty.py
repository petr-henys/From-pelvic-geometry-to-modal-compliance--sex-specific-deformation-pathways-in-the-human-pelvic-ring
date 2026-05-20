#!/usr/bin/env python3
"""Tests for the hierarchical ligament uncertainty model."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_ligament_uncertainty_sweep import (
    ALL_BUNDLES,
    BILATERAL_TYPES,
    BUNDLE_SIDE_SIGN,
    LIGAMENT_TYPES,
    N_LATENT,
    TYPE_TO_BUNDLES,
    UncertaintySpec,
    build_input_covariance,
    build_sample_configs,
    first_order_propagation,
    generate_lhs_samples,
    latent_to_ligaments,
)

# Default config LIGAMENTS dict for testing
BASELINE_LIGAMENTS: dict[str, dict[str, float]] = {
    "symphisys": {"stiffness": 1000.0, "pretension": 20.0},
    "right_anterior_SIJ_ligaments": {"stiffness": 1400.0, "pretension": 20.0},
    "left_anterior_SIJ_ligaments": {"stiffness": 1400.0, "pretension": 20.0},
    "right_posterior_SIJ_ligaments": {"stiffness": 5600.0, "pretension": 20.0},
    "left_posterior_SIJ_ligaments": {"stiffness": 5600.0, "pretension": 20.0},
    "right_INL": {"stiffness": 200.0, "pretension": 20.0},
    "left_INL": {"stiffness": 200.0, "pretension": 20.0},
    "right_SS": {"stiffness": 3000.0, "pretension": 20.0},
    "left_SS": {"stiffness": 3000.0, "pretension": 20.0},
    "right_ST": {"stiffness": 3000.0, "pretension": 20.0},
    "left_ST": {"stiffness": 3000.0, "pretension": 20.0},
}


class TestTopology:
    """Verify the ligament type topology constants."""

    def test_type_count(self) -> None:
        assert len(LIGAMENT_TYPES) == 6

    def test_bilateral_count(self) -> None:
        assert len(BILATERAL_TYPES) == 5

    def test_all_bundles_count(self) -> None:
        assert len(ALL_BUNDLES) == 11

    def test_bundles_cover_baseline(self) -> None:
        assert set(ALL_BUNDLES) == set(BASELINE_LIGAMENTS.keys())

    def test_side_signs(self) -> None:
        for bundle, sign in BUNDLE_SIDE_SIGN.items():
            if "right" in bundle:
                assert sign == +1.0
            elif "left" in bundle:
                assert sign == -1.0
            else:
                assert sign == 0.0


class TestUncertaintySpec:
    """Verify UncertaintySpec helper methods."""

    def test_total_sigma_k(self) -> None:
        spec = UncertaintySpec()
        expected = np.sqrt(0.25**2 + 0.20**2 + 0.10**2)
        assert abs(spec.total_sigma_k() - expected) < 1e-10

    def test_total_sigma_T(self) -> None:
        spec = UncertaintySpec()
        expected = np.sqrt(0.35**2 + 0.25**2 + 0.12**2)
        assert abs(spec.total_sigma_T() - expected) < 1e-10

    def test_correlation_bounds(self) -> None:
        spec = UncertaintySpec()
        rho_same = spec.correlation_same_type_opposite_side()
        rho_diff = spec.correlation_different_type()
        assert 0.0 < rho_diff < rho_same < 1.0

    def test_zero_asym_gives_perfect_lr_correlation(self) -> None:
        spec = UncertaintySpec(sigma_asym_k=0.0)
        assert abs(spec.correlation_same_type_opposite_side() - 1.0) < 1e-10


class TestLatentSampling:
    """Verify LHS sample generation."""

    def test_shape(self) -> None:
        spec = UncertaintySpec(n_samples=50, seed=0)
        Z = generate_lhs_samples(spec)
        assert Z.shape == (50, N_LATENT)

    def test_marginal_normality(self) -> None:
        """Each marginal should be approximately standard normal."""
        spec = UncertaintySpec(n_samples=1000, seed=42)
        Z = generate_lhs_samples(spec)
        for d in range(N_LATENT):
            assert abs(Z[:, d].mean()) < 0.15, f"dim {d} mean off"
            assert abs(Z[:, d].std() - 1.0) < 0.15, f"dim {d} std off"

    def test_reproducibility(self) -> None:
        spec = UncertaintySpec(n_samples=20, seed=99)
        Z1 = generate_lhs_samples(spec)
        Z2 = generate_lhs_samples(spec)
        np.testing.assert_array_equal(Z1, Z2)


class TestLatentMapping:
    """Verify latent → physical parameter mapping."""

    def test_zero_latent_gives_baseline(self) -> None:
        spec = UncertaintySpec()
        z = np.zeros(N_LATENT)
        result = latent_to_ligaments(z, BASELINE_LIGAMENTS, spec)
        for bundle in ALL_BUNDLES:
            assert abs(result[bundle]["stiffness"] - BASELINE_LIGAMENTS[bundle]["stiffness"]) < 1e-10
            assert abs(result[bundle]["pretension"] - BASELINE_LIGAMENTS[bundle]["pretension"]) < 1e-10

    def test_all_positive(self) -> None:
        """Physical parameters must always be positive (log-normal guarantee)."""
        spec = UncertaintySpec(n_samples=200, seed=7)
        configs, _ = build_sample_configs(spec, BASELINE_LIGAMENTS)
        for cfg in configs:
            for bundle, props in cfg.items():
                assert props["stiffness"] > 0, f"{bundle} stiffness non-positive"
                assert props["pretension"] > 0, f"{bundle} pretension non-positive"

    def test_global_factor_scales_all(self) -> None:
        """A positive global-stiffness factor should increase all stiffnesses."""
        spec = UncertaintySpec(
            sigma_global_k=0.5, sigma_type_k=0.0, sigma_asym_k=0.0,
            sigma_global_T=0.0, sigma_type_T=0.0, sigma_asym_T=0.0,
        )
        z = np.zeros(N_LATENT)
        z[0] = 2.0  # strong positive global stiffness factor
        result = latent_to_ligaments(z, BASELINE_LIGAMENTS, spec)
        for bundle in ALL_BUNDLES:
            assert result[bundle]["stiffness"] > BASELINE_LIGAMENTS[bundle]["stiffness"]
            # Pretension unchanged (all pretension σ = 0)
            assert abs(result[bundle]["pretension"] - BASELINE_LIGAMENTS[bundle]["pretension"]) < 1e-10

    def test_asymmetry_creates_lr_difference(self) -> None:
        """Non-zero asymmetry factor should create L/R mismatch."""
        spec = UncertaintySpec(
            sigma_global_k=0.0, sigma_type_k=0.0, sigma_asym_k=0.3,
            sigma_global_T=0.0, sigma_type_T=0.0, sigma_asym_T=0.0,
        )
        z = np.zeros(N_LATENT)
        z[8] = 1.5  # asymmetry factor for first bilateral type (anterior_SIJ)
        result = latent_to_ligaments(z, BASELINE_LIGAMENTS, spec)
        k_right = result["right_anterior_SIJ_ligaments"]["stiffness"]
        k_left = result["left_anterior_SIJ_ligaments"]["stiffness"]
        assert k_right != k_left
        # Right should be higher (positive z, positive side sign)
        assert k_right > k_left

    def test_symmetry_preserved_for_zero_asym_spec(self) -> None:
        """With σ_asym = 0, bilateral pairs should always be equal."""
        spec = UncertaintySpec(
            sigma_global_k=0.3, sigma_type_k=0.2, sigma_asym_k=0.0,
            sigma_global_T=0.3, sigma_type_T=0.2, sigma_asym_T=0.0,
        )
        rng = np.random.default_rng(42)
        for _ in range(20):
            z = rng.standard_normal(N_LATENT)
            result = latent_to_ligaments(z, BASELINE_LIGAMENTS, spec)
            for lig_type in BILATERAL_TYPES:
                bundles = TYPE_TO_BUNDLES[lig_type]
                assert abs(result[bundles[0]]["stiffness"] - result[bundles[1]]["stiffness"]) < 1e-10
                assert abs(result[bundles[0]]["pretension"] - result[bundles[1]]["pretension"]) < 1e-10


class TestCovarianceMatrix:
    """Verify analytical covariance matrix construction."""

    def test_shape(self) -> None:
        spec = UncertaintySpec()
        Sigma = build_input_covariance(spec, ALL_BUNDLES)
        assert Sigma.shape == (22, 22)

    def test_symmetric(self) -> None:
        spec = UncertaintySpec()
        Sigma = build_input_covariance(spec, ALL_BUNDLES)
        np.testing.assert_allclose(Sigma, Sigma.T, atol=1e-15)

    def test_positive_semidefinite(self) -> None:
        spec = UncertaintySpec()
        Sigma = build_input_covariance(spec, ALL_BUNDLES)
        eigenvalues = np.linalg.eigvalsh(Sigma)
        assert np.all(eigenvalues >= -1e-12)

    def test_diagonal_matches_total_variance(self) -> None:
        """Diagonal elements should match total σ² for midline bundles."""
        spec = UncertaintySpec()
        Sigma = build_input_covariance(spec, ALL_BUNDLES)
        # Symphysis is the first bundle (index 0)
        expected_var_k = spec.sigma_global_k**2 + spec.sigma_type_k**2
        # No asymmetry for midline
        assert abs(Sigma[0, 0] - expected_var_k) < 1e-12

    def test_bilateral_stiffness_correlation(self) -> None:
        """Verify the implied correlation for L/R bundles of same type."""
        spec = UncertaintySpec()
        Sigma = build_input_covariance(spec, ALL_BUNDLES)
        # anterior_SIJ: right=index 1, left=index 2
        var_r = Sigma[1, 1]
        var_l = Sigma[2, 2]
        cov_rl = Sigma[1, 2]
        rho = cov_rl / np.sqrt(var_r * var_l)
        expected = spec.correlation_same_type_opposite_side()
        assert abs(rho - expected) < 0.02  # small tolerance for asym sign effects

    def test_cross_block_correlation(self) -> None:
        """Verify stiffness–pretension cross-correlation at global level."""
        spec = UncertaintySpec(
            sigma_type_k=0.0, sigma_asym_k=0.0,
            sigma_type_T=0.0, sigma_asym_T=0.0,
        )
        Sigma = build_input_covariance(spec, ALL_BUNDLES)
        n = len(ALL_BUNDLES)
        # Symphysis (index 0) stiffness ↔ pretension
        cov_kT = Sigma[0, n + 0]
        var_k = Sigma[0, 0]
        var_T = Sigma[n + 0, n + 0]
        rho = cov_kT / np.sqrt(var_k * var_T)
        assert abs(rho - spec.rho_kT) < 1e-10


class TestMCSamplingCorrelation:
    """Verify that MC samples reproduce expected correlation structure."""

    def test_stiffness_correlation_structure(self) -> None:
        """Check that sample correlations approximate analytical predictions."""
        spec = UncertaintySpec(n_samples=5000, seed=123)
        configs, _ = build_sample_configs(spec, BASELINE_LIGAMENTS)

        # Collect log-stiffness for right and left anterior SIJ
        log_kr = np.array([np.log(c["right_anterior_SIJ_ligaments"]["stiffness"]) for c in configs])
        log_kl = np.array([np.log(c["left_anterior_SIJ_ligaments"]["stiffness"]) for c in configs])
        # Same-type L/R correlation
        rho_lr = np.corrcoef(log_kr, log_kl)[0, 1]
        expected_lr = spec.correlation_same_type_opposite_side()
        assert abs(rho_lr - expected_lr) < 0.1, f"L/R corr {rho_lr:.3f} vs expected {expected_lr:.3f}"

        # Different-type correlation
        log_k_inl = np.array([np.log(c["right_INL"]["stiffness"]) for c in configs])
        rho_diff = np.corrcoef(log_kr, log_k_inl)[0, 1]
        expected_diff = spec.correlation_different_type()
        assert abs(rho_diff - expected_diff) < 0.1, f"diff-type corr {rho_diff:.3f} vs expected {expected_diff:.3f}"


class TestFirstOrderPropagation:
    """Verify first-order variance propagation."""

    def test_zero_sensitivity_gives_zero_variance(self) -> None:
        spec = UncertaintySpec()
        n_modes, n_lig = 3, 11
        eigenvalues = np.array([1.0, 2.0, 3.0])
        zeros = np.zeros((n_modes, n_lig))
        stiffness_baseline = np.ones(n_lig)
        pretension_baseline = np.ones(n_lig)

        df = first_order_propagation(
            zeros, zeros, eigenvalues,
            stiffness_baseline, pretension_baseline,
            ALL_BUNDLES, spec,
        )
        assert len(df) == n_modes
        np.testing.assert_allclose(df["var_log_lambda"].values, 0.0, atol=1e-15)

    def test_single_sensitive_mode(self) -> None:
        """A mode sensitive to only one parameter should have predictable variance."""
        spec = UncertaintySpec(
            sigma_global_k=0.3, sigma_type_k=0.0, sigma_asym_k=0.0,
            sigma_global_T=0.0, sigma_type_T=0.0, sigma_asym_T=0.0,
            rho_kT=0.0,
        )
        n_modes, n_lig = 1, 11
        eigenvalues = np.array([100.0])
        # Uniform sensitivity to all stiffnesses
        stiffness_sens = np.ones((n_modes, n_lig))
        pretension_sens = np.zeros((n_modes, n_lig))
        stiffness_baseline = np.full(n_lig, 100.0)
        pretension_baseline = np.full(n_lig, 20.0)

        df = first_order_propagation(
            stiffness_sens, pretension_sens, eigenvalues,
            stiffness_baseline, pretension_baseline,
            ALL_BUNDLES, spec,
        )
        # With only global σ_k and uniform J = k/λ · ∂λ/∂k = 100/100 · 1 = 1,
        # all 11 bundles are perfectly correlated → Var = (11 · σ_g)² = (11 · 0.3)² = 10.89
        expected_var = (11 * spec.sigma_global_k) ** 2
        assert abs(df["var_log_lambda"].iloc[0] - expected_var) < 0.01

    def test_variance_decomposition_sums_to_total(self) -> None:
        """frac_global + frac_type + frac_asym should equal 1."""
        spec = UncertaintySpec()
        n_modes, n_lig = 5, 11
        rng = np.random.default_rng(42)
        eigenvalues = rng.uniform(10, 1000, n_modes)
        stiffness_sens = rng.standard_normal((n_modes, n_lig))
        pretension_sens = rng.standard_normal((n_modes, n_lig))
        stiffness_baseline = rng.uniform(100, 5000, n_lig)
        pretension_baseline = rng.uniform(10, 50, n_lig)

        df = first_order_propagation(
            stiffness_sens, pretension_sens, eigenvalues,
            stiffness_baseline, pretension_baseline,
            ALL_BUNDLES, spec,
        )
        fracs = df["frac_global"] + df["frac_type"] + df["frac_asym"]
        np.testing.assert_allclose(fracs.values, 1.0, atol=1e-10)

    def test_pretension_only_gives_nonzero_variance(self) -> None:
        """Pretension-only sensitivity (stiffness=0) should produce nonzero variance."""
        spec = UncertaintySpec(
            sigma_global_k=0.0, sigma_type_k=0.0, sigma_asym_k=0.0,
            sigma_global_T=0.35, sigma_type_T=0.20, sigma_asym_T=0.10,
            rho_kT=0.0,
        )
        n_modes, n_lig = 1, 11
        eigenvalues = np.array([100.0])
        stiffness_sens = np.zeros((n_modes, n_lig))
        pretension_sens = np.ones((n_modes, n_lig))
        stiffness_baseline = np.full(n_lig, 100.0)
        pretension_baseline = np.full(n_lig, 20.0)

        df = first_order_propagation(
            stiffness_sens, pretension_sens, eigenvalues,
            stiffness_baseline, pretension_baseline,
            ALL_BUNDLES, spec,
        )
        assert df["var_log_lambda"].iloc[0] > 0.0

    def test_pretension_only_global_variance(self) -> None:
        """With only global σ_T, pretension-only variance = (sum J_T · σ_gT)²."""
        spec = UncertaintySpec(
            sigma_global_k=0.0, sigma_type_k=0.0, sigma_asym_k=0.0,
            sigma_global_T=0.3, sigma_type_T=0.0, sigma_asym_T=0.0,
            rho_kT=0.0,
        )
        n_modes, n_lig = 1, 11
        eigenvalues = np.array([100.0])
        stiffness_sens = np.zeros((n_modes, n_lig))
        pretension_sens = np.ones((n_modes, n_lig))
        stiffness_baseline = np.full(n_lig, 100.0)
        pretension_baseline = np.full(n_lig, 20.0)

        df = first_order_propagation(
            stiffness_sens, pretension_sens, eigenvalues,
            stiffness_baseline, pretension_baseline,
            ALL_BUNDLES, spec,
        )
        # J_T = (T/λ)·∂λ/∂T = 20/100 · 1 = 0.2 for each bundle
        # Global-only: all 11 perfectly correlated → Var = (11·0.2·0.3)² = (0.66)² = 0.4356
        j_t = 20.0 / 100.0
        expected_var = (11 * j_t * spec.sigma_global_T) ** 2
        assert abs(df["var_log_lambda"].iloc[0] - expected_var) < 1e-6

    def test_stiffness_pretension_cross_correlation(self) -> None:
        """Non-zero ρ_kT should produce different variance than ρ_kT=0."""
        n_modes, n_lig = 3, 11
        rng = np.random.default_rng(99)
        eigenvalues = rng.uniform(50, 500, n_modes)
        stiffness_sens = rng.standard_normal((n_modes, n_lig)) * 10
        pretension_sens = rng.standard_normal((n_modes, n_lig)) * 5
        stiffness_baseline = rng.uniform(200, 5000, n_lig)
        pretension_baseline = rng.uniform(10, 50, n_lig)

        spec_no_cross = UncertaintySpec(rho_kT=0.0)
        spec_cross = UncertaintySpec(rho_kT=0.5)

        df_no_cross = first_order_propagation(
            stiffness_sens, pretension_sens, eigenvalues,
            stiffness_baseline, pretension_baseline,
            ALL_BUNDLES, spec_no_cross,
        )
        df_cross = first_order_propagation(
            stiffness_sens, pretension_sens, eigenvalues,
            stiffness_baseline, pretension_baseline,
            ALL_BUNDLES, spec_cross,
        )
        # Variances must differ when cross-correlation is nonzero
        assert not np.allclose(
            df_no_cross["var_log_lambda"].values,
            df_cross["var_log_lambda"].values,
            atol=1e-10,
        )

    def test_pretension_variance_increases_with_sigma(self) -> None:
        """Larger pretension σ_global should produce larger variance, all else equal."""
        n_modes, n_lig = 2, 11
        eigenvalues = np.array([100.0, 200.0])
        stiffness_sens = np.zeros((n_modes, n_lig))
        pretension_sens = np.ones((n_modes, n_lig))
        stiffness_baseline = np.full(n_lig, 1000.0)
        pretension_baseline = np.full(n_lig, 20.0)

        spec_small = UncertaintySpec(
            sigma_global_k=0.0, sigma_type_k=0.0, sigma_asym_k=0.0,
            sigma_global_T=0.10, sigma_type_T=0.05, sigma_asym_T=0.02,
            rho_kT=0.0,
        )
        spec_large = UncertaintySpec(
            sigma_global_k=0.0, sigma_type_k=0.0, sigma_asym_k=0.0,
            sigma_global_T=0.50, sigma_type_T=0.25, sigma_asym_T=0.10,
            rho_kT=0.0,
        )

        df_small = first_order_propagation(
            stiffness_sens, pretension_sens, eigenvalues,
            stiffness_baseline, pretension_baseline,
            ALL_BUNDLES, spec_small,
        )
        df_large = first_order_propagation(
            stiffness_sens, pretension_sens, eigenvalues,
            stiffness_baseline, pretension_baseline,
            ALL_BUNDLES, spec_large,
        )
        for m in range(n_modes):
            assert df_large["var_log_lambda"].iloc[m] > df_small["var_log_lambda"].iloc[m]

    def test_pretension_covariance_block_present(self) -> None:
        """The 22×22 covariance matrix should have non-zero pretension block."""
        spec = UncertaintySpec()
        Sigma = build_input_covariance(spec, ALL_BUNDLES)
        n = len(ALL_BUNDLES)
        # Pretension block is Sigma[n:, n:] (lower-right 11×11)
        pretension_block = Sigma[n:, n:]
        assert np.any(pretension_block > 0), "Pretension sub-block is all zeros"
        # Cross-block Sigma[:n, n:] should be non-zero when rho_kT != 0
        cross_block = Sigma[:n, n:]
        if spec.rho_kT != 0:
            assert np.any(cross_block != 0), "Cross-block is zero despite rho_kT != 0"
