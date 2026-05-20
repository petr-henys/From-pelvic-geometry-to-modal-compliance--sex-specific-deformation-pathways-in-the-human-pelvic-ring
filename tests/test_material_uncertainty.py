#!/usr/bin/env python3
"""Tests for cartilage and bone density-law uncertainty propagation."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_material_uncertainty import (
    BONE_PARAMS,
    CARTILAGE_PARAMS,
    CART_SIDE_SIGN,
    N_BONE,
    N_CART,
    SIJ_INDICES,
    SYM_INDEX,
    BoneUncertaintySpec,
    CartilageUncertaintySpec,
    bone_dimensionless_jacobian,
    bone_first_order_propagation,
    build_bone_covariance,
    build_cartilage_covariance,
    cartilage_dimensionless_jacobian,
    cartilage_first_order_propagation,
    combine_variances,
)


# ── Constants ────────────────────────────────────────────────────────────

class TestCartilageTopology:
    """Verify cartilage parameter constants."""

    def test_param_count(self) -> None:
        assert N_CART == 3

    def test_param_names(self) -> None:
        assert CARTILAGE_PARAMS == ["SIJCartilageLeft", "SIJCartilageRight", "PubicSymphysis"]

    def test_sij_indices(self) -> None:
        assert SIJ_INDICES == [0, 1]

    def test_sym_index(self) -> None:
        assert SYM_INDEX == 2

    def test_side_signs(self) -> None:
        assert CART_SIDE_SIGN["SIJCartilageLeft"] == +1.0
        assert CART_SIDE_SIGN["SIJCartilageRight"] == -1.0
        assert CART_SIDE_SIGN["PubicSymphysis"] == 0.0


class TestBoneTopology:
    def test_param_count(self) -> None:
        assert N_BONE == 2

    def test_param_names(self) -> None:
        assert BONE_PARAMS == ["alpha", "beta"]


# ── CartilageUncertaintySpec ─────────────────────────────────────────────

class TestCartilageSpec:
    def test_total_sigma_sij(self) -> None:
        spec = CartilageUncertaintySpec()
        expected = np.sqrt(0.35**2 + 0.25**2 + 0.10**2)
        assert abs(spec.total_sigma_sij() - expected) < 1e-10

    def test_total_sigma_sym(self) -> None:
        spec = CartilageUncertaintySpec()
        expected = np.sqrt(0.35**2 + 0.25**2)
        assert abs(spec.total_sigma_sym() - expected) < 1e-10

    def test_sym_smaller_than_sij(self) -> None:
        """Symphysis has no asymmetry contribution → smaller σ."""
        spec = CartilageUncertaintySpec()
        assert spec.total_sigma_sym() < spec.total_sigma_sij()


# ── Cartilage covariance ─────────────────────────────────────────────────

class TestCartilageCovariance:
    def test_shape(self) -> None:
        Sigma = build_cartilage_covariance(CartilageUncertaintySpec())
        assert Sigma.shape == (3, 3)

    def test_symmetric(self) -> None:
        Sigma = build_cartilage_covariance(CartilageUncertaintySpec())
        np.testing.assert_allclose(Sigma, Sigma.T)

    def test_positive_definite(self) -> None:
        Sigma = build_cartilage_covariance(CartilageUncertaintySpec())
        eigvals = np.linalg.eigvalsh(Sigma)
        assert np.all(eigvals > 0)

    def test_diagonal_sij(self) -> None:
        """SIJ diagonal = σ_g² + σ_t² + σ_a² (side² = 1)."""
        spec = CartilageUncertaintySpec()
        Sigma = build_cartilage_covariance(spec)
        expected = spec.sigma_global**2 + spec.sigma_type**2 + spec.sigma_asym**2
        np.testing.assert_allclose(Sigma[0, 0], expected)
        np.testing.assert_allclose(Sigma[1, 1], expected)

    def test_diagonal_sym(self) -> None:
        """Symphysis diagonal = σ_g² + σ_t² (no asym)."""
        spec = CartilageUncertaintySpec()
        Sigma = build_cartilage_covariance(spec)
        expected = spec.sigma_global**2 + spec.sigma_type**2
        np.testing.assert_allclose(Sigma[2, 2], expected)

    def test_sij_lr_correlation_positive(self) -> None:
        """Same-type SIJ L/R: Cov = σ_g² + σ_t² − σ_a² > 0."""
        spec = CartilageUncertaintySpec()
        Sigma = build_cartilage_covariance(spec)
        cov_lr = Sigma[0, 1]
        expected = spec.sigma_global**2 + spec.sigma_type**2 - spec.sigma_asym**2
        np.testing.assert_allclose(cov_lr, expected)
        assert cov_lr > 0, "L/R SIJ should be positively correlated"

    def test_sij_sym_covariance(self) -> None:
        """Cross-type (SIJ ↔ Sym) covariance = σ_g² only."""
        spec = CartilageUncertaintySpec()
        Sigma = build_cartilage_covariance(spec)
        np.testing.assert_allclose(Sigma[0, 2], spec.sigma_global**2)
        np.testing.assert_allclose(Sigma[1, 2], spec.sigma_global**2)

    def test_zero_asym_makes_sij_identical(self) -> None:
        """With σ_a=0 the L/R SIJ rows/cols are identical."""
        spec = CartilageUncertaintySpec(sigma_asym=0.0)
        Sigma = build_cartilage_covariance(spec)
        np.testing.assert_allclose(Sigma[0], Sigma[1])

    def test_implied_correlation_lr(self) -> None:
        """ρ(L,R) = (σ_g² + σ_t²) / (σ_g² + σ_t² + σ_a²)."""
        spec = CartilageUncertaintySpec()
        Sigma = build_cartilage_covariance(spec)
        # Actual correlation from covariance matrix
        rho = Sigma[0, 1] / np.sqrt(Sigma[0, 0] * Sigma[1, 1])
        # Since both SIJ have same variance, this simplifies
        expected_cov = spec.sigma_global**2 + spec.sigma_type**2 - spec.sigma_asym**2
        expected_var = spec.sigma_global**2 + spec.sigma_type**2 + spec.sigma_asym**2
        np.testing.assert_allclose(rho, expected_cov / expected_var, atol=1e-10)


# ── Bone covariance ──────────────────────────────────────────────────────

class TestBoneCovariance:
    def test_shape(self) -> None:
        Sigma = build_bone_covariance(BoneUncertaintySpec())
        assert Sigma.shape == (2, 2)

    def test_symmetric(self) -> None:
        Sigma = build_bone_covariance(BoneUncertaintySpec())
        np.testing.assert_allclose(Sigma, Sigma.T)

    def test_positive_definite(self) -> None:
        Sigma = build_bone_covariance(BoneUncertaintySpec())
        eigvals = np.linalg.eigvalsh(Sigma)
        assert np.all(eigvals > 0)

    def test_diagonal_values(self) -> None:
        spec = BoneUncertaintySpec()
        Sigma = build_bone_covariance(spec)
        np.testing.assert_allclose(Sigma[0, 0], spec.sigma_alpha**2)
        np.testing.assert_allclose(Sigma[1, 1], spec.sigma_beta**2)

    def test_off_diagonal(self) -> None:
        spec = BoneUncertaintySpec()
        Sigma = build_bone_covariance(spec)
        expected = spec.rho_alpha_beta * spec.sigma_alpha * spec.sigma_beta
        np.testing.assert_allclose(Sigma[0, 1], expected)

    def test_zero_correlation(self) -> None:
        spec = BoneUncertaintySpec(rho_alpha_beta=0.0)
        Sigma = build_bone_covariance(spec)
        np.testing.assert_allclose(Sigma[0, 1], 0.0)


# ── Dimensionless Jacobians ─────────────────────────────────────────────

class TestDimensionlessJacobians:
    """Test Jacobian computations with known inputs."""

    def test_cartilage_jacobian_shape(self) -> None:
        cart_sens = np.ones((5, 3))
        eigenvalues = np.arange(1, 6, dtype=float) * 100
        moduli = np.array([1.0, 1.0, 1.0])
        J = cartilage_dimensionless_jacobian(cart_sens, eigenvalues, moduli)
        assert J.shape == (5, 3)

    def test_cartilage_jacobian_values(self) -> None:
        """J = (E/λ) * ∂λ/∂E; with E=2, λ=4, sens=3 → J = 2/4 * 3 = 1.5."""
        cart_sens = np.array([[3.0, 3.0, 3.0]])
        eigenvalues = np.array([4.0])
        moduli = np.array([2.0, 2.0, 2.0])
        J = cartilage_dimensionless_jacobian(cart_sens, eigenvalues, moduli)
        np.testing.assert_allclose(J, 1.5)

    def test_bone_jacobian_shape(self) -> None:
        alpha_sens = np.ones(5)
        beta_sens = np.ones(5)
        eigenvalues = np.arange(1, 6, dtype=float)
        J = bone_dimensionless_jacobian(alpha_sens, beta_sens, eigenvalues, 10200.0, 2.0)
        assert J.shape == (5, 2)

    def test_bone_jacobian_values(self) -> None:
        """Ja = α/λ * ∂λ/∂α; Jb = β/λ * ∂λ/∂β."""
        alpha_sens = np.array([0.01])
        beta_sens = np.array([-5.0])
        eigenvalues = np.array([100.0])
        J = bone_dimensionless_jacobian(alpha_sens, beta_sens, eigenvalues, 10200.0, 2.0)
        np.testing.assert_allclose(J[0, 0], 10200.0 / 100.0 * 0.01)
        np.testing.assert_allclose(J[0, 1], 2.0 / 100.0 * (-5.0))

    def test_near_zero_eigenvalue_gives_nan(self) -> None:
        """When eigenvalue ≈ 0, Jacobian should produce NaN (safe division)."""
        cart_sens = np.array([[1.0, 1.0, 1.0]])
        eigenvalues = np.array([1e-15])
        moduli = np.array([1.0, 1.0, 1.0])
        J = cartilage_dimensionless_jacobian(cart_sens, eigenvalues, moduli)
        assert np.all(np.isnan(J))


# ── First-order propagation ─────────────────────────────────────────────

class TestCartilageFirstOrder:
    """Test cartilage first-order propagation."""

    @pytest.fixture
    def synthetic_data(self) -> tuple:
        """Create synthetic sensitivities for 5 modes."""
        rng = np.random.default_rng(42)
        n_modes = 5
        cart_sens = rng.uniform(0.001, 0.05, (n_modes, 3))
        eigenvalues = np.arange(100, 600, 100, dtype=float)
        moduli = np.array([1.0, 1.0, 1.0])
        return cart_sens, eigenvalues, moduli

    def test_returns_dataframe(self, synthetic_data) -> None:
        cart_sens, eigenvalues, moduli = synthetic_data
        df = cartilage_first_order_propagation(cart_sens, eigenvalues, moduli,
                                                CartilageUncertaintySpec())
        assert len(df) == 5
        assert "mode" in df.columns
        assert "CoV_lambda_cart" in df.columns

    def test_variance_nonnegative(self, synthetic_data) -> None:
        cart_sens, eigenvalues, moduli = synthetic_data
        df = cartilage_first_order_propagation(cart_sens, eigenvalues, moduli,
                                                CartilageUncertaintySpec())
        assert (df["var_log_lambda_cart"] >= 0).all()

    def test_fractions_sum_to_one(self, synthetic_data) -> None:
        cart_sens, eigenvalues, moduli = synthetic_data
        df = cartilage_first_order_propagation(cart_sens, eigenvalues, moduli,
                                                CartilageUncertaintySpec())
        total = df["cart_frac_global"] + df["cart_frac_type"] + df["cart_frac_asym"]
        np.testing.assert_allclose(total.values, 1.0, atol=1e-10)

    def test_mode_indexing_one_based(self, synthetic_data) -> None:
        cart_sens, eigenvalues, moduli = synthetic_data
        df = cartilage_first_order_propagation(cart_sens, eigenvalues, moduli,
                                                CartilageUncertaintySpec())
        assert list(df["mode"]) == [1, 2, 3, 4, 5]

    def test_zero_uncertainty_gives_zero_variance(self) -> None:
        spec = CartilageUncertaintySpec(sigma_global=0, sigma_type=0, sigma_asym=0)
        cart_sens = np.ones((3, 3))
        eigenvalues = np.array([100.0, 200.0, 300.0])
        moduli = np.array([1.0, 1.0, 1.0])
        df = cartilage_first_order_propagation(cart_sens, eigenvalues, moduli, spec)
        np.testing.assert_allclose(df["var_log_lambda_cart"].values, 0.0)

    def test_marginal_columns_present(self, synthetic_data) -> None:
        cart_sens, eigenvalues, moduli = synthetic_data
        df = cartilage_first_order_propagation(cart_sens, eigenvalues, moduli,
                                                CartilageUncertaintySpec())
        for name in CARTILAGE_PARAMS:
            assert f"marg_var_{name}" in df.columns


class TestBoneFirstOrder:
    """Test bone first-order propagation."""

    @pytest.fixture
    def synthetic_data(self) -> tuple:
        rng = np.random.default_rng(42)
        n_modes = 5
        alpha_sens = rng.uniform(1e-7, 1e-5, n_modes)
        beta_sens = rng.uniform(-0.01, -0.001, n_modes)
        eigenvalues = np.arange(100, 600, 100, dtype=float)
        return alpha_sens, beta_sens, eigenvalues

    def test_returns_dataframe(self, synthetic_data) -> None:
        alpha_sens, beta_sens, eigenvalues = synthetic_data
        df = bone_first_order_propagation(
            alpha_sens, beta_sens, eigenvalues,
            10200.0, 2.0, BoneUncertaintySpec(),
        )
        assert len(df) == 5
        assert "CoV_lambda_bone" in df.columns

    def test_variance_positive(self, synthetic_data) -> None:
        alpha_sens, beta_sens, eigenvalues = synthetic_data
        df = bone_first_order_propagation(
            alpha_sens, beta_sens, eigenvalues,
            10200.0, 2.0, BoneUncertaintySpec(),
        )
        # With default negative ρ and opposite-sign sensitivities,
        # total variance can still be positive
        # But var_alpha and var_beta are always >= 0
        assert (df["bone_var_alpha"] >= 0).all()
        assert (df["bone_var_beta"] >= 0).all()

    def test_decomposition_sums_to_total(self, synthetic_data) -> None:
        alpha_sens, beta_sens, eigenvalues = synthetic_data
        df = bone_first_order_propagation(
            alpha_sens, beta_sens, eigenvalues,
            10200.0, 2.0, BoneUncertaintySpec(),
        )
        recon = df["bone_var_alpha"] + df["bone_var_beta"] + df["bone_var_cross"]
        np.testing.assert_allclose(recon.values, df["var_log_lambda_bone"].values, atol=1e-15)

    def test_zero_correlation(self, synthetic_data) -> None:
        """With ρ=0 the cross-term should vanish."""
        alpha_sens, beta_sens, eigenvalues = synthetic_data
        spec = BoneUncertaintySpec(rho_alpha_beta=0.0)
        df = bone_first_order_propagation(
            alpha_sens, beta_sens, eigenvalues,
            10200.0, 2.0, spec,
        )
        np.testing.assert_allclose(df["bone_var_cross"].values, 0.0, atol=1e-15)


# ── Combined variance ────────────────────────────────────────────────────

class TestCombineVariances:
    """Test the combine_variances utility."""

    def test_additivity(self) -> None:
        """Total variance = sum of independent group variances."""
        import pandas as pd

        df_lig = pd.DataFrame({
            "mode": [1, 2],
            "eigenvalue": [100.0, 200.0],
            "var_log_lambda": [0.04, 0.01],
            "CoV_lambda": [0.2, 0.1],
        })
        df_cart = pd.DataFrame({
            "mode": [1, 2],
            "var_log_lambda_cart": [0.01, 0.005],
            "CoV_lambda_cart": [0.1, 0.07],
        })
        df_bone = pd.DataFrame({
            "mode": [1, 2],
            "var_log_lambda_bone": [0.005, 0.002],
            "CoV_lambda_bone": [0.07, 0.04],
        })

        merged = combine_variances(df_lig, df_cart, df_bone)
        np.testing.assert_allclose(
            merged["var_total"].values,
            [0.04 + 0.01 + 0.005, 0.01 + 0.005 + 0.002],
        )

    def test_fractions_sum_to_one(self) -> None:
        import pandas as pd

        df_lig = pd.DataFrame({
            "mode": [1], "eigenvalue": [100.0],
            "var_log_lambda": [0.04], "CoV_lambda": [0.2],
        })
        df_cart = pd.DataFrame({
            "mode": [1], "var_log_lambda_cart": [0.01], "CoV_lambda_cart": [0.1],
        })
        df_bone = pd.DataFrame({
            "mode": [1], "var_log_lambda_bone": [0.005], "CoV_lambda_bone": [0.07],
        })

        merged = combine_variances(df_lig, df_cart, df_bone)
        total_frac = merged["frac_lig"] + merged["frac_cart"] + merged["frac_bone"]
        np.testing.assert_allclose(total_frac.values, 1.0, atol=1e-10)


# ── Physical consistency ─────────────────────────────────────────────────

class TestPhysicalConsistency:
    """Sanity checks for typical parameter ranges."""

    def test_cartilage_cov_order_of_magnitude(self) -> None:
        """With realistic sensitivities (~0.01), CoV should be non-zero and < 100 %."""
        cart_sens = np.full((5, 3), 0.01)
        eigenvalues = np.full(5, 200.0)
        moduli = np.array([1.0, 1.0, 1.0])
        df = cartilage_first_order_propagation(
            cart_sens, eigenvalues, moduli, CartilageUncertaintySpec(),
        )
        # CoV should be small but non-zero
        assert (df["CoV_lambda_cart"] > 0).all()
        assert (df["CoV_lambda_cart"] < 1.0).all()

    def test_bone_negative_beta_sens_physical(self) -> None:
        """Bone β sensitivity is typically negative (higher exponent → stiffer).

        The variance should still be well-defined and positive.
        """
        alpha_sens = np.array([1e-6])
        beta_sens = np.array([-0.005])
        eigenvalues = np.array([200.0])
        df = bone_first_order_propagation(
            alpha_sens, beta_sens, eigenvalues, 10200.0, 2.0, BoneUncertaintySpec(),
        )
        assert df["var_log_lambda_bone"].iloc[0] > 0
