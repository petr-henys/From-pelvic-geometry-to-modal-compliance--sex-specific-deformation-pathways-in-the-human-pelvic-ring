"""Shared statistical utilities for pelvic analysis.

Provides reusable statistical functions including:
- Bootstrap confidence intervals
- Effect size calculations (Cohen's d, rank-biserial)
- Multiple comparison corrections (FDR)
- Power analysis helpers

Eliminates code duplication across eigen_analysis.py, cartilage_sensitivity_analysis.py,
and ligament_sensitivity_analysis.py.
"""
from __future__ import annotations

import itertools
import math
import numpy as np
import statsmodels.api as sm
from scipy import stats
from typing import Tuple, Callable, Optional


class StatisticalAnalyzer:
    """Perform statistical analyses with confidence intervals and corrections."""
    
    @staticmethod
    def bootstrap_ci(
        data: np.ndarray,
        n_bootstrap: int = 1000,
        confidence: float = 0.95,
        statistic: Callable = np.mean
    ) -> Tuple[float, float]:
        """Compute bootstrap confidence interval.
        
        Parameters
        ----------
        data : array-like
            Sample data
        n_bootstrap : int
            Number of bootstrap iterations (default: 1000)
        confidence : float
            Confidence level (default: 0.95 for 95% CI)
        statistic : callable
            Function to compute on bootstrap samples (default: np.mean)
            
        Returns
        -------
        tuple
            (lower_bound, upper_bound)
        """
        if len(data) == 0:
            return (np.nan, np.nan)
        
        rng = np.random.default_rng(42)
        bootstrap_stats = np.empty(n_bootstrap)
        
        for i in range(n_bootstrap):
            sample = rng.choice(data, size=len(data), replace=True)
            bootstrap_stats[i] = statistic(sample)
        
        alpha = 1 - confidence
        lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
        upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))
        
        return lower, upper
    
    @staticmethod
    def bootstrap_ci_with_distribution(
        data: np.ndarray,
        n_bootstrap: int = 1000,
        confidence: float = 0.95,
        statistic: Callable = np.mean
    ) -> Tuple[float, float, np.ndarray]:
        """Compute bootstrap confidence interval with full distribution.
        
        Same as bootstrap_ci but also returns the bootstrap distribution.
        
        Returns
        -------
        tuple
            (lower_bound, upper_bound, bootstrap_distribution)
        """
        if len(data) == 0:
            return (np.nan, np.nan, np.array([]))
        
        rng = np.random.default_rng(42)
        bootstrap_stats = np.empty(n_bootstrap)
        
        for i in range(n_bootstrap):
            sample = rng.choice(data, size=len(data), replace=True)
            bootstrap_stats[i] = statistic(sample)
        
        alpha = 1 - confidence
        lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
        upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))
        
        return lower, upper, bootstrap_stats
    
    @staticmethod
    def cohens_d_ci(
        group1: np.ndarray,
        group2: np.ndarray,
        n_bootstrap: int = 1000,
        confidence: float = 0.95
    ) -> Tuple[float, float, float]:
        """Compute Cohen's d effect size with bootstrap confidence interval.
        
        Parameters
        ----------
        group1, group2 : array-like
            Two groups to compare
        n_bootstrap : int
            Number of bootstrap iterations (default: 1000)
        confidence : float
            Confidence level (default: 0.95)
            
        Returns
        -------
        tuple
            (cohens_d, ci_lower, ci_upper)
        """
        if len(group1) == 0 or len(group2) == 0:
            return np.nan, np.nan, np.nan
        
        # Point estimate
        pooled_std = np.sqrt((np.std(group1)**2 + np.std(group2)**2) / 2)
        d = (np.mean(group1) - np.mean(group2)) / pooled_std if pooled_std > 0 else 0
        
        # Bootstrap CI
        rng = np.random.default_rng(42)
        bootstrap_d = []
        
        for _ in range(n_bootstrap):
            sample1 = rng.choice(group1, size=len(group1), replace=True)
            sample2 = rng.choice(group2, size=len(group2), replace=True)
            pooled = np.sqrt((np.std(sample1)**2 + np.std(sample2)**2) / 2)
            if pooled > 0:
                bootstrap_d.append((np.mean(sample1) - np.mean(sample2)) / pooled)
        
        if len(bootstrap_d) == 0:
            return d, np.nan, np.nan
        
        bootstrap_d = np.array(bootstrap_d)
        alpha = 1 - confidence
        ci_lower = np.percentile(bootstrap_d, 100 * alpha / 2)
        ci_upper = np.percentile(bootstrap_d, 100 * (1 - alpha / 2))
        
        return d, ci_lower, ci_upper
    
    @staticmethod
    def rank_biserial_correlation(
        group1: np.ndarray,
        group2: np.ndarray,
        u_stat: Optional[float] = None
    ) -> float:
        """Compute rank-biserial correlation effect size for Mann-Whitney U test.
        
        Parameters
        ----------
        group1, group2 : array-like
            Two groups to compare
        u_stat : float, optional
            Pre-computed U statistic. If None, will be computed.
            
        Returns
        -------
        float
            Rank-biserial correlation coefficient
        """
        n1, n2 = len(group1), len(group2)
        
        if u_stat is None:
            u_stat, _ = stats.mannwhitneyu(group1, group2, alternative='two-sided')
        
        r_rb = 1 - (2 * u_stat) / (n1 * n2)
        return r_rb
    
    @staticmethod
    def fdr_correction(p_values: np.ndarray, alpha: float = 0.05) -> Tuple[np.ndarray, float]:
        """Benjamini-Hochberg FDR correction for multiple comparisons.
        
        Parameters
        ----------
        p_values : array-like
            Array of p-values to correct
        alpha : float
            Family-wise error rate (default: 0.05)
            
        Returns
        -------
        tuple
            (rejected, threshold) where rejected is boolean array of 
            significant tests and threshold is the critical value
        """
        p_values = np.asarray(p_values)
        n = len(p_values)
        
        # Sort p-values and track original indices
        sorted_idx = np.argsort(p_values)
        sorted_p = p_values[sorted_idx]
        
        # BH procedure: find largest k where p[k] <= (k/n)*alpha
        critical_vals = (np.arange(1, n + 1) / n) * alpha
        rejected = sorted_p <= critical_vals
        
        if rejected.any():
            max_idx = np.where(rejected)[0][-1]
            threshold = critical_vals[max_idx]
        else:
            threshold = 0.0
        
        # Restore original order
        adjusted = np.zeros(n, dtype=bool)
        adjusted[sorted_idx] = rejected
        
        return adjusted, threshold


# Convenience aliases for backward compatibility
StatisticalUtils = StatisticalAnalyzer


# ============================================================================
# OLS and Variance Decomposition Helpers
# ============================================================================

def add_const(X: np.ndarray) -> np.ndarray:
    """Add intercept column to design matrix."""
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    return np.column_stack([np.ones(X.shape[0]), X])


def ols_r2(y: np.ndarray, X: np.ndarray) -> float:
    """Compute OLS R^2 with intercept."""
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    model = sm.OLS(y, add_const(X)).fit()
    return float(model.rsquared)


def residualize_ols(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Compute residuals of y ~ 1 + X (OLS)."""
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    model = sm.OLS(y, add_const(X)).fit()
    return y - model.fittedvalues


def group_lmg_r2(y: np.ndarray, groups: list[tuple[str, np.ndarray]]) -> dict[str, float]:
    """Compute group-wise LMG/Shapley contributions to R^2.

    Parameters
    ----------
    y : np.ndarray
        Response vector, shape (n,)
    groups : list[tuple[str, np.ndarray]]
        Disjoint predictor groups: [(name, X_group), ...] where X_group has shape (n, p_g)

    Returns
    -------
    dict[str, float]
        Keys: group names + 'R2_total'. Values: absolute contributions (sum to R2_total)
    """
    y = np.asarray(y, float)
    groups_clean: list[tuple[str, np.ndarray]] = []
    for name, Xg in groups:
        Xg = np.asarray(Xg, float)
        if Xg.ndim == 1:
            Xg = Xg.reshape(-1, 1)
        groups_clean.append((name, Xg))

    names = [g[0] for g in groups_clean]
    G = len(groups_clean)
    if G == 0:
        return {"R2_total": 0.0}

    # Precompute full R^2
    X_full = np.concatenate([g[1] for g in groups_clean], axis=1)
    R2_full = ols_r2(y, X_full)

    contrib = {name: 0.0 for name in names}
    
    # Average over all permutations
    for order in itertools.permutations(range(G)):
        X_so_far = None
        R2_prev = 0.0
        for idx in order:
            name, Xg = groups_clean[idx]
            X_next = Xg if X_so_far is None else np.concatenate([X_so_far, Xg], axis=1)
            R2_next = ols_r2(y, X_next)
            inc = R2_next - R2_prev
            if np.isfinite(inc) and inc > 0:
                contrib[name] += float(inc)
            R2_prev = R2_next
            X_so_far = X_next

    # Average
    n_perm = float(math.factorial(G))
    for k in contrib:
        contrib[k] /= n_perm

    contrib["R2_total"] = float(R2_full)
    return contrib


def fit_beta(log_scale: np.ndarray, sex_codes: np.ndarray, eigenvalues: np.ndarray) -> float:
    """Fit log(eigenvalue) ~ log(scale) + sex and return scale coefficient.
    
    Parameters
    ----------
    log_scale : np.ndarray
        Log-transformed scale values
    sex_codes : np.ndarray
        Binary sex codes (0/1)
    eigenvalues : np.ndarray
        Eigenvalues to regress
        
    Returns
    -------
    float
        Scale coefficient (beta)
    """
    valid_mask = np.isfinite(log_scale) & np.isfinite(sex_codes) & np.isfinite(eigenvalues) & (eigenvalues > 0)
    if valid_mask.sum() < 10:
        return np.nan
    
    y = np.log(eigenvalues[valid_mask])
    X = np.column_stack([
        np.ones(valid_mask.sum()),
        log_scale[valid_mask],
        sex_codes[valid_mask],
    ])
    
    model = sm.OLS(y, X).fit()
    return float(model.params[1])
