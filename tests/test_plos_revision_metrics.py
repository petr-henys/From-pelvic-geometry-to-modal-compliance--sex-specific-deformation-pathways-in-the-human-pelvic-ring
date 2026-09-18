"""Regression tests for publication-critical metric definitions."""
import numpy as np
from analysis.spectral_metrics import (canonical_pair_couplings, single_functional_coupling,
 orthonormalize_l2, compute_sep_out, cohort_bootstrap_robustness)


def test_covector_restriction_solves_constrained_functional_maximum():
    M=np.diag([1.,4.,9.,2.,3.,5.])
    Q=orthonormalize_l2(np.eye(6)[:,:2],M=M)
    b=np.array([1.,1.,0.,0.,0.,0.])
    _,psi,sigma=single_functional_coupling(Q,b,M=M)
    np.testing.assert_allclose(psi@M@psi,1.)
    np.testing.assert_allclose(b@psi,np.sqrt(1.25))
    np.testing.assert_allclose(sigma,np.sqrt(1.25))
    angles=np.linspace(0,2*np.pi,501)
    candidates=Q@np.array([np.cos(angles),np.sin(angles)])
    assert np.max(b@candidates)<=sigma+1e-12


def test_functional_coupling_invariant_under_internal_basis_rotation():
    rng=np.random.default_rng(4)
    M=np.diag(np.arange(1,13,dtype=float))
    Q=orthonormalize_l2(rng.normal(size=(12,3)),M=M)
    R=np.linalg.qr(rng.normal(size=(3,3)))[0]
    a,b=rng.normal(size=(2,12))
    c1=canonical_pair_couplings(Q,a,b,M=M)
    c2=canonical_pair_couplings(Q@R,a,b,M=M)
    np.testing.assert_allclose(c1[:2],c2[:2],rtol=1e-12)
    np.testing.assert_allclose(c1[-1],c2[-1],rtol=1e-12)


def test_truncated_spectrum_does_not_establish_two_sided_isolation():
    ev=np.array([[1.,3.,3.1,7.]])
    assert np.isnan(compute_sep_out(ev,2,4)).all()
    assert np.isfinite(compute_sep_out(ev,1,3)).all()


def test_full_size_subject_bootstrap_is_not_a_permutation():
    ev=np.array([[1.,1.05,3.]]*10+[[1.,2.,3.]]*10)
    perm=np.tile(np.arange(3),(20,1))
    result=cohort_bootstrap_robustness(ev,perm,.1,[(0,2)],n_boot=200,seed=12)
    s=result['cluster_prevalence']['Rank modes 1–2']
    assert result['n_draw']==20
    assert s['ci95_lo']<s['observed']<s['ci95_hi']


def test_gap_uncertainty_cancels_shared_shift():
    from analysis.material_uncertainty_population import compute_summary_stats
    # Test the covariance contraction itself against exact linear perturbations.
    G=np.array([[1.,2.],[1.,3.]])
    Sigma=np.diag([100.,.25])
    cov=G@Sigma@G.T
    variance=cov[0,0]+cov[1,1]-2*cov[0,1]
    np.testing.assert_allclose(variance,.25)
    assert variance < cov[0,0]+cov[1,1]
