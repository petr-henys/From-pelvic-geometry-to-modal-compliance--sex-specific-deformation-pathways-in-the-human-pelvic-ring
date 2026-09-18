import numpy as np
from analysis.global_mode_metrics import global_motion_profile


def test_exact_rigid_motion_and_sign_invariance():
    rng=np.random.default_rng(41)
    x=rng.normal(size=(100,3)); w=rng.uniform(.1,2,100)
    r=x-np.average(x,axis=0,weights=w)
    omega=np.array([.4,-.2,.7]); u=np.array([1.,2.,3.])+np.cross(omega,r)
    p=global_motion_profile(x,u,w); q=global_motion_profile(x,-5*u,w)
    np.testing.assert_allclose(p['omega'],omega,atol=1e-12)
    assert p['f_nonrigid'] < 1e-25
    for key in ['axis','f_translation','f_rotation','f_nonrigid']:
        np.testing.assert_allclose(p[key],q[key],atol=1e-12)


def test_expansion_has_no_rotation_axis():
    x=np.array([[a,b,c] for a in [-1,1] for b in [-1,1] for c in [-1,1]],float)
    p=global_motion_profile(x,x,np.ones(len(x)))
    assert np.isnan(p['axis']).all()
    np.testing.assert_allclose(p['f_nonrigid'],1)


def test_coordinate_rotation_covariance():
    rng=np.random.default_rng(2); x=rng.normal(size=(80,3)); u=rng.normal(size=x.shape)
    q,_=np.linalg.qr(rng.normal(size=(3,3)))
    p=global_motion_profile(x,u,np.ones(80)); r=global_motion_profile(x@q,u@q,np.ones(80))
    for key in ['f_translation','f_rotation','f_nonrigid']:
        np.testing.assert_allclose(p[key],r[key],atol=1e-12)
    np.testing.assert_allclose(abs(np.dot(p['axis']@q,r['axis'])),1)
