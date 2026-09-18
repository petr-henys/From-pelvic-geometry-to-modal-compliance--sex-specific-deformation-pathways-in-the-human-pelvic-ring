import numpy as np
import pytest
from analysis.global_strain_mechanisms import GlobalStrainModel, modal_cell_strain, MECHANISMS


def geometry():
    return np.array([[x,y,z] for x in [-2,-1,1,2] for y in [-2,-1,1,2]
                     for z in np.linspace(-3,3,20)],float)


@pytest.mark.parametrize('mechanism',['axial','bending','shear','twist'])
def test_canonical_whole_domain_mechanisms(mechanism):
    x=geometry(); eps=np.zeros((len(x),3,3))
    if mechanism=='axial': eps[:,2,2]=.2
    if mechanism=='bending': eps[:,2,2]=.1*x[:,0]-.2*x[:,1]
    if mechanism=='shear': eps[:,0,2]=eps[:,2,0]=.3
    if mechanism=='twist':
        eps[:,0,2]=eps[:,2,0]=-.1*x[:,1]
        eps[:,1,2]=eps[:,2,1]=.1*x[:,0]
    model=GlobalStrainModel(x,np.ones(len(x)),[0,0,1],8)
    p=model.decompose(eps); q=model.decompose(-eps*1e-15)
    np.testing.assert_allclose(p['f_'+mechanism],1,atol=1e-10)
    for key in MECHANISMS: np.testing.assert_allclose(p['f_'+key],q['f_'+key],atol=1e-10)


def test_rigid_rotation_has_zero_strain_and_is_not_torsion():
    x=np.array([[0.,0,0],[1,0,0],[0,1,0],[0,0,1]])
    u=np.array([2.,3.,4.])+np.cross([0.,0.,2.],x)
    eps=modal_cell_strain(u,np.array([[0,1,2,3]]),np.eye(3)[None])
    np.testing.assert_allclose(eps,0,atol=1e-12)
    model=GlobalStrainModel(geometry(),np.ones(len(geometry())),[0,0,1])
    assert not model.decompose(np.zeros((len(geometry()),3,3)))['is_valid']


def test_mixed_patterns_close_and_rotate_with_coordinates():
    x=geometry(); rng=np.random.default_rng(20)
    a=rng.normal(size=(len(x),3,3)); eps=(a+a.transpose(0,2,1))/2
    w=rng.uniform(.5,2,len(x)); q,_=np.linalg.qr(rng.normal(size=(3,3)))
    p=GlobalStrainModel(x,w,[0,0,1]).decompose(eps)
    e=np.einsum('ki,nkl,lj->nij',q,eps,q)
    r=GlobalStrainModel(x@q,w,np.array([0,0,1])@q).decompose(e)
    np.testing.assert_allclose(sum(p['f_'+k] for k in MECHANISMS),1)
    for k in MECHANISMS: np.testing.assert_allclose(p['f_'+k],r['f_'+k],atol=1e-10)
