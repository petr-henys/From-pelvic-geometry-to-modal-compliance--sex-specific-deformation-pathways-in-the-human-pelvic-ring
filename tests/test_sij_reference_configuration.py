"""Regression checks for anatomy/load separation and rotation reconstruction."""
import ast
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from anatomy_analyser import _euler_xyz_in_frame, _chunked_min_dists


def test_cardan_reconstructs_regular_and_both_gimbal_branches():
    for angles in ([13,21,-17],[31,90,-23],[31,-90,-23],[31,89.999,-23],[31,-89.999,-23],[0,0,0]):
        matrix=Rotation.from_euler('xyz',angles,degrees=True).as_matrix()
        extracted=_euler_xyz_in_frame(matrix,np.eye(3))
        np.testing.assert_allclose(Rotation.from_euler('xyz',extracted,degrees=True).as_matrix(),matrix,atol=1e-12)


def test_nearest_neighbors_equal_direct_distances():
    rng=np.random.default_rng(4);a=rng.normal(size=(37,3));b=rng.normal(size=(53,3))
    d,i=_chunked_min_dists(a,b)
    direct=np.linalg.norm(a[:,None,:]-b[None,:,:],axis=2)
    np.testing.assert_allclose(d,direct.min(axis=1),atol=1e-12)
    np.testing.assert_array_equal(i,direct.argmin(axis=1))
    empty_d, empty_i = _chunked_min_dists(a, np.empty((0, 3)))
    assert np.isinf(empty_d).all()
    assert (empty_i == -1).all()


def test_load_extractor_uses_subject_reference_not_template():
    # Exercise actual function without importing the optional FE runtime.
    source=Path('simulation/simulation_loads.py').read_text()
    tree=ast.parse(source)
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_compute_sij_stats')
    fn.returns=None
    for arg in fn.args.args:arg.annotation=None
    class Mesh:
        def __init__(self,points):self.points=np.array(points,dtype=float)
        def copy(self,deep=True):return Mesh(self.points.copy())
    class Solver:
        def solve(self):return 'solution'
        def mapped_coordinates(self,p):return p + np.array([3.,-2.,7.])
        def deform(self,p):return self.mapped_coordinates(p) + np.array([0.01,0.02,-0.03])
    observed={}
    def capture(ref,loaded):
        observed['ref']=ref.points;observed['loaded']=loaded.points
        return {}
    scope={'sij_relative_angles':capture}
    exec(compile(ast.Module(body=[fn],type_ignores=[]),'<extractor>','exec'),scope)
    template=Mesh([[1,2,3],[4,5,6]])
    scope['_compute_sij_stats'](Solver(),template)
    np.testing.assert_allclose(observed['ref'],template.points+[3,-2,7])
    np.testing.assert_allclose(observed['loaded']-observed['ref'],np.tile([.01,.02,-.03],(2,1)))


def test_cached_rbf_weights_match_direct_interpolation():
    import importlib.util
    from scipy.interpolate import RBFInterpolator
    spec=importlib.util.spec_from_file_location('sij_rbf_weights','manuscripts/paper_JAnat_SIJ_micromotion/analysis/rbf_weights.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    rng=np.random.default_rng(17)
    sites=rng.normal(size=(90,3));targets=rng.normal(size=(35,3));fields=rng.normal(size=(90,8))
    for smoothing in [0.,50.]:
        actual=module.weights(sites,targets,neighbors=10,smoothing=smoothing)@fields
        expected=RBFInterpolator(sites,fields,neighbors=10,smoothing=smoothing)(targets)
        np.testing.assert_allclose(actual,expected,atol=1e-10,rtol=1e-10)
