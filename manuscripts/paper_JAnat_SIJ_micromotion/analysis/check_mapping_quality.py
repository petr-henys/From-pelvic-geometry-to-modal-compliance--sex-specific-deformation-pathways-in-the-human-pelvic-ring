"""Exact cellwise Jacobian of the implemented P1 geometry map; no FE solves."""
from pathlib import Path
import sys,json,time
import numpy as np,pandas as pd,pyvista as pv
from scipy.interpolate import RBFInterpolator
from rbf_weights import weights
P=Path(__file__).resolve().parents[1];R=P.parents[1];O=P/'tables/revision';O.mkdir(exist_ok=True)
t0=time.monotonic();base=R/'results/ref_S1P_fixed_new2';cfg=json.loads((base/'simulation_metadata.json').read_text())['config']
m=pv.read(base/'paraview/reference_fields.vtk');T=pv.read(R/'data/pelvic.vtk');shapes=np.load(R/'data/X.npy',mmap_mode='r')
assert np.all(m.celltypes==10);cells=m.cells.reshape(-1,5);assert np.all(cells[:,0]==4);ids=cells[:,1:]
X=np.asarray(m.points,dtype=float);ref=np.linalg.det(X[ids[:,1:]]-X[ids[:,0,None]]);assert np.all(np.abs(ref)>1e-12)
W=weights(T.points,X,cfg['RBF_DATA_NEIGHBORS'],cfg['RBF_DATA_SMOOTHING']);rows=[];bad=[]
for i in range(len(shapes)):
 phi=W@(shapes[i]-T.points);Y=X+phi;J=np.linalg.det(Y[ids[:,1:]]-Y[ids[:,0,None]])/ref
 # Validate the worst cell independently of sparse weights for each subject.
 worst=int(np.argmin(J));v=ids[worst];direct=RBFInterpolator(T.points,shapes[i]-T.points,neighbors=cfg['RBF_DATA_NEIGHBORS'],smoothing=cfg['RBF_DATA_SMOOTHING'])(X[v]);np.testing.assert_allclose(direct,phi[v],rtol=1e-8,atol=1e-8)
 # F from the P1 gradient provides a second algebraic check.
 F=(Y[v][1:]-Y[v][0]).T@np.linalg.inv((X[v][1:]-X[v][0]).T)
 np.testing.assert_allclose(np.linalg.det(F),J[worst],rtol=1e-8,atol=1e-8)
 rows.append(dict(subject_idx=i,n_cells=len(J),min_J=float(J.min()),p001_J=float(np.quantile(J,.001)),p01_J=float(np.quantile(J,.01)),median_J=float(np.median(J)),max_J=float(J.max()),n_nonpositive=int((J<=0).sum()),n_nonfinite=int((~np.isfinite(J)).sum()),reference_volume_fraction_nonpositive=float(np.abs(ref[J<=0]).sum()/np.abs(ref).sum())))
 for c in np.where(J<=0)[0]:bad.append(dict(subject_idx=i,cell_index=int(c),J=float(J[c]),reference_volume_mm3=float(abs(ref[c])/6),x=float(X[ids[c]].mean(0)[0]),y=float(X[ids[c]].mean(0)[1]),z=float(X[ids[c]].mean(0)[2])))
 if i%50==0:print(i,rows[-1]['min_J'],flush=True)
 if time.monotonic()-t0>300:raise RuntimeError('Mapping check exceeded conservative five-minute cap; no incomplete summary published')
d=pd.DataFrame(rows);d.to_csv(O/'mapping_quality.csv',index=False);pd.DataFrame(bad).to_csv(O/'mapping_nonpositive_cells.csv',index=False)
summary=dict(n_subjects=len(d),n_cells_per_subject=len(ref),n_subject_cell_pairs=len(d)*len(ref),min_J=float(d.min_J.min()),max_J=float(d.max_J.max()),n_subjects_nonpositive=int((d.n_nonpositive>0).sum()),n_nonpositive=int(d.n_nonpositive.sum()),n_distinct_cells_nonpositive=len(set(x['cell_index'] for x in bad)),max_nonpositive_cells_per_subject=int(d.n_nonpositive.max()),n_nonfinite=int(d.n_nonfinite.sum()),min_subject_p01_J=float(d.p01_J.min()),max_subject_p01_J=float(d.p01_J.max()),max_reference_volume_fraction_nonpositive=float(d.reference_volume_fraction_nonpositive.max()),elapsed_seconds=time.monotonic()-t0,verification='Every subject: direct scipy RBF at worst-cell vertices; independent det(I+grad phi) check',scope='Full and geometry-preserved variants share these maps; density-preserved/template-geometry J=1')
(O/'mapping_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
