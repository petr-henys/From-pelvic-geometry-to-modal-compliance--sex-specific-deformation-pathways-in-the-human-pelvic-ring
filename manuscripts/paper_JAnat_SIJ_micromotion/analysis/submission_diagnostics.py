"""Serial, bounded checks of existing P1 mappings and saved displacements; no FE solve.

Writes separate revision outputs; never overwrites primary estimates or fields.
"""
from pathlib import Path
import argparse,json,time,sys
import numpy as np
import pandas as pd
import pyvista as pv
from scipy.interpolate import RBFInterpolator
import recompute_sij_reference as r
from anatomy_analyser import sij_relative_angles
OUT=Path(__file__).resolve().parents[1]/'tables'/'revision'/'extraction_pilot'

def main():
 p=argparse.ArgumentParser();p.add_argument('--limit',type=int,default=3);p.add_argument('--max-seconds',type=float,default=120);a=p.parse_args()
 OUT.mkdir(parents=True,exist_ok=True); t0=time.monotonic();r.init()
 mesh=pv.read(r.ROOT/'results'/r.CHANNELS['full']/'paraview/reference_fields.vtk')
 assert np.all(mesh.celltypes==10), 'Only linear tetrahedra supported'
 assert np.array_equal(mesh.points,r.DOFS['full'])
 cells=mesh.cells.reshape(-1,5);assert np.all(cells[:,0]==4);ids=cells[:,1:]
 X=np.asarray(mesh.points,dtype=float);D=X[ids[:,1:]]-X[ids[:,0,None]];det0=np.linalg.det(D)
 assert np.all(np.abs(det0)>1e-12)
 summary=[];sens=[]
 for i in range(min(a.limit,len(r.SHAPES))):
  f=OUT/f'subject_{i:03d}.json'
  if f.exists():
   saved=json.loads(f.read_text());summary.append(saved['mapping']);sens.extend(saved['sensitivity']);continue
  start=time.monotonic();phi=r.W_SHAPE['full']@(r.SHAPES[i]-r.TEMPLATE.points)
  Y=X+phi;J=np.linalg.det(Y[ids[:,1:]]-Y[ids[:,0,None]])/det0
  row=dict(subject_idx=i,n_cells=len(J),min_J=float(J.min()),p01_J=float(np.quantile(J,.01)),median_J=float(np.median(J)),max_J=float(J.max()),n_nonpositive=int((J<=0).sum()),n_nonfinite=int((~np.isfinite(J)).sum()))
  # Independent direct RBF evaluation at nodes of the worst cell.
  worst=ids[np.argmin(J)];cfg=r.CFG['full']
  direct=RBFInterpolator(r.TEMPLATE.points,r.SHAPES[i]-r.TEMPLATE.points,neighbors=cfg['RBF_DATA_NEIGHBORS'],smoothing=cfg['RBF_DATA_SMOOTHING'])(X[worst])
  np.testing.assert_allclose(direct,phi[worst],rtol=1e-8,atol=1e-8)
  row['direct_min_cell_J']=float(np.linalg.det((X[worst]+direct)[1:]-(X[worst]+direct)[0])/det0[np.argmin(J)])
  rows=[]
  fields=np.concatenate([phi]+[np.asarray(r.STORES['full'][l][i]) for l in r.LOADS],axis=1)
  surface=r.W_SURFACE['full']@fields
  unloaded=r.TEMPLATE.copy(deep=True);unloaded.points=r.TEMPLATE.points+surface[:,:3]
  with np.load(r.OUT/f'subject_{i:03d}.npz') as base:
   for j,load in enumerate(r.LOADS):
    loaded=unloaded.copy(deep=True);loaded.points=unloaded.points+surface[:,3*(j+1):3*(j+2)]
    angles,trans=r.arrays(sij_relative_angles(unloaded,loaded,sacrum_correction='rigid'))
    ba=base[f'full_{load}_angles'];bt=base[f'full_{load}_trans']
    if i==0:
     aa,at=r.arrays(sij_relative_angles(unloaded,loaded))
     np.testing.assert_allclose(aa,ba,atol=1e-8,rtol=1e-8);np.testing.assert_allclose(at,bt,atol=1e-8,rtol=1e-8)
    rows.append(dict(subject_idx=i,load_case=load,affine_rotation=float(np.linalg.norm(ba,axis=1).mean()),rigid_rotation=float(np.linalg.norm(angles,axis=1).mean()),affine_translation=float(np.linalg.norm(bt,axis=1).mean()),rigid_translation=float(np.linalg.norm(trans,axis=1).mean())))
  f.write_text(json.dumps(dict(mapping=row,sensitivity=rows,seconds=time.monotonic()-start),indent=2));summary.append(row);sens.extend(rows)
  print(f'{i+1}/{a.limit}: {time.monotonic()-start:.2f}s; min J={J.min():.6g}; nonpositive={row["n_nonpositive"]}',flush=True)
  if time.monotonic()-t0>a.max_seconds:
   print('Time budget reached; retaining completed diagnostic rows only.',flush=True);break
 pd.DataFrame(summary).to_csv(OUT/'mapping_quality.csv',index=False)
 df=pd.DataFrame(sens);df.to_csv(OUT/'rigid_correction_sensitivity.csv',index=False)
 out=[]
 for load,g in [('All',df)]+list(df.groupby('load_case',sort=False)):
  for metric in ['rotation','translation']:
   d=(g[f'rigid_{metric}']-g[f'affine_{metric}']).abs()
   out.append(dict(load_case=load,metric=metric,n=len(g),median=d.median(),p95=d.quantile(.95),maximum=d.max()))
 pd.DataFrame(out).to_csv(OUT/'rigid_correction_summary.csv',index=False)
 print(json.dumps(dict(subjects=len(summary),elapsed_seconds=time.monotonic()-t0,min_J=min(x['min_J'] for x in summary),nonpositive=sum(x['n_nonpositive'] for x in summary)),indent=2),flush=True)
if __name__=='__main__':main()
