from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.interpolate import RBFInterpolator
from analysis.spectral_data import _open_zarr_array,load_template_and_shapes,load_reference_space
from analysis.spectral_config import DATA_DIR_FULL
from simulation.data_mapper import calculate_bone_modulus
from dolfinx import fem
out=Path(__file__).parent;r={}
f=_open_zarr_array(DATA_DIR_FULL/'mode_energy_fraction.zarr');g=_open_zarr_array(DATA_DIR_FULL/'mode_energy_fraction_incremental.zarr')
valid=np.isfinite(g).all(axis=(1,2))&(abs(g.sum(axis=1)-1).max(axis=1)<1e-6)
ids=np.where(valid)[0];r['incremental']={'valid_subject_ids':ids.tolist(),'top_mode_agreement_per_load':(f[valid].argmax(axis=1)==g[valid].argmax(axis=1)).mean(axis=0).tolist() if len(ids) else []}
if len(ids):
 d=[]
 for cs,ce in [(0,3),(3,8),(8,10),(10,15)]:
  delta=g[valid,cs:ce].sum(axis=1)-f[valid,cs:ce].sum(axis=1)
  d.append({'block':f'{cs+1}-{ce}','max_individual_abs_pp':float(abs(delta).max()*100),'mean_change_pp':(delta.mean(axis=0)*100).tolist()})
 r['incremental']['blocks']=d
# Direct derivative of interpolate(modulus(density)) versus archived interpolation-of-density approximation.
p,V=load_reference_space();W=fem.functionspace(V.mesh,('DG',0));coords=W.tabulate_dof_coordinates()[::100]
tpl,_=load_template_and_shapes();density=np.load(ROOT/'data/HA.npy',mmap_mode='r')[0];rho=(np.maximum(density,0)+.09)/1.14
interp=lambda y:RBFInterpolator(tpl,y,smoothing=50.,neighbors=10)(coords)
E=interp(calculate_bone_modulus(density));rhom=interp(rho)
for param in ['alpha','beta']:
 base=10200. if param=='alpha' else 2.;h=base*1e-5
 kw1={param:base+h};kw2={param:base-h}
 fd=(interp(calculate_bone_modulus(density,**kw1))-interp(calculate_bone_modulus(density,**kw2)))/(2*h)
 approx=E*np.where(rhom>.486,(1/10200. if param=='alpha' else np.log(np.maximum(rhom,1e-12))),0.)
 mask=abs(fd)>1e-8
 r[param+'_material_derivative']={'n_query':len(coords),'relative_l2_error':float(np.linalg.norm(fd-approx)/np.linalg.norm(fd)),'median_relative_error_nonzero':float(np.median(abs(fd[mask]-approx[mask])/abs(fd[mask])))}
(out/'additional_checks.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
