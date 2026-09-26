"""Independent audit of cached statistics, provenance and modal-capacity optima.
Does not modify original analyses or FE arrays.
"""
from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd
from scipy import stats,optimize
from scipy.spatial import cKDTree
from statsmodels.stats.multitest import multipletests
from analysis.spectral_data import *
from analysis.spectral_config import DATA_DIR_FULL
from analysis.spectral_metrics import orthonormalize_l2,build_patient_measurement_vectors,single_functional_capacity_max_nodal,compute_gap_in,detect_clusters
out=Path(__file__).parent
report={}
prov=json.loads((ROOT/'analysis_outputs/plos_revision/provenance.json').read_text())
report['source_hash_matches']={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in prov['source_hashes'].items()}
sex,age=load_metadata();report['demography']={'n':len(age),'female':int((sex=='F').sum()),'min_age':float(age.min()),'max_age':float(age.max()),'mean_age':float(age.mean()),'sd_age':float(age.std(ddof=1))}
ev=load_eigenvalues(DATA_DIR_FULL);gap=compute_gap_in(ev);eps=float(np.percentile(gap,25));cl=detect_clusters(ev,eps)
report['spectra']={'all_positive':bool((ev>0).all()),'all_sorted':bool((np.diff(ev,axis=1)>=0).all()),'threshold':eps,'cluster_9_10':sum((8,10) in c for c in cl)}
df=pd.read_csv(ROOT/'analysis_outputs/plos_revision/tables/subject_metrics.csv');key=df[(df.block=='9-10')&df.member];rng=np.random.default_rng(20260918);rows=[]
for metric in ['rank9','rank10','AP','AP_pair']:
 a=key.loc[key.swap==0,metric].to_numpy();b=key.loc[key.swap==1,metric].to_numpy();u,p=stats.mannwhitneyu(a,b)
 ai=rng.integers(len(a),size=(5000,len(a)));bi=rng.integers(len(b),size=(5000,len(b)))
 d=np.median(b[bi],axis=1)-np.median(a[ai],axis=1)
 rows.append(dict(metric=metric,n0=len(a),n1=len(b),median0=float(np.median(a)),median1=float(np.median(b)),p=float(p),ci_lo=float(np.quantile(d,.025)),ci_hi=float(np.quantile(d,.975)),rank_biserial=float(1-2*u/(len(a)*len(b)))))
report['label_effects_recomputed']=rows
report['q_recomputed']=multipletests([r['p'] for r in rows],method='fdr_bh')[1].tolist()
from analysis.spectral_data import _open_zarr_array
fr=_open_zarr_array(DATA_DIR_FULL/'mode_energy_fraction.zarr')
report['energy']={'max_sum_error':float(abs(fr.sum(axis=1)-1).max()),'SP2_mode2':float(fr[:,1,0].mean()),'SP1_modes1_3':float(fr[:,:3,1].sum(axis=1).mean()),'LAB1_modes9_10':float(fr[:,8:10,2].sum(axis=1).mean())}
tpl,shapes=load_template_and_shapes();sizes=np.sqrt(((shapes-shapes.mean(axis=1,keepdims=True))**2).sum(axis=2).mean(axis=1));sc=pd.read_excel(DATA_DIR_FULL/'allometry.xlsx').scale.to_numpy()
report['geometry_size']={'rms_min_mm':float(sizes.min()),'rms_max_mm':float(sizes.max()),'rms_cv':float(sizes.std()/sizes.mean()),'rms_scale_correlation':float(np.corrcoef(sizes,sc)[0,1]),'scale_min':float(sc.min()),'scale_max':float(sc.max())}
(out/'results_checks.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
# Convex QCQP: maximise c^T a under one quadratic bound per mesh node.
# Positive dual multipliers provide an independent upper bound on the optimum.
coords=load_fe_mesh_coords();M=load_mass_matrix();eig=load_eigenvectors_obj(DATA_DIR_FULL);lm=load_inlet_landmarks();ids=[0,101,120,206]
bv=build_patient_measurement_vectors(cKDTree(coords),len(coords),tpl,shapes[ids],*lm['AP']);checks=[]
for ii,i in enumerate(ids):
 for cs,ce in [(8,10),(11,15)]:
  Q=orthonormalize_l2(np.asarray(eig[i,cs:ce]).reshape(ce-cs,-1).T,M=M)
  # Rescaling keeps the small optimisation in reasonable units.
  Q=Q/np.max(np.linalg.norm(Q.reshape(-1,3,ce-cs),axis=1));b=bv[ii];c=Q.T@b
  H=np.einsum('nki,nkj->nij',Q.reshape(-1,3,ce-cs),Q.reshape(-1,3,ce-cs))
  def fun(a):return -c@a
  def cons(a):return 1-np.einsum('i,nij,j->n',a,H,a)
  def jac(a):return -2*np.einsum('nij,j->ni',H,a)
  a0=c/np.linalg.norm(c);a0/=np.sqrt(np.max(1-cons(a0)))
  res=optimize.minimize(fun,a0,jac=lambda a:-c,constraints=[{'type':'ineq','fun':cons,'jac':jac}],method='SLSQP',options={'ftol':1e-11,'maxiter':100})
  a=res.x;active=np.where(cons(a)<1e-6)[0];lam,_=optimize.nnls((-jac(a)[active]).T,c)
  HH=np.einsum('n,nij->ij',lam,H[active]);dual=float(lam.sum()+.25*c@np.linalg.pinv(HH)@c)
  cap,_,_=single_functional_capacity_max_nodal(Q,b);primal=float(c@a)
  row={'subject':i,'block':f'{cs+1}-{ce}','implementation':cap,'convex_primal':primal,'dual_upper':dual,'constraint_min':float(cons(a).min()),'success':bool(res.success),'absolute_difference':float(abs(primal-cap))}
  checks.append(row);print(row,flush=True)
(out/'capacity_checks.json').write_text(json.dumps(checks,indent=2))
