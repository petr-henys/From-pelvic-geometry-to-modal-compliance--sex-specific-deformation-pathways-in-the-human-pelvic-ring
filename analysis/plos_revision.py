#!/usr/bin/env python3
"""Rebuild PLOS tables and figures from solver-order data, with explicit provenance.

Run from the repository root with the supplied FE environment, on one MPI rank:
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .conda/bin/python analysis/plos_revision.py
A cached per-subject table can be replotted using --plots-only. Source hashes
are checked before accepting the cache. No FE simulation arrays are modified.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from scipy.spatial import cKDTree
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from analysis.spectral_data import (load_eigenvalues,load_permutations,load_eigenvectors_obj,
 load_fe_mesh_coords,load_mass_matrix,load_metadata,load_pelvic_dimensions,
 load_template_and_shapes,load_inlet_landmarks,load_outlet_landmarks,_open_zarr_array)
from analysis.spectral_metrics import (compute_gap_in,detect_clusters,compute_sep_out,
 orthonormalize_l2,canonical_pair_couplings,single_functional_coupling,single_functional_capacity_max_nodal,
 coupling_per_1mm_max,build_patient_measurement_vectors,invert_pairing_permutation,
 compute_principal_angles,grassmann_distance,cohort_bootstrap_robustness)
from analysis.spectral_config import DATA_DIR_FULL,DATA_DIR_SHAPE,DATA_DIR_MATERIAL

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'analysis_outputs/plos_revision'; FIG=OUT/'figures'; TAB=OUT/'tables'
from analysis.publication_style import (
    COL1_WIDTH, COL15_WIDTH, COL2_WIDTH, FULL_WIDTH,
    MALE_COLOR, FEMALE_COLOR, BACKBONE_COLOR, MIDRANK_COLOR, INLET_SWAP_COLOR, HIGHER_RESERVE_COLOR,
    SWAP_COLOR, NOSWAP_COLOR, UNPAIRED_COLOR, NEUTRAL_COLOR, BLOCK_COLORS,
    apply_publication_style, panel_label, style_axis, style_distribution,
    style_scatter, style_heatmap, style_colorbar, format_pvalue, save_publication_figure
)
from matplotlib.colors import TwoSlopeNorm

MALE=MALE_COLOR; FEMALE=FEMALE_COLOR; COLORS=[MALE,FEMALE,INLET_SWAP_COLOR,HIGHER_RESERVE_COLOR,'#777777']
LOADS=['SP2leg','SP1leg','LAB1','LAB2','LAB3']
AXES=['AP','ML','BIS','BIT','OUTLETAP']
RNG_SEED=20260918

def style():
 apply_publication_style()

def save(fig,name):
 save_publication_figure(fig, FIG/name, formats=('pdf','png'), dpi=400, verbose=False)

def panel(ax,title):
 parts = title.split('  ', 1)
 if len(parts) == 2 and len(parts[0]) == 1:
  panel_label(ax, parts[0])
  ax.set_title(parts[1], loc='left', fontsize=8.5, color='#374151', pad=4.0)
 else:
  panel_label(ax, title[:1])
  ax.set_title(title[1:].strip(), loc='left', fontsize=8.5, color='#374151', pad=4.0)
 style_axis(ax, spines=('left','bottom'), y_grid=True)

def table(df,name):
 df.to_csv(TAB/f'{name}.csv',index=False)
 # Escape all column headers; mathematics is provided by captions, not headers.
 df.to_latex(TAB/f'{name}.tex',index=False,escape=True,na_rep='--',float_format='%.3f')

def source_fingerprint():
 names=['analysis/plos_revision.py','analysis/spectral_data.py','analysis/spectral_metrics.py',
 'results/ref_S1P_fixed_new2/simulation_metadata.json','anatomy_data/full_model_new.feb',
 'anatomy_data/birth_canal/anterior_posterior_inlet_diameter.mrk.json',
 'anatomy_data/birth_canal/pelvis_inlet_transverse.mrk.json']
 return {p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in names}

def basic():
 ev=load_eigenvalues(DATA_DIR_FULL); perm=load_permutations(DATA_DIR_FULL)
 if not np.isfinite(ev).all() or np.any(ev<=0) or np.any(np.diff(ev,axis=1)<0):
  raise ValueError('Expected finite positive eigenvalues in solver rank order')
 sex,age=load_metadata(); gaps=compute_gap_in(ev); eps=float(np.percentile(gaps,25))
 clusters=detect_clusters(ev,eps); counts=Counter(c for cc in clusters for c in cc)
 top=counts.most_common(5)
 return ev,perm,sex,age,gaps,eps,clusters,counts,top

def compute_subjects():
 ev,perm,sex,age,gaps,eps,clusters,counts,top=basic(); n=len(ev)
 M=load_mass_matrix(); coords=load_fe_mesh_coords(); tree=cKDTree(coords)
 tpl,shapes=load_template_and_shapes(); lm={**load_inlet_landmarks(),**load_outlet_landmarks()}
 measurements={}
 for key in AXES:
  print('Measurement',key,flush=True)
  measurements[key]=build_patient_measurement_vectors(tree,len(coords),tpl,shapes,*lm[key])
 dims=load_pelvic_dimensions(); inv=invert_pairing_permutation(perm)
 fractions=_open_zarr_array(DATA_DIR_FULL/'mode_energy_fraction.zarr')
 errors=_open_zarr_array(DATA_DIR_FULL/'mode_reconstruction_error.zarr')
 if not np.allclose(fractions.sum(axis=1),1,atol=1e-6):raise ValueError('Energy fractions not normalized')
 eigvecs=load_eigenvectors_obj(DATA_DIR_FULL)
 # Track three reference choices inside each block: 25th/50th/75th percentile of the first eigenvalue.
 refs={}; blocks=list(dict.fromkeys([c for c,_ in top]+[(7,9),(10,12)]))
 for cs,ce in blocks:
  ids=np.array([i for i,c in enumerate(clusters) if (cs,ce) in c])
  if len(ids)==0:ids=np.arange(n)
  vals=ev[ids,cs]; chosen=[int(ids[np.argmin(abs(vals-np.percentile(vals,q)))]) for q in [25,50,75]]
  refs[(cs,ce)]=[(j,orthonormalize_l2(np.asarray(eigvecs[j,cs:ce]).reshape(ce-cs,-1).T,M=M)) for j in chosen]
 rows=[]
 for i in range(n):
  # A single Zarr chunk contains all 15 modes for this subject.
  modes=np.asarray(eigvecs[i],dtype=float)
  for cs,ce in blocks:
   block=modes[cs:ce].reshape(ce-cs,-1).T; Q=orthonormalize_l2(block,M=M)
   ap,ml,*_=canonical_pair_couplings(Q,measurements['AP'][i],measurements['ML'][i],M=M)
   bis,bit,*_=canonical_pair_couplings(Q,measurements['BIS'][i],measurements['BIT'][i],M=M)
   out,_,_=single_functional_capacity_max_nodal(Q,measurements['OUTLETAP'][i],M=M)
   scalar,_,sigma=single_functional_capacity_max_nodal(Q,measurements['AP'][i],M=M)
   ml_scalar,_,ml_sigma=single_functional_capacity_max_nodal(Q,measurements['ML'][i],M=M)
   bis_scalar,_,bis_sigma=single_functional_capacity_max_nodal(Q,measurements['BIS'][i],M=M)
   bit_scalar,_,bit_sigma=single_functional_capacity_max_nodal(Q,measurements['BIT'][i],M=M)
   rec={'subject':i,'sex':sex[i],'age':age[i],'block':f'{cs+1}-{ce}',
    'member':(cs,ce) in clusters[i],
    'AP':scalar,'ML':ml_scalar,'BIS':bis_scalar,'BIT':bit_scalar,'OUTLETAP':out,
    'AP_pair':ap,'ML_pair':ml,'BIS_pair':bis,'BIT_pair':bit,
    'AP_scalar':scalar,'AP_mass_norm':sigma,'ML_scalar':ml_scalar,'BIS_scalar':bis_scalar,'BIT_scalar':bit_scalar,
    'AP_diameter':dims['AnteriorPosteriorInletDiameter'][i],
    'gap':gaps[i,cs], 'rank9':coupling_per_1mm_max(measurements['AP'][i],modes[8].ravel()),
    'rank10':coupling_per_1mm_max(measurements['AP'][i],modes[9].ravel()),
    'swap':int(inv[i,8]==9 and inv[i,9]==8)}
   for k,(idx,qr) in enumerate(refs[(cs,ce)]):
    rec[f'dG_ref{k}']=grassmann_distance(compute_principal_angles(qr,Q,M=M))
    rec[f'ref{k}_subject']=idx
   for l,load in enumerate(LOADS):
    rec[load]=fractions[i,cs:ce,l].sum(); rec[f'error_{load}']=errors[i,l]
   rows.append(rec)
  if i%25==0:print('Subjects',i+1,'/',n,flush=True)
 df=pd.DataFrame(rows); df.to_csv(TAB/'subject_metrics.csv',index=False)
 # Anatomy-independent source identifiers are positional; confirm metadata join before regressions.
 allo=pd.read_excel(DATA_DIR_FULL/'allometry.xlsx'); demo=pd.read_excel(DATA_DIR_FULL/'demography.xlsx')
 print('Allometry columns:',list(allo.columns),flush=True)
 merged=demo[['patient_id']].merge(allo,on='patient_id',how='left',validate='one_to_one')
 if len(merged)!=n: raise ValueError('Demographic order does not match cohort')
 if not np.array_equal(demo.sex.str.upper().to_numpy(),sex):raise ValueError('Demographic order mismatch')
 scale_col=next((c for c in ['scale','s','scale_factor','isotropic_scale'] if c in merged),None)
 if scale_col is None:raise ValueError('No documented scale column')
 df['scale']=df.subject.map(dict(enumerate(merged[scale_col].to_numpy())))
 df.to_csv(TAB/'subject_metrics.csv',index=False)
 (OUT/'provenance.json').write_text(json.dumps({'source_hashes':source_fingerprint(),
  'seed':RNG_SEED,'n_subjects':n,'n_modes':ev.shape[1],
  'metric':'reference P1 L2 consistent mass','coordinates':'serial DOLFINx DOF order',
  'functional':'b.T @ u; projection Q.T @ b','scale_column':scale_col,
  'eigenvector_shape':list(eigvecs.shape),'eigenvector_chunks':list(eigvecs.chunks),
  'data_directory':str(DATA_DIR_FULL.relative_to(ROOT))},indent=2))
 return df

def statistics(df):
 ev,perm,sex,age,gaps,eps,clusters,counts,top=basic()
 rows=[]
 for (cs,ce),cnt in counts.most_common():
  if cnt/len(ev)<.03:continue
  mem=np.array([(cs,ce) in cc for cc in clusters]); sep=compute_sep_out(ev,cs,ce)[mem]
  rows.append({'Block':f'{cs+1}-{ce}','n':cnt,'Prevalence (%)':100*cnt/len(ev),
   'Median min gap':np.median(gaps[mem,cs:ce-1].min(axis=1)),
   'Two-sided separation':np.median(sep) if np.isfinite(sep).any() else np.nan})
 table(pd.DataFrame(rows),'clusters')
 boot=cohort_bootstrap_robustness(ev,invert_pairing_permutation(perm),eps,[c for c,_ in top],
  n_boot=2000,subsample_frac=1.0,threshold_percentile=25,seed=RNG_SEED)
 table(pd.DataFrame([{'Block':k.replace('Rank modes ',''),'Prevalence':v['observed'],
  'CI low':v['ci95_lo'],'CI high':v['ci95_hi']} for k,v in boot['cluster_prevalence'].items()]),'prevalence_bootstrap')
 rows=[]
 for p in [15,20,25,30,35]:
  e=np.percentile(gaps,p); cl=detect_clusters(ev,e)
  for c,_ in top:rows.append({'Percentile':p,'Threshold':e,'Block':f'{c[0]+1}-{c[1]}','Prevalence':np.mean([c in cc for cc in cl])})
 table(pd.DataFrame(rows),'threshold_sensitivity')
 key=df[(df.block=='9-10')&df.member]; rng=np.random.default_rng(RNG_SEED)
 rows=[]
 for metric in ['rank9','rank10','AP','AP_pair']:
  a=key.loc[key.swap==0,metric].to_numpy(); b=key.loc[key.swap==1,metric].to_numpy()
  u,p=stats.mannwhitneyu(a,b,alternative='two-sided')
  ai=rng.integers(len(a),size=(5000,len(a))); bi=rng.integers(len(b),size=(5000,len(b)))
  delta=np.median(b[bi],axis=1)-np.median(a[ai],axis=1)
  ratio=np.median(b[bi],axis=1)/np.median(a[ai],axis=1)
  rows.append({'Metric':metric,'n no exchange':len(a),'n exchange':len(b),'Median no exchange':np.median(a),
   'Median exchange':np.median(b),'Difference':np.median(b)-np.median(a),
   'CI low':np.quantile(delta,.025),'CI high':np.quantile(delta,.975),
   'Ratio':np.median(b)/np.median(a),'Ratio CI low':np.quantile(ratio,.025),'Ratio CI high':np.quantile(ratio,.975),
   'p':p,'Rank biserial':1-2*u/(len(a)*len(b))})
 effects=pd.DataFrame(rows); effects['q']=multipletests(effects.p,method='fdr_bh')[1]; table(effects,'label_effects')

 # Full-cohort comparison (N = 278) for the 9-10 block regardless of gap clustering
 full910=df[df.block=='9-10']
 full_rows=[]
 for metric in ['rank9','rank10','AP','AP_pair']:
  a=full910.loc[full910.swap==0,metric].to_numpy(); b=full910.loc[full910.swap==1,metric].to_numpy()
  u,p=stats.mannwhitneyu(a,b,alternative='two-sided')
  ai=rng.integers(len(a),size=(5000,len(a))); bi=rng.integers(len(b),size=(5000,len(b)))
  delta=np.median(b[bi],axis=1)-np.median(a[ai],axis=1)
  ratio=np.median(b[bi],axis=1)/np.median(a[ai],axis=1)
  full_rows.append({'Metric':metric,'n no exchange':len(a),'n exchange':len(b),'Median no exchange':np.median(a),
   'Median exchange':np.median(b),'Difference':np.median(b)-np.median(a),
   'CI low':np.quantile(delta,.025),'CI high':np.quantile(delta,.975),
   'Ratio':np.median(b)/np.median(a),'Ratio CI low':np.quantile(ratio,.025),'Ratio CI high':np.quantile(ratio,.975),
   'p':p,'Rank biserial':1-2*u/(len(a)*len(b))})
 effects_cohort=pd.DataFrame(full_rows); effects_cohort['q']=multipletests(effects_cohort.p,method='fdr_bh')[1]; table(effects_cohort,'label_effects_full_cohort')

 # Descriptive variation, deliberately not a proof of equivalence.
 table(key[['rank9','rank10','AP','AP_pair']].agg(['median','std']).T.reset_index(names='Metric'),'coupling_variation')
 coupling_summary_df=df[df.member].groupby('block')[AXES].median().reset_index()
 table(coupling_summary_df,'coupling_summary')
 # Size-matched adjacent-pair controls on the same 152 subjects.
 ids=set(key.subject); controls=df[df.subject.isin(ids)&df.block.isin(['8-9','9-10','11-12'])]
 table(controls.groupby('block')[['AP','AP_pair']].median().reset_index(),'control_pairs')
 rows=[]
 for block,g in df[df.member].groupby('block'):
  for ref in range(3):rows.append({'Block':block,'Reference quantile':[25,50,75][ref],
   'Median distance':g[f'dG_ref{ref}'].median(), 'n':len(g)})
 table(pd.DataFrame(rows),'reference_sensitivity')
 rows=[]
 full=df[df.block=='9-10'].copy()
 for load in LOADS[2:]:
  for label,group in [('All',full),('Female',full[full.sex=='F']),('Male',full[full.sex=='M'])]:
   r,p=stats.spearmanr(group.AP_diameter,group[load]); rows.append({'Load':load,'Group':label,'n':len(group),'rho':r,'p':p})
 table(pd.DataFrame(rows),'routing_correlations')
 rows=[]
 for metric,data in [(l,full) for l in LOADS[2:]]+[('AP',key)]:
  X=pd.DataFrame({'Female':(data.sex=='F').astype(float),'Age':data.age,'Scale':data.scale,
    'AP diameter':data.AP_diameter,'Exchange':data.swap},index=data.index)
  X=(X-X.mean())/X.std(ddof=0); fit=sm.OLS(data[metric],sm.add_constant(X)).fit(cov_type='HC3')
  for term in X.columns:rows.append({'Outcome':metric,'Term':term,'Coefficient per SD':fit.params[term],
   'CI low':fit.conf_int().loc[term,0],'CI high':fit.conf_int().loc[term,1],'p':fit.pvalues[term],
   'n':int(fit.nobs),'Condition number':np.linalg.cond(sm.add_constant(X))})
 adj=pd.DataFrame(rows);adj['q']=multipletests(adj.p,method='fdr_bh')[1];table(adj,'adjusted_associations')
 fractions=_open_zarr_array(DATA_DIR_FULL/'mode_energy_fraction.zarr'); means=fractions.mean(axis=0)
 rows=[]
 for l,load in enumerate(LOADS):
  order=np.argsort(means[:,l])[::-1]; cum=np.cumsum(means[order,l]); n80=np.searchsorted(cum,.8)+1
  rows.append({'Load':load,'Top mode':order[0]+1,'Top share (%)':100*means[order[0],l],
   'N80':n80,'Modes':','.join(str(i+1) for i in order[:n80]),'Cumulative (%)':100*cum[n80-1]})
 table(pd.DataFrame(rows),'dominant_modes')
 errors=_open_zarr_array(DATA_DIR_FULL/'mode_reconstruction_error.zarr')
 table(pd.DataFrame([{'Load':l,'Median relative error':np.median(errors[:,i]),
  '95th percentile':np.percentile(errors[:,i],95),'Maximum':errors[:,i].max()} for i,l in enumerate(LOADS)]),'reconstruction')
 # Machine-readable values are also used to generate manuscript macros.
 vals={'threshold':eps,'n':len(ev),'coupling_effects':effects.to_dict('records'),'coupling_effects_cohort':effects_cohort.to_dict('records')}
 (OUT/'results.json').write_text(json.dumps(vals,indent=2))

 def format_p_macro(p_val: float) -> str:
  if not np.isfinite(p_val): return "--"
  if p_val < 0.001:
   exp = int(np.floor(np.log10(p_val)))
   c = p_val / (10 ** exp)
   return rf"{c:.1f} \times 10^{{{exp}}}"
  return f"{p_val:.3f}"

 macros=[]
 def def_macro(name, val):
  macros.append(f'\\newcommand{{\\{name}}}{{{val}}}')

 # Cohort and sample sizes
 def_macro('CohortN', f'{len(ev)}')
 def_macro('CohortMaleN', f'{int((sex == "M").sum())}')
 def_macro('CohortFemaleN', f'{int((sex == "F").sum())}')
 def_macro('CohortNoSwapN', f'{int((full910.swap == 0).sum())}')
 def_macro('CohortSwapN', f'{int((full910.swap == 1).sum())}')

 def_macro('ClusterMemberN', f'{len(key)}')
 def_macro('ClusterMaleN', f'{int((key.sex == "M").sum())}')
 def_macro('ClusterFemaleN', f'{int((key.sex == "F").sum())}')
 def_macro('ClusterNoSwapN', f'{int((key.swap == 0).sum())}')
 def_macro('ClusterSwapN', f'{int((key.swap == 1).sum())}')

 # Cluster member effects
 names={'rank9':'RankNine','rank10':'RankTen','AP':'ScalarAP','AP_pair':'PairAP'}
 for _,r in effects.iterrows():
  m_name = names[r.Metric]
  def_macro(f'{m_name}No', f'{r["Median no exchange"]:.3f}')
  def_macro(f'{m_name}Yes', f'{r["Median exchange"]:.3f}')
  def_macro(f'{m_name}Difference', f'{r["Difference"]:.3f}')
  def_macro(f'{m_name}Low', f'{r["CI low"]:.3f}')
  def_macro(f'{m_name}High', f'{r["CI high"]:.3f}')
  def_macro(f'{m_name}Ratio', f'{r["Ratio"]:.2f}')
  def_macro(f'{m_name}Pval', f'{r["p"]:.2e}')
  def_macro(f'{m_name}Pformatted', format_p_macro(r['p']))
  def_macro(f'{m_name}RankBiserial', f'{r["Rank biserial"]:.2f}')

 # Full cohort effects
 for _,r in effects_cohort.iterrows():
  m_name = 'Cohort' + names[r.Metric]
  def_macro(f'{m_name}No', f'{r["Median no exchange"]:.3f}')
  def_macro(f'{m_name}Yes', f'{r["Median exchange"]:.3f}')
  def_macro(f'{m_name}Difference', f'{r["Difference"]:.3f}')
  def_macro(f'{m_name}Low', f'{r["CI low"]:.3f}')
  def_macro(f'{m_name}High', f'{r["CI high"]:.3f}')
  def_macro(f'{m_name}Ratio', f'{r["Ratio"]:.2f}')
  def_macro(f'{m_name}Pval', f'{r["p"]:.2e}')
  def_macro(f'{m_name}Pformatted', format_p_macro(r['p']))
  def_macro(f'{m_name}RankBiserial', f'{r["Rank biserial"]:.2f}')

 # Prevalences
 boot_dict = {k.replace('Rank modes ', '').replace('–', '-'): v['observed'] * 100 for k, v in boot['cluster_prevalence'].items()}
 def_macro('PrevalenceFiveSix', f"{boot_dict.get('5-6', 0.0):.1f}")
 def_macro('PrevalenceNineTen', f"{boot_dict.get('9-10', 0.0):.1f}")
 def_macro('PrevalenceTwelveThirteen', f"{boot_dict.get('12-13', 0.0):.1f}")
 def_macro('PrevalenceTwelveFifteen', f"{boot_dict.get('12-15', 0.0):.1f}")
 def_macro('PrevalenceFourteenFifteen', f"{boot_dict.get('14-15', 0.0):.1f}")

 # Block 5-6 and 9-10 couplings
 cs_idx = coupling_summary_df.set_index('block')
 if '5-6' in cs_idx.index:
  def_macro('BlockFiveSixAP', f"{cs_idx.loc['5-6', 'AP']:.2f}")
  def_macro('BlockFiveSixML', f"{cs_idx.loc['5-6', 'ML']:.2f}")
  def_macro('BlockFiveSixBIS', f"{cs_idx.loc['5-6', 'BIS']:.2f}")
  def_macro('BlockFiveSixBIT', f"{cs_idx.loc['5-6', 'BIT']:.2f}")
  def_macro('BlockFiveSixOUTLETAP', f"{cs_idx.loc['5-6', 'OUTLETAP']:.2f}")
 if '9-10' in cs_idx.index:
  def_macro('BlockNineTenAP', f"{cs_idx.loc['9-10', 'AP']:.2f}")
  def_macro('BlockNineTenML', f"{cs_idx.loc['9-10', 'ML']:.2f}")
  def_macro('BlockNineTenBIS', f"{cs_idx.loc['9-10', 'BIS']:.2f}")
  def_macro('BlockNineTenBIT', f"{cs_idx.loc['9-10', 'BIT']:.2f}")
  def_macro('BlockNineTenOUTLETAP', f"{cs_idx.loc['9-10', 'OUTLETAP']:.2f}")

 # Polynomial algebra and continuum constants
 def_macro('TotalPolyDOFs', '30')
 def_macro('RigidDOFs', '6')
 def_macro('NonrigidDOFs', '24')
 def_macro('GramConditionNumber', '20.98')
 def_macro('DesignConditionNumber', '4.58')
 def_macro('CartilageEffectiveModulus', '1.0')

 macro_str = '\n'.join(macros) + '\n'
 (TAB/'values.tex').write_text(macro_str)
 (ROOT/'analysis_outputs/tables/values.tex').write_text(macro_str)
 print(effects.to_string(index=False),flush=True)

def plots(df):
 style();ev,perm,sex,age,gaps,eps,clusters,counts,top=basic()
 
 # 1. Spectral structure (Figure 2)
 fig,axs=plt.subplots(3,1,figsize=(COL2_WIDTH,7.4),layout='constrained')
 gap_data=[gaps[:,i] for i in range(14)]
 gap_labels=[f'{i}–{i+1}' for i in range(1,15)]
 style_distribution(axs[0],gap_data,positions=np.arange(1,15),labels=gap_labels,
                    color=NEUTRAL_COLOR,width=0.46,pt_alpha=0.22,pt_size=6,rng_seed=42)
 axs[0].axhline(eps,color=FEMALE_COLOR,ls='--',lw=1.2,label=f'25th percentile threshold ($\\epsilon = {eps:.3f}$)')
 axs[0].set_ylabel('Relative gap')
 axs[0].set_xlabel('Adjacent solver ranks')
 panel(axs[0],'A  Consecutive eigenvalue gaps')
 axs[0].legend(loc='upper right',frameon=False,fontsize=8)
 
 x=np.arange(1,16); sw=((perm!=np.arange(15))&(perm>=0)).mean(axis=0); un=(perm<0).mean(axis=0)
 axs[1].bar(x,sw,color=SWAP_COLOR,label='Exchanged',width=0.62,edgecolor='none')
 axs[1].bar(x,un,bottom=sw,color=UNPAIRED_COLOR,label='Unpaired',width=0.62,edgecolor='none')
 axs[1].set_xticks(x);axs[1].set_ylabel('Subject fraction');axs[1].set_xlabel('Reference modal rank')
 panel(axs[1],'B  Label exchange and pairing failures');axs[1].legend(loc='upper left',ncol=2,frameon=False,fontsize=8)
 
 bar_w=0.26
 for j,(path,label,color) in enumerate([(DATA_DIR_FULL,'Combined',MALE_COLOR),(DATA_DIR_SHAPE,'Shape only',MIDRANK_COLOR),(DATA_DIR_MATERIAL,'Material only',INLET_SWAP_COLOR)]):
  v=load_eigenvalues(path);axs[2].bar(x+(j-1)*bar_w,v.std(axis=0)/v.mean(axis=0),width=bar_w,label=label,color=color,edgecolor='none')
 axs[2].set_xticks(x);axs[2].set_xlabel('Solver rank');axs[2].set_ylabel('Coefficient of variation');axs[2].set_ylim(0,.24)
 panel(axs[2],'C  Between-subject eigenvalue variability');axs[2].legend(loc='upper right',ncol=3,frameon=False,fontsize=8)
 save(fig,'spectral_structure')
 
 # 2. Label stability (Figure 4 - Central Result)
 key=df[(df.block=='9-10')&df.member]
 fig,axs=plt.subplots(1,3,figsize=(COL2_WIDTH,3.4),sharey=True,layout='constrained')
 g0=key[key.swap==0]; g1=key[key.swap==1]
 pcols0=[MALE_COLOR if s=='M' else FEMALE_COLOR for s in g0.sex]
 pcols1=[MALE_COLOR if s=='M' else FEMALE_COLOR for s in g1.sex]
 axs[0].set_ylim(0,0.88)
 
 eff_tab = pd.read_csv(TAB/'label_effects.csv').set_index('Metric')
 panels_cfg = []
 for m, title in [('rank9','A  Rank mode 9'), ('rank10','B  Rank mode 10'), ('AP','C  Subspace 9–10 (AP capacity)')]:
  r = eff_tab.loc[m]
  pval = r['p']
  p_str = format_pvalue(pval)
  if pval >= 0.05:
   p_str = p_str[:-1] + r'\ \mathrm{(n.s.)}$'
  annot_txt = rf'$\Delta = {r["Difference"]:+.2f}$' + '\n' + p_str
  panels_cfg.append((m, title, annot_txt))

 for ax,(m,title,annot_txt) in zip(axs,panels_cfg):
  data_m=[g0[m].to_numpy(),g1[m].to_numpy()]
  style_distribution(ax,data_m,positions=[1,2],labels=[f'No exchange\n(n = {len(g0)})',f'Exchange\n(n = {len(g1)})'],
                     color=NEUTRAL_COLOR,width=0.48,pt_alpha=0.38,pt_size=10,
                     point_colors=[pcols0,pcols1],rng_seed=42)
  panel(ax,title)
  ax.text(0.96,0.94,annot_txt,transform=ax.transAxes,ha='right',va='top',fontsize=7.5,color='#374151',
          bbox=dict(boxstyle='round,pad=0.2',facecolor='white',edgecolor='#e5e7eb',alpha=0.85,lw=0.6))
 axs[0].set_ylabel('AP inlet capacity (mm/mm)')
 n_male = int((key.sex == 'M').sum())
 n_female = int((key.sex == 'F').sum())
 fig.legend(handles=[plt.Line2D([],[],marker='o',ls='',color=MALE_COLOR,markersize=4.5,label=f'Male (n = {n_male})'),
                     plt.Line2D([],[],marker='o',ls='',color=FEMALE_COLOR,markersize=4.5,label=f'Female (n = {n_female})')],
            loc='outside lower center',ncol=2,frameon=False,fontsize=8)
 save(fig,'label_stability')
 
 # 3. Functional coupling (Figure 3)
 blocks=[f'{c[0]+1}-{c[1]}' for c,_ in top]
 fig,axs=plt.subplots(2,2,figsize=(COL2_WIDTH,5.4),layout='constrained')
 coupling_cfgs=[
  ('AP','A  AP inlet','Basis-invariant capacity (mm/mm)'),
  ('ML','B  ML inlet','Basis-invariant capacity (mm/mm)'),
  ('BIS','C  Biischiadic outlet','Basis-invariant capacity (mm/mm)'),
  ('BIT','D  Bituberous outlet','Basis-invariant capacity (mm/mm)'),
 ]
 for ax,(m,title,ylbl) in zip(axs.flat,coupling_cfgs):
  block_data=[df.loc[(df.block==b)&df.member,m].to_numpy() for b in blocks]
  b_colors=[BLOCK_COLORS.get(b,NEUTRAL_COLOR) for b in blocks]
  style_distribution(ax,block_data,positions=np.arange(len(blocks)),labels=blocks,
                     color=b_colors,width=0.46,pt_alpha=0.28,pt_size=7,rng_seed=42)
  ax.set_ylabel(ylbl);ax.set_xlabel('Rank block');panel(ax,title)
 save(fig,'functional_coupling')
 
 # 4. Load routing (Figure 6)
 fractions=_open_zarr_array(DATA_DIR_FULL/'mode_energy_fraction.zarr'); means=fractions.mean(axis=0)
 fig,axs=plt.subplots(2,1,figsize=(COL2_WIDTH,5.4),layout='constrained')
 im=axs[0].imshow(means.T,vmin=0,vmax=1,cmap='viridis',aspect='auto',interpolation='nearest')
 axs[0].set_xticks(np.arange(15),np.arange(1,16));axs[0].set_yticks(np.arange(5),LOADS);axs[0].set_xlabel('Solver rank')
 panel(axs[0],'A  Mean retained modal energy share')
 style_axis(axs[0],spines=('left','bottom','top','right'),y_grid=False,x_grid=False)
 style_colorbar(fig,im,ax=axs[0],label='Mean share',orientation='vertical',shrink=0.88,fraction=0.035,pad=0.02)
 bottom=np.zeros(5)
 for a,b,label,c in [(0,3,'1–3 (Backbone)',BACKBONE_COLOR),(3,8,'4–8 (Mid-rank)',MIDRANK_COLOR),(8,10,'9–10 (Inlet swap)',INLET_SWAP_COLOR),(10,15,'11–15 (Higher reserve)',HIGHER_RESERVE_COLOR)]:
  y=means[a:b].sum(axis=0);axs[1].bar(np.arange(5),y,bottom=bottom,label=label,color=c,width=0.55,edgecolor='none');bottom+=y
 axs[1].set_xticks(np.arange(5),LOADS);axs[1].set_ylabel('Mean energy share');axs[1].set_ylim(0,1.05)
 panel(axs[1],'B  Routing across modal blocks')
 axs[1].legend(title='Modal blocks',loc='upper center',bbox_to_anchor=(0.5,-0.18),ncol=4,frameon=False,fontsize=8)
 save(fig,'load_routing')
 
 # 5. Routing distributions (Figure S1)
 fig,axs=plt.subplots(3,2,figsize=(COL2_WIDTH,7.2),layout='constrained')
 for l,ax in enumerate(axs.flat):
  if l==5:ax.axis('off');continue
  for k,(s,c) in enumerate([('M',MALE_COLOR),('F',FEMALE_COLOR)]):
   ar=fractions[sex==s,:,l];pos=np.arange(1,16)+(k-.5)*.32
   vp=ax.violinplot(list(ar.T),positions=pos,widths=.28,showextrema=False,showmedians=True)
   for body in vp['bodies']:body.set_facecolor(c);body.set_alpha(.55);body.set_edgecolor('none')
   vp['cmedians'].set_color(c);vp['cmedians'].set_linewidth(1.4)
  ax.set_xticks([1,3,5,7,9,11,13,15]);ax.set_xlabel('Solver rank');ax.set_ylabel('Energy share');ax.set_ylim(0,1)
  panel(ax,f'{chr(65+l)}  {LOADS[l]}')
 fig.legend(handles=[plt.Line2D([],[],color=MALE_COLOR,lw=3.0,label='Male (n = 128)'),plt.Line2D([],[],color=FEMALE_COLOR,lw=3.0,label='Female (n = 150)')],loc='outside lower center',ncol=2,frameon=False,fontsize=8)
 save(fig,'routing_distributions')
 
 # 6. Routing associations (Figure 5)
 fig,axs=plt.subplots(2,2,figsize=(COL2_WIDTH,5.6),layout='constrained')
 full=df[df.block=='9-10']
 for ax,load,letter in zip(axs.flat[:3],LOADS[2:],['A','B','C']):
  for s,c in [('M',MALE_COLOR),('F',FEMALE_COLOR)]:
   g=full[full.sex==s];style_scatter(ax,g.AP_diameter,g[load],c=c,s=10,alpha=0.45)
   fit=np.polyfit(g.AP_diameter,g[load],1);xx=np.linspace(g.AP_diameter.min(),g.AP_diameter.max(),50);ax.plot(xx,np.polyval(fit,xx),color=c,ls='--',lw=1.3)
  ax.set_xlabel('AP inlet diameter (mm)');ax.set_ylabel('9–10 energy share');panel(ax,f'{letter}  {load}')
 y_min_bc=min(axs[0,1].get_ylim()[0],axs[1,0].get_ylim()[0]);y_max_bc=max(axs[0,1].get_ylim()[1],axs[1,0].get_ylim()[1])
 axs[0,1].set_ylim(y_min_bc,y_max_bc);axs[1,0].set_ylim(y_min_bc,y_max_bc)
 groups_d=[
  key.loc[(key.sex=='M')&(key.swap==0),'AP'].to_numpy(),
  key.loc[(key.sex=='F')&(key.swap==0),'AP'].to_numpy(),
  key.loc[(key.sex=='M')&(key.swap==1),'AP'].to_numpy(),
  key.loc[(key.sex=='F')&(key.swap==1),'AP'].to_numpy(),
 ]
 cols_d=[MALE_COLOR,FEMALE_COLOR,MALE_COLOR,FEMALE_COLOR]
 style_distribution(axs[1,1],groups_d,positions=[1,2,3,4],labels=['M / no','F / no','M / yes','F / yes'],
                    color=cols_d,width=0.48,pt_alpha=0.40,pt_size=9,rng_seed=42)
 axs[1,1].set_xlabel('Sex / exchange status');axs[1,1].set_ylabel('AP inlet capacity (mm/mm)');panel(axs[1,1],'D  Subspace capacity')
 fig.legend(handles=[plt.Line2D([],[],marker='o',ls='',color=MALE_COLOR,label='Male'),plt.Line2D([],[],marker='o',ls='',color=FEMALE_COLOR,label='Female')],loc='outside lower center',ncol=2,frameon=False,fontsize=8)
 save(fig,'routing_associations')
 
 # 7. Material uncertainty (Figure 9)
 uq=pd.read_csv(ROOT/'analysis_outputs/material_uncertainty/population_material_summary.csv')
 fig,axs=plt.subplots(2,1,figsize=(COL2_WIDTH,5.6),layout='constrained')
 for j,(s,c) in enumerate([('Male',MALE_COLOR),('Female',FEMALE_COLOR)]):
  g=uq[uq.sex==s].sort_values('mode');axs[0].plot(g['mode'],g.mean_CoV_total*100,'o-',c=c,label=s,ms=3.5,lw=1.2)
  g14=g[g['mode']<15];axs[1].plot(g14['mode']+(j-.5)*.12,g14.mean_gap_z,'o-',c=c,ms=3.5,lw=1.2,label=s)
  axs[1].scatter(g14['mode']+(j-.5)*.12,g14.p10_gap_z,marker='v',c=c,s=14,zorder=3)
 axs[0].set_xticks(np.arange(1,16));axs[0].set_ylabel('Eigenvalue CoV (%)');axs[0].set_xlabel('Solver rank');axs[0].legend(loc='upper right',ncol=2,frameon=False,fontsize=8)
 panel(axs[0],'A  First-order material uncertainty')
 axs[1].axvspan(8.6,9.4,color='#e5e7eb',alpha=0.65,zorder=0);axs[1].set_xticks(np.arange(1,15),[f'{i}–{i+1}' for i in range(1,15)],rotation=30,ha='right')
 axs[1].set_ylabel('Gap / propagated SD ($Z$)');axs[1].set_xlabel('Adjacent solver ranks');axs[1].set_ylim(bottom=0)
 panel(axs[1],'B  Gap uncertainty including shared-parameter covariance')
 save(fig,'material_uncertainty')
 
 # 8. Robustness (Figure 13 / S3)
 th=pd.read_csv(TAB/'threshold_sensitivity.csv');fig,axs=plt.subplots(2,2,figsize=(COL2_WIDTH,5.6),layout='constrained')
 for b in blocks:
  c=BLOCK_COLORS.get(b,NEUTRAL_COLOR);g=th[th.Block==b];axs[0,0].plot(g.Percentile,g.Prevalence,'o-',c=c,label=b,ms=3.5,lw=1.2)
 axs[0,0].set_xlabel('Gap percentile threshold');axs[0,0].set_ylabel('Block prevalence');panel(axs[0,0],'A  Threshold sensitivity')
 axs[0,0].legend(ncol=2,fontsize=7.5,frameon=False)
 rr=pd.read_csv(TAB/'reference_sensitivity.csv')
 for b in blocks:
  c=BLOCK_COLORS.get(b,NEUTRAL_COLOR);g=rr[rr.Block==b];axs[0,1].plot(g['Reference quantile'],g['Median distance'],'o-',c=c,ms=3.5,lw=1.2)
 axs[0,1].set_xlabel('Reference eigenvalue percentile');axs[0,1].set_ylabel('Median Grassmann distance ($d_G$)');panel(axs[0,1],'B  Reference sensitivity')
 full=df[df.block=='9-10']
 for s,c in [('M',MALE_COLOR),('F',FEMALE_COLOR)]:
  g=full[full.sex==s];style_scatter(axs[1,0],g.scale,g.gap,c=c,s=10,alpha=0.45)
 axs[1,0].set_xlabel('Isotropic scale ($s$)');axs[1,0].set_ylabel('Relative gap ($9$–$10$)');panel(axs[1,0],'C  Size and solver-rank gap')
 errors=_open_zarr_array(DATA_DIR_FULL/'mode_reconstruction_error.zarr')
 err_data=[errors[:,l]*100 for l in range(5)]
 style_distribution(axs[1,1],err_data,positions=np.arange(5),labels=LOADS,color=NEUTRAL_COLOR,width=0.48,pt_alpha=0.30,pt_size=8,rng_seed=42)
 axs[1,1].set_xlabel('Load condition');axs[1,1].set_ylabel('Displacement error (%)');panel(axs[1,1],'D  15-mode displacement reconstruction error')
 save(fig,'robustness')
 
 # 9. Coupling routing heatmap (Figure S2)
 fig,axs=plt.subplots(2,3,figsize=(COL2_WIDTH,5.4),layout='constrained');corr=[];im=None
 for ax,m in zip(axs.flat,AXES):
  mat=np.empty((len(blocks),5))
  for j,b in enumerate(blocks):
   g=df[(df.block==b)&df.member]
   for l,load in enumerate(LOADS):
    rho,p=stats.spearmanr(g[m],g[load]);mat[j,l]=rho;corr.append({'Block':b,'Functional':m,'Load':load,'n':len(g),'rho':rho,'p':p})
  im=style_heatmap(ax,mat,cmap='RdBu_r',norm=TwoSlopeNorm(vmin=-1,vcenter=0,vmax=1),x_labels=LOADS,y_labels=blocks,annot=True,annot_fmt='{:.2f}',annot_size=7.5)
  ax.tick_params(axis='x',rotation=35);panel(ax,m)
 axs[1,2].axis('off');style_colorbar(fig,im,ax=axs[:,2],label=r'Spearman correlation ($\rho$)',shrink=0.75)
 save(fig,'coupling_routing_heatmap');corr=pd.DataFrame(corr);corr['q']=multipletests(corr.p,method='fdr_bh')[1];table(corr,'coupling_routing_correlations')
 
 # 10. Toy model (Figure 7)
 from analysis.toy_subspace_dashboard import ToyConfig,make_main_cohort,make_summary_curves
 cfg=ToyConfig(); toy=make_main_cohort(cfg);curves=make_summary_curves(cfg)
 fig,axs=plt.subplots(2,2,figsize=(COL2_WIDTH,5.4),layout='constrained')
 for i,c in enumerate([BACKBONE_COLOR,MALE_COLOR,FEMALE_COLOR]):style_scatter(axs[0,0],toy['mu'],toy['evals'][:,i],c=c,s=6,alpha=0.5)
 axs[0,0].set_ylabel(r'Eigenvalue ($\lambda$)');axs[0,0].set_xlabel(r'Model parameter $\mu$');panel(axs[0,0],'A  Illustrative spectrum (avoided crossing)')
 for k,c,label in [('label_low',MALE_COLOR,'Lower rank'),('label_high',FEMALE_COLOR,'Upper rank'),('subspace','#111827','Pair')]:
  style_scatter(axs[0,1],toy['mu'],toy[k],c=c,s=6,alpha=0.55,label=label)
 axs[0,1].set_ylabel('Squared functional projection');axs[0,1].set_xlabel(r'Model parameter $\mu$');panel(axs[0,1],'B  Functional projection')
 axs[0,1].legend(loc='upper center',bbox_to_anchor=(0.5,-0.22),ncol=3,fontsize=8,frameon=False)
 y=np.vstack([toy['standing'].mean(axis=0),toy['task'].mean(axis=0)]);bottom=np.zeros(2)
 for k,c,label in [(0,BACKBONE_COLOR,'Backbone'),(1,MALE_COLOR,'Lower rank'),(2,FEMALE_COLOR,'Upper rank')]:
  axs[1,0].bar([0,1],y[:,k],bottom=bottom,color=c,label=label,width=0.50,edgecolor='none');bottom+=y[:,k]
 axs[1,0].set_xticks([0,1]);axs[1,0].set_xticklabels(['Standing-like','Task-like']);axs[1,0].set_ylabel('Mean energy share');panel(axs[1,0],'C  Constructed load contrast')
 axs[1,0].legend(loc='upper center',bbox_to_anchor=(0.5,-0.22),ncol=3,fontsize=8,frameon=False)
 for k,c,label in [('mismatch',FEMALE_COLOR,'Carrier change'),('subspace_retained','#111827','Pair projection'),('task_cluster',INLET_SWAP_COLOR,'Pair routing')]:
  axs[1,1].plot(curves['eta'],curves[k],'o-',c=c,label=label,ms=3.5,lw=1.2)
 axs[1,1].set_xlabel(r'Nuisance scale $\eta$');axs[1,1].set_ylabel('Fraction');panel(axs[1,1],'D  Nuisance sensitivity')
 axs[1,1].legend(loc='upper center',bbox_to_anchor=(0.5,-0.22),ncol=3,fontsize=8,frameon=False)
 save(fig,'toy_model')


def main():
 p=argparse.ArgumentParser();p.add_argument('--plots-only',action='store_true');p.add_argument('--statistics-only',action='store_true');a=p.parse_args()
 for d in [OUT,FIG,TAB]:d.mkdir(parents=True,exist_ok=True)
 if a.plots_only or a.statistics_only:
  provenance=json.loads((OUT/'provenance.json').read_text())
  if provenance['source_hashes']!=source_fingerprint():
   raise RuntimeError('Source changed since computation; recompute rather than silently reuse stale metrics')
  df=pd.read_csv(TAB/'subject_metrics.csv')
 else:df=compute_subjects()
 if not a.plots_only:statistics(df)
 if not a.statistics_only:plots(df)
 print('PLOS revision outputs:',OUT,flush=True)

if __name__=='__main__':main()
