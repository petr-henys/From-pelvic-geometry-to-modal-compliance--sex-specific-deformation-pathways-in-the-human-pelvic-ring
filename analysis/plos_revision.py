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
 orthonormalize_l2,canonical_pair_couplings,single_functional_coupling,
 coupling_per_1mm_max,build_patient_measurement_vectors,invert_pairing_permutation,
 compute_principal_angles,grassmann_distance,cohort_bootstrap_robustness)
from analysis.spectral_config import DATA_DIR_FULL,DATA_DIR_SHAPE,DATA_DIR_MATERIAL

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'analysis_outputs/plos_revision'; FIG=OUT/'figures'; TAB=OUT/'tables'
MALE='#0072B2'; FEMALE='#D99000'; COLORS=[MALE,FEMALE,'#009E73','#CC79A7','#777777']
LOADS=['SP2leg','SP1leg','LAB1','LAB2','LAB3']
AXES=['AP','ML','BIS','BIT','OUTLETAP']
RNG_SEED=20260918

def style():
 plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
  'font.size':10,'axes.labelsize':10,'axes.titlesize':10,'xtick.labelsize':9,
  'ytick.labelsize':9,'legend.fontsize':9,'axes.spines.top':False,'axes.spines.right':False,
  'pdf.fonttype':42,'savefig.dpi':400})

def save(fig,name):
 fig.savefig(FIG/f'{name}.pdf',bbox_inches='tight',pad_inches=.08)
 fig.savefig(FIG/f'{name}.png',bbox_inches='tight',pad_inches=.08,dpi=180)
 plt.close(fig)

def panel(ax,title):
 ax.set_title(title,loc='left',fontweight='bold',pad=12)
 ax.grid(axis='y',alpha=.18); ax.set_axisbelow(True)

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
   out,_,_=single_functional_coupling(Q,measurements['OUTLETAP'][i],M=M)
   scalar,_,sigma=single_functional_coupling(Q,measurements['AP'][i],M=M)
   rec={'subject':i,'sex':sex[i],'age':age[i],'block':f'{cs+1}-{ce}',
    'member':(cs,ce) in clusters[i], 'AP':ap,'ML':ml,'BIS':bis,'BIT':bit,'OUTLETAP':out,
    'AP_scalar':scalar,'AP_mass_norm':sigma,'AP_diameter':dims['AnteriorPosteriorInletDiameter'][i],
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
 for metric in ['rank9','rank10','AP','AP_scalar']:
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
 # Descriptive variation, deliberately not a proof of equivalence.
 table(key[['rank9','rank10','AP','AP_scalar']].agg(['median','std']).T.reset_index(names='Metric'),'coupling_variation')
 table(df[df.member].groupby('block')[AXES].median().reset_index(),'coupling_summary')
 # Size-matched adjacent-pair controls on the same 152 subjects.
 ids=set(key.subject); controls=df[df.subject.isin(ids)&df.block.isin(['8-9','9-10','11-12'])]
 table(controls.groupby('block')[['AP','AP_scalar']].median().reset_index(),'control_pairs')
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
 vals={'threshold':eps,'n':len(ev),'coupling_effects':effects.to_dict('records')}
 (OUT/'results.json').write_text(json.dumps(vals,indent=2))
 macros=[]
 names={'rank9':'RankNine','rank10':'RankTen','AP':'PairAP','AP_scalar':'ScalarAP'}
 for _,r in effects.iterrows():
  for col,suf in [('Median no exchange','No'),('Median exchange','Yes'),('Difference','Difference'),('CI low','Low'),('CI high','High')]:
   macros.append('\\newcommand{\\'+names[r.Metric]+suf+'}{'+f'{r[col]:.3f}'+'}')
 (TAB/'values.tex').write_text('\n'.join(macros)+'\n')
 print(effects.to_string(index=False),flush=True)

def plots(df):
 style();ev,perm,sex,age,gaps,eps,clusters,counts,top=basic()
 fig,axs=plt.subplots(3,1,figsize=(6.4,7.6),layout='constrained')
 axs[0].boxplot(gaps,tick_labels=[f'{i}-{i+1}' for i in range(1,15)],showfliers=False)
 axs[0].axhline(eps,color='#D55E00',ls='--',label=f'25th percentile = {eps:.3f}')
 axs[0].set_ylabel('Relative gap');panel(axs[0],'A  Consecutive eigenvalue gaps')
 axs[0].legend(loc='upper right',frameon=False)
 x=np.arange(1,16); sw=((perm!=np.arange(15))&(perm>=0)).mean(axis=0); un=(perm<0).mean(axis=0)
 axs[1].bar(x,sw,color=MALE,label='Exchanged');axs[1].bar(x,un,bottom=sw,color='.75',label='Unpaired')
 axs[1].set_xticks(x);axs[1].set_ylabel('Subject fraction');axs[1].set_xlabel('Reference label')
 panel(axs[1],'B  Label exchange and pairing failures');axs[1].legend(ncol=2,frameon=False)
 for j,(path,label,color) in enumerate([(DATA_DIR_FULL,'Combined',MALE),(DATA_DIR_SHAPE,'Shape only',FEMALE),(DATA_DIR_MATERIAL,'Material only','#009E73')]):
  v=load_eigenvalues(path);axs[2].bar(x+(j-1)*.25,v.std(axis=0)/v.mean(axis=0),width=.25,label=label,color=color)
 axs[2].set_xticks(x);axs[2].set_xlabel('Solver rank');axs[2].set_ylabel('Coefficient of variation');axs[2].set_ylim(0,.25)
 panel(axs[2],'C  Between-subject eigenvalue variability');axs[2].legend(ncol=3,frameon=False)
 save(fig,'spectral_structure')
 key=df[(df.block=='9-10')&df.member]
 fig,axs=plt.subplots(1,3,figsize=(6.4,3.5),layout='constrained')
 for ax,m,title in zip(axs,['rank9','rank10','AP'],['A  Rank 9','B  Rank 10','C  Subspace 9–10']):
  arrays=[key.loc[key.swap==v,m].to_numpy() for v in [0,1]]
  ax.boxplot(arrays,showfliers=False,widths=.55,medianprops={'color':'black'})
  rng=np.random.default_rng(42)
  for v in [0,1]:
   g=key[key.swap==v];ax.scatter(v+1+rng.uniform(-.12,.12,len(g)),g[m],c=[MALE if s=='M' else FEMALE for s in g.sex],s=9,alpha=.55)
  ax.set_xticks([1,2],['No exchange\n(n=95)','Exchange\n(n=57)']); ax.set_ylim(0,max(key[m])*1.12)
  panel(ax,title)
 axs[0].set_ylabel('AP coupling (mm/mm)')
 fig.legend(handles=[plt.Line2D([],[],marker='o',ls='',color=MALE,label='Male'),plt.Line2D([],[],marker='o',ls='',color=FEMALE,label='Female')],loc='outside lower center',ncol=2,frameon=False)
 save(fig,'label_stability')
 # Functional specificity: all five prevalent blocks, uncluttered distribution panels.
 blocks=[f'{c[0]+1}-{c[1]}' for c,_ in top]
 fig,axs=plt.subplots(2,2,figsize=(6.4,5.8),layout='constrained')
 for ax,m,title in zip(axs.flat,['AP','ML','BIS','BIT'],['A  AP inlet','B  ML inlet','C  Biischiadic','D  Bituberous']):
  ax.boxplot([df.loc[(df.block==b)&df.member,m] for b in blocks],tick_labels=blocks,showfliers=False)
  ax.set_ylabel('Coupling (mm/mm)');ax.set_xlabel('Rank block');panel(ax,title)
 save(fig,'functional_coupling')
 fractions=_open_zarr_array(DATA_DIR_FULL/'mode_energy_fraction.zarr'); means=fractions.mean(axis=0)
 fig,axs=plt.subplots(2,1,figsize=(6.4,5.8),layout='constrained')
 im=axs[0].imshow(means.T,vmin=0,vmax=1,cmap='Blues',aspect='auto')
 axs[0].set_xticks(np.arange(15),np.arange(1,16));axs[0].set_yticks(np.arange(5),LOADS);axs[0].set_xlabel('Solver rank')
 panel(axs[0],'A  Mean retained modal energy share');fig.colorbar(im,ax=axs[0],label='Share',fraction=.035)
 bottom=np.zeros(5)
 for a,b,label,c in [(0,3,'1–3',MALE),(3,8,'4–8',FEMALE),(8,10,'9–10','#009E73'),(10,15,'11–15','#CC79A7')]:
  y=means[a:b].sum(axis=0);axs[1].bar(np.arange(5),y,bottom=bottom,label=label,color=c);bottom+=y
 axs[1].set_xticks(np.arange(5),LOADS);axs[1].set_ylabel('Mean energy share');panel(axs[1],'B  Routing across modal blocks')
 axs[1].legend(title='Ranks',loc='outside upper center' if False else 'upper center',bbox_to_anchor=(.5,-.16),ncol=4,frameon=False)
 save(fig,'load_routing')
 # Full sex-specific distributions: 3 x 2, explicit legend, all 15 ranks.
 fig,axs=plt.subplots(3,2,figsize=(6.4,7.3),layout='constrained')
 for l,ax in enumerate(axs.flat):
  if l==5:ax.axis('off');continue
  for k,(s,c) in enumerate([('M',MALE),('F',FEMALE)]):
   ar=fractions[sex==s,:,l];pos=np.arange(1,16)+(k-.5)*.3
   vp=ax.violinplot(list(ar.T),positions=pos,widths=.28,showextrema=False,showmedians=True)
   for body in vp['bodies']:body.set_facecolor(c);body.set_alpha(.65)
   vp['cmedians'].set_color(c)
  ax.set_xticks([1,3,5,7,9,11,13,15]);ax.set_xlabel('Rank');ax.set_ylabel('Energy share');ax.set_ylim(0,1)
  panel(ax,f'{chr(65+l)}  {LOADS[l]}')
 fig.legend(handles=[plt.Line2D([],[],color=MALE,lw=4,label='Male (n=128)'),plt.Line2D([],[],color=FEMALE,lw=4,label='Female (n=150)')],loc='outside lower center',ncol=2,frameon=False)
 save(fig,'routing_distributions')
 # Associations: pooled statistics are tabulated, panels show separate within-sex trends.
 fig,axs=plt.subplots(2,2,figsize=(6.4,5.8),layout='constrained')
 for ax,load in zip(axs.flat,LOADS[2:]):
  full=df[df.block=='9-10']
  for s,c in [('M',MALE),('F',FEMALE)]:
   g=full[full.sex==s];ax.scatter(g.AP_diameter,g[load],s=9,color=c,alpha=.5)
   fit=np.polyfit(g.AP_diameter,g[load],1);xx=np.linspace(g.AP_diameter.min(),g.AP_diameter.max(),50);ax.plot(xx,np.polyval(fit,xx),c=c,ls='--')
  ax.set_xlabel('AP inlet diameter (mm)');ax.set_ylabel('9–10 energy share');panel(ax,f'{chr(65+LOADS[2:].index(load))}  {load}')
 axs[1,1].boxplot([key.loc[(key.sex==s)&(key.swap==v),'AP'] for v in [0,1] for s in ['M','F']],tick_labels=['M/no','F/no','M/yes','F/yes'],showfliers=False)
 axs[1,1].set_xlabel('Sex / exchange');axs[1,1].set_ylabel('AP coupling (mm/mm)');panel(axs[1,1],'D  Subspace coupling')
 fig.legend(handles=[plt.Line2D([],[],color=MALE,label='Male'),plt.Line2D([],[],color=FEMALE,label='Female')],loc='outside lower center',ncol=2,frameon=False)
 save(fig,'routing_associations')
 # Correctly indexed, covariance-aware UQ: no inference of 95% confidence from Z.
 uq=pd.read_csv(ROOT/'analysis_outputs/material_uncertainty/population_material_summary.csv')
 fig,axs=plt.subplots(2,1,figsize=(6.4,5.8),layout='constrained')
 for j,(s,c) in enumerate([('Male',MALE),('Female',FEMALE)]):
  g=uq[uq.sex==s].sort_values('mode');axs[0].plot(g['mode'],g.mean_CoV_total*100,'o-',c=c,label=s,ms=3)
  g=g[g['mode']<15];axs[1].plot(g['mode']+(j-.5)*.1,g.mean_gap_z,'o-',c=c,ms=3)
  axs[1].scatter(g['mode']+(j-.5)*.1,g.p10_gap_z,marker='v',c=c,s=14)
 axs[0].set_xticks(np.arange(1,16));axs[0].set_ylabel('Eigenvalue CoV (%)');axs[0].set_xlabel('Solver rank');axs[0].legend(ncol=2,frameon=False)
 panel(axs[0],'A  First-order material uncertainty')
 axs[1].axvspan(8.7,9.3,color='#999999',alpha=.15);axs[1].set_xticks(np.arange(1,15),[f'{i}–{i+1}' for i in range(1,15)],rotation=45)
 axs[1].set_ylabel('Gap / propagated SD');axs[1].set_xlabel('Adjacent solver ranks');axs[1].set_ylim(bottom=0)
 panel(axs[1],'B  Gap uncertainty including shared-parameter covariance')
 save(fig,'material_uncertainty')
 # Additional robustness replaces unsupported allometric recovery claims.
 th=pd.read_csv(TAB/'threshold_sensitivity.csv');fig,axs=plt.subplots(2,2,figsize=(6.4,5.9),layout='constrained')
 for b,c in zip(blocks,COLORS):
  g=th[th.Block==b];axs[0,0].plot(g.Percentile,g.Prevalence,'o-',c=c,label=b,ms=3)
 axs[0,0].set_xlabel('Gap percentile');axs[0,0].set_ylabel('Block prevalence');panel(axs[0,0],'A  Threshold sensitivity')
 axs[0,0].legend(ncol=2,fontsize=8,frameon=False)
 rr=pd.read_csv(TAB/'reference_sensitivity.csv')
 for b,c in zip(blocks,COLORS):
  g=rr[rr.Block==b];axs[0,1].plot(g['Reference quantile'],g['Median distance'],'o-',c=c,ms=3)
 axs[0,1].set_xlabel('Reference eigenvalue percentile');axs[0,1].set_ylabel('Median Grassmann distance');panel(axs[0,1],'B  Reference sensitivity')
 full=df[df.block=='9-10']
 for s,c in [('M',MALE),('F',FEMALE)]:
  g=full[full.sex==s];axs[1,0].scatter(g.scale,g.gap,s=9,color=c,alpha=.5)
 axs[1,0].set_xlabel('Isotropic scale');axs[1,0].set_ylabel('Relative gap (9–10)');panel(axs[1,0],'C  Size and solver-rank gap')
 errors=_open_zarr_array(DATA_DIR_FULL/'mode_reconstruction_error.zarr')
 axs[1,1].boxplot(errors*100,tick_labels=LOADS,showfliers=False);axs[1,1].tick_params(axis='x',rotation=45)
 axs[1,1].set_ylabel('Relative displacement error (%)');panel(axs[1,1],'D  Fifteen-mode reconstruction')
 save(fig,'robustness')
 # Correlation heatmaps: fixed [-1,1] scale, no stars crossing cell boundaries.
 fig,axs=plt.subplots(2,3,figsize=(6.4,5.8),layout='constrained');corr=[]
 for ax,m in zip(axs.flat,AXES):
  mat=np.empty((len(blocks),5))
  for j,b in enumerate(blocks):
   g=df[(df.block==b)&df.member]
   for l,load in enumerate(LOADS):
    rho,p=stats.spearmanr(g[m],g[load]);mat[j,l]=rho;corr.append({'Block':b,'Functional':m,'Load':load,'n':len(g),'rho':rho,'p':p})
  im=ax.imshow(mat,vmin=-1,vmax=1,cmap='RdBu_r',aspect='auto')
  for j in range(len(blocks)):
   for l in range(5):ax.text(l,j,f'{mat[j,l]:.2f}',ha='center',va='center',fontsize=8,color='white' if abs(mat[j,l])>.55 else 'black')
  ax.set_xticks(np.arange(5),LOADS,rotation=45,ha='right');ax.set_yticks(np.arange(len(blocks)),blocks);ax.set_title(m,loc='left',fontweight='bold')
 axs[1,2].axis('off');fig.colorbar(im,ax=axs[:,2],label='Spearman correlation',shrink=.75)
 save(fig,'coupling_routing_heatmap');corr=pd.DataFrame(corr);corr['q']=multipletests(corr.p,method='fdr_bh')[1];table(corr,'coupling_routing_correlations')
 # Toy model is illustrative and not fitted to anatomy.
 from analysis.toy_subspace_dashboard import ToyConfig,make_main_cohort,make_summary_curves
 cfg=ToyConfig(); toy=make_main_cohort(cfg);curves=make_summary_curves(cfg)
 fig,axs=plt.subplots(2,2,figsize=(6.4,5.7),layout='constrained')
 for i,c in enumerate(['#555555',MALE,FEMALE]):axs[0,0].scatter(toy['mu'],toy['evals'][:,i],s=4,c=c,alpha=.5)
 axs[0,0].set_ylabel('Eigenvalue');axs[0,0].set_xlabel('Model parameter μ');panel(axs[0,0],'A  Illustrative spectrum')
 for k,c,label in [('label_low',MALE,'Lower rank'),('label_high',FEMALE,'Upper rank'),('subspace','#222222','Pair')]:
  axs[0,1].scatter(toy['mu'],toy[k],s=4,c=c,alpha=.6,label=label)
 axs[0,1].set_ylabel('Squared functional projection');axs[0,1].set_xlabel('Model parameter μ');panel(axs[0,1],'B  Functional projection')
 axs[0,1].legend(loc='upper center',bbox_to_anchor=(.5,-.24),ncol=3,fontsize=8,frameon=False)
 # Means, unlike componentwise medians, sum exactly to one.
 y=np.vstack([toy['standing'].mean(axis=0),toy['task'].mean(axis=0)]);bottom=np.zeros(2)
 for k,c,label in [(0,'#555555','Backbone'),(1,MALE,'Lower rank'),(2,FEMALE,'Upper rank')]:
  axs[1,0].bar([0,1],y[:,k],bottom=bottom,color=c,label=label);bottom+=y[:,k]
 axs[1,0].set_xticks([0,1],['Standing-like','Task-like']);axs[1,0].set_ylabel('Mean energy share');panel(axs[1,0],'C  Constructed load contrast')
 for k,c,label in [('mismatch',FEMALE,'Carrier change'),('subspace_retained','#222222','Pair projection'),('task_cluster','#009E73','Pair routing')]:axs[1,1].plot(curves['eta'],curves[k],'o-',c=c,label=label,ms=3)
 axs[1,1].set_xlabel('Nuisance scale η');axs[1,1].set_ylabel('Fraction');panel(axs[1,1],'D  Nuisance sensitivity')
 axs[1,1].legend(loc='upper center',bbox_to_anchor=(.5,-.25),frameon=False,fontsize=8)
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
