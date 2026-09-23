"""Quadratic-age sensitivity of existing statistical data; leaves primary fits intact."""
from pathlib import Path
import json
import numpy as np,pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
P=Path(__file__).resolve().parents[1];T=P/'tables/generated';O=P/'tables/revision';O.mkdir(exist_ok=True)
d=pd.read_csv(T/'subject_level.csv');baseline=pd.read_csv(T/'nested_sex_models.csv');allom=pd.read_csv(T/'allometry_models.csv')
rows=[];slopes=[]
for load,g in d.groupby('load_case',sort=False):
 for outcome in ['rot_mag_deg','trans_mag_mm']:
  if load.startswith('LAB'):
   for name,rhs in [('M0','sex_F + age_z'),('M1','sex_F + age_z + log_total_volume_z'),('M2','sex_F + age_z + log_total_volume_z + AP_z + BiischiadicWidth_z + SubpubicAngle_z')]:
    fit0=smf.ols(f'{outcome} ~ {rhs}',g).fit(cov_type='HC3');b=baseline[(baseline.load_case==load)&(baseline.outcome==outcome)&(baseline.model==name)].iloc[0]
    np.testing.assert_allclose(fit0.params.sex_F,b.beta,atol=1e-10)
    fit=smf.ols(f'{outcome} ~ {rhs} + I(age_z ** 2)',g).fit(cov_type='HC3');ci=fit.conf_int().loc['sex_F']
    assert np.linalg.matrix_rank(fit.model.exog)==fit.model.exog.shape[1]
    rows.append(dict(load_case=load,outcome=outcome,model=name,beta_linear=b.beta,beta_quadratic=fit.params.sex_F,ci95_low=ci[0],ci95_high=ci[1],p=fit.pvalues.sex_F,n=int(fit.nobs)))
  rhs='log_total_volume * sex_F + age_z + AP_z + BiischiadicWidth_z + SubpubicAngle_z'
  fit0=smf.ols(f'np.log({outcome}) ~ {rhs}',g).fit(cov_type='HC3');b=allom[(allom.load_case==load)&(allom.outcome==outcome)].iloc[0]
  np.testing.assert_allclose([fit0.params.log_total_volume,fit0.params.log_total_volume+fit0.params['log_total_volume:sex_F']],[b.male_exponent,b.female_exponent],atol=1e-10)
  fit=smf.ols(f'np.log({outcome}) ~ {rhs} + I(age_z ** 2)',g).fit(cov_type='HC3');assert np.linalg.matrix_rank(fit.model.exog)==fit.model.exog.shape[1]
  for sex in ['male','female']:
   c=np.zeros(len(fit.params));c[fit.params.index.get_loc('log_total_volume')]=1
   if sex=='female':c[fit.params.index.get_loc('log_total_volume:sex_F')]=1
   test=fit.t_test(c);ci=np.asarray(test.conf_int()).ravel()
   slopes.append(dict(load_case=load,outcome=outcome,sex=sex,slope_linear=b[f'{sex}_exponent'],slope_quadratic=float(np.asarray(test.effect).item()),ci95_low=ci[0],ci95_high=ci[1],p=float(test.pvalue),n=int(fit.nobs)))
x=pd.DataFrame(rows);sel=x.model=='M2';x.loc[sel,'q']=multipletests(x.loc[sel,'p'],method='fdr_bh')[1];x.to_csv(O/'age_quadratic_sex_models.csv',index=False)
y=pd.DataFrame(slopes);y['q']=multipletests(y.p,method='fdr_bh')[1];y.to_csv(O/'age_quadratic_volume_slopes.csv',index=False)
summary=dict(n_subjects=int(d.patient_id.nunique()),n_rows=len(d),M2_intervals_include_zero=int(((x.loc[sel,'ci95_low']<=0)&(x.loc[sel,'ci95_high']>=0)).sum()),n_M2=int(sel.sum()),max_abs_change_M2_translation=float((x.loc[sel&(x.outcome=='trans_mag_mm'),'beta_quadratic']-x.loc[sel&(x.outcome=='trans_mag_mm'),'beta_linear']).abs().max()),n_negative_slopes=int((y.slope_quadratic<0).sum()),n_slopes_q_below_05=int((y.q<.05).sum()),max_slope_q=float(y.q.max()),max_abs_slope_change=float((y.slope_quadratic-y.slope_linear).abs().max()),baseline_reproduction_atol=1e-10)
(O/'age_quadratic_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
