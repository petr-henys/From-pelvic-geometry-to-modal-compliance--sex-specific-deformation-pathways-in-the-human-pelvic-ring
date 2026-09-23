"""Format the quadratic-age diagnostic without changing the primary estimates."""
from pathlib import Path
import pandas as pd
P=Path(__file__).resolve().parents[1];R=P/'tables/revision';T=P/'tables/generated'
loads={'SP2leg':'SP2leg','SP1leg':'SP1leg','LAB_phase1':'LAB1','LAB_phase2':'LAB2','LAB_phase3':'LAB3'}
metrics={'rot_mag_deg':'Rotation (deg)','trans_mag_mm':'Translation (mm)'}
def q(x):return '$<0.001$' if x<.001 else f'{x:.3f}'
x=pd.read_csv(R/'age_quadratic_sex_models.csv');rows=[]
for _,r in x[x.model=='M2'].iterrows():
 g=x[(x.load_case==r.load_case)&(x.outcome==r.outcome)].set_index('model')
 rows.append({'Load':loads[r.load_case],'Outcome':metrics[r.outcome],'M0 (quadratic age)':f'{g.loc["M0","beta_quadratic"]:+.4f}','M2 (linear age)':f'{r.beta_linear:+.4f}','M2 (quadratic age) [95\\% CI]':f'{r.beta_quadratic:+.4f} [{r.ci95_low:+.4f}, {r.ci95_high:+.4f}]','BH $q$':q(r.q)})
pd.DataFrame(rows).to_latex(T/'supp_age_sex.tex',index=False,escape=False)
y=pd.read_csv(R/'age_quadratic_volume_slopes.csv');rows=[]
for _,r in y.iterrows():
 rows.append({'Load':loads[r.load_case],'Outcome':metrics[r.outcome].split()[0],'Sex':r.sex.title(),'Linear-age slope':f'{r.slope_linear:.3f}','Quadratic-age slope [95\\% CI]':f'{r.slope_quadratic:.3f} [{r.ci95_low:.3f}, {r.ci95_high:.3f}]','BH $q$':q(r.q)})
pd.DataFrame(rows).to_latex(T/'supp_age_volume.tex',index=False,escape=False)
