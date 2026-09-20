"""Readable supplementary tables; no opaque raw statistical column names."""
from pathlib import Path
import pandas as pd
P=Path(__file__).resolve().parents[1];T=P/'tables/generated'
loads={'SP2leg':'SP2leg','SP1leg':'SP1leg','LAB_phase1':'LAB1','LAB_phase2':'LAB2','LAB_phase3':'LAB3'}
metrics={'rot_mag_deg':'Rotation (deg)','trans_mag_mm':'Translation (mm)','ap_trans_abs_mm':'AP (mm)','ml_trans_abs_mm':'ML (mm)','cc_trans_abs_mm':'CC (mm)'}
def q(x):return '$<0.001$' if x<.001 else f'{x:.3f}'
def ci(r,b='beta',low='ci95_low',high='ci95_high',digits=3):return f'{r[b]:.{digits}f} [{r[low]:.{digits}f}, {r[high]:.{digits}f}]'
def write(rows,name):
    pd.DataFrame(rows).to_latex(T/name,index=False,escape=False)
def main():
    d=pd.read_csv(T/'directional_rerouting.csv')
    write([{'Load':loads[r.load_case],'Component':metrics[r.metric],'Median difference [95\\% CI]':ci(r,'median_delta'),'Raw $p$':q(r.p_value),'BH $q$':q(r.p_fdr_bh)} for _,r in d.iterrows()],'supp_directional_rerouting.tex')
    d=pd.read_csv(T/'variance_channels.csv')
    write([{'Load':loads[r.load_case],'Endpoint':metrics[r.metric],'Shape/full \\% [95\\% CI]':ci(r,'shape_over_full_pct','shape_over_full_ci95_low','shape_over_full_ci95_high',2),'Material/full \\% [95\\% CI]':ci(r,'material_over_full_pct','material_over_full_ci95_low','material_over_full_ci95_high',3)} for _,r in d.iterrows()],'supp_variance_channels.tex')
    write([{'Load':loads[r.load_case],'Endpoint':metrics[r.metric],'Identity $R^2$':f'{r.identity_r2:.4f}','RMSE':f'{r.rmse:.4f}','MAE':f'{r.mae:.4f}'} for _,r in d.iterrows()],'supp_variant_agreement.tex')
    d=pd.read_csv(T/'sex_models_lab.csv')
    write([{'Load':loads[r.load_case],'Outcome':metrics[r.outcome],'Female minus male [95\\% CI]':ci(r,'beta_female_minus_male'),'Raw $p$':q(r.p_value),'BH $q$':q(r.p_fdr_bh)} for _,r in d.iterrows()],'supp_sex_models.tex')
    d=pd.read_csv(T/'allometry_models_scale.csv')
    write([{'Load':loads[r.load_case],'Outcome':metrics[r.outcome],'Male scale slope [95\\% CI]':ci(r,'male_exponent_scale','male_ci95_low','male_ci95_high'),'Female scale slope [95\\% CI]':ci(r,'female_exponent_scale','female_ci95_low','female_ci95_high')} for _,r in d.iterrows()],'supp_allometry_scale.tex')
    d=pd.read_csv(P/'tables/corrected/rigid_correction_sensitivity.csv')
    d['rotation_delta']=d.rigid_rotation-d.affine_rotation;d['translation_delta']=d.rigid_translation-d.affine_translation
    write([{'Load':loads[l],'Number':len(a),'Max $|\\Delta$ rotation$|$ (deg)':f'{a.rotation_delta.abs().max():.4f}','Max $|\\Delta$ translation$|$ (mm)':f'{a.translation_delta.abs().max():.4f}'} for l,a in d.groupby('load_case',sort=False)],'supp_extraction_sensitivity.tex')
    a=pd.read_csv(T/'allometry_models.csv')
    write([{'Load':loads[r.load_case],'Outcome':metrics[r.outcome],'Male exponent [95\\% CI]':ci(r,'male_exponent','male_ci95_low','male_ci95_high'),'Male $q$':q(r.male_q),'Female exponent [95\\% CI]':ci(r,'female_exponent','female_ci95_low','female_ci95_high'),'Female $q$':q(r.female_q),'Interaction $q$':q(r.interaction_p_fdr_bh)} for _,r in a.iterrows()],'table4b_allometry.tex')
if __name__=='__main__':main()
