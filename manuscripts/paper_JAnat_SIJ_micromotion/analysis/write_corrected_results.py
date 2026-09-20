"""Generate numerical manuscript passages from the corrected statistical tables."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
P=Path(__file__).resolve().parents[1];T=P/'tables/generated'
L=['SP2leg','SP1leg','LAB_phase1','LAB_phase2','LAB_phase3']
N=dict(zip(L,['SP2leg','SP1leg','LAB1','LAB2','LAB3']))
def num(x):return f'{x:.3f}'
def prob(x):return '<0.001' if x<.001 else f'={x:.3f}'
def interval(r,coef='median_delta'):return f"${r[coef]:+.3f}$ (95\\% CI {r.ci95_low:+.3f} to {r.ci95_high:+.3f})"
def figure(file,label,caption,height=None):
    opts=r'width=\textwidth'+(f',height={height}\\textheight,keepaspectratio' if height else '')
    return '\n'+r'\begin{figure}[p]'+'\n'+r'\centering'+'\n'+f'\\includegraphics[{opts}]{{figures/{file}.pdf}}\n'+f'\\caption{{{caption}}}\n\\label{{{label}}}\n'+r'\end{figure}'+'\n'
def table(file,label,caption):
    return '\n'+r'\begin{table}[htbp]'+'\n'+r'\centering'+'\n'+f'\\caption{{{caption}}}\n\\label{{{label}}}\n\\StdTableInput{{tables/generated/{file}.tex}}\n'+r'\end{table}'+'\n'
def main():
    d=pd.read_csv(T/'subject_level.csv');v=pd.read_csv(T/'variance_channels.csv');sex=pd.read_csv(T/'sex_models_lab.csv')
    nested=pd.read_csv(T/'nested_sex_models.csv');direc=pd.read_csv(T/'directional_rerouting.csv');h1=pd.read_csv(T/'standing_contrasts.csv');allo=pd.read_csv(T/'allometry_models.csv')
    med=d.groupby('load_case').median(numeric_only=True)
    text=r'\section{Results}'+'\n'
    text+=figure('Fig1_anatomy_coordinates_loads','fig:overview',r'\textbf{Model anatomy and load definitions.} (A) Pelvic assembly with schematic morphometric overlays; their drawn locations illustrate dimensions and are not quantitative landmark measurements. Green denotes the S1 constraint, orange the SIJ cartilage and purple the symphysis. (B) Exploded articulation and pelvis-aligned coordinate definitions; arrows illustrate axes, not measured trajectories. Rotations describe ilium relative to sacrum. (C) Bilateral and unilateral acetabular loads. (D) Independent ring, ischial and AP force pairs. Arrows represent applied global force directions; they do not demonstrate joint opening. The penalty approximates a fixed boundary.',.67)
    text+=r'\subsection{Cohort and corrected extraction}'+'\n'
    text+=f'The corrected dataset contains {d.patient_id.nunique()} subjects and {len(d)} subject--load observations in the full variant, with corresponding records in both auxiliary variants. Cohort demographics and dimensions are unchanged (Table~\\ref{{tab:cohort}}). All numerical results below use the unloaded subject reference.\n'
    zero=[]
    for f in sorted((P/'tables/corrected').glob('subject_*.npz')):
        with np.load(f) as z:zero.append([np.linalg.norm(z['zero_old_angles'],axis=1).mean(),np.linalg.norm(z['zero_old_trans'],axis=1).mean()])
    z=np.asarray(zero)
    text+='The unloaded-to-unloaded checks were below $10^{-8}$ in degrees and mm for every subject.\n'
    text+=r'\subsection{Standing load contrasts (H1)}'+'\n'
    text+=f'Under SP2leg and SP1leg, median rotation norms were {med.loc["SP2leg","rot_mag_deg"]:.3f}$^\\circ$ and {med.loc["SP1leg","rot_mag_deg"]:.3f}$^\\circ$, and median translations were {med.loc["SP2leg","trans_mag_mm"]:.3f} and {med.loc["SP1leg","trans_mag_mm"]:.3f}~mm, respectively. Median bilateral translation asymmetry was {med.loc["SP2leg","lr_trans_asym_mm"]:.3f}~mm in SP2leg and {med.loc["SP1leg","lr_trans_asym_mm"]:.3f}~mm in SP1leg (Fig.~\\ref{{fig:loadprofiles}}).\n'
    for metric,label,unit in [('rot_mag_deg','rotation norm','degrees'),('trans_mag_mm','translation magnitude','mm'),('lr_trans_asym_mm','translation asymmetry','mm')]:
        r=h1.set_index('metric').loc[metric]
        text+=f'The paired SP1leg-minus-SP2leg median difference in {label} was {interval(r)} {unit}, $q{prob(r.q)}$. '
    text+='These comparisons test motion and asymmetry; absolute ML rotation does not establish self-bracing nutation.\n'
    text+=table('table3_micromotion_summary','tab:microsummary','Corrected SIJ motion summaries across 278 subjects. Values are median [interquartile range]; component columns are absolute magnitudes.')
    text+=figure('Fig2_primary_load_profiles','fig:loadprofiles',r'\textbf{Corrected motion distributions.} Rotation-triplet norm, relative translation norm, absolute ML rotation and bilateral translation-magnitude asymmetry. Points are subjects; boxplots show medians and IQRs, with whiskers at 1.5 IQR. Female and male groups use vermilion and blue. Jitter is deterministic. No signed nutation direction is inferred.')
    text+=r'\subsection{Component contrasts (H2)}'+'\n'
    for load in L[2:]:
        text+=N[load]+': '
        bits=[]
        for metric,label in [('ml_trans_abs_mm','ML'),('ap_trans_abs_mm','AP'),('cc_trans_abs_mm','CC')]:
            r=direc[(direc.load_case==load)&(direc.metric==metric)].iloc[0]
            bits.append(f'the {label} absolute-component contrast was {interval(r)}~mm, $q{prob(r.p_fdr_bh)}$')
        text+='; '.join(bits)+'.\n\n'
    text+='All contrasts are paired differences in absolute magnitudes relative to SP2leg (Fig.~\\ref{fig:directional}; Supplementary Table S1). The state-space arrows connect the baseline to componentwise median contrasts; they are not physical displacement vectors or trajectories.\n'
    text+=figure('Fig3_directional_rerouting','fig:directional',r'\textbf{Paired absolute-component contrasts.} (A) Median within-subject differences from SP2leg, with bootstrap 95\% CIs. (B) Individual ML/CC contrast pairs and arrows to their componentwise medians. Positive values mean larger component magnitude, not outward joint motion. Corrected $q$ values are reported in Supplementary Table S1; intervals are pointwise, not simultaneous.')
    text+=r'\subsection{Sex and morphometric associations (H3)}'+'\n'
    for load in L[2:]:
        n=nested[(nested.load_case==load)&(nested.outcome=='trans_mag_mm')].set_index('model')
        r=sex[(sex.load_case==load)&(sex.outcome=='trans_mag_mm')].iloc[0]
        text+=f'{N[load]} female-minus-male translation coefficients were ${n.loc["M0","beta"]:+.3f}$, ${n.loc["M1","beta"]:+.3f}$ and ${n.loc["M2","beta"]:+.3f}$~mm in M0, M1 and M2. The M2 interval was {r.ci95_low:+.3f} to {r.ci95_high:+.3f}~mm ($p{prob(r.p_value)}$, $q{prob(r.p_fdr_bh)}$). '
    text+=f'The maximum non-intercept VIF in M2 was {nested.loc[nested.model=="M2","max_vif"].max():.2f}.\n'
    sub=d[d.load_case=='LAB_phase1'];r,p=spearmanr(sub.SubpubicAngle,sub.trans_mag_mm)
    text+=f'For LAB1, the pooled Spearman correlation between subpubic angle and translation magnitude was $r_s={r:.3f}$ ($p{prob(p)}$). '
    for flag,label in [(0,'male'),(1,'female')]:
        ss=sub[sub.sex_F==flag];r,p=spearmanr(ss.SubpubicAngle,ss.trans_mag_mm)
        text+=f'The {label}-only correlation was $r_s={r:.3f}$ ($p{prob(p)}$). '
    text+='These exploratory correlations are uncorrected for multiplicity and do not estimate a geometric mechanism. Figure~\\ref{fig:sexeffects} and Supplementary Table S3 report M2 estimates for both outcomes.\n'
    text+=figure('Fig4_sex_effects_lab','fig:sexeffects',r'\textbf{Conditional sex estimates and morphology.} (A,B) M2 female-minus-male coefficients with HC3 95\% CIs, shown on separate translation and rotation axes. Labels give BH-adjusted $q$ values across six tests. (C) Nested translation coefficients; connecting lines show changes between model specifications, not causal pathways. (D) LAB1 translation versus subpubic angle, with pooled and sex-stratified Spearman correlations. All panels use corrected endpoints.')
    text+=r'\subsection{Restricted geometry/material comparison (H4)}'+'\n'
    text+=f'Across five loads and five endpoints, shape-only/full variance ratios ranged from {v.shape_over_full_pct.min():.2f}\\% to {v.shape_over_full_pct.max():.2f}\\% (median {v.shape_over_full_pct.median():.2f}\\%). Material-only/full ratios ranged from {v.material_over_full_pct.min():.3f}\\% to {v.material_over_full_pct.max():.3f}\\% (median {v.material_over_full_pct.median():.3f}\\%). '
    for metric,label,unit in [('rot_mag_deg','rotation','degrees'),('trans_mag_mm','translation','mm')]:
        a=v[v.metric==metric]
        text+=f'For {label}, minimum shape/full identity-line $R^2$ was {a.identity_r2.min():.4f}, and maximum RMSE was {a.rmse.max():.4f}~{unit}. '
    text+='These compare model variants and do not decompose an additive population variance budget (Fig.~\\ref{fig:variance}; Supplementary Table S2).\n'
    text+=figure('Fig5_variance_channels','fig:variance',r'\textbf{Variance ratios across matched variants.} (A) Shape-only/full and (B) material-only/full sample-variance ratios, in percent. Five endpoints are shown. Color scales differ between panels and follow their actual data ranges. Values above 100\% are allowed. Bootstrap intervals and paired agreement diagnostics are provided in Supplementary Table S2; neither panel represents variance explained by an independent causal factor.')
    text+=r'\subsection{Exploratory volume associations}'+'\n'
    sig=[]
    for _,r in allo.iterrows():
        for sexlabel in ['male','female']:
            if r[sexlabel+'_q']<.05:
                sig.append(f'{N[r.load_case]} {r.outcome.replace("rot_mag_deg","rotation").replace("trans_mag_mm","translation")} in {sexlabel} models ($b={r[sexlabel+"_exponent"]:.3f}$, 95\\% CI {r[sexlabel+"_ci95_low"]:.3f} to {r[sexlabel+"_ci95_high"]:.3f}, $q{prob(r[sexlabel+"_q"])}$)')
    slopes=np.r_[allo.male_exponent,allo.female_exponent]
    slope_q=np.r_[allo.male_q,allo.female_q]
    text+=f'{len(sig)} of 20 sex-specific slopes survived FDR correction. The exponents ranged from {slopes.min():.3f} to {slopes.max():.3f}; the largest adjusted slope $q$ was {slope_q.max():.3f}. Thus negative conditional volume associations extended across standing and all three localized force-pair configurations, rather than being restricted to unilateral standing. Full estimates and intervals are in Table~\\ref{{tab:allometry}}.\n'
    text+=f'Sex-by-volume interaction $q$ values ranged from {allo.interaction_p_fdr_bh.min():.3f} to {allo.interaction_p_fdr_bh.max():.3f}; nonsignificance does not establish equivalence. Figure~\\ref{{fig:allometry}} and Table~\\ref{{tab:allometry}} use the same adjusted models. Pooled cluster-robust coefficients are reported in Supplementary Table S5.\n'
    text+=figure('Fig6_allometry_loglog','fig:allometry',r'\textbf{Conditional log-volume slopes.} Male and female exponents with HC3 95\% CIs from the same adjusted log--log models as Table~\ref{tab:allometry}. The vertical zero line indicates no conditional association. Confidence intervals are pointwise; labels give slope $q$ values from the 20-test family. Raw scatterplots and adjusted conditional geometric-mean curves are supplied as Supplementary Figure S1.')
    text+=table('table4b_allometry','tab:allometry','Adjusted log--log exponents by load and sex. Slope $q$ values use the 20-test family; interaction $q$ values use a separate ten-test family.')
    sens=pd.read_csv(P/'tables/corrected/rigid_correction_sensitivity.csv')
    text+=r'\subsection{Extraction sensitivity and hypothesis assessment}'+'\n'
    text+=f'In six subjects selected at approximately evenly spaced archive indices (30 subject--load comparisons), replacing affine sacral correction with rigid-only correction changed bilateral rotation norms by at most {(sens.rigid_rotation-sens.affine_rotation).abs().max():.4f}$^\\circ$ and translation magnitudes by at most {(sens.rigid_translation-sens.affine_translation).abs().max():.4f}~mm. This subset diagnostic does not bound errors in the remaining cohort (Supplementary Table S6).\n'
    h1ok=bool(((h1.median_delta>0)&(h1.q<.05)).all())
    h2a=direc[(direc.load_case.isin(L[2:4]))&(direc.metric=='ml_trans_abs_mm')]
    h2b=direc[(direc.load_case.isin(L[2:4]))&(direc.metric=='cc_trans_abs_mm')]
    h2ok=bool(((h2a.median_delta>0)&(h2a.p_fdr_bh<.05)).all() and ((h2b.median_delta<0)&(h2b.p_fdr_bh<.05)).all())
    text+=r'\begin{itemize}'+'\n'
    text+=r'\item \textbf{H1:} '+('Supported for the specified static endpoints.' if h1ok else 'Only partly supported; the individual paired contrasts above determine which endpoints change.')+' This does not verify a nutation or force-closure mechanism.\n'
    text+=r'\item \textbf{H2:} '+('Supported for the specified component pattern in LAB1/LAB2 within this retrospective analysis.' if h2ok else 'The full proposed LAB1/LAB2 component pattern is not supported after multiplicity correction.')+' LAB3 is a separate contrast; signed opening remains untested.\n'
    supported_sex = sex[(sex.outcome=='trans_mag_mm') & (sex.beta_female_minus_male>0) & (sex.p_fdr_bh<.05)]
    sex_names=', '.join(N[l] for l in supported_sex.load_case)
    text+=r'\item \textbf{H3:} '+(f'Positive fully adjusted female translation associations survived FDR correction in {sex_names}. ' if sex_names else 'No positive fully adjusted female translation association survived FDR correction. ')+'The nested coefficient changes are descriptive and do not establish causal mediation.\n'
    text+=r'\item \textbf{H4:} '+'Shape/full variance retention and paired errors support only the stated comparison with a reference bone-density field. They do not rank geometry against unmodeled soft-tissue variability.\n'+r'\end{itemize}'+'\n'
    (P/'sections/results.tex').write_text(text)
    ts=sex[sex.outcome=='trans_mag_mm']
    abstract=(f'The relation between pelvic anatomy and sacroiliac joint (SIJ) response depends on both loading and the reference configuration used to measure motion. We reanalyzed saved finite-element fields from 278 pelves (150 female, 128 male; age 16--91 years), comparing bilateral standing, unilateral standing and three localized force-pair proxies. Kinematics were measured between unloaded and loaded versions of the same mapped anatomy, separating anatomical variation from elastic displacement. Median bilateral translation was {med.loc["SP2leg","trans_mag_mm"]:.3f}~mm under bilateral standing and {med.loc["SP1leg","trans_mag_mm"]:.3f}~mm under unilateral standing; translation-magnitude asymmetry was {med.loc["SP2leg","lr_trans_asym_mm"]:.3f} and {med.loc["SP1leg","lr_trans_asym_mm"]:.3f}~mm, respectively. The three force-pair cases yielded median translations of '+', '.join(f'{med.loc[l,"trans_mag_mm"]:.3f}' for l in L[2:])+f'~mm. Age-, size- and morphology-adjusted female-minus-male translation coefficients ranged from {ts.beta_female_minus_male.min():+.3f} to {ts.beta_female_minus_male.max():+.3f}~mm; '+f'{int((ts.p_fdr_bh<.05).sum())} of the three translation comparisons survived FDR correction across six translation and rotation tests. Shape-only models retained {v.shape_over_full_pct.min():.2f}--{v.shape_over_full_pct.max():.2f}\\% of full-model variance across the five reported endpoints, while material-only/full ratios reached {v.material_over_full_pct.max():.3f}\\%. These ratios are non-additive and concern bone density under fixed soft-tissue assumptions. Component contrasts describe absolute motion redistribution, not signed joint opening or birth-canal expansion. The resulting dataset supports comparisons among specified passive mechanical configurations; physiological, obstetric and clinical prediction requires independent validation.')
    (P/'sections/abstract.tex').write_text(abstract+'\n')
    summary=dict(n=278,zero_mapping_rotation_median=float(np.median(z[:,0])),zero_mapping_translation_median=float(np.median(z[:,1])),h1_supported=h1ok,h2_supported=h2ok,significant_translation_sex_tests=int((ts.p_fdr_bh<.05).sum()),medians=med[['rot_mag_deg','trans_mag_mm','lr_trans_asym_mm']].to_dict(),variance_min=float(v.shape_over_full_pct.min()),variance_max=float(v.shape_over_full_pct.max()))
    import json
    (P/'review/corrected_summary.json').write_text(json.dumps(summary,indent=2))
if __name__=='__main__':main()
