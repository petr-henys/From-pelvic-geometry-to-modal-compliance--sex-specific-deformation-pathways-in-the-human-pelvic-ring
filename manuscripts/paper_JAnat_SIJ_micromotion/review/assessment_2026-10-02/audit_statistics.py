"""Read-only audit of archived SIJ endpoints; writes only this review directory.

No FE solves or changes to manuscript/primary analysis. New sensitivity estimates
are review diagnostics, not replacements for the manuscript's estimates.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import binomtest, norm, spearmanr
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

OUT = Path(__file__).resolve().parent
PAPER = OUT.parents[1]
G = PAPER / "tables/generated"
subjects = pd.read_csv(G / "subject_level.csv")
base = pd.read_csv(G / "subject_base.csv")
kin = pd.read_csv(PAPER / "tables/corrected/kinematics.csv")
assert len(base) == 278 and len(subjects) == 1390
assert not subjects.duplicated(["subject_idx", "load_case"]).any()
assert (subjects.groupby("subject_idx").size() == 5).all()
summary = {"scope": "Archived endpoint/statistical audit; no new FE solution",
           "n_subjects": len(base), "n_subject_load_rows": len(subjects)}

# Independently refit the ten published HC3 models and reproduce all slope tests.
saved = pd.read_csv(G / "allometry_models.csv")
rows = []
coef_err = []
for _, row in saved.iterrows():
    sub = subjects.loc[subjects.load_case == row.load_case].copy()
    outcome = row.outcome
    for name, covariates in [
        ("published", "age_z + AP_z + BiischiadicWidth_z + SubpubicAngle_z"),
        ("without_triad", "age_z"),
    ]:
        fit = smf.ols(f"np.log({outcome}) ~ log_total_volume * sex_F + {covariates}", sub).fit(cov_type="HC3")
        cov = fit.cov_params()
        a, b = "log_total_volume", "log_total_volume:sex_F"
        for sex in ["male", "female"]:
            slope = fit.params[a] + (fit.params[b] if sex == "female" else 0)
            var = cov.loc[a, a]
            if sex == "female":
                var += cov.loc[b, b] + 2 * cov.loc[a, b]
            se = np.sqrt(var)
            rows.append(dict(specification=name, load_case=row.load_case, outcome=outcome,
                             sex=sex, slope=slope, low=slope-1.96*se, high=slope+1.96*se,
                             p=2*norm.sf(abs(slope/se))))
            if name == "published":
                coef_err.append(abs(slope-row[f"{sex}_exponent"]))
        if name == "published":
            coef_err.append(abs(fit.pvalues[b]-row.sex_slope_interaction_p))
slopes = pd.DataFrame(rows)
slopes["q"] = np.nan
for spec in slopes.specification.unique():
    mask = slopes.specification == spec
    slopes.loc[mask, "q"] = multipletests(slopes.loc[mask, "p"], method="fdr_bh")[1]
slopes.to_csv(OUT / "volume_model_audit.csv", index=False)
summary["max_abs_published_slope_or_interaction_p_error"] = max(coef_err)
summary["published_slopes_negative"] = bool((slopes.loc[slopes.specification == "published", "slope"] < 0).all())
summary["published_max_slope_q"] = float(slopes.loc[slopes.specification == "published", "q"].max())

nested = pd.read_csv(G / "nested_sex_models.csv")
sex_err = []
for _, row in nested.iterrows():
    sub = subjects.loc[subjects.load_case == row.load_case]
    covariates = {"M0": "sex_F + age_z", "M1": "sex_F + age_z + log_total_volume_z",
                  "M2": "sex_F + age_z + log_total_volume_z + AP_z + BiischiadicWidth_z + SubpubicAngle_z"}[row.model]
    fit = smf.ols(f"{row.outcome} ~ {covariates}", sub).fit(cov_type="HC3")
    sex_err.append(abs(fit.params.sex_F-row.beta))
    sex_err.extend(abs(fit.conf_int().loc["sex_F"].to_numpy()-row[["ci95_low", "ci95_high"]].to_numpy(dtype=float)))
summary["max_abs_nested_sex_estimate_or_CI_error"] = float(max(sex_err))

# Marginal medians and correlations not independently checked by the older review.
summary["standing_asymmetry_medians_mm"] = subjects.loc[subjects.load_case.isin(["SP2leg", "SP1leg"])].groupby("load_case").lr_trans_asym_mm.median().to_dict()
lab1 = subjects.loc[subjects.load_case == "LAB_phase1"]
summary["LAB1_subpubic_translation_spearman"] = {}
for label, sub in [("pooled", lab1), ("male", lab1.loc[lab1.sex == "M"]), ("female", lab1.loc[lab1.sex == "F"])]:
    r, p = spearmanr(sub.SubpubicAngle, sub.trans_mag_mm)
    summary["LAB1_subpubic_translation_spearman"][label] = {"r": float(r), "p": float(p)}

# Agreement ratios and errors from the three archived variants.
agreement_errors = []
tails = []
variance = pd.read_csv(G / "variance_channels.csv")
for _, row in variance.iterrows():
    w = kin.loc[kin.load_case == row.load_case].pivot(index="subject_idx", columns="channel", values=row.metric)
    f, g, d = w.full.to_numpy(), w.shape_only.to_numpy(), w.material_only.to_numpy()
    err = g-f
    calculated = dict(identity_r2=1-np.sum(err**2)/np.sum((f-f.mean())**2), rmse=np.sqrt(np.mean(err**2)),
                      mae=np.mean(abs(err)), shape_over_full_pct=100*np.var(g, ddof=1)/np.var(f, ddof=1),
                      material_over_full_pct=100*np.var(d, ddof=1)/np.var(f, ddof=1))
    agreement_errors.extend(abs(calculated[k]-row[k]) for k in calculated)
    tails.append(dict(load_case=row.load_case, metric=row.metric, **calculated,
                      p95_absolute_error=float(np.quantile(abs(err), .95)),
                      maximum_absolute_error=float(max(abs(err))),
                      mean_bias=float(err.mean())))
pd.DataFrame(tails).to_csv(OUT / "agreement_audit.csv", index=False)
summary["max_abs_agreement_summary_error"] = float(max(agreement_errors))

# Explicit application-site contrasts absent from the manuscript, exploratory only.
site = []
for metric in ["ml_trans_abs_mm", "ap_trans_abs_mm", "cc_trans_abs_mm"]:
    w = subjects.pivot(index="subject_idx", columns="load_case", values=metric)
    delta = (w.LAB_phase2-w.LAB_phase1).to_numpy()
    rng = np.random.default_rng(20261002)
    medians = np.median(delta[rng.integers(0, len(delta), size=(3000, len(delta)))], axis=1)
    nonzero = delta[delta != 0]
    site.append(dict(metric=metric, comparison="LAB2 minus LAB1", median_delta_mm=float(np.median(delta)),
                     low=float(np.quantile(medians, .025)), high=float(np.quantile(medians, .975)),
                     p=float(binomtest(int((nonzero > 0).sum()), len(nonzero), .5).pvalue)))
site = pd.DataFrame(site)
site["q"] = multipletests(site.p, method="fdr_bh")[1]
site.to_csv(OUT / "exploratory_site_contrasts.csv", index=False)
summary["site_contrasts_status"] = "New exploratory review diagnostic, BH family of three component tests"
summary["source_manuscript_unchanged"] = True
(OUT / "statistics_audit_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
print(site.to_string(index=False))
print(slopes.loc[slopes.specification == "without_triad", ["load_case", "outcome", "sex", "slope"]].to_string(index=False))
