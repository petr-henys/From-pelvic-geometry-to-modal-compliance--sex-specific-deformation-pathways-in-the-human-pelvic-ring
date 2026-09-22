# Numerical and interpretive consistency review

Reviewed 22 September 2026. Scope: manuscript text, published tables, seven main figures and two supplementary figures, figure-generation code and existing numerical outputs. No simulations, subject-level analyses, model fits, correlations or bootstrap intervals were recomputed. Selecting extrema and the middle value from saved summary rows was used to check reported ranges.

## Numerical correspondence

| Claim or display | Existing evidence | Finding |
| --- | --- | --- |
| 278 subjects; 150 female, 128 male; age 16–91 | `analysis_summary.json`, Table 1 | Consistent. |
| Standing rotation 0.956/2.040 degrees and translation 1.058/1.627 mm | `table3_micromotion_summary.csv`, Table 3, Figure 3 | Consistent. Figure 3 shows sex-specific distributions; Table 3 and text pool both sexes. Caption clarified. |
| Paired standing changes +1.048 degrees, +0.582 mm translation, +1.509 mm asymmetry | `standing_contrasts.csv` | Estimates, intervals and q < 0.001 agree with text. These are medians of within-subject differences, not differences between marginal medians. |
| Directional changes relative to SP2leg | `directional_rerouting.csv`, Table S1, Figure 4 | Displayed signs, estimates and intervals agree. LAB1 ML: −0.006891 mm, CI −0.017532 to +0.001566, q = 0.254434. LAB2 ML: +0.052249 mm, CI +0.046564 to +0.060683, q < 0.001. LAB3 AP and ML increases are supported by the saved paired contrasts. |
| Translation coefficient attenuation M0 → M1 → M2 | `nested_sex_models.csv`, `sex_models_lab.csv`, Figure 5, Table S3 | Rounded text and figure estimates agree. M2 translation: +0.012969, −0.031246, +0.001967 mm. All six localized-load M2 intervals (translation and rotation) span zero. Translation q values: 0.566005, 0.566005, 0.950371. Maximum M2 VIF rounds to 5.55. |
| Geometry/density comparison | `variance_channels.csv`, Figure 6, Table S2 | Shape/full range 96.432854–120.611472%, median 100.675791%; material/full range 0.013797–5.788682%, median 0.243695%. Text rounding agrees. Translation minimum identity R² 0.945707, maximum RMSE 0.027968 mm; rotation 0.877658 and 0.031786 degrees. |
| Twenty negative conditional slopes | `allometry_models.csv`, Table 4, Figure 7 | All negative; range −1.062121 to −0.365556. Largest slope q = 0.00187363. Interaction q range 0.343283–0.698681. Estimates and intervals agree with displayed values. Reference lines are correctly distinguished from fitted slopes. |
| Linear-scale parameterization | `allometry_models_scale.csv`, supplementary slope table | Reparameterization uses three times the volume slope and interval endpoints, consistent with log V = 3 log L. |
| Extraction sensitivity | Saved `tables/corrected/rigid_correction_sensitivity.csv`, Table S6 | Reported maxima 0.0229 degrees and 0.0055 mm are consistent with the saved six-subject comparison; inference remains restricted to that subset. |

Paths in the table refer to `tables/generated/` unless otherwise specified. Figure numbers are manuscript numbers, not filename prefixes.

## Figure review

- Figures 1–2: anatomical measurement definitions and load labels were inspected. Morphometric values displayed on the template are not cohort means. Standing loads total 800 N; the localized loads are opposing 400 N force pairs.
- Figure 3: sex colors, units, load ordering and distribution displays are consistent with their descriptions. Separate sex-specific boxes should not be read as the pooled medians in Table 3.
- Figure 4: printed contrasts agree after rounding; arrows represent componentwise median contrasts, not physical trajectories or signed joint opening.
- Figure 5: adjusted coefficients, intervals, q labels and nested translation coefficient paths agree with saved model summaries. Printed correlation coefficients agree with the text at the figure's precision.
- Figure 6: heatmap values correspond to saved variance ratios; the two panels use different color scales. Ratios exceeding 100% do not denote variance explained.
- Figure 7: load/sex assignments, slope signs, intervals and dimensional reference lines are consistent with Table 4 and the saved model summaries.
- Supplementary Figure S1: plotted directions and stated conditional adjustment agree with the model description. Full prediction curves and bands could not be independently reconstructed from the saved slope-only summaries without refitting; no refit was performed.
- Supplementary Figure S2: schematics agree with the stated rotation-triplet, translation and bone-volume conventions.

## Discussion and conclusion

The main conclusions are supported within the passive static model: inverse conditional size–motion associations, close retention of responses after density standardization, increased motion/asymmetry under unilateral support, and attenuation of sex-associated translation contrasts after anatomical adjustment under localized loading.

Targeted corrections:

1. Discussion and conclusion now identify the outcome and loading scope of the attenuation claim explicitly.
2. The load-path paragraph states the LAB1 and LAB2 findings relative to SP2leg. Different significance outcomes do not themselves establish a direct LAB1-versus-LAB2 contrast; the absence of that reported contrast is stated once.
3. The functional-anatomy synthesis now describes the observed inverse size association and geometry-preserving comparison directly, avoiding a stronger mechanistic interpretation of the conditional regression.
4. Figure 3 caption distinguishes sex-specific boxplots from pooled numerical summaries.

Preserved qualifications: no equivalence or causal mediation conclusion; no out-of-sample prediction claim; variance ratios are not an additive causal decomposition; between-load slope comparisons are descriptive; dimensional references do not validate a scaling law; fixed tissue properties limit the density comparison; SIJ motion does not establish birth-canal expansion.

## Limits of independent verification

The reported pooled asymmetry medians (0.065 and 1.602 mm) and exact Spearman statistics/p values were not located in separate saved summary outputs. The corresponding figure/text agreement was checked, and the saved paired asymmetry contrast was verified, but these marginal medians and correlation statistics were not independently recalculated. Likewise, Supplementary Figure S1 prediction bands lack saved prediction/covariance outputs sufficient for an independent numerical reconstruction. These are provenance/verification limits, not demonstrated numerical errors. Resolving them would require additional archived outputs or analysis that was excluded by the task constraints.
