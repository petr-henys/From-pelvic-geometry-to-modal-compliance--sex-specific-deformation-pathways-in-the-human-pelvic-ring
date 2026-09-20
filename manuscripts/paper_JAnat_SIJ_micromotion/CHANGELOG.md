# Comprehensive Revision Changelog: Journal of Anatomy Submission

**Manuscript Title:** Sex-Specific Sacroiliac Joint Micromotion Under Locomotor and Obstetric Loading Proxies: A Cohort Finite-Element Study  
**Target Journal:** *Journal of Anatomy*  
**Date:** September 20, 2026  
**Authors:** Petr Henyš, Niels Hammer, et al.  
**Repository Directory:** `manuscripts/paper_JAnat_SIJ_micromotion/`

---

## Executive Summary of Scientific & Editorial Audits

This document provides a systematic record of all changes, statistical validations, anatomical clarifications, and editorial revisions implemented across the manuscript, tables, and figures in accordance with the rigorous standards of the *Journal of Anatomy*. 

Zero finite-element models were rerun or recomputed; all adjustments rely strictly on existing simulation outputs, cached datasets, and verified post-processing code.

---

## 1. Load Cases and Boundary Conditions Rectification
- **Correction:** Eradicated erroneous draft statements claiming that 500 N or 780 N downward vertical loads were applied at the S1 superior plateau with acetabular restraints.
- **Audited Truth:** The S1 superior facet is fixed via a high Dirichlet penalty ($\gamma = 10^6$) across all load regimes. Stance configurations apply upward acetabular ground reaction forces:
  - **SP2leg:** Bilateral acetabular upward force ($+400\text{ N}$ each, $800\text{ N}$ total craniocaudal).
  - **SP1leg:** Unilateral acetabular upward force ($+800\text{ N}$ right acetabulum).
- **Parturition Proxies:** Correctly characterized as localized mechanical proxies rather than physiologic fetal head dynamics:
  - **LAB1:** Internal mediolateral ring compression/shear proxy ($\pm 400\text{ N}$ transverse at pubic rami).
  - **LAB2:** Ischial tuberosity distraction proxy ($\pm 400\text{ N}$ transverse outward).
  - **LAB3:** Subpubic outlet AP distraction proxy ($\pm 400\text{ N}$ sagittal).
- **Files Modified:** `main.tex`, `sections/methods.tex`, `sections/results.tex`, `sections/discussion.tex`, `tables/generated/table2_loads.tex`, `figures/Fig1_anatomy_coordinates_loads.pdf`.

---

## 2. Kinematic Metrics and Sign Conventions
- **Clarification:** Joint micromotion is reported as the bilateral mean of absolute component magnitudes ($|\text{nutation}|$, $|\text{AP}|$, $|\text{ML}|$, $|\text{CC}|$, and resultant displacement/rotation). This prevents lateral cancellation across bilateral joints while accurately capturing total motion magnitude.
- **Files Modified:** `sections/methods.tex`.

---

## 3. Cohort Baseline and Numerical Reconciliation
- **Table 1 vs. Section 3.1:** Fully synchronized all morphometric values between Table 1 and Section 3.1:
  - Total Pelvic Volume: Female $736.79 \pm 105.99\text{ cm}^3$ vs. Male $925.28 \pm 126.99\text{ cm}^3$ ($p = 2.4 \times 10^{-29}$).
  - AP Inlet Diameter: Female $121.04 \pm 10.02\text{ mm}$ vs. Male $112.34 \pm 9.07\text{ mm}$ ($p = 1.1 \times 10^{-12}$).
  - Biischiadic Width: Female $109.87 \pm 7.13\text{ mm}$ vs. Male $97.71 \pm 7.06\text{ mm}$ ($p = 8.5 \times 10^{-35}$).
  - Subpubic Arch Angle: Female $85.80 \pm 6.14^\circ$ vs. Male $69.11 \pm 5.26^\circ$ ($p = 1.1 \times 10^{-58}$).
- **Table 3 vs. Section 3.2:** Reconciled median [IQR] kinematic values:
  - SP2leg Nutation: $1.062^\circ$ [0.653, 1.522]; SP2leg Asymmetry: $0.215\text{ mm}$ [0.093, 0.347].
  - SP1leg Nutation: $1.160^\circ$ [0.819, 1.619]; SP1leg Asymmetry: $2.013\text{ mm}$ [1.423, 2.548].
  - LAB1 Nutation: $0.432^\circ$ [0.257, 0.731]; LAB2 Nutation: $0.412^\circ$ [0.247, 0.630].
- **Files Modified:** `sections/results.tex`.

---

## 4. Nested Regression Models and Removal of Causal Claims
- **Removal of Mediation Claims:** Eradicated all causal mediation terminology ("mediator", "indirect path", "causal mechanism").
- **Implementation of Nested Regressions:** Implemented hierarchical regression sequence evaluating conditional differences (female minus male):
  - **Model 0 (Age-adjusted):** LAB1 $\beta_{\text{sex}} = +0.364\text{ mm}$ ($p < 10^{-30}$); LAB2 $+0.414\text{ mm}$; LAB3 $+0.344\text{ mm}$.
  - **Model 1 (Age + Volume):** LAB1 $\beta_{\text{sex}} = +0.295\text{ mm}$ ($p < 10^{-12}$); LAB2 $+0.329\text{ mm}$; LAB3 $+0.279\text{ mm}$.
  - **Model 2 (Age + Volume + Morphometric Triad):** LAB1 $\beta_{\text{sex}} = +0.224\text{ mm}$ ($p = 0.0042$); LAB2 $+0.193\text{ mm}$ ($p = 0.0196$); LAB3 $+0.231\text{ mm}$ ($p = 0.0038$).
- **Multicollinearity Diagnostics:** Screened Variance Inflation Factors in Model 2: Female sex (5.55), subpubic angle (4.76), biischiadic width (3.03), pelvic volume (2.62), AP diameter (1.59), and age (1.55)—all well below the standard concern threshold of 10.
- **Files Modified:** `sections/methods.tex`, `sections/results.tex`, `tables/generated/table4a_primary_models.tex`.

---

## 5. Paired Subject-Level Diagnostics (Full vs. Shape Models)
- **Addition of Paired Statistics:** Accompanied population variance ratios with paired within-subject diagnostics across all load regimes:
  - Resultant Translation: $R^2 \ge 0.994$, Pearson $r \ge 0.997$, Spearman $r_s \ge 0.996$, $\text{RMSE} \le 0.031\text{ mm}$, $\text{MAE} \le 0.021\text{ mm}$.
  - Resultant Rotation: $R^2 \ge 0.999$, Pearson $r \ge 0.999$, Spearman $r_s \ge 0.999$, $\text{RMSE} \le 0.023^\circ$, $\text{MAE} \le 0.015^\circ$.
- **Clarification of Variance Ratios:** Explicitly clarified that $V_{\text{shape}} / V_{\text{full}}$ and $V_{\text{material}} / V_{\text{full}}$ represent sample variance retention ratios from separate datasets rather than an orthogonal additive decomposition.
- **Files Modified:** `sections/methods.tex`, `sections/results.tex`.

---

## 6. Secondary Allometric Scaling Exponents
- **Precision:** Corrected text claiming no scaling under parturition loads. Clarified that female pelves in LAB2 retain a statistically significant negative allometric scaling slope ($\beta_{\text{female}} = -0.614$, 95% CI [$-1.061$, $-0.166$], raw $p = 0.0072$, BH-FDR $q = 0.0360$).
- **Interaction Testing:** Confirmed that sex-by-size interaction terms are non-significant after FDR correction across all load cases ($q \ge 0.958$).
- **Files Modified:** `sections/results.tex`, `tables/generated/table4b_allometry.tex`.

---

## 7. Tone, Style, and Restrained Scientific Language
- **Tone Calibration:** Systematically purged promotional, anthropomorphic, and hyperbolic language (*quintessential, profound, exquisite, remarkable, stark contrast, tenfold surge, astonishing*).
- **Calibrated Descriptions:** Replaced with objective biomechanical descriptions (*pronounced difference, preferential recruitment, localized compliance, order-of-magnitude shift*).
- **Discussion Restraint:** Reframed clinical and surgical implications as mechanical insights, cautioning against direct surgical translation in the absence of muscular tone, multi-ligament pretension, and dynamic physiological loads.
- **Files Modified:** `sections/introduction.tex`, `sections/results.tex`, `sections/discussion.tex`, `sections/conclusion.tex`.

---

## 8. Journal of Anatomy Abstract & Formatting
- **Abstract Compliance:** Rewritten into a single cohesive paragraph of 442 words (strictly under the journal's 500-word limit) without any subheadings.
- **Formatting Standards:** Formatted $p$-values as $p < 0.001$ instead of $p = 0.000$; removed raw Python statsmodels strings (`C(...)`) from all LaTeX tables.
- **BibTeX Diacritics:** Fixed corrupted diacritic in `bib/refs.bib` (`No{\"e}`).
- **Files Modified:** `main.tex`, `bib/refs.bib`, `analysis/format_publication_tables.py`.

---

## 9. Visual Figures Update and Regeneration
- **Fig 1 (Anatomy, Coordinates, Loads):** Corrected boundary condition schematics (Dirichlet fixity at superior S1 facet) and upward acetabular ground reaction force arrows.
- **Fig 3 (Directional Re-routing):** Updated Panel B title to "2D kinematic state space".
- **Vector PDF Regeneration:** Regenerated Figs 1–6 at 300 DPI vector PDF standards matching the revised numerical outputs.
- **Files Modified:** `analysis/generate_publication_figures.py`, `figures/Fig1_anatomy_coordinates_loads.pdf` through `Fig6_allometry_loglog.pdf`.

---

## 10. Document Typesetting & Compilation
- **Typesetting Cleanup:** Resolved all overfull `\hbox` warnings in `methods.tex` and `results.tex` by optimizing line breaking around math strings and hyphenated compounds.
- **Log Verification:** Fully compiled `main.pdf` (36 pages) via `pdflatex` and `bibtex` with **0 errors, 0 warnings, and 0 overfull/underfull hboxes**.

---

## 11. Mathematical Appendix A: Complete Algorithmic Formulation
- **New Section Created:** `sections/appendix.tex` added to `main.tex` as **Appendix A: Mathematical Formulation of Sacroiliac Joint Micromotion Analysis**.
- **Equilibrium & Variational Boundary Value Problem:** Posed the weak form of 3D small-strain elasticity with Dirichlet penalty parameter $\gamma = 10^6\text{ N/mm}$ on the superior S1 facet $\Gamma_{S1}$, and surface traction integrals defining SP2leg, SP1leg, LAB1, LAB2, and LAB3.
- **Pelvic Coordinate Frame $\mathbf{P}$:** Formulated optimal midsagittal symmetry plane minimization ($\hat{\mathbf{x}}_{\text{ML}}$), in-plane SVD/PCA for craniocaudal orientation ($\hat{\mathbf{z}}_{\text{CC}}$), and orthogonal cross-product for anteroposterior orientation ($\hat{\mathbf{y}}_{\text{AP}}$).
- **Global Sacrum Affine Drift Removal:** Rigorously specified regularized least-squares affine transformation $(\mathbf{A}_S, \mathbf{b}_S)$ mapping deformed sacral points back to reference space.
- **Articular ROI Segmentation:** Segmented opposing joint facets via Euclidean proximity threshold $g = 5.0\text{ mm}$ and minimum support threshold $N_{\min} \ge 80$.
- **Trimmed Kabsch Registration:** Specified cross-covariance SVD with reflection prevention ($\det(\mathbf{R}_B) = +1$) and 10% residual trimming over 2 iterations.
- **Cardan Angle Decomposition:** Closed-form singularity-robust Tait-Bryan $XYZ$ factorization for nutation $\theta_{\text{nut}} = \alpha_x$, abduction $\theta_{\text{abd}} = \beta_y$, and axial twist $\theta_{\text{rot}} = \gamma_z$.
- **Contact Centroid Displacement:** Projected centroid translation $\mathbf{d}_{\text{pelvis}} = \mathbf{P}^T [(\mathbf{R}_I - \mathbf{R}_S)\mathbf{p}_c + (\mathbf{t}_I - \mathbf{t}_S)]$ into local pelvic axes.
- **Morphometric Triad Landmarks:** Formalized geometric distance and angle equations for $d_{\text{AP}}$ (conjugate vera), $w_{\text{biisch}}$ (interspinous width), and $\alpha_{\text{subpubic}}$ (subpubic arch angle).
- **Files Modified:** `sections/appendix.tex`, `main.tex`, `sections/methods.tex`, `sections/results.tex`.

---

## 12. Figure 1 3D Anatomy Modernization & 4-Panel Layout
- **Elimination of Rendering Defects:** Replaced the coplanar volumetric tetrahedral chunk of the S1 facet with an oriented elliptical cylinder plate ($r_{\text{maj}} = 22.5\text{ mm}, r_{\text{min}} = 14.5\text{ mm}, h = 3.0\text{ mm}$) centered along the facet normal, completely eliminating black edges and SSAO occlusion artifacts. Removed 1D ligament line elements to clear pelvic outlet view.
- **Panel A (Anatomy & Morphometric Triad):** Superimposed landmark vectors and dashed measurement overlays for $d_{\text{AP}}$ (inlet AP conjugate vera), $w_{\text{biisch}}$ (biischiadic interspinous width), and $\alpha_{\text{subpubic}}$ (subpubic arch angle) with non-overlapping callout annotations.
- **Panel B (Locomotor Stance & Boundary Conditions):** Structured comparative cards for SP2leg and SP1leg with boundary fixity schematics, bilateral ground reaction arrows, and load-specific mechanical summaries.
- **Panel C (Parturition Mechanical Expansion Proxies):** Added designated cards and directional force schematics for LAB1 (inner ring compression/shear proxy), LAB2 (ischial tuberosity distraction proxy), and LAB3 (outlet AP distraction proxy).
- **Panel D (Applied Force Components):** Generated horizontal 3D force component bar chart (ML, AP, CC) for every load case and contact surface.
- **Files Modified:** `analysis/generate_publication_figures.py`, `figures/Fig1_anatomy_coordinates_loads.pdf`, `figures/Fig1_anatomy_coordinates_loads.png`, `sections/results.tex`.

