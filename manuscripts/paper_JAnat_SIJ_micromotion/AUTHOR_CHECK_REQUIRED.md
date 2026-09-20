# Author Check Required: Key Decisions & Verification Points

**Manuscript:** Sex-Specific Sacroiliac Joint Micromotion Under Locomotor and Obstetric Loading Proxies: A Cohort Finite-Element Study  
**Authors:** Petr Henyš, Niels Hammer, et al.  
**Target:** *Journal of Anatomy*  
**Date:** September 20, 2026

Please review the following three pivotal scientific, anatomical, and methodological items that were refined during the comprehensive revision.

---

### 1. Boundary Condition Clarification (Ground Reaction vs. Sacral Restraint)
- **Previous Draft Wording:** Stated that 500 N (or 780 N) downward vertical loads were applied at the S1 superior plateau with bilateral acetabular restraints.
- **Audited Simulation Code & Metadata:** 
  The actual FE solver scripts (`run_cohort_simulation.py`, `simulation/boundary_conditions.py`) constrain the **superior S1 facet** using a Dirichlet penalty ($\gamma = 10^6$) and apply **upward ground reaction forces at the acetabula** (+400 N each for bilateral stance, +800 N unilateral).
- **Manuscript Update:** 
  We completely aligned the text, equations, Table 2, and Figure 1 (Panel B & C) with the ground reaction formulation.
- **Action for Authors:** 
  Confirm that this upward ground reaction description accurately represents the physiological boundary conditions you intended to communicate for Journal of Anatomy reviewers.

---

### 2. Elimination of Causal Mediation Language & Adoption of Nested Regressions
- **Previous Framing:** Used classical mediation terms ("mediates the sex effect", "indirect pathway"), which would draw severe scrutiny from reviewers given that skeletal shape and sex are biologically non-separable and non-manipulable.
- **Revised Methodological Framing:** 
  We re-framed this analysis as **nested multivariable regressions** (Model 0: Age-adjusted $\to$ Model 1: Age + Pelvic volume $\to$ Model 2: Age + Pelvic volume + Morphometric triad: AP diameter, biischiadic width, and subpubic arch angle).
- **Key Finding:** 
  Conditioning on total volume and the morphometric triad attenuates the female mobility surplus from $+0.364\text{ mm}$ to $+0.224\text{ mm}$ (a ~38% attenuation), demonstrating that enhanced joint compliance is substantially associated with anterior arch geometry and outlet dimensions, while leaving a modest residual conditional difference. Multicollinearity was verified with VIF $< 6.0$.
- **Action for Authors:** 
  Verify that this nested conditional association framework fulfills your intended anatomical narrative without over-claiming causality.

---

### 3. Female Negative Allometric Scaling in LAB2 (Ischial Distraction)
- **Previous Text:** Stated that allometric scaling with pelvic volume disappears under all parturition configurations.
- **Empirical Re-check:** 
  In LAB2 (ischial tuberosity distraction), female translation retains a statistically significant negative allometric scaling slope:
  $$\beta_{\text{female}} = -0.614,\quad 95\%\;\text{CI } [-1.061, -0.166],\quad \text{raw } p = 0.0072,\quad \text{BH-FDR } q = 0.0360$$
  whereas male scaling is non-significant ($\beta_{\text{male}} = -0.367, p = 0.1809$).
- **Manuscript Update:** 
  We accurately reported this specific female scaling retention in Section 3.6 and Table 4B, noting that the sex-by-size interaction term remains non-significant ($q \ge 0.958$).
- **Action for Authors:** 
  Note this nuance in interpretation: larger female pelves demonstrate slightly less ischial distraction compliance than smaller female pelves under an identical 400 N load.

---

### 4. Parturition Configurations as "Mechanical Proxies"
- **Revised Framing:** 
  Throughout the manuscript, LAB1–LAB3 are consistently designated as "localized mechanical proxies" (internal ring compression/shear, ischial tuberosity distraction, and subpubic outlet distraction) rather than dynamic simulations of obstetric labor. This prevents objections regarding the lack of hormone-mediated ligamentous laxity (relaxin), dynamic fetal passage, and soft tissue/pelvic floor active tone.
- **Action for Authors:** 
  Confirm agreement with this cautious framing.

---

### 5. Mathematical Appendix A & Modernized Figure 1
- **New Appendix A:** 
  Added a 6-page comprehensive mathematical appendix (`sections/appendix.tex`) deriving the weak form of linear elasticity, Dirichlet boundary penalty ($\gamma = 10^6\text{ N/mm}$), pelvic coordinate system construction via symmetry plane and PCA, regularized affine sacrum drift elimination, trimmed Kabsch registration, Cardan angle extractions, and landmark formulas for the morphometric triad.
- **Modernized Figure 1:** 
  Upgraded Figure 1 to a 4-panel publication visual:
  - **Panel A:** 3D pelvic anatomy with artifact-free elliptical cylinder S1 facet plate (eliminating previous Z-fighting tetrahedral edges and ligament wire clutters) and morphometric triad overlays ($d_{\text{AP}}$, $w_{\text{biisch}}$, $\alpha_{\text{subpubic}}$).
  - **Panel B:** Symmetrical (SP2leg) and asymmetrical (SP1leg) stance configurations with boundary condition diagrams and resultant ground reaction arrows.
  - **Panel C:** Parturition expansion proxies (LAB1, LAB2, LAB3) with force direction arrows and anatomical targets.
  - **Panel D:** 3D applied force component bar chart across ML, AP, and CC axes.
- **Action for Authors:** 
  Inspect the generated `Fig1_anatomy_coordinates_loads.pdf` and Appendix A in `main.pdf` to confirm visual layout and mathematical notations meet your expectations.

