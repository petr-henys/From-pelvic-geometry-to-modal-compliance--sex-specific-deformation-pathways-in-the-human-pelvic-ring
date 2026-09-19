# Pre-Submission Scientific Audit and Verification Report

**Date:** September 19, 2026  
**Status:** Pre-Submission Verification and Synchronization Complete  
**Manuscript:** *Stable and recurrent deformation subspaces preserve pelvic mechanical function across anatomical variability* (`manuscripts/natcomm/main_plos.tex`)  
**Target Journal:** *PLOS Computational Biology*

---

## 1. Issues Found

1. **Max-Nodal Anatomical Capacity Formulation:**
   The previously reported capacity was calculated by optimizing the functional on the unit modal sphere ($\max_{\|\boldsymbol\alpha\|_2 \le 1} |\mathbf b_\ell^\top Q \boldsymbol\alpha| = \|Q^\top \mathbf b_\ell\|_2$) and dividing post-hoc by the maximum nodal displacement. This did not strictly equal the maximum anatomical diameter change per 1\,mm maximum nodal displacement.
2. **Optimizer Characterization for $r=4$ Subspaces:**
   Section 4.5 claimed that Nelder–Mead optimization over the coefficient vector “yields the global maximum”, which is not mathematically guaranteed for general non-convex simplex searches.
3. **Ambiguity in Load Routing Methodology:**
   The manuscript text did not explicitly distinguish between the primary cohort-wide modal routing (computed from total static displacements) and the scale-invariant incremental load routing sensitivity check (subtracting baseline ligament pretension).
4. **Shapley Additivity Axiom Wording:**
   The additivity axiom in Appendix~E previously described linear decomposition of orthogonal fields rather than the standard mathematical game-theoretic axiom $\phi_i(v+w) = \phi_i(v) + \phi_i(w)$.
5. **Overstated Stability and Preservation Language:**
   Several passages used absolute claims (e.g., “is exactly preserved”), which oversimplified the biological mechanism.
6. **Degeneracy vs. Near-Degeneracy Distinction:**
   Operational clustering thresholds ($\mathrm{gap}_{\rm in} \le 10^{-6}$) were conflated in earlier phrasing with exact mathematical multiplicity ($\lambda_i = \lambda_{i+1}$).
7. **Eigensolve Spectral Truncation:**
   Cluster 12–15 was described symmetrically with cluster 9–10, omitting that 12–15 is right-truncated by the modal solver cutoff ($N_{\rm ev}=15$).
8. **Shapley Decomposition Terminology:**
   The term “algebraic overlap” improperly implied rank deficiency in the 24-dimensional nonrigid polynomial dictionary, whereas the basis is full-rank and well-conditioned ($\kappa(\mathbf G) \approx 20.98$).
9. **Load Case Generalization:**
   Proxy forces of 400\,N and 800\,N were at risk of being overinterpreted as direct clinical labour simulations.
10. **Uncertainty Quantification Scope:**
    First-order parameter sensitivities risked being misconstrued as full non-linear Monte Carlo propagation.

---

## 2. Fixes Made

1. **Exact Max-Nodal Capacity Implementation:**
   - Implemented $\kappa_{\ell,\infty}(Q) = \max_{\boldsymbol\alpha \ne \mathbf 0} |\mathbf b_\ell^\top Q \boldsymbol\alpha| / \max_a \|(Q\boldsymbol\alpha)_a\|_2$ in `analysis/spectral_metrics.py` (`single_functional_capacity_max_nodal`).
   - For $r=2$: 90-point dense angular search on $[0, \pi)$ with bracketed Brent refinement on exact full-mesh nodal displacements.
   - For $r \ge 3$: Simplex search initialized with the $L_2$-extremal vector and coordinate axes.
   - Verified 100% exact constraint satisfaction ($\max_a \|\boldsymbol\psi_a\|_2 = 1.00000000$), strict rotational invariance under $\mathrm O(r)$, and sign invariance.
2. **Validation and Textual Correction of $r=4$ Optimizer:**
   - Benchmarked Nelder–Mead on modes 12–15 against a 59-seed global multi-start Powell optimization across representative subjects: differences were exactly $0.0000\%$ ($< 10^{-6}$ discrepancy).
   - Removed any claim that Nelder–Mead guarantees the global maximum; updated Section 4.5 to use the exact formulation: “Nelder–Mead simplex optimisation was used to estimate the optimum; multi-start validation on representative subjects agreed to $<10^{-6}$.”
3. **Clarification of Primary Routing vs. Incremental Sensitivity Check:**
   - Explicitly defined the total static load vector $\mathbf b = \mathbf f_{\rm ext} + \mathbf f_{\rm pret}$ in Section 4.4 for the primary total-response analysis ($K_s \mathbf u = \mathbf b$).
   - Declared that primary full-cohort routing (Figure 6, Table 2) evaluates total static displacements, whereas incremental routing ($\delta\mathbf u_{\rm ext} = K^{-1}\mathbf f_{\rm ext}$) was evaluated as an $N=10$ sensitivity check, proving that baseline pretension subtraction shifts block energies by $\le 3.5\%$, preserves 100% of proxy top modes, and leaves $N_{80}$ invariant.
4. **Standard Shapley Additivity Axiom:**
   - Rewrote the additivity axiom in Appendix~E to the standard mathematical definition:
     $$\phi_m(v + w) = \phi_m(v) + \phi_m(w) \quad \text{for all } (v+w)(S) = v(S) + w(S).$$
5. **Toned Down Stability Language:**
   - Softened claims throughout the abstract, introduction, results, and discussion, replacing absolute preservation terms with “remains remarkably stable”, “is substantially more stable at the subspace level”, or “maintains mechanical capability”.
6. **Multiplicity vs. Near-Degeneracy Clarification:**
   - Explicitly separated operational near-degeneracy ($\mathrm{gap}_{\rm in} \le 10^{-6}$) from mathematical multiplicity ($\lambda_i = \lambda_{i+1}$), stating in Section 2.1 and Section 4.1: “No obvious spatial symmetry was identified that would enforce repeated eigenvalues, although accidental parameter-dependent crossings are not mathematically excluded.”
7. **Cluster Boundary Clarification:**
   - Clarified in Section 2.1 and 2.3 that modes 9–10 are externally separated on both sides, whereas modes 12–15 are left-separated but right-truncated by the eigensolve boundary ($N_{\rm ev}=15$).
8. **Shapley Terminology Update:**
   - Replaced “algebraic overlap” and “intrinsic mathematical overlap” with “intrinsic non-orthogonality”, “shared explained norm”, and “correlated explanatory subspaces”.
9. **Standardized Comparative Probes:**
   - Framed 400\,N and 800\,N loads strictly as standardized linear comparative probes, explicitly stating limitations regarding clinical labour simulation.
10. **First-Order Sensitivity Framing:**
    - Explicitly designated UQ analyses as local linear first-order approximations, highlighting the absence of non-linear Monte Carlo as a documented limitation.
11. **Cohort Recomputations and LaTeX Synchronization:**
    - Regenerated `subject_metrics.csv`, `values.tex`, and Figures 2–6 using serial execution.
    - Full-cohort AP capacity median difference is minimal ($\Delta = -0.006$\,mm/mm, 95% CI [$-0.048, +0.019$], Mann–Whitney $p = 0.106$), contrasting with single-rank collapse ($\Delta \approx 0.30$--$0.32$\,mm/mm, $p < 10^{-33}$).

---

## 3. Confirmation of Computational Safety & No Expensive Recomputations

- **Zero Broad Cohort FE Solves:**
  No broad cohort finite-element reruns were launched. The incremental load routing sensitivity check was conducted on an $N=10$ representative subset (5 exchange, 5 non-exchange) using existing stored displacement fields.
- **Strictly Serial Execution:**
  All algebraic recomputations (`analysis/plos_revision.py`) ran strictly serially in a single Python process on 1 CPU core.
- **Thread Limits Enforced:**
  `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`, `VECLIB_MAXIMUM_THREADS=1` were active throughout.
- **Resource Footprint:**
  Peak memory was restricted to 2.8\,GB ($4.3\%$ of system RAM), with 0\,B swap usage and zero workstation instability.

---

## 4. Confirmation of Manuscript Quality, PDF Build & Consistency

- **Automated Results Audit (`validate_manuscript_results.py`):**
  - Stale keywords: **PASSED** (0 occurrences of banned phrases).
  - Values macro concordance: **PASSED** (100% concordance between `values.tex` and CSV tables).
  - Table E1 and analytical cube overlap ($-22/45$ and $-11/180$): **PASSED**.
- **Pytest Suite:**
  - `tests/test_scientific_audit_math.py`, `tests/test_plos_revision_metrics.py`, `tests/test_manuscript_consistency.py`: **21 passed in 1.23s** (100% pass).
- **Text and Cross-Reference Audit:**
  - Duplicate words: **0** found.
  - Labels and references: **72 labels, 26 refs, 0 unresolved/missing**.
  - Citations: **33 cited keys, 33 .bbl entries, 0 missing, 0 unused**.
  - Broken references in compiled PDF (`??` search): **0** occurrences.
  - LaTeX compilation (`latexmk -pdf`): **Clean build (exit code 0, 0 undefined references)**, generating `main_plos.pdf` (50 pages, 5.7\,MB).
- **Visual Inspection:**
  - Figures 1–9, Figures B1, C1, D1, D2, and Tables 1–2, A1–A6, B1–B2, C1–C3, D1–D2, E1–E2 inspected and confirmed free of clipped text, bad line breaks, or overlapping labels.
