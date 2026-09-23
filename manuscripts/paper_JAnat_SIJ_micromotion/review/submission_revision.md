# Submission revision — 23 September 2026

> Subsequent author-directed editorial change (23 September 2026): the mapping-diagnostic paragraph and specific discussion of negative Jacobians were removed from the manuscript. The underlying measurements below remain an internal diagnostic record; their exclusion does not establish that they are artifacts or validate the mapping. No further calculations listed in section C were performed.

The current `main.tex` and included sources were revised. No separate file named `main(5)` was present in the supplied attachment or workspace; the attachment contained the revision instructions. The existing section structure, central scale–geometry–load-path story, primary estimates and prior removal of unrelated project links were preserved.

**Submission status: author review is required before submission because the mapping check found local inversions. This revision does not certify the numerical model as fully validated.**

## A. Manuscript-only changes

- Replaced model names throughout text, captions, Supporting Information, table headers and plotting labels: **geometry-preserved / standardized-density (G)** and **density-preserved / template-geometry (D)**. G retains scale and uses the spatially varying population-median density field, not homogeneous bone. Existing `shape_only`, `material_only`, `shape_over_full_pct` and related internal keys remain unchanged; they map to G and D respectively.
- Clarified the zero-elastic-displacement anatomical reference in main Methods. Motion includes prescribed pretension plus external loading; no separate equilibrated pretension-only reference is used. Added the corresponding proportionate limitation.
- Added verified P1 tetrahedral displacement/geometry, DG0 material fields, degree-6 quadrature, PETSc PREONLY/LU/MUMPS and configured tolerance settings. Explained that configured iterative tolerances are not an iterative stopping test for the selected direct solve.
- Defined architecture, total mapped bony pelvic volume and the selected morphometric triad. Replaced broad adjustment claims with volume plus selected dimensions, retaining the limits on causal and equivalence interpretations.
- Strengthened the endpoint-specific density interpretation under the constitutive law, low-density modulus floor and fixed cartilage/ligament properties and pretension.
- Corrected the constrained region to **superior S1 endplate**. Inspected the actual `S1_facet` surface in `anatomy_data/full_model_new.feb` (631 triangular faces, 351 nodes, one approximately planar central patch) and the anatomical rendering. The internal model region name was not changed.
- Expanded Data Availability to distinguish source imaging/anatomy, derived morphometry/endpoints, machine-readable estimates, code and Supporting Information. Verified the [BoneDat source-data record](https://zenodo.org/records/15189761) against the [data-descriptor article](https://www.nature.com/articles/s41597-025-05161-y). It is explicitly not presented as a deposit of this study's FE results or code. No study-specific DOI, repository or sharing promise was invented.

## B. Inexpensive analyses/post-processing performed

### Mapping verification

`analysis/check_mapping_quality.py` evaluated the exact cellwise determinant of the implemented P1 geometry map from existing geometry arrays and FE connectivity. Single-threaded serial run: approximately 17 seconds; no FE assembly or solve.

- 278 anatomies × 166,808 tetrahedra = **46,372,624 subject–cell pairs**.
- **425 negative Jacobians in 224 anatomies**; minimum **−2.5834614973**.
- At most 7 affected cells in one anatomy; 97 distinct reference cells affected across the cohort.
- No non-finite values. Subject-specific first percentiles of J ranged from 0.205972 to 0.901564.
- Maximum fraction of reference mesh volume in affected cells: **0.000527306%**. This small fraction does not demonstrate a negligible mechanical effect.
- Independently reproduced each subject's worst cell using direct SciPy RBF evaluation and a separate determinant of `I + grad(phi)`.
- The full and G variants use the same maps; D uses the template geometry and J = 1. Code and configurations were checked for this correspondence.

Results are in `tables/revision/mapping_quality.csv`, `mapping_nonpositive_cells.csv` (with cell indices and reference locations), and `mapping_summary.json`. The actual finding is reported in Appendix A.1 and the limitation paragraph. No geometry or FE result was repaired or replaced.

### Age functional form

`analysis/check_age_function.py` added squared standardized age, retaining HC3 covariance and the same subjects, other predictors and multiplicity families. The original nested-sex coefficients and volume slopes were reproduced to absolute tolerance 1e−10 before adding the new term.

- All 6 M2 confidence intervals still include zero.
- Maximum absolute M2 translation-coefficient change: **0.000333404 mm**.
- All 20 conditional volume slopes remain negative and pass BH correction; maximum q = **0.00190232**.
- Maximum absolute slope change: **0.0195832**.
- Translation attenuation from M0 to M2 persists.

Separate diagnostic CSVs and summary JSON are in `tables/revision/age_quadratic_*`. Supporting Tables S7–S8 report the estimates. The primary linear-age models were not replaced.

### Extraction pilot

`analysis/submission_diagnostics.py` conservatively timed rigid-only extraction from saved displacements for subjects 0–2, all five loads, using one computational thread. The first subject's affine extraction was also checked against existing endpoints to 1e−8. Subsequent subjects took 8.2–8.5 seconds each, projecting about **39 minutes** for 278 subjects. The pilot stopped after three subjects in accordance with the cost constraint. Pilot results are separate in `tables/revision/extraction_pilot`; they are not pooled with or substituted for the existing six-subject sensitivity in the manuscript.

### Validation and outputs

- Clean isolated LaTeX builds of main text and supplement succeeded, with no undefined references, duplicate labels or overfull boxes.
- Regenerated only the anatomical-overview and variance-comparison figures whose labels changed; statistical plotting values are unchanged.
- 14 primary CSV files were byte-identical after revision. The cohort-table CSV changed labels only; every other cell is identical. See `primary_output_hashes_before.json` and `review/submission_validation.json`.
- DOCX exports contain 7 figures / 4 tables in the main text and 2 figures / 9 tables in the supplement, with native editable equations. Selected PDF tables and revised figures were inspected visually. Word pagination was not rendered in Microsoft Word/LibreOffice in this environment.
- No FE cohort, parallel simulation job, pretension-only solve or mesh-convergence study was run.

## C. Requested checks not completed

1. **Full-cohort rigid-only extraction:** not run after the pilot projected roughly 39 minutes. Section 3.6 and Table S6 retain the original six-subject / 30-observation scope and maxima, with the subset limitation intact.
2. **Fraction of FE elements/volume below the density threshold:** not reported. `MaterialMapper` applies the piecewise density–modulus law at source density samples and then RBF-interpolates modulus to DG0 cell locations. The mapped modulus is therefore not a stored cellwise density-threshold classification. Inferring a unique branch or volume fraction from it would conflate the pre-interpolation density law with the interpolated FE property. No surrogate metric was invented.
3. **Effect or correction of the negative Jacobians:** not evaluated by new FE solves, which were prohibited. A claim of universally positive J would be false.
4. **Mesh convergence:** not undertaken, as requested. Local orientation checks are not an accuracy or convergence study.

## D. Issues for author review

1. **Resolve the mapping defects before submission.** The implementation uses signed J in the volume and surface pullbacks; local inversions cannot be dismissed on the basis of their small volume fraction. Inspect the listed cells, determine a sound mapping correction and assess the resulting effect on the mechanical endpoints under a separately authorized numerical plan. No subjects were silently excluded and no primary result was changed.
2. **Study-specific data/code release:** source imaging has a verified public record, but this study's derived data, model outputs and code have no confirmed public deposit. Decide on the release/access terms before submission; the current statement accurately reports the present status.
3. **Version identity:** confirm that the current source corresponds to the locally named `main(5)` copy, if that name refers to a separate version not supplied here.
4. Review the revised PDF and DOCX, including the newly explicit pretension-reference convention. DOCX pagination may differ from the compiled PDF.

## Implementation provenance

- `simulation/simulation_core.py`, `_create_function_spaces`: P1 vector and DG0 spaces.
- `simulation/febio_parser.py`, `_create_dolfinx_mesh`: linear tetrahedral geometry; the saved VTK connectivity was independently checked to contain only VTK linear tetrahedra.
- `simulation/solver_base.py`, `_build_forms` and `_create_ksp_solver`: signed J, inverse mapping, degree-6 integration, PREONLY/LU/MUMPS and configured tolerances.
- `results/ref_S1P_fixed_new2/simulation_metadata.json` and the two variant metadata files: mesh dimensions, RBF parameters, tolerance values, model variant configuration.
- `simulation/data_mapper.py`, `calculate_bone_modulus`, `MaterialMapper.apply_sample` and `_reference_density`: threshold/floor before modulus interpolation; pointwise median reference density.
- `process_allometry_data.py`: selected bodies `[0, 1, 2]`, labels LI/RI/S; geometric volumes converted from mm³ to cm³ and summed.
