# Reproduction of the anatomical-prediction revision (2 October 2026)

The current manuscript addresses prediction from pelvic size and three anatomical
measurements, followed by a controlled test of scalar load transfer. The 278
subjects are a subset of the earlier 281-subject CT cohort; the author confirmed
this provenance. Source simulations, historical endpoint tables and constitutive
settings were not changed. All new outputs are in `tables/prediction/`.

## Current analysis and build

Run these commands serially from the repository root:

```bash
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/sij_matplotlib
python manuscripts/paper_JAnat_SIJ_micromotion/analysis/predict_anatomical_response.py
python manuscripts/paper_JAnat_SIJ_micromotion/analysis/test_load_mixture_mechanism.py
python manuscripts/paper_JAnat_SIJ_micromotion/analysis/audit_supporting_comparisons.py
python manuscripts/paper_JAnat_SIJ_micromotion/analysis/format_prediction_results.py
python manuscripts/paper_JAnat_SIJ_micromotion/analysis/validate_prediction_revision.py
latexmk -pdf -interaction=nonstopmode -halt-on-error -cd manuscripts/paper_JAnat_SIJ_micromotion/main.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error -cd manuscripts/paper_JAnat_SIJ_micromotion/supplementary/supplement.tex
python manuscripts/paper_JAnat_SIJ_micromotion/analysis/export_docx.py
python manuscripts/paper_JAnat_SIJ_micromotion/analysis/prepare_submission_files.py
python manuscripts/paper_JAnat_SIJ_micromotion/analysis/audit_formal_requirements.py
```

For an editorial-only build, use just the last five commands. All narrative
sections are manually maintained and authoritative. **Do not run
`write_corrected_results.py` or the historical full workflow to build this
revision:** that generator overwrites the revised abstract and Results.

The formal audit records reproducible prose counts, declarations, citation pairs,
Highlights character counts and fonts in both PDFs and their figure assets. It
does not certify live journal rules: the current JTB Guide for Authors could not
be retrieved. See `review/formal_requirements_2026-10-02/kontrola_cs.md` for the
verified publisher policies and remaining journal-specific checks. The manuscript
discloses Codex assistance with prose, analysis and visualization code; the author
confirmed that no other AI tools were used, the funder had no role and both authors
reviewed and approved the text and outputs.

Legacy Type-3 figure fonts were converted to vector outlines without recomputing
analyses. `normalize_figure_fonts.py` retains originals under the formal-review
backup, checks rendering at 1600 pixels and exact decoded bitmap preservation,
and writes the conversion record. It is only needed when legacy PDFs are restored.
New Matplotlib exports specify embedded TrueType fonts. `prepare_submission_files.py`
copies the eight main figures in manuscript order and creates a separate editable
caption sheet; it uses current LaTeX reference numbers.

Prediction uses identical subject folds across loads, outcomes and predictor sets;
fits log outcomes with training-only standardization and arithmetic-scale
smearing; and resamples subjects for paired error comparisons. The triad was
selected retrospectively before this validation, so this is internal validation
of a fixed description, not nested selection or external validation.

The mechanism analysis mixes archived LAB1/LAB2 displacement fields at fixed
stiffness, redistributing each side's 400 N force between the original patches.
Weights sum to one, retaining common pretension once. Joint kinematics are
re-extracted at five force fractions. Endpoint vectors and magnitudes are FE
information diagnostics, not predictions from anatomy alone. No FE solver is
required for this exact linear superposition.

`load_mixture_checkpoints/subject_*.npz` enables extraction to resume. Move these
checkpoints aside and rerun the complete 278-subject extraction if any source
fields, geometry, interpolation settings, ROI or extraction definitions change.
`--limit` produces separate pilot summaries and must not substitute for the full
cohort. The synchronous, read-only `zarr_field_reader.py` accepts only the archived
Zarr-v3 float64/little-endian bytes-plus-zstd layout; it checks metadata, required
chunks, decoded size and finite values. This avoids an asynchronous LocalStore
hang in the current postprocessing runtime.

`validate_prediction_revision.py` independently refits all 4,000 held-out
regressions with statsmodels in raw predictor units, checks the saved metrics and
subject folds, compares both mixture endpoints against all 278 original corrected
archives, and verifies vector/scalar definitions and summary errors. Results are
in `tables/prediction/validation_checks.json`. Paired intervals condition on fitted
validation predictions and do not include refitting or feature-selection
uncertainty. Source alignment retains the original archive row-order limitation
reported in the manuscript.

Tables S9 and Figure S3 now contain the exploratory conditional volume analysis;
its heuristic dimensional reference exponents are supplementary context. Tables
S10--S13 contain prediction benchmarks, density-standardization error tails,
direct site contrasts and volume-adjustment sensitivity. The historical notes
below document earlier revisions and are not the current run order.

---

# Reproduction of the corrected manuscript

## Editorial build (no analysis)

The 2026-09-21 narrative revision uses the existing numerical outputs. To build that version, run only the following serial command from the repository root:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error -cd manuscripts/paper_JAnat_SIJ_micromotion/main.tex
```

The edited `sections/abstract.tex` and `sections/results.tex` are now the authoritative prose. The historical `write_corrected_results.py` predates this restructuring and would overwrite it; do not run that generator to build the revised manuscript. Its numerical outputs have not been recomputed.

## Historical numerical reproduction workflow

The following is an analysis workflow, not an editorial build command. Run only when numerical recomputation is explicitly intended. Original FE displacement archives are inputs and are not overwritten.

```bash
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
python manuscripts/paper_JAnat_SIJ_micromotion/analysis/recompute_sij_reference.py --workers 6
python manuscripts/paper_JAnat_SIJ_micromotion/analysis/check_reference_sensitivity.py
python manuscripts/paper_JAnat_SIJ_micromotion/analysis/run_analysis.py
python manuscripts/paper_JAnat_SIJ_micromotion/analysis/format_publication_tables.py
python manuscripts/paper_JAnat_SIJ_micromotion/analysis/format_corrected_supplement.py
python manuscripts/paper_JAnat_SIJ_micromotion/analysis/write_corrected_results.py
python manuscripts/paper_JAnat_SIJ_micromotion/analysis/generate_publication_figures.py
python manuscripts/paper_JAnat_SIJ_micromotion/analysis/generate_measurement_atlas.py
python -m pytest -q tests/test_sij_reference_configuration.py
latexmk -pdf -interaction=nonstopmode -halt-on-error -cd manuscripts/paper_JAnat_SIJ_micromotion/main.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error -cd manuscripts/paper_JAnat_SIJ_micromotion/supplementary/supplement.tex
```

Extraction resumes from `tables/corrected/subject_*.npz`. These checkpoints must be moved aside and regenerated if source fields, interpolation settings, or extraction definitions change. The `--limit` option is for diagnostics only; the statistical pipeline requires the complete cohort.

`write_corrected_results.py` generates the earlier abstract and Results narrative. After any future numerical recomputation, reconcile changed estimates with the manually revised manuscript rather than overwriting its prose. Discussion, hypotheses and methods remain manually maintained. The pipeline intentionally refuses to substitute the legacy SIJ endpoints for missing corrected data.

`review/environment.json` records the postprocessing environment. The FE runtime (DOLFINx 0.10) is described by the original simulation metadata; it was not rerun during this revision. The broader existing FE-dependent test module could not be collected because mpi4py is absent in the current postprocessing environment. The four added portable mathematical/extraction regression tests passed.

Auxiliary archives share the indexed shape/density inputs and sample loop with the full archive but do not carry independent subject-ID manifests. Pairing is therefore based on the archived row order and common inputs; see `review/subject_alignment.json`.

The measurement atlas reads the original `anatomy_data/palpace/*.mrk.json` landmarks and reproduces all 2,224 archived morphometric values using the original 10-neighbor, smoothing-100 interpolation. The check is stored in `review/measurement_landmark_validation.json`. It does not overwrite measurements.

## Second conceptual revision: saved-estimate Figure 7 only

For annotation-only reproduction, run from this manuscript directory:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -c "from analysis.generate_publication_figures import build_scaling_benchmarks; build_scaling_benchmarks()"
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

This function reads only `tables/generated/allometry_models.csv` and writes the historical `Fig6_allometry_loglog` PDF/PNG assets (Figure 7 in manuscript numbering). Do not invoke the full generator or `build_figure_6` for this task: its supplementary-scatter path fits models. No numerical regeneration is needed for the second conceptual revision. `write_corrected_results.py` is a historical numerical-text generator and must not overwrite the revised Results narrative/caption.

## Submission revision (23 September 2026)

The current prose and primary statistics remain authoritative. Do not run the historical text generator or rerun the FE cohort to reproduce editorial changes.

The lightweight revision diagnostics run serially from the manuscript directory:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python analysis/check_mapping_quality.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python analysis/check_age_function.py
python analysis/format_revision_tables.py
```

The mapping script has a five-minute time cap. Its actual result includes negative Jacobians; see `review/submission_revision.md`. The three-subject extraction pilot (`analysis/submission_diagnostics.py --limit 3 --max-seconds 120`) is optional and is not the source for manuscript Table S6. Do not automatically extend it to the full cohort.

Figure-label-only regeneration:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -c "import pandas as pd; from analysis.generate_publication_figures import build_figure_1, build_figure_5, TABLE_DIR; build_figure_1(); build_figure_5(pd.read_csv(TABLE_DIR/'variance_channels.csv'))"
```

Then build the manuscript and Supporting Information serially and export DOCX:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error -cd supplementary/supplement.tex
python analysis/export_docx.py
```

If an editor is already compiling the same files, use distinct `-outdir` directories and pass their `.aux` paths with `--main-aux` and `--supp-aux` to the DOCX exporter. This avoids concurrent writers to LaTeX auxiliary files. The exporter requires `pandoc` and `pdftoppm`; equations remain native Word mathematics and figure PDFs are embedded as high-resolution PNGs.
