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
