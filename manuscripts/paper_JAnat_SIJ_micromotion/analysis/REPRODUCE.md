# Reproduction of the corrected manuscript

Run from the repository root. Original FE displacement archives are inputs and are not overwritten.

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

`write_corrected_results.py` regenerates the abstract and numerical Results section; edit that script when changing generated prose. Discussion, hypotheses and methods remain manually maintained. The pipeline intentionally refuses to substitute the legacy SIJ endpoints for missing corrected data.

`review/environment.json` records the postprocessing environment. The FE runtime (DOLFINx 0.10) is described by the original simulation metadata; it was not rerun during this revision. The broader existing FE-dependent test module could not be collected because mpi4py is absent in the current postprocessing environment. The four added portable mathematical/extraction regression tests passed.

Auxiliary archives share the indexed shape/density inputs and sample loop with the full archive but do not carry independent subject-ID manifests. Pairing is therefore based on the archived row order and common inputs; see `review/subject_alignment.json`.

The measurement atlas reads the original `anatomy_data/palpace/*.mrk.json` landmarks and reproduces all 2,224 archived morphometric values using the original 10-neighbor, smoothing-100 interpolation. The check is stored in `review/measurement_landmark_validation.json`. It does not overwrite measurements.
