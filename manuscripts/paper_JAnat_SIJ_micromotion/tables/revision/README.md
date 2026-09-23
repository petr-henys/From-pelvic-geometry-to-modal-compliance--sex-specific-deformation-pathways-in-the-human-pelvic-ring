# Submission diagnostics

These outputs are separate from the unchanged primary analysis in `../generated` and `../corrected`.

- `mapping_quality.csv`: one row per anatomy, exact cellwise P1 mapping determinants. Percentiles refer to cells within that anatomy; `reference_volume_fraction_nonpositive` is a fraction, not a percentage.
- `mapping_nonpositive_cells.csv`: every subject–cell pair with J <= 0; reference cell indices, locations in mm, and volumes in mm³. These are not additional independent subjects.
- `mapping_summary.json`: full-cohort summary and verification method.
- `age_quadratic_sex_models.csv`: HC3 nested models, original vs quadratic-age coefficients. `q` is defined only for the six M2 sex coefficients; outcome units are degrees or mm.
- `age_quadratic_volume_slopes.csv`: 20 sex-specific dimensionless log–log slopes; BH correction across all 20 tests.
- `age_quadratic_summary.json`: measured sensitivity summary.
- `extraction_pilot/`: three-subject timing/feasibility pilot only. It does not replace the published six-subject sensitivity, is not a full-cohort result, and must not be used to generate Table S6.
- `primary_output_hashes_before.json`: fingerprints before editorial relabeling; cohort-table labels are the sole permitted change.

All diagnostics read source geometry/displacement/statistical data without solving FE systems. No primary endpoints or model estimates are overwritten.
