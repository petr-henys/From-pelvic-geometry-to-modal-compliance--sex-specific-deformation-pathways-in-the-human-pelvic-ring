# Data Dictionary

## Source files
- `results/ref_S1P_fixed_new2/data/demography.xlsx`: patient-level age and sex metadata.
- `results/ref_S1P_fixed_new2/data/anatomy_data.xlsx`: morphology sheet (8 pelvic dimensions) and SIJ kinematics sheets for 5 load cases x 2 sides.
- `results/ref_S1P_fixed_new2/data/allometry.xlsx`: true-size metrics (scale, surface, volume, mass).
- `results/ref_S1P_fixed_new2[_shape_only|_material_only]/data/sij_angles_*.zarr`, `sij_trans_*.zarr`: variance-channel decomposition inputs.

## Cohort
- Subjects: 278
- Sex counts: F=150, M=128
- Age range: 16-91 years

## Load cases
- SP2leg: bilateral standing load
- SP1leg: unilateral standing load
- LAB_phase1, LAB_phase2, LAB_phase3: parturition-motivated loading proxies

## SIJ metrics (per subject x load)
- `rot_mag_deg`: mean left-right 3D rotation magnitude (deg)
- `trans_mag_mm`: mean left-right 3D translation magnitude (mm)
- `nut_abs_deg`, `ap_rot_abs_deg`, `cc_rot_abs_deg`: absolute rotational components (deg)
- `ap_trans_abs_mm`, `ml_trans_abs_mm`, `cc_trans_abs_mm`: absolute translation components (mm)
- `lr_rot_asym_deg`, `lr_trans_asym_mm`: left-right asymmetry magnitudes

## Size and morphometric covariates
- `total_volume` (cm^3), `total_surface` (cm^2), `scale` (-) from allometry workbook
- `log_total_volume`, `log_scale`: log-transformed true-size proxies
- Morphology dimensions (mm unless angle): AP, BiacetabularWidth, BiischiadicWidth, BituberousWidth, IliopectinealEminenceWidth, PIT, SacralWidth, SubpubicAngle (deg)

## Missingness
- `age`: 0 missing values
- `sex`: 0 missing values
- `AP`: 0 missing values
- `BiacetabularWidth`: 0 missing values
- `BiischiadicWidth`: 0 missing values
- `BituberousWidth`: 0 missing values
- `IliopectinealEminenceWidth`: 0 missing values
- `PIT`: 0 missing values
- `SacralWidth`: 0 missing values
- `SubpubicAngle`: 0 missing values
- `scale`: 0 missing values
- `total_volume`: 0 missing values
- `rot_mag_deg`: 0 missing values
- `trans_mag_mm`: 0 missing values

## Variance-channel metrics
- `shape_over_full_pct`: variance ratio (%) using shape-only model relative to full model
- `material_over_full_pct`: variance ratio (%) using material-only model relative to full model
- 95% CIs estimated by bootstrap resampling across subjects
