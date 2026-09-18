# Whole-pelvis reference mode atlas

Run `.conda/bin/python analysis/compute_mode_mechanisms.py` from the repository root.
The default analyses the stored reference solution only. The legacy regional
cohort analysis requires `--cohort` (optionally `--max-subjects N`).

Current outputs:
- `analysis_outputs/plos_revision/figures/reference_mode_atlas.pdf` and `.png`:
  15 whole-pelvis modes, anterior and cranial views, stacked deformation profile per mode.
- `analysis_outputs/mode_mechanisms/reference/global_deformation_profiles.csv`:
  axial, bending, shear, torsion and residual shape fractions across all modes.
- `global_deformation_provenance.json` in the same directory: source and definitions.

### Rationale and Methodology
1. **Limitation of local strain fitting**: Fitting simple kinematic beam patterns directly to
   local strain fields leaves 97–99% of the strain norm in the residual across all eigenmodes.
   This occurs because compliance in the pelvic ring is heavily concentrated in the cartilaginous
   joints (sacroiliac joints and pubic symphysis), which drowns out global ring deformation and
   prevents effective mode differentiation.
2. **Global 3D shape change without 1D axes**: Forcing a single 1D beam axis (such as ML, AP,
   or CC) over a closed 3D pelvic ring is unphysical and leads to artificial residuals (e.g. 84%
   unexplained in Mode 2). We therefore evaluate the complete continuous shape change in 3D
   Cartesian space without imposing any 1D reference axis or line.
3. **Removal of rigid motion**: Infinitesimal rigid-body translation (3 DOF) and rotation (3 DOF)
   are mathematically projected out upfront via reduced QR decomposition of the rigid design matrix.
   Rigid rotation is explicitly eliminated and is **not** one of the reported deformation categories.
4. **Full 3D Canonical deformation pattern dictionary**: The nonrigid shape change is evaluated
   against canonical 3D polynomial deformation families:
   - **Axial** (normal stretching and stretch gradients along all coordinate axes)
   - **Bending** (6 pure Euler-Bernoulli flexural modes with zero shear strain)
   - **Shear** (uniform transverse shears and transverse shear gradients)
   - **Torsion** (Saint-Venant twist gradients with zero normal strain)
5. **Shapley allocation**: Subspace overlap on irregular pelvic geometry is partitioned using Shapley
   value decomposition across the 4 families, ensuring fair, order-independent distribution of the
   explained nonrigid squared norm. Fractions and residual strictly sum to 100%.

Validation: `tests/test_global_deformation_patterns.py` validates rigid-motion elimination,
backward recovery of pure synthetic deformations, exact 100% orthogonality on symmetric domains,
overlap closure, and scale/sign invariance.
