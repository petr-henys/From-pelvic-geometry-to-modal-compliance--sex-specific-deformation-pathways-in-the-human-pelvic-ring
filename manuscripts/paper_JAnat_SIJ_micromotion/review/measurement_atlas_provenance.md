# Measurement atlas provenance

The source landmarks were recovered from `anatomy_data/palpace/`, and the original calculation from `generator/morphology_michal.py` (ignored by the default repository search). The previous statement that landmark definitions could not be found is superseded.

| Archived variable | Source markup in anatomy_data/palpace |
|---|---|
| AP | AP.mrk.json |
| BiacetabularWidth | biacetabular.mrk.json |
| BiischiadicWidth | biischiadic.mrk.json |
| BituberousWidth | bituberous.mrk.json |
| IliopectinealEminenceWidth | iliopectineal_eminence.mrk.json |
| PIT | PIT.mrk.json |
| SacralWidth | sacral_width.mrk.json |
| SubpubicAngle | subpubic.mrk.json |

The generator uses the stored LPS coordinates without a coordinate conversion, matching the original workflow and template geometry. The seven lengths are 3D Euclidean distances; the subpubic angle is the angle p1–p2–p3 with vertex p2. Each view is orthographic and contains the corresponding measurement segment; the angle view is perpendicular to the plane of its arms. Landmarks are overlaid to remain visible through bone. Values in the figure are template values, not population means. The reference pelvis rendering excludes the lumbar bodies.

All 278 geometries in data/X.npy were used with the original thin-plate-spline interpolation (10 neighbors, smoothing=100). All eight resulting variables were compared against data/dimensions_michal.zarr. This checks 2,224 values; maximum absolute discrepancy: 1.99e-13 mm or degrees. This is computational reproduction, not independent anatomical validation of the originally selected landmarks.

Supplementary Figure S2 distinguishes isolated schematic positive rotations from the combined Cardan decomposition, illustrates the common-point displacement definition, and identifies the three bodies used for geometric volume and area, matching BODY_INDICES=[0,1,2] in process_allometry_data.py.

The main and supplementary PDFs were rebuilt successfully, with no undefined references or overfull boxes. Both new figures were visually checked as standalone images and in their final PDF pages. No cohort statistics were modified.
