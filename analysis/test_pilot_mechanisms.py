"""Pilot evaluation of Level A and Level B mechanism metrics on Subject 0."""
from pathlib import Path
import sys
import numpy as np
import pyvista as pv
import zarr
import dolfinx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.spectral_data import load_reference_space, load_template_and_shapes
from analysis.mode_mechanism_metrics import compute_level_a_profile, RegionalBeamModel
from simulation.data_mapper import ShapeMapper

parser, V = load_reference_space()
mesh = parser.mesh_dolfinx
num_cells = mesh.topology.index_map(3).size_local
tdim = mesh.topology.dim
mesh.topology.create_connectivity(tdim, 0)
c2v = mesh.topology.connectivity(tdim, 0)
cells = np.array([c2v.links(c) for c in range(num_cells)], dtype=int)
mat_labels = np.asarray(parser.material_labels, dtype=str)

coords_ref = mesh.geometry.x.copy()
v0, v1, v2, v3 = coords_ref[cells[:,0]], coords_ref[cells[:,1]], coords_ref[cells[:,2]], coords_ref[cells[:,3]]
vols_ref = np.einsum("ij,ij->i", np.cross(v1 - v0, v2 - v0), v3 - v0) / 6.0
cell_centroids = (v0 + v1 + v2 + v3) / 4.0

# Load Subject 0 eigenvectors
data_dir = Path("results/ref_S1P_fixed_new2/data")
ev_store = zarr.open_group(data_dir / "eigenvectors.zarr", mode="r")["data"]
u0 = ev_store[0, 0]  # mode 1 of subj 0

# Compute mapping using ShapeMapper
tpl_pts, shapes = load_template_and_shapes()
phi = dolfinx.fem.Function(V, name="phi")
tpl_pv = pv.PolyData(tpl_pts)
config = {"RBF_DATA_SMOOTHING": 50.0, "RBF_DATA_NEIGHBORS": 10}
mapper = ShapeMapper(phi, tpl_pv, shapes, config)
mapper.apply_sample(0)
phi_arr = phi.x.array.reshape(-1, 3)

coords_phys = coords_ref + phi_arr
vd0, vd1, vd2, vd3 = coords_phys[cells[:,0]], coords_phys[cells[:,1]], coords_phys[cells[:,2]], coords_phys[cells[:,3]]
vols_phys = np.einsum("ij,ij->i", np.cross(vd1 - vd0, vd2 - vd0), vd3 - vd0) / 6.0
J_cells = vols_phys / vols_ref

# Compute F and strain
X_mat = np.stack([v1 - v0, v2 - v0, v3 - v0], axis=-1)
x_mat = np.stack([vd1 - vd0, vd2 - vd0, vd3 - vd0], axis=-1)
X_inv = np.linalg.inv(X_mat)
F_cells = x_mat @ X_inv
F_inv = np.linalg.inv(F_cells)

u_mat = np.stack([u0[cells[:,1]] - u0[cells[:,0]], u0[cells[:,2]] - u0[cells[:,0]], u0[cells[:,3]] - u0[cells[:,0]]], axis=-1)
grad_u_ref = u_mat @ X_inv
grad_u = grad_u_ref @ F_inv
eps = 0.5 * (grad_u + np.swapaxes(grad_u, -1, -2))

weights = J_cells * vols_ref
weights[J_cells <= 0] = 0.0

# Level A on whole pelvis
prof_all = compute_level_a_profile(eps, weights)
print("Subject 0, Mode 1 Level A (All):")
for k, v in prof_all.items():
    if k.startswith("f_"):
        print(f"  {k}: {v*100:.2f}%")

# Level B on Right SPR
bone_mask = (mat_labels == "PelvisBone")
mask_spr_r = bone_mask & (cell_centroids[:, 0] > 35) & (cell_centroids[:, 0] < 70) & (cell_centroids[:, 1] < -155) & (cell_centroids[:, 2] > -1415)
beam_spr_r = RegionalBeamModel("RightSPR", cell_indices=np.where(mask_spr_r)[0], centroids=cell_centroids[mask_spr_r])
res_spr_r = beam_spr_r.decompose(eps, weights)
print("\nSubject 0, Mode 1 Level B (Right SPR):")
mech_keys = ["f_axial", "f_bending", "f_shear", "f_twist", "f_residual"]
for k in mech_keys:
    print(f"  {k}: {res_spr_r[k]*100:.2f}%")
total_pct = sum(res_spr_r[k] for k in mech_keys) * 100
print(f"  Total sum: {total_pct:.2f}%")

