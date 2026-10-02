"""Controlled transverse load redistribution using exact FE superposition.

For unchanged geometry, density, constraints and linearized stiffness,
u(w)=(1-w)u_LAB1+w*u_LAB2 retains the same pretension and splits each 400 N
side force between the original patches. Re-extract kinematics from mixed
displacement fields; never interpolate scalar motion endpoints as FE truth.
All original simulation archives are read-only.
"""
from pathlib import Path
import argparse
import json
import sys
import time
import numpy as np
import pandas as pd
import pyvista as pv
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

PAPER = Path(__file__).resolve().parents[1]
ROOT = PAPER.parents[1]
sys.path.insert(0, str(ROOT))
from anatomy_analyser import (_build_pelvis_frame, _ensure_lr, _sij_roi_masks,
                             _affine_fit, _rigid_from_kabsch_trimmed, _euler_xyz_in_frame)
from simulation.utils import find_symmetry_plane
from rbf_weights import weights
from zarr_field_reader import ArchivedVectorFields

OUT = PAPER / "tables/prediction"
MIX = np.array([0., .25, .5, .75, 1.])


def body_indices(template):
    """Connected tetrahedral components, in descending cell-count order.

    This reproduces split_bodies ordering without repeatedly extracting VTK
    cells. Endpoint comparisons to the original extractor validate its use.
    """
    assert (template.celltypes == 10).all()
    c = template.cells.reshape(-1, 5)[:, 1:]
    graph = coo_matrix((np.ones(len(c)*3, np.int8),
                       (np.repeat(c[:, 0], 3), c[:, 1:].ravel())),
                      shape=(template.n_points, template.n_points)).tocsr()
    n, labels = connected_components(graph, directed=False)
    assert n == 5
    order = np.argsort(-np.bincount(labels[c[:, 0]]), kind="stable")
    return [np.where(labels == r)[0] for r in order[:3]]


class PreparedKinematics:
    def __init__(self, unloaded, bodies):
        self.unloaded = unloaded
        self.left, self.right, self.sac = bodies
        normal, origin = find_symmetry_plane(unloaded)
        frame = _build_pelvis_frame(unloaded, normal, origin)
        self.frame = _ensure_lr(frame, unloaded[self.left], unloaded[self.right], origin)
        self.regions = []
        for name, ili in [("left", self.left), ("right", self.right)]:
            s, i = _sij_roi_masks(unloaded[self.sac], unloaded[ili], self.frame, origin,
                                  name, 5., min_pts=80)
            sidx, iidx = self.sac[s], ili[i]
            self.regions.append((sidx, iidx, unloaded[sidx].mean(axis=0)))

    def extract(self, displacement):
        loaded = self.unloaded+displacement
        a, b = _affine_fit(self.unloaded[self.sac], loaded[self.sac])
        ainv = np.linalg.pinv(a)
        angles, translations = [], []
        for sac, ili, centroid in self.regions:
            q_s = (loaded[sac]-b)@ainv.T
            q_i = (loaded[ili]-b)@ainv.T
            rs, ts = _rigid_from_kabsch_trimmed(self.unloaded[sac], q_s)
            ri, ti = _rigid_from_kabsch_trimmed(self.unloaded[ili], q_i)
            angles.append(_euler_xyz_in_frame(rs.T@ri, self.frame))
            translations.append(self.frame.T@((ri-rs)@centroid+ti-ts))
        return np.asarray(angles), np.asarray(translations)


def run(limit):
    OUT.mkdir(parents=True, exist_ok=True)
    checkpoint = OUT / "load_mixture_checkpoints"
    checkpoint.mkdir(exist_ok=True)
    print("Preparing template and archived mesh", flush=True)
    template = pv.read(ROOT / "data/pelvic.vtk")
    base = ROOT / "results/ref_S1P_fixed_new2"
    cfg = json.loads((base / "simulation_metadata.json").read_text())["config"]
    mesh = pv.read(base / "paraview/reference_fields.vtk")
    print("Checking saved displacement ordering", flush=True)
    reference = np.load(base / "reference_solution.npz")
    for load in ["LAB_phase1", "LAB_phase2"]:
        np.testing.assert_array_equal(mesh["ref_u_"+load], reference["displacement_"+load])
    print("Preparing common body indices and interpolation operators", flush=True)
    bodies = body_indices(template)
    shape_map = weights(template.points, mesh.points, cfg["RBF_DATA_NEIGHBORS"], cfg["RBF_DATA_SMOOTHING"])
    surface_map = weights(mesh.points, template.points, cfg["RBF_SOLVER_NEIGHBORS"], cfg["RBF_SOLVER_SMOOTHING"])
    print("Opening archived subject fields", flush=True)
    shapes = np.load(ROOT / "data/X.npy", mmap_mode="r")
    stores = [ArchivedVectorFields(base / "data" / f"displacements_{l}.zarr")
              for l in ["LAB_phase1", "LAB_phase2"]]
    start = time.monotonic()
    for i in range(limit):
        path = checkpoint / f"subject_{i:03d}.npz"
        if not path.exists():
            phi = shape_map@(shapes[i]-template.points)
            fields = np.concatenate([phi, np.asarray(stores[0][i]), np.asarray(stores[1][i])], axis=1)
            mapped = surface_map@fields
            unloaded = template.points+mapped[:, :3]
            extractor = PreparedKinematics(unloaded, bodies)
            u0, u1 = mapped[:, 3:6], mapped[:, 6:9]
            angles, trans = zip(*(extractor.extract((1-w)*u0+w*u1) for w in MIX))
            angles, trans = np.asarray(angles), np.asarray(trans)
            with np.load(PAPER / "tables/corrected" / f"subject_{i:03d}.npz") as archived:
                for j, load in [(0, "LAB_phase1"), (-1, "LAB_phase2")]:
                    np.testing.assert_allclose(angles[j], archived[f"full_{load}_angles"], atol=1e-8, rtol=1e-8)
                    np.testing.assert_allclose(trans[j], archived[f"full_{load}_trans"], atol=1e-8, rtol=1e-8)
            np.savez_compressed(path, weights=MIX, angles=angles, translations=trans)
        if i == 0 or (i+1) % 20 == 0 or i+1 == limit:
            print(f"Mixture extraction {i+1}/{limit}, {time.monotonic()-start:.1f} s", flush=True)
    rows = []
    for i in range(limit):
        with np.load(checkpoint / f"subject_{i:03d}.npz") as d:
            trans = d["translations"]
            true = np.linalg.norm(trans, axis=2).mean(axis=1)
            scalar = (1-MIX)*true[0]+MIX*true[-1]
            vector = np.linalg.norm((1-MIX[:, None, None])*trans[0]+MIX[:, None, None]*trans[-1], axis=2).mean(axis=1)
            cosine = np.sum(trans[0]*trans[-1], axis=1)/(np.linalg.norm(trans[0], axis=1)*np.linalg.norm(trans[-1], axis=1))
            for j, w in enumerate(MIX):
                rows.append(dict(subject_idx=i, weight=w, translation_mm=true[j],
                                 scalar_oracle_mm=scalar[j], vector_oracle_mm=vector[j],
                                 directional_reduction_pct=100*(1-true[j]/scalar[j]),
                                 mean_endpoint_cosine=cosine.mean(),
                                 rotation_deg=np.linalg.norm(d["angles"][j], axis=1).mean()))
    data = pd.DataFrame(rows)
    prefix = "mixture" if limit == 278 else "mixture_pilot"
    data.to_csv(OUT / f"{prefix}_subject_endpoints.csv", index=False)
    summary = []
    rng = np.random.default_rng(20261002)
    sample = rng.integers(0, limit, size=(3000, limit))
    for w, sub in data.groupby("weight", sort=True):
        observed = sub.translation_mm.to_numpy()
        reduction = sub.directional_reduction_pct.to_numpy()
        boot = np.median(reduction[sample], axis=1)
        summary.append(dict(weight=float(w), median_translation_mm=float(np.median(observed)),
                            median_directional_reduction_pct=float(np.median(reduction)),
                            reduction_ci95_low=float(np.quantile(boot, .025)),
                            reduction_ci95_high=float(np.quantile(boot, .975)),
                            scalar_oracle_rmse_mm=float(np.sqrt(np.mean((sub.scalar_oracle_mm-observed)**2))),
                            vector_oracle_rmse_mm=float(np.sqrt(np.mean((sub.vector_oracle_mm-observed)**2))),
                            vector_oracle_max_error_mm=float(np.max(abs(sub.vector_oracle_mm-observed))),
                            median_endpoint_cosine=float(np.median(sub.mean_endpoint_cosine))))
    pd.DataFrame(summary).to_csv(OUT / f"{prefix}_summary.csv", index=False)
    protocol = dict(n_subjects=limit, weights=MIX.tolist(),
                    intervention="Each bilateral 400 N transverse force is split between LAB1 and LAB2 patches",
                    equilibrium="Exact convex combination of full displacement fields for the fixed linearized stiffness",
                    pretension="Retained once because weights sum to one",
                    extraction="Original affine-corrected, trimmed-Kabsch definitions; same unloaded ROI across weights",
                    validation="Endpoint angles/translations reproduce corrected archives at both weights 0 and 1 to atol/rtol 1e-8 for every subject",
                    oracle="Scalar and vector comparisons both use subject-specific FE endpoint information, not anatomy-only or external predictions",
                    scope="Tests a directional limit under a new combination of represented force patches; does not identify causes of cross-subject CV residuals")
    (OUT / f"{prefix}_protocol.json").write_text(json.dumps(protocol, indent=2))
    print(pd.DataFrame(summary).to_string(index=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=278)
    args = parser.parse_args()
    if not 1 <= args.limit <= 278:
        raise ValueError("limit must be between 1 and 278")
    run(args.limit)
