"""Reference-first computation (use --cohort for the legacy population analysis).

Cohort-wide computation of kinematic deformation mechanisms for all 15 eigenmodes.

Vectorized and optimized for high throughput across all 278 cohort subjects:
1. Level A: Exact 6-component strain tensor fractions across whole pelvis,
   individual bone bodies (Sacrum, Left/Right Innominate), cartilages (Pubic Symphysis,
   Left/Right SIJ), and branches using sparse matrix-vector products.
2. Level B: Regional orthogonal mechanism decomposition (axial, bending, transverse shear,
   circulatory twist, and residual) on elongated branches (superior/inferior pubic rami,
   iliac crests).
3. 1D Ligament bundle kinematics (axial elongation/strain and transverse slip) across 11 bundles.
4. Invariant aggregate profiles for near-degenerate subspaces (Modes 1-3, 5-6, 9-10, 12-15)
   via exact quadratic energy summation.
5. Population summaries in both solver order and reference-paired order.
6. Sensitivity analysis: N=54 (100% clean) vs N=278 (admissible integration domain).

All outputs are written to analysis_outputs/mode_mechanisms/.
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import dolfinx
import numpy as np
import pandas as pd
import pyvista as pv
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.spatial import KDTree
import zarr

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from analysis.spectral_config import OUT_DIR, REPO_ROOT
from analysis.spectral_data import (
    load_metadata,
    load_reference_space,
    load_template_and_shapes,
)
from analysis.mode_mechanism_metrics import (
    RegionalBeamModel,
)
from simulation.data_mapper import ShapeMapper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = OUT_DIR / "mode_mechanisms"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def build_anatomical_regions(parser, mesh, cells, cell_centroids):
    """Build masks, sparse incidence matrix, and beam models for anatomical regions."""
    num_cells = len(cells)
    mat_labels = np.asarray(parser.material_labels, dtype=str)

    # 1. Tissues
    bone_mask = (mat_labels == "PelvisBone")
    sij_l_mask = (mat_labels == "SIJCartilageLeft")
    sij_r_mask = (mat_labels == "SIJCartilageRight")
    sym_mask = (mat_labels == "PubicSymphysis")

    # 2. Bone connected components (Sacrum, Left Innominate, Right Innominate)
    bone_idx = np.where(bone_mask)[0]
    bone_cells = cells[bone_idx]
    num_nodes = mesh.geometry.x.shape[0]

    rows = np.repeat(np.arange(len(bone_cells)), 4)
    cols = bone_cells.ravel()
    data = np.ones(len(rows), dtype=np.int32)
    A = sp.csr_matrix((data, (rows, cols)), shape=(len(bone_cells), num_nodes))
    Face_adj = (A @ A.T) >= 3
    _, comp_labels = connected_components(Face_adj, directed=False)

    sacrum_mask = np.zeros(num_cells, dtype=bool)
    innom_r_mask = np.zeros(num_cells, dtype=bool)
    innom_l_mask = np.zeros(num_cells, dtype=bool)

    # Component 0: Sacrum (X ~ 29, Y ~ -70)
    # Component 1: Right Innominate (X > 30)
    # Component 2: Left Innominate (X < 30)
    sacrum_mask[bone_idx[comp_labels == 0]] = True
    innom_r_mask[bone_idx[comp_labels == 1]] = True
    innom_l_mask[bone_idx[comp_labels == 2]] = True

    # 3. Elongated branches
    # Superior Pubic Rami
    spr_r_mask = bone_mask & (cell_centroids[:, 0] > 35) & (cell_centroids[:, 0] < 70) & (cell_centroids[:, 1] < -155) & (cell_centroids[:, 2] > -1415)
    spr_l_mask = bone_mask & (cell_centroids[:, 0] > -10) & (cell_centroids[:, 0] < 25) & (cell_centroids[:, 1] < -155) & (cell_centroids[:, 2] > -1415)

    # Inferior Pubic Rami
    ipr_r_mask = bone_mask & (cell_centroids[:, 0] > 35) & (cell_centroids[:, 0] < 85) & (cell_centroids[:, 1] > -155) & (cell_centroids[:, 1] < -105) & (cell_centroids[:, 2] < -1415)
    ipr_l_mask = bone_mask & (cell_centroids[:, 0] > -25) & (cell_centroids[:, 0] < 25) & (cell_centroids[:, 1] > -155) & (cell_centroids[:, 1] < -105) & (cell_centroids[:, 2] < -1415)

    # Iliac Crests
    crest_r_mask = bone_mask & (cell_centroids[:, 0] > 70) & (cell_centroids[:, 2] > -1270)
    crest_l_mask = bone_mask & (cell_centroids[:, 0] < -10) & (cell_centroids[:, 2] > -1270)

    region_names = [
        "WholePelvis",
        "PelvisBone",
        "Sacrum",
        "InnominateLeft",
        "InnominateRight",
        "PubicSymphysis",
        "SIJCartilageLeft",
        "SIJCartilageRight",
        "SuperiorPubicRamusLeft",
        "SuperiorPubicRamusRight",
        "InferiorPubicRamusLeft",
        "InferiorPubicRamusRight",
        "IliacCrestLeft",
        "IliacCrestRight",
    ]

    region_masks = {
        "WholePelvis": np.ones(num_cells, dtype=bool),
        "PelvisBone": bone_mask,
        "Sacrum": sacrum_mask,
        "InnominateLeft": innom_l_mask,
        "InnominateRight": innom_r_mask,
        "PubicSymphysis": sym_mask,
        "SIJCartilageLeft": sij_l_mask,
        "SIJCartilageRight": sij_r_mask,
        "SuperiorPubicRamusLeft": spr_l_mask,
        "SuperiorPubicRamusRight": spr_r_mask,
        "InferiorPubicRamusLeft": ipr_l_mask,
        "InferiorPubicRamusRight": ipr_r_mask,
        "IliacCrestLeft": crest_l_mask,
        "IliacCrestRight": crest_r_mask,
    }

    # Sparse incidence matrix R_mat (n_regions, n_cells)
    stacked_masks = np.vstack([region_masks[name] for name in region_names]).astype(float)
    R_mat = sp.csr_matrix(stacked_masks)

    # Beam models for the 6 branches
    beam_models = {}
    beam_regions = [
        ("SuperiorPubicRamusLeft", spr_l_mask),
        ("SuperiorPubicRamusRight", spr_r_mask),
        ("InferiorPubicRamusLeft", ipr_l_mask),
        ("InferiorPubicRamusRight", ipr_r_mask),
        ("IliacCrestLeft", crest_l_mask),
        ("IliacCrestRight", crest_r_mask),
    ]

    for name, mask in beam_regions:
        c_idx = np.where(mask)[0]
        centroids = cell_centroids[c_idx]
        cov = np.cov(centroids.T)
        _, eigvecs = np.linalg.eigh(cov)
        long_axis = eigvecs[:, 2]
        beam_models[name] = RegionalBeamModel(
            name=name,
            cell_indices=c_idx,
            centroids=centroids,
            e_long=long_axis,
        )

    return region_names, region_masks, R_mat, beam_models


def run_cohort_mechanisms_pipeline(max_subjects: int | None = None):
    """Execute complete mode mechanism analysis on cohort."""
    logger.info("Initializing reference finite-element mesh and geometry...")
    parser, V = load_reference_space()
    mesh = parser.mesh_dolfinx
    num_cells = mesh.topology.index_map(3).size_local
    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim, 0)
    c2v = mesh.topology.connectivity(tdim, 0)
    cells = np.array([c2v.links(c) for c in range(num_cells)], dtype=int)

    coords_ref = mesh.geometry.x.copy()
    v0 = coords_ref[cells[:, 0]]
    v1 = coords_ref[cells[:, 1]]
    v2 = coords_ref[cells[:, 2]]
    v3 = coords_ref[cells[:, 3]]
    vols_ref = np.einsum("ij,ij->i", np.cross(v1 - v0, v2 - v0), v3 - v0) / 6.0
    cell_centroids = (v0 + v1 + v2 + v3) / 4.0

    X_mat = np.stack([v1 - v0, v2 - v0, v3 - v0], axis=-1)
    X_inv = np.linalg.inv(X_mat)

    # Build anatomical regions, sparse incidence matrix, and beam models
    region_names, region_masks, R_mat, beam_models = build_anatomical_regions(parser, mesh, cells, cell_centroids)
    n_regions = len(region_names)
    logger.info("Defined %d anatomical regions and %d beam models", n_regions, len(beam_models))

    # Build exact FEBio -> DolfinX node mapping for ligaments
    feb_nodes = np.vstack(list(parser.nodes.values()))
    node_tree = KDTree(coords_ref)
    _, f2d = node_tree.query(feb_nodes)
    logger.info("Built FEBio to DolfinX node mapping (f2d) for %d nodes", len(f2d))

    discrete_sets = parser.discrete_sets
    logger.info("Loaded %d discrete ligament sets (%s)", len(discrete_sets), ", ".join(discrete_sets.keys()))

    # Load templates, shapes, and metadata
    tpl_pts, shapes = load_template_and_shapes()
    n_subjects, _, _ = shapes.shape
    if max_subjects is not None:
        n_subjects = min(n_subjects, max_subjects)
    sex_arr, age_arr = load_metadata()

    # Load Zarr results
    data_dir = REPO_ROOT / "results" / "ref_S1P_fixed_new2" / "data"
    ev_store = zarr.open_group(data_dir / "eigenvectors.zarr", mode="r")["data"]
    eval_store = zarr.open_group(data_dir / "eigenvalues.zarr", mode="r")["data"]
    perm_store = zarr.open_group(data_dir / "eig_permutations.zarr", mode="r")["data"]

    # Setup ShapeMapper
    phi = dolfinx.fem.Function(V, name="phi")
    tpl_pv = pv.PolyData(tpl_pts)
    rbf_config = {"RBF_DATA_SMOOTHING": 50.0, "RBF_DATA_NEIGHBORS": 10}
    mapper = ShapeMapper(phi, tpl_pv, shapes, rbf_config)

    # Subspaces of interest (0-based mode indices):
    subspace_defs = {
        "Subspace_1_3": [0, 1, 2],
        "Subspace_5_6": [4, 5],
        "Subspace_9_10": [8, 9],
        "Subspace_12_15": [11, 12, 13, 14],
    }

    detailed_records = []
    subspace_records = []
    ligament_records = []

    logger.info("Processing %d subjects across 15 modes...", n_subjects)
    t_pipeline_start = time.perf_counter()

    for s_idx in range(n_subjects):
        t0 = time.perf_counter()
        sex = str(sex_arr[s_idx]) if sex_arr is not None else "unknown"
        age = float(age_arr[s_idx]) if age_arr is not None else np.nan

        # 1. Morphological mapping
        mapper.apply_sample(s_idx)
        phi_arr = phi.x.array.reshape(-1, 3)
        coords_phys = coords_ref + phi_arr

        vd0 = coords_phys[cells[:, 0]]
        vd1 = coords_phys[cells[:, 1]]
        vd2 = coords_phys[cells[:, 2]]
        vd3 = coords_phys[cells[:, 3]]
        vols_phys = np.einsum("ij,ij->i", np.cross(vd1 - vd0, vd2 - vd0), vd3 - vd0) / 6.0
        J_cells = vols_phys / vols_ref

        # Admissible weights (strictly exclude J <= 0)
        admissible_mask = (J_cells > 0.0)
        weights = J_cells * vols_ref
        weights[~admissible_mask] = 0.0

        # Deformation gradient F
        x_mat = np.stack([vd1 - vd0, vd2 - vd0, vd3 - vd0], axis=-1)
        F_cells = x_mat @ X_inv
        F_inv = np.linalg.inv(F_cells)

        # Permutation array for subject
        perm = np.asarray(perm_store[s_idx], dtype=int)
        solver_to_ref = np.full(15, -1, dtype=int)
        for r_m, s_m in enumerate(perm[:15]):
            if 0 <= s_m < 15:
                solver_to_ref[s_m] = r_m

        evals_subj = eval_store[s_idx]

        # Per-mode regional energy arrays for rapid subspace aggregation
        # shapes: (15, n_regions)
        mode_D11 = np.zeros((15, n_regions))
        mode_D22 = np.zeros((15, n_regions))
        mode_D33 = np.zeros((15, n_regions))
        mode_D12 = np.zeros((15, n_regions))
        mode_D13 = np.zeros((15, n_regions))
        mode_D23 = np.zeros((15, n_regions))
        mode_Dvol = np.zeros((15, n_regions))
        mode_Dtot = np.zeros((15, n_regions))

        for m_idx in range(15):
            u_mode = ev_store[s_idx, m_idx]
            u_mat = np.stack([
                u_mode[cells[:, 1]] - u_mode[cells[:, 0]],
                u_mode[cells[:, 2]] - u_mode[cells[:, 0]],
                u_mode[cells[:, 3]] - u_mode[cells[:, 0]],
            ], axis=-1)
            grad_u_ref = u_mat @ X_inv
            grad_u = grad_u_ref @ F_inv
            eps_m = 0.5 * (grad_u + np.swapaxes(grad_u, -1, -2))

            # Strain components
            e11 = eps_m[:, 0, 0]
            e22 = eps_m[:, 1, 1]
            e33 = eps_m[:, 2, 2]
            e12 = eps_m[:, 0, 1]
            e13 = eps_m[:, 0, 2]
            e23 = eps_m[:, 1, 2]
            tr_eps = e11 + e22 + e33

            # Volume-weighted quadratic components
            w_e11 = weights * (e11 ** 2)
            w_e22 = weights * (e22 ** 2)
            w_e33 = weights * (e33 ** 2)
            w_e12 = 2.0 * weights * (e12 ** 2)
            w_e13 = 2.0 * weights * (e13 ** 2)
            w_e23 = 2.0 * weights * (e23 ** 2)
            w_vol = weights * (tr_eps ** 2) / 3.0

            # Rapid sparse multiplication across all 14 regions
            d11_all = R_mat @ w_e11
            d22_all = R_mat @ w_e22
            d33_all = R_mat @ w_e33
            d12_all = R_mat @ w_e12
            d13_all = R_mat @ w_e13
            d23_all = R_mat @ w_e23
            dvol_all = R_mat @ w_vol
            dtot_all = d11_all + d22_all + d33_all + d12_all + d13_all + d23_all
            ddev_all = dtot_all - dvol_all

            mode_D11[m_idx] = d11_all
            mode_D22[m_idx] = d22_all
            mode_D33[m_idx] = d33_all
            mode_D12[m_idx] = d12_all
            mode_D13[m_idx] = d13_all
            mode_D23[m_idx] = d23_all
            mode_Dvol[m_idx] = dvol_all
            mode_Dtot[m_idx] = dtot_all

            D_pelvis_total = dtot_all[0]  # WholePelvis is index 0
            mode_solver = m_idx + 1
            mode_ref = int(solver_to_ref[m_idx] + 1) if solver_to_ref[m_idx] >= 0 else -1
            eval_val = float(evals_subj[m_idx])

            # Populate detailed region records
            for r_idx, reg_name in enumerate(region_names):
                D_reg = dtot_all[r_idx]
                if D_reg > 1e-15:
                    f_ML = d11_all[r_idx] / D_reg
                    f_AP = d22_all[r_idx] / D_reg
                    f_CC = d33_all[r_idx] / D_reg
                    f_ML_AP = d12_all[r_idx] / D_reg
                    f_ML_CC = d13_all[r_idx] / D_reg
                    f_AP_CC = d23_all[r_idx] / D_reg
                    f_vol = dvol_all[r_idx] / D_reg
                    f_dev = ddev_all[r_idx] / D_reg
                else:
                    f_ML = f_AP = f_CC = f_ML_AP = f_ML_CC = f_AP_CC = f_vol = f_dev = 0.0

                coverage = D_reg / D_pelvis_total if D_pelvis_total > 1e-15 else 0.0

                row = {
                    "subject_idx": s_idx,
                    "sex": sex,
                    "age": age,
                    "mode_solver": mode_solver,
                    "mode_ref": mode_ref,
                    "eigenvalue": eval_val,
                    "region": reg_name,
                    "D_region": D_reg,
                    "coverage": coverage,
                    "f_ML": f_ML,
                    "f_AP": f_AP,
                    "f_CC": f_CC,
                    "f_ML_AP": f_ML_AP,
                    "f_ML_CC": f_ML_CC,
                    "f_AP_CC": f_AP_CC,
                    "f_volumetric": f_vol,
                    "f_deviatoric": f_dev,
                    "f_axial": np.nan,
                    "f_bending": np.nan,
                    "f_shear": np.nan,
                    "f_twist": np.nan,
                    "f_residual": np.nan,
                }

                # Level B beam decomposition on the 6 branches
                if reg_name in beam_models:
                    beam_res = beam_models[reg_name].decompose(eps_m, weights)
                    if beam_res["is_valid"]:
                        row["f_axial"] = beam_res["f_axial"]
                        row["f_bending"] = beam_res["f_bending"]
                        row["f_shear"] = beam_res["f_shear"]
                        row["f_twist"] = beam_res["f_twist"]
                        row["f_residual"] = beam_res["f_residual"]

                detailed_records.append(row)

            # Evaluate 1D Ligament bundles
            for lig_name, pairs in discrete_sets.items():
                p0_idx = f2d[pairs[:, 0] - 1]
                p1_idx = f2d[pairs[:, 1] - 1]

                r_vecs = coords_phys[p1_idx] - coords_phys[p0_idx]
                L_deformed = np.linalg.norm(r_vecs, axis=1)
                e_dir = r_vecs / np.maximum(L_deformed[:, None], 1e-12)

                delta_u = u_mode[p1_idx] - u_mode[p0_idx]
                delta_L = np.sum(delta_u * e_dir, axis=1)
                eps_axial_lig = delta_L / np.maximum(L_deformed, 1e-12)

                delta_u_trans = delta_u - delta_L[:, None] * e_dir
                norm_trans = np.linalg.norm(delta_u_trans, axis=1)

                ligament_records.append({
                    "subject_idx": s_idx,
                    "sex": sex,
                    "mode_solver": mode_solver,
                    "mode_ref": mode_ref,
                    "bundle": lig_name,
                    "num_springs": len(pairs),
                    "mean_length_mm": float(np.mean(L_deformed)),
                    "rms_axial_strain": float(np.sqrt(np.mean(eps_axial_lig ** 2))),
                    "max_abs_axial_strain": float(np.max(np.abs(eps_axial_lig))),
                    "rms_trans_disp_mm": float(np.sqrt(np.mean(norm_trans ** 2))),
                })

        # Evaluate invariant clustered subspaces via direct quadratic summation
        for sub_name, m_indices in subspace_defs.items():
            sub_D11 = np.sum(mode_D11[m_indices], axis=0)
            sub_D22 = np.sum(mode_D22[m_indices], axis=0)
            sub_D33 = np.sum(mode_D33[m_indices], axis=0)
            sub_D12 = np.sum(mode_D12[m_indices], axis=0)
            sub_D13 = np.sum(mode_D13[m_indices], axis=0)
            sub_D23 = np.sum(mode_D23[m_indices], axis=0)
            sub_Dtot = np.sum(mode_Dtot[m_indices], axis=0)

            for r_idx, reg_name in enumerate(region_names):
                D_reg_sub = sub_Dtot[r_idx]
                if D_reg_sub > 1e-15:
                    sub_f_ML = sub_D11[r_idx] / D_reg_sub
                    sub_f_AP = sub_D22[r_idx] / D_reg_sub
                    sub_f_CC = sub_D33[r_idx] / D_reg_sub
                    sub_f_ML_AP = sub_D12[r_idx] / D_reg_sub
                    sub_f_ML_CC = sub_D13[r_idx] / D_reg_sub
                    sub_f_AP_CC = sub_D23[r_idx] / D_reg_sub
                else:
                    sub_f_ML = sub_f_AP = sub_f_CC = sub_f_ML_AP = sub_f_ML_CC = sub_f_AP_CC = 0.0

                subspace_records.append({
                    "subject_idx": s_idx,
                    "sex": sex,
                    "subspace": sub_name,
                    "region": reg_name,
                    "D_total": D_reg_sub,
                    "f_ML": sub_f_ML,
                    "f_AP": sub_f_AP,
                    "f_CC": sub_f_CC,
                    "f_ML_AP": sub_f_ML_AP,
                    "f_ML_CC": sub_f_ML_CC,
                    "f_AP_CC": sub_f_AP_CC,
                })

        if (s_idx + 1) % 25 == 0 or (s_idx + 1) == n_subjects:
            elapsed = time.perf_counter() - t_pipeline_start
            rate = (s_idx + 1) / elapsed
            logger.info("Processed %d/%d subjects (%.1fs elapsed, %.2f subj/s)", s_idx + 1, n_subjects, elapsed, rate)

    # 4. Save and Aggregate
    logger.info("Compiling results into DataFrames...")
    df_detailed = pd.DataFrame(detailed_records)
    df_subspace = pd.DataFrame(subspace_records)
    df_ligaments = pd.DataFrame(ligament_records)

    parquet_path = OUTPUT_DIR / "mode_mechanisms_cohort_detailed.parquet"
    df_detailed.to_parquet(parquet_path, index=False)
    logger.info("Saved detailed records parquet: %s (%d rows)", parquet_path, len(df_detailed))

    df_subspace.to_csv(OUTPUT_DIR / "subspace_mechanisms_cohort.csv", index=False)
    df_ligaments.to_csv(OUTPUT_DIR / "ligament_mechanisms_cohort.csv", index=False)

    def safe_iqr(x):
        v = x.dropna()
        if len(v) == 0:
            return np.nan
        return float(np.percentile(v, 75) - np.percentile(v, 25))

    # 5. Summaries
    metric_cols = ["f_ML", "f_AP", "f_CC", "f_ML_AP", "f_ML_CC", "f_AP_CC", "f_volumetric", "f_deviatoric",
                   "f_axial", "f_bending", "f_shear", "f_twist", "f_residual", "coverage"]

    summary_solver = df_detailed.groupby(["mode_solver", "region"])[metric_cols].agg(
        ["mean", "std", "median", safe_iqr]
    )
    summary_solver.columns = [f"{col}_{stat}" if stat != "safe_iqr" else f"{col}_iqr" for col, stat in summary_solver.columns]
    summary_solver.reset_index().to_csv(OUTPUT_DIR / "mode_mechanisms_summary_solver.csv", index=False)
    logger.info("Saved solver order summary CSV")

    df_ref_valid = df_detailed[df_detailed["mode_ref"] > 0]
    summary_ref = df_ref_valid.groupby(["mode_ref", "region"])[metric_cols].agg(
        ["mean", "std", "median", safe_iqr]
    )
    summary_ref.columns = [f"{col}_{stat}" if stat != "safe_iqr" else f"{col}_iqr" for col, stat in summary_ref.columns]
    summary_ref.reset_index().to_csv(OUTPUT_DIR / "mode_mechanisms_summary_ref.csv", index=False)
    logger.info("Saved reference order summary CSV")

    sub_metric_cols = ["f_ML", "f_AP", "f_CC", "f_ML_AP", "f_ML_CC", "f_AP_CC"]
    summary_subspace = df_subspace.groupby(["subspace", "region"])[sub_metric_cols].agg(
        ["mean", "std", "median", safe_iqr]
    )
    summary_subspace.columns = [f"{col}_{stat}" if stat != "safe_iqr" else f"{col}_iqr" for col, stat in summary_subspace.columns]
    summary_subspace.reset_index().to_csv(OUTPUT_DIR / "subspace_mechanisms_summary.csv", index=False)
    logger.info("Saved subspace summary CSV")

    summary_lig = df_ligaments.groupby(["mode_solver", "bundle"])[["rms_axial_strain", "max_abs_axial_strain", "rms_trans_disp_mm"]].agg(["mean", "std", "median"]).reset_index()
    summary_lig.columns = [f"{col}_{stat}" if stat else col for col, stat in summary_lig.columns]
    summary_lig.to_csv(OUTPUT_DIR / "ligament_mechanisms_summary.csv", index=False)
    logger.info("Saved ligament summary CSV")

    # 6. Sensitivity Analysis: Clean (N=54) vs Full Cohort (N=278)
    audit_csv = OUTPUT_DIR / "geometry_audit.csv"
    if audit_csv.exists():
        audit_df = pd.read_csv(audit_csv)
        clean_subj_ids = set(audit_df.loc[audit_df["is_strictly_valid"], "subject_idx"])
        df_clean = df_detailed[df_detailed["subject_idx"].isin(clean_subj_ids)]

        clean_means = df_clean.groupby(["mode_solver", "region"])[metric_cols].mean()
        full_means = df_detailed.groupby(["mode_solver", "region"])[metric_cols].mean()
        diff_means = (full_means - clean_means).abs()
        max_diff = float(diff_means.max().max())
        logger.info("Sensitivity check: Max absolute difference between N=54 clean and N=%d full cohort across all modes and regions = %.6f (%.4f%%)", n_subjects, max_diff, max_diff * 100)

        diff_summary = pd.DataFrame({
            "metric": metric_cols,
            "mean_abs_diff": [diff_means[m].mean() for m in metric_cols],
            "max_abs_diff": [diff_means[m].max() for m in metric_cols],
        })
        diff_summary.to_csv(OUTPUT_DIR / "sensitivity_clean_vs_cohort.csv", index=False)
        logger.info("Saved sensitivity comparison CSV")

    # 7. Generate Classification Table
    logger.info("Generating publication mode classification table...")
    wp_summary = df_detailed[df_detailed["region"] == "WholePelvis"].groupby("mode_solver")[
        ["f_ML", "f_AP", "f_CC", "f_ML_AP", "f_ML_CC", "f_AP_CC", "f_volumetric", "f_deviatoric"]
    ].mean()

    class_rows = []
    for m in range(1, 16):
        row = wp_summary.loc[m]
        comps = {
            "ML (Normal)": row["f_ML"],
            "AP (Normal)": row["f_AP"],
            "CC (Normal)": row["f_CC"],
            "ML-AP (Shear)": row["f_ML_AP"],
            "ML-CC (Shear)": row["f_ML_CC"],
            "AP-CC (Shear)": row["f_AP_CC"],
        }
        sorted_comps = sorted(comps.items(), key=lambda x: x[1], reverse=True)
        top1, val1 = sorted_comps[0]
        top2, val2 = sorted_comps[1]

        total_shear = row["f_ML_AP"] + row["f_ML_CC"] + row["f_AP_CC"]
        total_normal = row["f_ML"] + row["f_AP"] + row["f_CC"]

        if total_shear > 0.70:
            mech_class = f"Predominant Shear ({top1.split()[0]} + {top2.split()[0]})"
        elif total_normal > 0.60:
            mech_class = f"Predominant Normal ({top1.split()[0]})"
        else:
            mech_class = f"Coupled Normal-Shear ({top1.split()[0]} / {top2.split()[0]})"

        class_rows.append({
            "mode_solver": m,
            "classification": mech_class,
            "top_component_1": f"{top1} ({val1*100:.1f}%)",
            "top_component_2": f"{top2} ({val2*100:.1f}%)",
            "total_normal_pct": f"{total_normal*100:.1f}%",
            "total_shear_pct": f"{total_shear*100:.1f}%",
            "deviatoric_pct": f"{row['f_deviatoric']*100:.1f}%",
        })

    df_class = pd.DataFrame(class_rows)
    df_class.to_csv(OUTPUT_DIR / "mode_classification_table.csv", index=False)
    logger.info("Saved mode classification table: %s", OUTPUT_DIR / "mode_classification_table.csv")

    total_pipeline_time = time.perf_counter() - t_pipeline_start
    logger.info("Mode mechanisms pipeline finished successfully in %.1f seconds!", total_pipeline_time)


if __name__ == "__main__":
    import argparse
    cli = argparse.ArgumentParser(description="Reference mode atlas by default; optional cohort analysis")
    cli.add_argument("--cohort", action="store_true", help="Run the legacy cohort pipeline")
    cli.add_argument("--max-subjects", type=int, help="Limit the optional cohort run")
    args = cli.parse_args()
    if args.max_subjects is not None and not args.cohort:
        cli.error("--max-subjects requires --cohort")
    if args.cohort:
        run_cohort_mechanisms_pipeline(args.max_subjects)
    else:
        from analysis.reference_mode_atlas import main
        main()

