"""Dataset processing for the eigenstiffness simulation."""
from __future__ import annotations

from typing import TYPE_CHECKING
from dolfinx import fem

import numpy as np
from tqdm import trange


from mode_pairing import pair_modes
from simulation.simulation_analysis import (
    compute_mode_contributions,
    sij_angles_to_array,
    sij_trans_to_array,
)
from simulation.simulation_loads import (
    labor_load_phase_1,
    labor_load_phase_2,
    labor_load_phase_3,
    static_post_load_two_legs,
    static_post_load_one_leg,
)

if TYPE_CHECKING:
    from simulation.simulation_core import EigenstiffnessSimulation


class DatasetProcessor:
    """Handle dataset-level solves and storage."""

    def __init__(self, simulation: "EigenstiffnessSimulation") -> None:
        self.sim = simulation
        self.logger = simulation.logger

    def process(self) -> None:
        sim = self.sim
        if not sim.config["FULL_DATASET_FLAG"]:
            self.logger.info("FULL_DATASET_FLAG is False. Skipping dataset processing.")
            return

        dataset_mode = sim.config.get("DATASET_MODE", "full")
        self.logger.info("Dataset mode: %s", dataset_mode)
        
        num_samples = sim.material_mapper.num_samples()
        num_elastic_modes = len(sim.ref_eigvals)
        nev_requested = num_elastic_modes + 10
        self.logger.info("Processing %d samples, %d modes requested", num_samples, nev_requested)
        
        self._init_storage(num_samples, num_elastic_modes)
        for i in trange(num_samples, desc="Processing Samples"):
            self._process_single_sample(i, nev_requested, num_elastic_modes)

        self.logger.info("Dataset phase done")

    def _init_storage(self, num_samples: int, num_elastic_modes: int) -> None:
        sim = self.sim
        self.logger.debug("Initialising dataset storage (%d samples, %d modes)", num_samples, num_elastic_modes)

        sim.eigvals_ds = sim.storage_manager.add_field_storage(
            field_name="eigenvalues",
            shape=(num_samples, num_elastic_modes),
            dims_description="samples, modes",
            max_chunk_mb=sim.config.get("ZARR_EIGVAL_CHUNK_SIZE_MB", 4),
        )

        sim.eig_permutations_ds = sim.storage_manager.add_field_storage(
            field_name="eig_permutations",
            shape=(num_samples, num_elastic_modes),
            dims_description="samples, modes",
            dtype=np.int32,
            max_chunk_mb=sim.config.get("ZARR_EIGVAL_CHUNK_SIZE_MB", 4),
        )

        nodes = sim.ref_eigvecs.shape[1]
        gdim = sim.ref_eigvecs.shape[2]
        sim.eigvecs_ds = sim.storage_manager.add_field_storage(
            field_name="eigenvectors",
            shape=(num_samples, num_elastic_modes, nodes, gdim),
            dims_description="samples, modes, nodes, spatial_dims",
            max_chunk_mb=sim.config.get("ZARR_EIGVEC_MAX_CHUNK_MB", 16),
        )

        if (
            sim.config.get("COMPUTE_SENSITIVITIES", True)
            and sim.ligament_system is not None
            and sim.ligament_system.num_bundles > 0
        ):
            sim.lig_sens_ds = sim.storage_manager.add_field_storage(
                field_name="lig_sensitivities",
                shape=(num_samples, num_elastic_modes, sim.ligament_system.num_bundles),
                dims_description="samples, modes, ligaments",
                max_chunk_mb=sim.config.get("ZARR_LIG_SENS_MAX_CHUNK_MB", 8),
            )
            sim.lig_sens_ds.attrs.update({"ligament_names": sim.ligament_system.bundle_names})
            
            sim.pretension_sens_ds = sim.storage_manager.add_field_storage(
                field_name="pretension_sensitivities",
                shape=(num_samples, num_elastic_modes, sim.ligament_system.num_bundles),
                dims_description="samples, modes, ligaments",
                max_chunk_mb=sim.config.get("ZARR_PRETENSION_SENS_MAX_CHUNK_MB", 8),
            )
            sim.pretension_sens_ds.attrs.update({"ligament_names": sim.ligament_system.bundle_names})
        else:
            sim.lig_sens_ds = None
            sim.pretension_sens_ds = None

        if sim.config.get("COMPUTE_SENSITIVITIES", True):
            # Bone density-law parameter sensitivities (scale α, exponent β)
            sim.bone_alpha_sens_ds = sim.storage_manager.add_field_storage(
                field_name="bone_alpha_sensitivities",
                shape=(num_samples, num_elastic_modes),
                dims_description="samples, modes",
                max_chunk_mb=sim.config.get("ZARR_BONE_SENS_MAX_CHUNK_MB", 4),
            )
            sim.bone_beta_sens_ds = sim.storage_manager.add_field_storage(
                field_name="bone_beta_sensitivities",
                shape=(num_samples, num_elastic_modes),
                dims_description="samples, modes",
                max_chunk_mb=sim.config.get("ZARR_BONE_SENS_MAX_CHUNK_MB", 4),
            )
        else:
            sim.bone_alpha_sens_ds = None
            sim.bone_beta_sens_ds = None

        if sim.config.get("COMPUTE_SENSITIVITIES", True) and sim.cartilage_param_defs:
            num_cartilage = len(sim.cartilage_param_defs)
            sim.cartilage_sens_ds = sim.storage_manager.add_field_storage(
                field_name="cartilage_sensitivities",
                shape=(num_samples, num_elastic_modes, num_cartilage),
                dims_description="samples, modes, cartilage_parameters",
                max_chunk_mb=sim.config.get("ZARR_CART_SENS_MAX_CHUNK_MB", 4),
            )
            sim.cartilage_sens_ds.attrs.update({
                "parameter_names": [str(param["name"]) for param in sim.cartilage_param_defs],
            })
        else:
            sim.cartilage_sens_ds = None

        if sim.config.get("STATIC_ANAL", True):
            self._init_static_storage(num_samples, nodes, gdim, num_elastic_modes)
        else:
            sim.u_SP2leg_ds = None
            sim.u_SP1leg_ds = None
            sim.u_LAB_phase1_ds = None
            sim.u_LAB_phase2_ds = None
            sim.u_LAB_phase3_ds = None
            for load_name in sim.static_load_case_names:
                setattr(sim, f"strain_{load_name}_ds", None)
                setattr(sim, f"stress_{load_name}_ds", None)
            sim.mode_mac_ds = None
            sim.mode_energy_frac_ds = None
            sim.mode_energy_ds = None
            sim.mode_modal_amp_ds = None
            sim.mode_recon_error_ds = None
            sim.sij_angles_SP2leg_ds = None
            sim.sij_angles_SP1leg_ds = None
            sim.sij_angles_LAB_phase1_ds = None
            sim.sij_angles_LAB_phase2_ds = None
            sim.sij_angles_LAB_phase3_ds = None
            sim.sij_trans_SP2leg_ds = None
            sim.sij_trans_SP1leg_ds = None
            sim.sij_trans_LAB_phase1_ds = None
            sim.sij_trans_LAB_phase2_ds = None
            sim.sij_trans_LAB_phase3_ds = None

        stored_fields = {
            "num_samples": num_samples,
            "num_elastic_modes": num_elastic_modes,
            "reference_eigvec_shape": sim.ref_eigvecs.shape,
            "stored_fields": {
                "eigenvalues": "solver-order eigenvalues (unsorted)",
                "eig_permutations": "per-sample mapping from reference mode index to solver order (for optional alignment)",
                "eigenvectors": "solver-order eigenvectors (unsorted)",
            },
        }
        if sim.config.get("STATIC_ANAL", True):
            stored_fields["stored_fields"].update(
                {
                    "displacements_SP2leg": "static-post (two leg) load displacement field",
                    "displacements_SP1leg": "static-post (single leg) load displacement field",
                    "displacements_LAB_phase1": "labor load phase 1 displacement field",
                    "displacements_LAB_phase2": "labor load phase 2 displacement field",
                    "displacements_LAB_phase3": "labor load phase 3 displacement field",
                    "strain_*": "DG0 full 3x3 strain tensor ε = sym(∇u·F⁻¹) per load case",
                    "stress_*": "DG0 full 3x3 stress tensor σ = λ tr(ε)I + 2μ ε per load case",
                    "mode_mac": "MAC between static load responses and solver-order modes [per load case x mode]",
                    "mode_energy_fraction": "Modal strain energy fractions under static loads [per load case x mode] (solver order)",
                    "mode_energy": "Modal strain energies under static loads [per load case x mode] (solver order)",
                    "mode_modal_amplitude": "Modal coordinates from least-squares projection [per load case x mode] (solver order)",
                    "mode_reconstruction_error": "Relative reconstruction error of static load from modal expansion [per load case]",
                }
            )
            if sim.config.get("SAVE_SIJ_STATS", True):
                stored_fields["stored_fields"].update(
                    {
                        "sij_angles_SP2leg": "SIJ relative angles (deg) under SP two-leg load [L/R x nutation_alpha_x,rotation_AP_beta_y,rotation_CC_gamma_z]",
                        "sij_angles_SP1leg": "SIJ relative angles (deg) under SP single-leg load [L/R x nutation_alpha_x,rotation_AP_beta_y,rotation_CC_gamma_z]",
                        "sij_angles_LAB_phase1": "SIJ relative angles (deg) under labor phase 1 load [L/R x nutation_alpha_x,rotation_AP_beta_y,rotation_CC_gamma_z]",
                        "sij_angles_LAB_phase2": "SIJ relative angles (deg) under labor phase 2 load [L/R x nutation_alpha_x,rotation_AP_beta_y,rotation_CC_gamma_z]",
                        "sij_angles_LAB_phase3": "SIJ relative angles (deg) under labor phase 3 load [L/R x nutation_alpha_x,rotation_AP_beta_y,rotation_CC_gamma_z]",
                        "sij_trans_SP2leg": "SIJ translations (mm) under SP two-leg load [L/R x dx_ML,dy_AP,dz_CC]",
                        "sij_trans_SP1leg": "SIJ translations (mm) under SP single-leg load [L/R x dx_ML,dy_AP,dz_CC]",
                        "sij_trans_LAB_phase1": "SIJ translations (mm) under labor phase 1 load [L/R x dx_ML,dy_AP,dz_CC]",
                        "sij_trans_LAB_phase2": "SIJ translations (mm) under labor phase 2 load [L/R x dx_ML,dy_AP,dz_CC]",
                        "sij_trans_LAB_phase3": "SIJ translations (mm) under labor phase 3 load [L/R x dx_ML,dy_AP,dz_CC]",
                    }
                )
        stored_fields["mode_indices_report_order"] = list(range(1, num_elastic_modes + 1))

        if sim.lig_sens_ds is not None:
            bundle_names = sim.ligament_system.bundle_names if sim.ligament_system else []
            if bundle_names:
                desc = "eigenvalue sensitivity to ligament stiffness (order: " + ", ".join(bundle_names) + ")"
            else:
                desc = "eigenvalue sensitivity to ligament stiffness"
            stored_fields["stored_fields"]["lig_sensitivities"] = desc
        if sim.pretension_sens_ds is not None:
            bundle_names = sim.ligament_system.bundle_names if sim.ligament_system else []
            if bundle_names:
                desc_pretension = "eigenvalue sensitivity to ligament pretension (order: " + ", ".join(bundle_names) + ")"
            else:
                desc_pretension = "eigenvalue sensitivity to ligament pretension"
            stored_fields["stored_fields"]["pretension_sensitivities"] = desc_pretension
        if sim.cartilage_sens_ds is not None:
            names = ", ".join(str(param["name"]) for param in sim.cartilage_param_defs)
            stored_fields["stored_fields"]["cartilage_sensitivities"] = (
                "eigenvalue sensitivity to cartilage moduli (order: " + names + ")"
            )

        # Build base metadata from simulation_core (includes ligament names, etc.)
        base_metadata = sim._build_complete_metadata()
        
        # Merge dataset-specific fields into base metadata
        base_metadata.update(stored_fields)

        sim.storage_manager.save_metadata(base_metadata)

    def _init_static_storage(
        self,
        num_samples: int,
        nodes: int,
        gdim: int,
        num_elastic_modes: int,
    ) -> None:
        sim = self.sim
        num_cells = sim.domain.topology.index_map(sim.domain.topology.dim).size_local

        sim.u_SP2leg_ds = sim.storage_manager.add_field_storage(
            field_name="displacements_SP2leg",
            shape=(num_samples, nodes, gdim),
            dims_description="samples, nodes, spatial_dims",
            max_chunk_mb=sim.config.get("ZARR_DISP_MAX_CHUNK_MB", 8),
        )
        sim.u_SP1leg_ds = sim.storage_manager.add_field_storage(
            field_name="displacements_SP1leg",
            shape=(num_samples, nodes, gdim),
            dims_description="samples, nodes, spatial_dims",
            max_chunk_mb=sim.config.get("ZARR_DISP_MAX_CHUNK_MB", 8),
        )
        sim.u_LAB_phase1_ds = sim.storage_manager.add_field_storage(
            field_name="displacements_LAB_phase1",
            shape=(num_samples, nodes, gdim),
            dims_description="samples, nodes, spatial_dims",
            max_chunk_mb=sim.config.get("ZARR_DISP_MAX_CHUNK_MB", 8),
        )
        sim.u_LAB_phase2_ds = sim.storage_manager.add_field_storage(
            field_name="displacements_LAB_phase2",
            shape=(num_samples, nodes, gdim),
            dims_description="samples, nodes, spatial_dims",
            max_chunk_mb=sim.config.get("ZARR_DISP_MAX_CHUNK_MB", 8),
        )
        sim.u_LAB_phase3_ds = sim.storage_manager.add_field_storage(
            field_name="displacements_LAB_phase3",
            shape=(num_samples, nodes, gdim),
            dims_description="samples, nodes, spatial_dims",
            max_chunk_mb=sim.config.get("ZARR_DISP_MAX_CHUNK_MB", 8),
        )

        # Strain tensors (DG0, full 3x3 per cell)
        for load_name in sim.static_load_case_names:
            ds = sim.storage_manager.add_field_storage(
                field_name=f"strain_{load_name}",
                shape=(num_samples, num_cells, gdim, gdim),
                dims_description="samples, cells, i, j",
                max_chunk_mb=sim.config.get("ZARR_TENSOR_MAX_CHUNK_MB", 16),
            )
            ds.attrs.update({"tensor_type": "strain", "space": "DG0", "representation": "full_3x3"})
            setattr(sim, f"strain_{load_name}_ds", ds)

        # Stress tensors (DG0, full 3x3 per cell)
        for load_name in sim.static_load_case_names:
            ds = sim.storage_manager.add_field_storage(
                field_name=f"stress_{load_name}",
                shape=(num_samples, num_cells, gdim, gdim),
                dims_description="samples, cells, i, j",
                max_chunk_mb=sim.config.get("ZARR_TENSOR_MAX_CHUNK_MB", 16),
            )
            ds.attrs.update({"tensor_type": "stress", "space": "DG0", "representation": "full_3x3"})
            setattr(sim, f"stress_{load_name}_ds", ds)

        num_loads = len(sim.static_load_case_names)
        sim.mode_mac_ds = sim.storage_manager.add_field_storage(
            field_name="mode_mac",
            shape=(num_samples, num_elastic_modes, num_loads),
            dims_description="samples, mode, load_case",
            max_chunk_mb=sim.config.get("ZARR_MODE_MAC_MAX_CHUNK_MB", 4),
        )
        sim.mode_mac_ds.attrs.update({"load_case_names": list(sim.static_load_case_names)})

        sim.mode_energy_frac_ds = sim.storage_manager.add_field_storage(
            field_name="mode_energy_fraction",
            shape=(num_samples, num_elastic_modes, num_loads),
            dims_description="samples, mode, load_case",
            max_chunk_mb=sim.config.get("ZARR_MODE_ENERGY_FRAC_CHUNK_MB", 4),
        )
        sim.mode_energy_frac_ds.attrs.update({"load_case_names": list(sim.static_load_case_names)})

        sim.mode_energy_ds = sim.storage_manager.add_field_storage(
            field_name="mode_energy",
            shape=(num_samples, num_elastic_modes, num_loads),
            dims_description="samples, mode, load_case",
            max_chunk_mb=sim.config.get("ZARR_MODE_ENERGY_CHUNK_MB", 4),
        )
        sim.mode_energy_ds.attrs.update({"load_case_names": list(sim.static_load_case_names)})

        sim.mode_modal_amp_ds = sim.storage_manager.add_field_storage(
            field_name="mode_modal_amplitude",
            shape=(num_samples, num_elastic_modes, num_loads),
            dims_description="samples, mode, load_case",
            max_chunk_mb=sim.config.get("ZARR_MODE_MODAL_AMP_CHUNK_MB", 4),
        )
        sim.mode_modal_amp_ds.attrs.update({"load_case_names": list(sim.static_load_case_names)})

        sim.mode_recon_error_ds = sim.storage_manager.add_field_storage(
            field_name="mode_reconstruction_error",
            shape=(num_samples, num_loads),
            dims_description="samples, load_case",
            max_chunk_mb=sim.config.get("ZARR_MODE_RECON_ERR_CHUNK_MB", 1),
        )
        sim.mode_recon_error_ds.attrs.update({"load_case_names": list(sim.static_load_case_names)})

        if sim.config.get("SAVE_SIJ_STATS", True):
            # Common attributes for SIJ stats
            angles_attrs = {
                "sides": ["left", "right"],
                "angle_names": ["nutation_alpha_x", "rotation_AP_beta_y", "rotation_CC_gamma_z"],
            }
            trans_attrs = {
                "sides": ["left", "right"],
                "components": ["dx_ML", "dy_AP", "dz_CC"],
                "units": "mm",
            }
            # Create angles/trans datasets for all load cases in a loop
            for load_name in sim.static_load_case_names:
                angles_ds = sim.storage_manager.add_field_storage(
                    field_name=f"sij_angles_{load_name}",
                    shape=(num_samples, 2, 3),
                    dims_description="samples, side(L/R), angles(nutation_alpha_x,rotation_AP_beta_y,rotation_CC_gamma_z)",
                    max_chunk_mb=sim.config.get("ZARR_SIJ_ANGLES_MAX_CHUNK_MB", 1),
                )
                angles_ds.attrs.update(angles_attrs)
                setattr(sim, f"sij_angles_{load_name}_ds", angles_ds)

                trans_ds = sim.storage_manager.add_field_storage(
                    field_name=f"sij_trans_{load_name}",
                    shape=(num_samples, 2, 3),
                    dims_description="samples, side(L/R), components(dx_ML,dy_AP,dz_CC)",
                    max_chunk_mb=sim.config.get("ZARR_SIJ_TRANS_MAX_CHUNK_MB", 1),
                )
                trans_ds.attrs.update(trans_attrs)
                setattr(sim, f"sij_trans_{load_name}_ds", trans_ds)
        else:
            sim.sij_angles_SP2leg_ds = None
            sim.sij_angles_SP1leg_ds = None
            sim.sij_angles_LAB_phase1_ds = None
            sim.sij_angles_LAB_phase2_ds = None
            sim.sij_angles_LAB_phase3_ds = None
            sim.sij_trans_SP2leg_ds = None
            sim.sij_trans_SP1leg_ds = None
            sim.sij_trans_LAB_phase1_ds = None
            sim.sij_trans_LAB_phase2_ds = None
            sim.sij_trans_LAB_phase3_ds = None

    def _process_single_sample(self, idx: int, nev_requested: int, num_elastic_modes: int) -> None:
        sim = self.sim
        dataset_mode = sim.config.get("DATASET_MODE", "full")
        
        # Apply material properties
        if dataset_mode in ("full", "material_only"):
            sim._apply_sample_material_properties(idx)
        
        # Apply shape deformation
        if dataset_mode in ("full", "shape_only"):
            sim.shape_mapper.apply_sample(idx)
        else:
            sim.shape_mapper.apply_reference()

        # Update solvers with new phi/E/nu
        if sim.ligament_system:
            sim.ligament_system.update()
        sim.eigen_solver.update()
        sim.elastic_solver.update()

        # Warm start eigen solve from reference modes (initial space)
        init_space = sim.ref_eigvecs if sim.ref_eigvecs is not None else None
        eigvals, eigvecs = sim.eigen_solver.solve(
            nev=nev_requested,
            target=sim.config["EIGENVALUE_TARGET"],
            initial_space_vecs=init_space,
        )
        
        # Truncate or pad to expected size
        if len(eigvals) > num_elastic_modes:
            eigvals = eigvals[:num_elastic_modes]
            eigvecs = eigvecs[:num_elastic_modes]
        elif len(eigvals) < num_elastic_modes:
            padded_eigvals = np.zeros(num_elastic_modes)
            padded_eigvals[: len(eigvals)] = eigvals
            eigvals = padded_eigvals
            padded_eigvecs = np.zeros((num_elastic_modes, *eigvecs.shape[1:]))
            padded_eigvecs[: len(eigvecs)] = eigvecs
            eigvecs = padded_eigvecs

        perm = pair_modes(sim.ref_eigvecs, sim.ref_eigvals, eigvecs, eigvals, sim.eigen_solver.M0)
        sim.eig_permutations_ds[idx] = perm.astype(np.int32)

        # Keep only solver-order arrays; permutation saved for optional downstream alignment

        sim.eigvals_ds[idx] = eigvals
        sim.eigvecs_ds[idx] = eigvecs

        if sim.lig_sens_ds and sim.config.get("COMPUTE_SENSITIVITIES", True):
            sim.lig_sens_ds[idx] = sim.eigen_solver.eigenvalue_sensitivities(eigvecs)
        
        if sim.pretension_sens_ds and sim.config.get("COMPUTE_SENSITIVITIES", True):
            sim.pretension_sens_ds[idx] = sim.eigen_solver.eigenvalue_pretension_sensitivities(eigvecs)
        
        if sim.cartilage_sens_ds and sim.config.get("COMPUTE_SENSITIVITIES", True):
            sim.cartilage_sens_ds[idx] = sim._compute_cartilage_sensitivities(eigvecs)

        if sim.bone_alpha_sens_ds and sim.bone_beta_sens_ds and sim.config.get("COMPUTE_SENSITIVITIES", True):
            # Per-sample bone-law sensitivities (use sample densities via sample_idx)
            bone_sens = sim._compute_bone_density_param_sensitivities(eigvecs, sample_idx=idx)
            sim.bone_alpha_sens_ds[idx] = bone_sens["alpha"]
            sim.bone_beta_sens_ds[idx] = bone_sens["beta"]

        if sim.config.get("STATIC_ANAL", True):
            # Save modal analysis using solver-order modes (unsorted)
            self._process_static_loads(idx, eigvecs, eigvals)

    def _process_static_loads(
        self,
        idx: int,
        eigvecs_unsorted: np.ndarray,
        eigvals_unsorted: np.ndarray,
    ) -> None:
        """Compute per-load modal metrics using solver-order (unsorted) modes.

        Args:
            idx: Sample index being processed
            eigvecs_unsorted: Eigenvectors in solver order (no MAC pairing applied)
            eigvals_unsorted: Eigenvalues in solver order (no MAC pairing applied)

        Notes:
            - The pairing permutation is still computed and saved separately as
              ``eig_permutations_ds`` for downstream alignment, but it is NOT
              applied here. All saved modal metrics are in solver order.
        """
        sim = self.sim
        # Use the reference template mesh. Patient geometry is represented implicitly by fu.
        template_ref = sim.template.copy(deep=True)

        # Map load case name to load function
        load_fns = {
            "SP2leg": static_post_load_two_legs,
            "SP1leg": static_post_load_one_leg,
            "LAB_phase1": labor_load_phase_1,
            "LAB_phase2": labor_load_phase_2,
            "LAB_phase3": labor_load_phase_3,
        }

        # Compute and store per-load displacements and SIJ stats
        gdim = sim.V.mesh.geometry.dim
        load_results: list[tuple[str, fem.Function]] = []
        for load_name in sim.static_load_case_names:
            fn = load_fns.get(load_name)
            if fn is None:
                continue
            u_disp, sij_stats = fn(sim.elastic_solver, sim.f2x_model.ds_named, template_ref)

            # Displacement dataset (if configured)
            u_ds = getattr(sim, f"u_{load_name}_ds", None)
            if u_ds is not None:
                u_ds[idx] = u_disp.x.array.reshape((-1, gdim))

            # Strain/stress tensors (DG0, full 3x3)
            strain_ds = getattr(sim, f"strain_{load_name}_ds", None)
            stress_ds = getattr(sim, f"stress_{load_name}_ds", None)
            if strain_ds is not None or stress_ds is not None:
                strain_arr, stress_arr = sim.elastic_solver.compute_strain_stress()
                if strain_ds is not None:
                    strain_ds[idx] = strain_arr
                if stress_ds is not None:
                    stress_ds[idx] = stress_arr

            # SIJ stats datasets (if configured via _init_storage loop)
            angles_ds = getattr(sim, f"sij_angles_{load_name}_ds", None)
            trans_ds = getattr(sim, f"sij_trans_{load_name}_ds", None)
            self._write_sij_dataset(idx, sij_stats, angles_ds, trans_ds)

            load_results.append((load_name, u_disp))

        # Modal analysis for each load case (solver-order modes, unsorted)
        eigvecs_clean = np.nan_to_num(eigvecs_unsorted, nan=0.0)
        eigvals_clean = np.nan_to_num(eigvals_unsorted, nan=0.0)
        load_index = {name: i for i, name in enumerate(sim.static_load_case_names)}
        for load_name, disp in load_results:
            analysis = compute_mode_contributions(
                eigvecs_clean,
                eigvals_clean,
                disp,
                mass_matrix=sim.eigen_solver.M,
            )
            idx_load = load_index[load_name]
            sim.mode_mac_ds[idx, :, idx_load] = analysis["mac"]
            sim.mode_energy_frac_ds[idx, :, idx_load] = analysis["energy_fraction"]
            sim.mode_energy_ds[idx, :, idx_load] = analysis["energy"]
            sim.mode_modal_amp_ds[idx, :, idx_load] = analysis["modal_amplitude"]
            sim.mode_recon_error_ds[idx, idx_load] = analysis["reconstruction_error"]

    def _write_sij_dataset(self, idx: int, stats: dict, angle_store, trans_store) -> None:
        if angle_store:
            angle_store[idx] = sij_angles_to_array(stats)
        if trans_store:
            trans_store[idx] = sij_trans_to_array(stats)
