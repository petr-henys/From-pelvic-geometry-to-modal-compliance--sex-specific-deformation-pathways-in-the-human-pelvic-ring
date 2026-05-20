"""Eigenstiffness simulation orchestrator.

Consumes FEBio model, interpolates template material fields onto mesh via RBF,
and runs reference/dataset eigenstiffness analyses.
"""
from __future__ import annotations

import logging
import numpy as np
import pyvista as pv
from dolfinx import fem
from pathlib import Path

from config import SimulationConfig
from logging_config import LoggingManager
from simulation.data_mapper import MaterialMapper, ShapeMapper
from simulation.storage_manager import StorageManager
from simulation.febio_parser import FEBio2Dolfinx
from simulation.utils import ensure_nonnegative
from simulation.simulation_dataset import DatasetProcessor
from simulation.simulation_reference import ReferenceSolver

logger = logging.getLogger(__name__)



class EigenstiffnessSimulation:
    """Main orchestrator for pelvic eigenstiffness analysis workflow.
    
    Workflow: Load meshes → Prepare materials → Solve reference → Process dataset
    Storage: Zarr arrays in RESULTS_DIR/data/
    """

    def __init__(self, config_file: str | None = None):
        self.config = SimulationConfig(config_file)
        if not self.config.validate():
            raise ValueError("Invalid configuration")

        self.logging_manager = LoggingManager(self.config["RESULTS_DIR"])
        self.logger = logging.getLogger(__name__)

        # Core components
        self.storage_manager = None
        self.material_mapper = None
        self.shape_mapper = None
        self.f2x_model = None
        
        # Material fields
        self.E_func = None
        self.nu_func = None
        
        # Mesh and geometry
        self.domain = None
        self.template = None
        self.ligament_pairs = []
        self.ligament_names = []
        self.cell_tags = None
        
        # Function spaces
        self.V = None
        self.W = None
        
        # RBF and data
        self.rbf_options = {}
        self.data_arrays = {}
        self.cartilage_param_defs = []
        
        # Solvers
        self.ligament_system = None
        self.eigen_solver = None
        self.elastic_solver = None
        
        # Reference solution
        self.ref_eigvals = None
        self.ref_eigvecs = None
        self.ref_lig_sens = None
        self.ref_pretension_sens = None
        self.ref_cartilage_sens = None
        self.ref_bone_param_sens = None
        
        # Static analysis storage
        self.static_load_case_names = ("SP2leg", "SP1leg", "LAB_phase1", "LAB_phase2", "LAB_phase3")
        self.ref_mode_mac = {}
        self.ref_mode_energy_frac = {}
        self.ref_mode_energy = {}
        self.ref_mode_modal_amp = {}
        self.ref_mode_recon_error = {}
        
        # Dataset storage (initialized by storage_manager)
        self.eigvals_ds = None
        self.eig_permutations_ds = None
        self.eigvecs_ds = None
        self.lig_sens_ds = None
        self.pretension_sens_ds = None
        self.cartilage_sens_ds = None
        self.bone_alpha_sens_ds = None
        self.bone_beta_sens_ds = None
        self.u_SP2leg_ds = None
        self.u_SP1leg_ds = None
        self.u_LAB_phase1_ds = None
        self.u_LAB_phase2_ds = None
        self.u_LAB_phase3_ds = None
        self.strain_SP2leg_ds = None
        self.strain_SP1leg_ds = None
        self.strain_LAB_phase1_ds = None
        self.strain_LAB_phase2_ds = None
        self.strain_LAB_phase3_ds = None
        self.stress_SP2leg_ds = None
        self.stress_SP1leg_ds = None
        self.stress_LAB_phase1_ds = None
        self.stress_LAB_phase2_ds = None
        self.stress_LAB_phase3_ds = None
        self.sij_angles_SP2leg_ds = None
        self.sij_angles_SP1leg_ds = None
        self.sij_angles_LAB_phase1_ds = None
        self.sij_angles_LAB_phase2_ds = None
        self.sij_angles_LAB_phase3_ds = None
        self.sij_trans_SP2leg_ds = None
        self.sij_trans_SP1leg_ds = None
        self.sij_trans_LAB_phase1_ds = None
        self.sij_trans_LAB_phase2_ds = None
        self.sij_trans_LAB_phase3_ds = None
        self.mode_mac_ds = None
        self.mode_energy_frac_ds = None
        self.mode_energy_ds = None
        self.mode_modal_amp_ds = None
        self.mode_recon_error_ds = None

    def run(self) -> None:
        self.logging_manager.setup()
        verbose = self.config.get("VERBOSE")
        self.logging_manager.reconfigure_console("INFO" if verbose else "WARNING")
        
        self.logger.info("Simulation start")
        if self.config.config_path and Path(self.config.config_path).exists():
            self.logger.info("Configuration: %s", self.config.config_path)

        self._setup_simulation()
        self._prepare_materials()

        ReferenceSolver(self).run()
        DatasetProcessor(self).process()
        self._finalize()

        self.logger.info("Simulation completed successfully")

    def _setup_simulation(self) -> None:
        self.logger.info("--- Starting Simulation Setup ---")
        self._load_data()
        self._setup_domain_mesh()
        self.storage_manager = StorageManager(self.config)

    def _load_data(self) -> None:
        self.logger.info("Load template: %s", self.config["TEMPLATE_MESH_PATH"])
        self.template = pv.read(self.config["TEMPLATE_MESH_PATH"])

        self.logger.info("Loading shape and density data")
        shapes = np.load(self.config["SHAPE_DATA_PATH"])
        densities = ensure_nonnegative(np.load(self.config["DENSITY_DATA_PATH"]), self.config["MIN_DENSITY_VALUE"])
        self.data_arrays = {
            "shapes": shapes,
            "densities": densities,
        }
        self.logger.info("Dataset samples: %d", shapes.shape[0] if shapes.ndim > 1 else 1)

    def _setup_domain_mesh(self) -> None:
        self.logger.info("Load domain mesh: %s", self.config['DOMAIN_MESH_PATH'])

        self.f2x_model = FEBio2Dolfinx(self.config["DOMAIN_MESH_PATH"])

        ligaments_dir = Path(self.config["RESULTS_DIR"]) / "ligaments"
        par_dir = Path(self.config["RESULTS_DIR"]) / "paraview"
        ligaments_dir.mkdir(parents=True, exist_ok=True)
        
        ligaments_path = ligaments_dir / "ligaments.vtm"
        self.f2x_model.save_mesh_with_tags(par_dir)
        ligaments_mb = self.f2x_model.save_discrete_sets_to_vtk(str(ligaments_path))
        
        self.ligament_names = self.f2x_model.discrete_set_names or []
        self.ligament_pairs = [ligaments_mb[name] for name in self.ligament_names] if ligaments_mb else []
        
        # Validate ligament config
        if self.ligament_names:
            ligament_config = self.config["LIGAMENTS"]
            for name in self.ligament_names:
                if name not in ligament_config:
                    raise ValueError(f"Ligament '{name}' from FEBio model not found in config")
            self.logger.info("Validated %d ligaments", len(self.ligament_names))
        
        self.domain = self.f2x_model.mesh_dolfinx
        self.cell_tags = self.f2x_model.cell_tag

    def _prepare_materials(self) -> None:
        self.logger.info("--- Preparing Material Properties (RBF template → FEBio mesh) ---")

        self._create_function_spaces()
        self.E_func, self.nu_func, self.phi_func = self._initialize_property_functions()

        self.material_mapper = MaterialMapper(
            E_func=self.E_func,
            nu_func=self.nu_func,
            template=self.template,
            densities=self.data_arrays["densities"],
            config=self.config,
            febio_parser=self.f2x_model,
        )

        self.shape_mapper = ShapeMapper(
            phi=self.phi_func,
            template=self.template,
            shapes=self.data_arrays["shapes"],
            config=self.config,
        )

        # Interpolate reference elasticity
        self.material_mapper.apply_reference()
        
        # Define cartilage/symphysis parameters for sensitivity analysis
        cell_values = self.cell_tags.values
        self.cartilage_param_defs = []
        # Use FEBio region names directly; include only cartilage/symphysis regions
        region_labels = list(self.f2x_model.region_labels)
        for rid, label in enumerate(region_labels):
            label_lower = str(label).lower()
            if ("cartilage" not in label_lower) and ("symph" not in label_lower):
                continue
            region_mask = cell_values == int(rid)
            if not np.any(region_mask):
                continue
            modulus = float(self.E_func.x.array[region_mask].mean())
            self.cartilage_param_defs.append({
                "name": str(label),
                "region_ids": [int(rid)],
                "modulus": modulus,
            })

    def _create_function_spaces(self) -> None:
        gdim = self.domain.geometry.dim
        self.V = fem.functionspace(self.domain, ("P", 1, (gdim,)))
        self.W = fem.functionspace(self.domain, ("DG", 0))

    def _initialize_property_functions(self):
        E = fem.Function(self.W, name="YoungsModulus")
        nu = fem.Function(self.W, name="PoissonsRatio")
        phi = fem.Function(self.V, name="ShapeField")
        return E, nu, phi

    def _compute_cartilage_sensitivities(self, eigenvectors: np.ndarray) -> np.ndarray:
        """Compute per-mode dλ/dE for cartilage/symphysis using current E (no reference means).

        Sums per-region elastic energies and divides by the current region modulus
        taken directly from ``E_func`` (assumed uniform per region). Normalizes by
        the mode mass ``xᵀ M x``.
        """
        if not self.cartilage_param_defs:
            return np.zeros((eigenvectors.shape[0], 0), dtype=float)

        unique_region_ids = sorted(
            {int(rid) for param in self.cartilage_param_defs for rid in param["region_ids"]}
        )
        energies = self.eigen_solver.mode_region_energies(
            eigenvectors, self.cell_tags, unique_region_ids
        )
        region_index = {rid: idx for idx, rid in enumerate(unique_region_ids)}
        out = np.zeros((eigenvectors.shape[0], len(self.cartilage_param_defs)), dtype=float)

        cell_values = self.cell_tags.values
        for j, param in enumerate(self.cartilage_param_defs):
            # Columns in energies corresponding to this parameter's regions
            ids = [int(rid) for rid in param["region_ids"] if int(rid) in region_index]
            if not ids:
                continue
            col_idx = [region_index[rid] for rid in ids]
            energy_sum = energies[:, col_idx].sum(axis=1)

            # Use current E in these regions as denominator (uniform per region)
            region_mask = np.isin(cell_values, ids)
            if not np.any(region_mask):
                continue
            region_E_vals = self.E_func.x.array[region_mask]
            denom = float(region_E_vals[0]) if region_E_vals.size > 0 else 0.0
            if abs(denom) > 0.0:
                out[:, j] = energy_sum / denom

        mass_norms = self.eigen_solver._mass_norms(eigenvectors)
        valid = mass_norms > 1e-12
        out[valid] /= mass_norms[valid, None]
        out[~valid] = 0.0
        return out

    def _compute_bone_density_param_sensitivities(self, eigenvectors: np.ndarray, *, sample_idx: int | None = None) -> dict[str, np.ndarray]:
        """Compute per-mode dλ/dα and dλ/dβ for E = α ρ_ash^β.

        If sample_idx is provided, weights are computed from that sample's
        density field; otherwise, the reference (median) density is used.
        """
        # Get sensitivity weight arrays from MaterialMapper (sample-specific when requested)
        weights = self.material_mapper.compute_bone_param_sensitivity_weights(sample_idx=sample_idx)
        
        # Compute weighted strain energies
        E_alpha = self.eigen_solver.mode_weighted_energies(eigenvectors, weights["alpha"])
        E_beta = self.eigen_solver.mode_weighted_energies(eigenvectors, weights["beta"])
        
        # Normalize by mass
        mass_norms = self.eigen_solver._mass_norms(eigenvectors)
        valid = mass_norms > 1e-12
        
        dlam_dalpha = np.zeros_like(E_alpha)
        dlam_dbeta = np.zeros_like(E_beta)
        dlam_dalpha[valid] = E_alpha[valid] / mass_norms[valid]
        dlam_dbeta[valid] = E_beta[valid] / mass_norms[valid]
        
        return {"alpha": dlam_dalpha, "beta": dlam_dbeta}

    def _apply_sample_material_properties(self, idx: int) -> None:
        self.material_mapper.apply_sample(idx)

    def cleanup(self) -> None:
        if self.eigen_solver:
            self.eigen_solver.K.destroy()
            if self.eigen_solver.M is not None:
                self.eigen_solver.M.destroy()
            if self.eigen_solver.M0 is not None:
                self.eigen_solver.M0.destroy()
        if self.elastic_solver:
            self.elastic_solver.K.destroy()
        if self.ligament_system:
            self.ligament_system.K.destroy()
        self.logger.debug("Cleanup complete")

    def _finalize(self) -> None:
        # Build comprehensive metadata combining config + runtime info
        if self.storage_manager:
            metadata = self._build_complete_metadata()
            self.storage_manager.save_metadata(metadata)
        
        self.cleanup()
    
    def _build_complete_metadata(self) -> dict:
        """Build complete metadata combining config and runtime information.
        
        Top-level fields are ONLY runtime info not in config:
        - timestamp, versions: runtime info
        - mesh, reference_eigvec_shape: runtime info
        - mode_load_cases: runtime list (not in config)
        
        All config parameters (including ligament names, stiffnesses) in config dict.
        """
        import datetime
        import numpy as np
        import zarr
        import dolfinx
        
        metadata = {
            # Runtime info (NOT in config)
            "timestamp": datetime.datetime.now().isoformat(),
            
            # Library versions (NOT in config)
            "versions": {
                "numpy": np.__version__,
                "zarr": zarr.__version__,
                "dolfinx": dolfinx.__version__,
            },
        }
        
        # Mesh information (runtime info, NOT in config)
        if self.domain:
            metadata["mesh"] = {
                "dim": int(self.domain.geometry.dim),
                "num_cells": int(self.domain.topology.index_map(self.domain.topology.dim).size_local),
                "num_nodes": int(self.domain.geometry.x.shape[0]),
            }
        
        # Reference eigenvector shape (runtime info, NOT in config)
        if self.ref_eigvecs is not None:
            metadata["reference_eigvec_shape"] = list(self.ref_eigvecs.shape)
        
        # Static load cases (runtime list, NOT in config)
        if self.config.get("STATIC_ANAL", False):
            metadata["mode_load_cases"] = list(self.static_load_case_names)
        
        # Full config (all configuration parameters including ligaments)
        # Add ligament names from FEBio to config before saving
        config_snapshot = self.storage_manager._config_snapshot()
        if self.ligament_names:
            config_snapshot["LIGAMENT_NAMES"] = self.ligament_names
        
        metadata["config"] = config_snapshot
        
        return metadata
