"""Reference problem handling for the eigenstiffness simulation."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pyvista as pv


from simulation.elasticsolver import ElasticSolver
from simulation.eigensolver import ElasticEigenSolver
from simulation.ligamentassembler import LigamentSpringSystem
from dolfinx import plot

from simulation.simulation_analysis import (
    compute_mode_contributions,
    format_sensitivity_debug,
    log_sij_stats,
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

if TYPE_CHECKING:  # pragma: no cover - typing only
    from simulation.simulation_core import EigenstiffnessSimulation


class ReferenceSolver:
    """Solve the reference eigenproblem and associated post-processing."""

    def __init__(self, simulation: "EigenstiffnessSimulation") -> None:
        self.sim = simulation
        self.logger = simulation.logger

    def run(self) -> None:
        sim = self.sim
        self.logger.info("Reference eigenproblem")

        sim.ligament_system = None
        if sim.ligament_pairs:
            self._initialize_ligaments()

        sim.eigen_solver = ElasticEigenSolver(
            sim.V,
            sim.E_func,
            sim.nu_func,
            sim.phi_func,
            sim.config.config,
            ligament_system=sim.ligament_system,
        )
        # Apply user-provided boundary penalty (required; no automatic fallback)
        # Boundary conditions on S1_facet (fixed sacrum)

        penalty = float(sim.config["SURFACE_PENALTY_GAMMA"])
        bnd_name = sim.config["BOUNDARY_FIXED"]
        sim.eigen_solver.fixed_dirichlet(gamma=penalty, ds=sim.f2x_model.ds_named(bnd_name))

        sim.elastic_solver = ElasticSolver(
            sim.V,
            sim.E_func,
            sim.nu_func,
            sim.phi_func,
            sim.config.config,
            ligament_system=sim.ligament_system,
        )
        # Apply user-provided boundary penalty (required; no automatic fallback)
        sim.elastic_solver.fixed_dirichlet(sim.f2x_model.ds_named(bnd_name), penalty)


        nev_requested = sim.config["NUM_EIGENVALUES"] + 10
        self.logger.info(
            "Requesting %d eigenvalues to ensure %d elastic modes",
            nev_requested,
            sim.config['NUM_EIGENVALUES'],
        )

        sim.ref_eigvals, sim.ref_eigvecs = sim.eigen_solver.solve(
            nev=nev_requested,
            target=sim.config["EIGENVALUE_TARGET"],
        )

        if len(sim.ref_eigvals) > sim.config["NUM_EIGENVALUES"]:
            self.logger.info(
                "Truncating from %d to %d elastic modes",
                len(sim.ref_eigvals),
                sim.config['NUM_EIGENVALUES'],
            )
            sim.ref_eigvals = sim.ref_eigvals[: sim.config["NUM_EIGENVALUES"]]
            sim.ref_eigvecs = sim.ref_eigvecs[: sim.config["NUM_EIGENVALUES"]]

        self.logger.info("Reference modes ready (%d elastic)", len(sim.ref_eigvals))

        if (
            sim.config["COMPUTE_SENSITIVITIES"]
            and sim.ligament_system is not None
            and sim.ligament_system.num_bundles > 0
        ):
            sim.ref_lig_sens = sim.eigen_solver.eigenvalue_sensitivities(sim.ref_eigvecs)
            sim.ref_pretension_sens = sim.eigen_solver.eigenvalue_pretension_sensitivities(sim.ref_eigvecs)

            self.logger.info(
                "Computed reference ligament sensitivities (shape %s)",
                sim.ref_lig_sens.shape,
            )
            if self.logger.isEnabledFor(logging.DEBUG):
                lig_labels = sim.ligament_system.bundle_names
                self.logger.debug(
                    "Reference ligament sensitivities\n%s",
                    format_sensitivity_debug(sim.ref_lig_sens, lig_labels),
                )
            
            self.logger.info(
                "Computed reference pretension sensitivities (shape %s)",
                sim.ref_pretension_sens.shape,
            )
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(
                    "Reference pretension sensitivities\n%s",
                    format_sensitivity_debug(sim.ref_pretension_sens, lig_labels),
                )
        else:
            sim.ref_lig_sens = None
            sim.ref_pretension_sens = None

        if sim.config["COMPUTE_SENSITIVITIES"] and sim.cartilage_param_defs:
            sim.ref_cartilage_sens = sim._compute_cartilage_sensitivities(sim.ref_eigvecs)
            self.logger.info(
                "Computed reference cartilage sensitivities (shape %s)",
                sim.ref_cartilage_sens.shape,
            )
            if self.logger.isEnabledFor(logging.DEBUG):
                cart_labels = [str(param["name"]) for param in sim.cartilage_param_defs]
                self.logger.debug(
                    "Reference cartilage sensitivities\n%s",
                    format_sensitivity_debug(sim.ref_cartilage_sens, cart_labels),
                )
        else:
            sim.ref_cartilage_sens = None

        # Bone density-law parameter sensitivities (scale α and exponent β)
        if sim.config["COMPUTE_SENSITIVITIES"]:
            sens = sim._compute_bone_density_param_sensitivities(sim.ref_eigvecs)
            sim.ref_bone_param_sens = sens
            self.logger.info(
                "Computed reference bone density-law sensitivities: keys=%s, per-mode length=%d",
                list(sens.keys()),
                len(next(iter(sens.values()))) if sens else 0,
            )

        self._run_static_loads()
        self._save_reference_visualization()
        self._save_reference_npz(Path(sim.config["RESULTS_DIR"]) / "reference_solution.npz")

    def _initialize_ligaments(self) -> None:
        sim = self.sim
        ligaments_config = sim.config["LIGAMENTS"]
        
        # Extract stiffnesses and pretensions in order of ligament_names
        stiffnesses = [ligaments_config[name]["stiffness"] for name in sim.ligament_names]
        pretensions = [ligaments_config[name]["pretension"] for name in sim.ligament_names]
        
        sim.ligament_system = LigamentSpringSystem(
            sim.V,
            sim.ligament_pairs,
            stiffnesses,
            sim.phi_func,
            sim.config.config,
            bundle_names=sim.ligament_names,
            pretensions=pretensions,
        )
        sim.ligament_names = sim.ligament_system.bundle_names

    def _run_static_loads(self) -> None:
        sim = self.sim
        sim.u_SP2leg_ref = None
        sim.u_SP1leg_ref = None
        sim.u_LAB_phase1_ref = None
        sim.u_LAB_phase2_ref = None
        sim.u_LAB_phase3_ref = None

        sim.ref_sij_angles_sp2leg = None
        sim.ref_sij_angles_sp1leg = None
        sim.ref_sij_angles_LAB_phase1 = None
        sim.ref_sij_angles_LAB_phase2 = None
        sim.ref_sij_angles_LAB_phase3 = None

        sim.ref_sij_trans_sp2leg = None
        sim.ref_sij_trans_sp1leg = None
        sim.ref_sij_trans_LAB_phase1 = None
        sim.ref_sij_trans_LAB_phase2 = None
        sim.ref_sij_trans_LAB_phase3 = None

        # Reference strain/stress tensors (DG0, full 3x3)
        sim.ref_strain = {}
        sim.ref_stress = {}

        if not sim.config["STATIC_ANAL"]:
            self.logger.info("STATIC_ANAL=False: skipping SP/LAB displacements and SIJ stats")
            return

        sim.ref_mode_mac.clear()
        sim.ref_mode_energy_frac.clear()
        sim.ref_mode_energy.clear()
        sim.ref_mode_modal_amp.clear()
        sim.ref_mode_recon_error.clear()

        (
            sim.u_SP2leg_ref,
            sij_sp2leg,
        ) = static_post_load_two_legs(sim.elastic_solver, sim.f2x_model.ds_named, sim.template)
        sim.ref_strain["SP2leg"], sim.ref_stress["SP2leg"] = sim.elastic_solver.compute_strain_stress()
        self.logger.info(
            "Static Post Load Displacement (mm) max: %.2f",
            sim.u_SP2leg_ref.x.array.max(),
        )

        (
            sim.u_SP1leg_ref,
            sij_sp1leg,
        ) = static_post_load_one_leg(
            sim.elastic_solver,
            sim.f2x_model.ds_named,
            sim.template,
        )
        sim.ref_strain["SP1leg"], sim.ref_stress["SP1leg"] = sim.elastic_solver.compute_strain_stress()
        self.logger.info(
            "Static Post one leg Load Displacement (mm) max: %.2f",
            sim.u_SP1leg_ref.x.array.max(),
        )

        sim.u_LAB_phase1_ref, sij_lab_phase1 = labor_load_phase_1(
            sim.elastic_solver,
            sim.f2x_model.ds_named,
            sim.template,
        )
        sim.ref_strain["LAB_phase1"], sim.ref_stress["LAB_phase1"] = sim.elastic_solver.compute_strain_stress()
        self.logger.info(
            "Labor Load Phase 1 Displacement (mm) max: %.2f",
            sim.u_LAB_phase1_ref.x.array.max(),
        )

        sim.u_LAB_phase2_ref, sij_lab_phase2 = labor_load_phase_2(
            sim.elastic_solver,
            sim.f2x_model.ds_named,
            sim.template,
        )
        sim.ref_strain["LAB_phase2"], sim.ref_stress["LAB_phase2"] = sim.elastic_solver.compute_strain_stress()
        self.logger.info(
            "Labor Load Phase 2 Displacement (mm) max: %.2f",
            sim.u_LAB_phase2_ref.x.array.max(),
        )

        sim.u_LAB_phase3_ref, sij_lab_phase3 = labor_load_phase_3(
            sim.elastic_solver,
            sim.f2x_model.ds_named,
            sim.template,
        )
        sim.ref_strain["LAB_phase3"], sim.ref_stress["LAB_phase3"] = sim.elastic_solver.compute_strain_stress()
        self.logger.info(
            "Labor Load Phase 3 Displacement (mm) max: %.2f",
            sim.u_LAB_phase3_ref.x.array.max(),
        )

        sim.ref_sij_angles_sp2leg = sij_angles_to_array(sij_sp2leg)
        sim.ref_sij_angles_sp1leg = sij_angles_to_array(sij_sp1leg)
        sim.ref_sij_angles_LAB_phase1 = sij_angles_to_array(sij_lab_phase1)
        sim.ref_sij_angles_LAB_phase2 = sij_angles_to_array(sij_lab_phase2)
        sim.ref_sij_angles_LAB_phase3 = sij_angles_to_array(sij_lab_phase3)

        sim.ref_sij_trans_sp2leg = sij_trans_to_array(sij_sp2leg)
        sim.ref_sij_trans_sp1leg = sij_trans_to_array(sij_sp1leg)
        sim.ref_sij_trans_LAB_phase1 = sij_trans_to_array(sij_lab_phase1)
        sim.ref_sij_trans_LAB_phase2 = sij_trans_to_array(sij_lab_phase2)
        sim.ref_sij_trans_LAB_phase3 = sij_trans_to_array(sij_lab_phase3)

        self.logger.info(log_sij_stats("SP 2legs", sim.ref_sij_angles_sp2leg, sim.ref_sij_trans_sp2leg))
        self.logger.info(log_sij_stats("SP 1leg", sim.ref_sij_angles_sp1leg, sim.ref_sij_trans_sp1leg))
        self.logger.info(log_sij_stats("LAB phase1", sim.ref_sij_angles_LAB_phase1, sim.ref_sij_trans_LAB_phase1))
        self.logger.info(log_sij_stats("LAB phase2", sim.ref_sij_angles_LAB_phase2, sim.ref_sij_trans_LAB_phase2))
        self.logger.info(log_sij_stats("LAB phase3", sim.ref_sij_angles_LAB_phase3, sim.ref_sij_trans_LAB_phase3))

        load_cases_ref = [
            ("SP2leg", sim.u_SP2leg_ref),
            ("SP1leg", sim.u_SP1leg_ref),
            ("LAB_phase1", sim.u_LAB_phase1_ref),
            ("LAB_phase2", sim.u_LAB_phase2_ref),
            ("LAB_phase3", sim.u_LAB_phase3_ref),
        ]
        for load_name, load_disp in load_cases_ref:
            if load_disp is None:
                continue
            analysis = compute_mode_contributions(
                sim.ref_eigvecs,
                sim.ref_eigvals,
                load_disp,
                mass_matrix=sim.eigen_solver.M if sim.eigen_solver is not None else None,
            )
            if analysis is None:
                continue
            sim.ref_mode_mac[load_name] = analysis["mac"]
            sim.ref_mode_energy_frac[load_name] = analysis["energy_fraction"]
            sim.ref_mode_energy[load_name] = analysis["energy"]
            sim.ref_mode_modal_amp[load_name] = analysis["modal_amplitude"]
            sim.ref_mode_recon_error[load_name] = analysis["reconstruction_error"]

            if self.logger.isEnabledFor(logging.DEBUG):
                sorted_idx = np.argsort(analysis["energy"])[::-1][:5]
                top_pairs = ", ".join(
                    f"mode {idx}: {analysis['energy_fraction'][idx]:.3f}"
                    for idx in sorted_idx
                )
                self.logger.debug(
                    "[%s] Top modal energy fractions: %s",
                    load_name,
                    top_pairs,
                )

    def _save_reference_visualization(self) -> None:
        sim = self.sim
        paraview_dir = Path(sim.config["RESULTS_DIR"]) / "paraview"
        paraview_dir.mkdir(parents=True, exist_ok=True)
        grid_ref = pv.UnstructuredGrid(*plot.vtk_mesh(sim.V))
        grid_ref.cell_data["E_ref"] = sim.E_func.x.array

        for idx_mode, vec in enumerate(sim.ref_eigvecs, start=1):
            grid_ref.point_data[f"ref_mode_{idx_mode}"] = vec
        if sim.config["STATIC_ANAL"]:
            if sim.u_SP2leg_ref is not None:
                grid_ref.point_data["ref_u_SP2leg"] = sim.u_SP2leg_ref.x.array.reshape(
                    (-1, sim.V.mesh.geometry.dim)
                )
            if sim.u_SP1leg_ref is not None:
                grid_ref.point_data["ref_u_SP1leg"] = sim.u_SP1leg_ref.x.array.reshape(
                    (-1, sim.V.mesh.geometry.dim)
                )
            if sim.u_LAB_phase1_ref is not None:
                grid_ref.point_data["ref_u_LAB_phase1"] = sim.u_LAB_phase1_ref.x.array.reshape(
                    (-1, sim.V.mesh.geometry.dim)
                )
            if sim.u_LAB_phase2_ref is not None:
                grid_ref.point_data["ref_u_LAB_phase2"] = sim.u_LAB_phase2_ref.x.array.reshape(
                    (-1, sim.V.mesh.geometry.dim)
                )
            if sim.u_LAB_phase3_ref is not None:
                grid_ref.point_data["ref_u_LAB_phase3"] = sim.u_LAB_phase3_ref.x.array.reshape(
                    (-1, sim.V.mesh.geometry.dim)
                )

            # Strain/stress tensors as cell data (flatten 3x3 → 9 components)
            for load_name in sim.static_load_case_names:
                if load_name in sim.ref_strain:
                    arr = sim.ref_strain[load_name]
                    grid_ref.cell_data[f"ref_strain_{load_name}"] = arr.reshape(-1, 9)
                if load_name in sim.ref_stress:
                    arr = sim.ref_stress[load_name]
                    grid_ref.cell_data[f"ref_stress_{load_name}"] = arr.reshape(-1, 9)

        ref_path = paraview_dir / "reference_fields.vtk"
        grid_ref.save(str(ref_path))
        self.logger.info("Reference fields saved to: %s", ref_path)

    def _save_reference_npz(self, filename: Path) -> None:
        sim = self.sim
        out: dict[str, object] = {
            "eigenvalues": sim.ref_eigvals,
            "eigenvectors": sim.ref_eigvecs,
        }
        out["mode_indices"] = np.arange(1, len(sim.ref_eigvals) + 1, dtype=np.int32)

        if (
            sim.config["STATIC_ANAL"]
            and (sim.u_SP2leg_ref is not None)
            and (sim.u_LAB_phase1_ref is not None)
        ):
            gdim = sim.V.mesh.geometry.dim
            out["displacement_SP2leg"] = sim.u_SP2leg_ref.x.array.reshape((-1, gdim))
            out["displacement_SP1leg"] = sim.u_SP1leg_ref.x.array.reshape((-1, gdim))
            out["displacement_LAB_phase1"] = sim.u_LAB_phase1_ref.x.array.reshape((-1, gdim))
            out["displacement_LAB_phase2"] = sim.u_LAB_phase2_ref.x.array.reshape((-1, gdim))
            out["displacement_LAB_phase3"] = sim.u_LAB_phase3_ref.x.array.reshape((-1, gdim))

        if sim.config["STATIC_ANAL"] and sim.config["SAVE_SIJ_STATS"]:
            self._write_sij_stats(out)

        if sim.config["STATIC_ANAL"] and sim.ref_mode_energy_frac:
            self._write_mode_analysis(out)

        # Save reference strain/stress tensors (DG0, full 3x3)
        if sim.config["STATIC_ANAL"]:
            for load_name in sim.static_load_case_names:
                if load_name in sim.ref_strain:
                    out[f"strain_{load_name}"] = sim.ref_strain[load_name]
                if load_name in sim.ref_stress:
                    out[f"stress_{load_name}"] = sim.ref_stress[load_name]

        lig_sens = sim.ref_lig_sens if sim.ref_lig_sens is not None else np.empty((0, 0))
        pretension_sens = sim.ref_pretension_sens if sim.ref_pretension_sens is not None else np.empty((0, 0))
        cart_sens = sim.ref_cartilage_sens if sim.ref_cartilage_sens is not None else np.empty((0, 0))
        cart_names = [param["name"] for param in sim.cartilage_param_defs] if sim.cartilage_param_defs else []
        out["ligament_sensitivities"] = lig_sens
        out["ligament_pretension_sensitivities"] = pretension_sens
        out["ligament_names"] = np.array(sim.ligament_names, dtype=str)
        # Extract pretensions from LIGAMENTS config
        ligament_config = sim.config["LIGAMENTS"]
        pretensions = [ligament_config[name]["pretension"] for name in sim.ligament_names]
        out["ligament_pretensions"] = np.array(pretensions, dtype=float)
        out["cartilage_sensitivities"] = cart_sens
        out["cartilage_parameter_names"] = np.array(cart_names)

        np.savez_compressed(filename, **out)

    def _write_sij_stats(self, out: dict[str, object]) -> None:
        sim = self.sim
        if sim.ref_sij_angles_sp2leg is not None:
            out["sij_angles_SP2leg"] = sim.ref_sij_angles_sp2leg
        if sim.ref_sij_angles_sp1leg is not None:
            out["sij_angles_SP1leg"] = sim.ref_sij_angles_sp1leg
        if sim.ref_sij_angles_LAB_phase1 is not None:
            out["sij_angles_LAB_phase1"] = sim.ref_sij_angles_LAB_phase1
        if sim.ref_sij_angles_LAB_phase2 is not None:
            out["sij_angles_LAB_phase2"] = sim.ref_sij_angles_LAB_phase2
        if sim.ref_sij_angles_LAB_phase3 is not None:
            out["sij_angles_LAB_phase3"] = sim.ref_sij_angles_LAB_phase3
        if any(
            arr is not None
            for arr in (
                sim.ref_sij_angles_sp2leg,
                sim.ref_sij_angles_sp1leg,
                sim.ref_sij_angles_LAB_phase1,
                sim.ref_sij_angles_LAB_phase2,
                sim.ref_sij_angles_LAB_phase3,
            )
        ):
            out["sij_angle_names"] = np.array(
                ["nutation_alpha_x", "rotation_AP_beta_y", "rotation_CC_gamma_z"],
                dtype=object,
            )
            out["sij_sides"] = np.array(["left", "right"], dtype=object)

        if sim.ref_sij_trans_sp2leg is not None:
            out["sij_trans_SP2leg"] = sim.ref_sij_trans_sp2leg
        if sim.ref_sij_trans_sp1leg is not None:
            out["sij_trans_SP1leg"] = sim.ref_sij_trans_sp1leg
        if sim.ref_sij_trans_LAB_phase1 is not None:
            out["sij_trans_LAB_phase1"] = sim.ref_sij_trans_LAB_phase1
        if sim.ref_sij_trans_LAB_phase2 is not None:
            out["sij_trans_LAB_phase2"] = sim.ref_sij_trans_LAB_phase2
        if sim.ref_sij_trans_LAB_phase3 is not None:
            out["sij_trans_LAB_phase3"] = sim.ref_sij_trans_LAB_phase3
        if any(
            arr is not None
            for arr in (
                sim.ref_sij_trans_sp2leg,
                sim.ref_sij_trans_sp1leg,
                sim.ref_sij_trans_LAB_phase1,
                sim.ref_sij_trans_LAB_phase2,
                sim.ref_sij_trans_LAB_phase3,
            )
        ):
            out["sij_trans_names"] = np.array(["dx_ML", "dy_AP", "dz_CC"], dtype=object)

    def _write_mode_analysis(self, out: dict[str, object]) -> None:
        sim = self.sim
        load_names = list(sim.static_load_case_names)
        num_loads = len(load_names)
        num_modes = len(sim.ref_eigvals)
        mode_mac = np.zeros((num_loads, num_modes), dtype=float)
        mode_energy_frac = np.zeros((num_loads, num_modes), dtype=float)
        mode_energy = np.zeros((num_loads, num_modes), dtype=float)
        mode_modal_amp = np.zeros((num_loads, num_modes), dtype=float)
        mode_recon_err = np.zeros((num_loads,), dtype=float)
        for i, name in enumerate(load_names):
            if name in sim.ref_mode_mac:
                mode_mac[i] = sim.ref_mode_mac[name]
            if name in sim.ref_mode_energy_frac:
                mode_energy_frac[i] = sim.ref_mode_energy_frac[name]
            if name in sim.ref_mode_energy:
                mode_energy[i] = sim.ref_mode_energy[name]
            if name in sim.ref_mode_modal_amp:
                mode_modal_amp[i] = sim.ref_mode_modal_amp[name]
            if name in sim.ref_mode_recon_error:
                mode_recon_err[i] = sim.ref_mode_recon_error[name]
        out["mode_analysis_loads"] = np.array(load_names, dtype=object)
        out["mode_mac"] = mode_mac
        out["mode_energy_fraction"] = mode_energy_frac
        out["mode_energy"] = mode_energy
        out["mode_modal_amplitude"] = mode_modal_amp
        out["mode_reconstruction_error"] = mode_recon_err
