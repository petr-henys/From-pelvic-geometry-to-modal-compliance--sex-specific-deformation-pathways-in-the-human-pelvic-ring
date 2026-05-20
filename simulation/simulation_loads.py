"""Static load definitions for standing and labor load cases."""
from __future__ import annotations

import numpy as np
import pyvista as pv
import ufl
from dolfinx import fem

from anatomy_analyser import sij_relative_angles
from simulation.elasticsolver import ElasticSolver


def _compute_sij_stats(
    elastic_solver: ElasticSolver, template: pv.PolyData
) -> tuple[fem.Function, dict]:
    """Solve elastic equilibrium and extract SIJ kinematics."""
    displacement = elastic_solver.solve()
    template_ref_mesh = template.copy(deep=True)
    deformed_mesh = template_ref_mesh.copy(deep=True)
    deformed_mesh.points = elastic_solver.deform(template_ref_mesh.points)
    sij_stats = sij_relative_angles(template_ref_mesh, deformed_mesh)
    return displacement, sij_stats


def static_post_load_two_legs(
    elastic_solver: ElasticSolver,
    ds: ufl.Measure,
    template: pv.PolyData,
) -> tuple[fem.Function, dict]:
    """Bilateral standing: 400 N vertical on each acetabulum (SP2leg).
    
    Args:
        elastic_solver: Elastic solver instance
        ds: Function that returns ds measure for a surface name (use sim.ds_named)
        template: Template mesh for SIJ statistics
    """
    elastic_solver.clear_loads()
    Fz = np.array([0.0, 0.0, 400.0])
    # Load on right and left acetabular notches
    elastic_solver.apply_load(Fz, ds("right_AC_notch"), defined_on="sample", input_type="total_force")
    elastic_solver.apply_load(Fz, ds("left_AC_notch"), defined_on="sample", input_type="total_force")
    return _compute_sij_stats(elastic_solver, template)


def static_post_load_one_leg(
    elastic_solver: ElasticSolver,
    ds: ufl.Measure,
    template: pv.PolyData,
) -> tuple[fem.Function, dict]:
    """Unilateral standing: 800 N vertical on one acetabulum (SP1leg)."""
    elastic_solver.clear_loads()
    Fz = np.array([0.0, 0.0, 800.0])
    elastic_solver.apply_load(Fz, ds("right_AC_notch"), defined_on="sample", input_type="total_force")
    return _compute_sij_stats(elastic_solver, template)


def labor_load_phase_1(
    elastic_solver: ElasticSolver,
    ds: ufl.Measure,
    template: pv.PolyData,
) -> tuple[fem.Function, dict]:
    """Labor phase 1: Bilateral mediolateral force (\u00b1400 N) on ring contact surfaces."""
    elastic_solver.clear_loads()
    force_total = 400
    Fx_pos = np.array([force_total, 0.0, 0.0])
    Fx_neg = np.array([-force_total * 1.0, 0.0, 0.0])
    # Mediolateral compression on left and right pelvic ring contact surfaces
    elastic_solver.apply_load(Fx_pos, ds("ring_contact_left"), defined_on="sample", input_type="total_force")
    elastic_solver.apply_load(Fx_neg, ds("ring_contact_right"), defined_on="sample", input_type="total_force")
    return _compute_sij_stats(elastic_solver, template)


def labor_load_phase_2(
    elastic_solver: ElasticSolver,
    ds: ufl.Measure,
    template: pv.PolyData,
) -> tuple[fem.Function, dict]:
    """Labor phase 2: Bilateral SIJ distraction (±400 N)."""
    elastic_solver.clear_loads()
    force_total = 400
    Fx_pos = np.array([force_total, 0.0, 0.0])
    Fx_neg = np.array([-force_total, 0.0, 0.0])
    # Distraction forces on left and right ischial tuberosities
    elastic_solver.apply_load(Fx_pos, ds("left_ischium_tuber"), defined_on="sample", input_type="total_force")
    elastic_solver.apply_load(Fx_neg, ds("right_ischium_tuber"), defined_on="sample", input_type="total_force")
    return _compute_sij_stats(elastic_solver, template)


def labor_load_phase_3(
    elastic_solver: ElasticSolver,
    ds: ufl.Measure,
    template: pv.PolyData,
) -> tuple[fem.Function, dict]:
    """Labor phase 3: AP distraction of outlet (±400 N on pubis_ins and SCJ)."""
    elastic_solver.clear_loads()
    force_total = 400
    Fy_pos = np.array([0.0, force_total, 0.0])
    Fy_neg = np.array([0.0, -force_total, 0.0])
    # Anteroposterior distraction: pubis forward, SCJ backward
    elastic_solver.apply_load(Fy_neg, ds("pubis_ins"), defined_on="sample", input_type="total_force")
    elastic_solver.apply_load(Fy_pos, ds("SCJ"), defined_on="sample", input_type="total_force")
    return _compute_sij_stats(elastic_solver, template)
