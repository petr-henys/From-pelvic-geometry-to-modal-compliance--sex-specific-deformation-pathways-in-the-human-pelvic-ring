"""Shared simulation runner for parameter-sweep scripts.

Centralises the boilerplate that every ``run_*_sweep.py`` script
repeats: copy base config, inject standard flags, run, clean up.
"""
from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def run_sweep_simulation(
    param_point: dict,
    output_path: Path,
    base_config: dict,
    *,
    full_dataset: bool = False,
    compute_sensitivities: bool = True,
    **extra_overrides: object,
) -> None:
    """Run a single FEM simulation inside a parameter sweep.

    Parameters
    ----------
    param_point : dict
        Parameters specific to this sweep point (merged last).
    output_path : Path
        Directory where results are written.
    base_config : dict
        Baseline simulation configuration dictionary.
    full_dataset : bool
        If *True*, process all subjects (``FULL_DATASET_FLAG``).
    compute_sensitivities : bool
        Whether to compute sensitivity fields.
    **extra_overrides
        Any additional config keys to override.
    """
    from simulation.simulation_core import EigenstiffnessSimulation

    output_path.mkdir(parents=True, exist_ok=True)

    config_dict = base_config.copy()
    config_dict["RESULTS_DIR"] = str(output_path)
    config_dict["FULL_DATASET_FLAG"] = full_dataset
    config_dict["STATIC_ANAL"] = True
    config_dict["SAVE_SIJ_STATS"] = True
    config_dict["COMPUTE_SENSITIVITIES"] = compute_sensitivities
    config_dict["VERBOSE"] = False
    config_dict.update(extra_overrides)
    config_dict.update(param_point)

    logger.info("Running: %s", output_path.name)

    sim = EigenstiffnessSimulation(config_dict)
    sim.run()
    sim.cleanup()
