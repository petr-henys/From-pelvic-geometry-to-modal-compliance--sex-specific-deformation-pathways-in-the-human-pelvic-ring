#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from parametrizer import ParameterSweep, Parametrizer
from config import SimulationConfig
from logging_config import get_logger
from utils.sweep_runner import run_sweep_simulation

logger = get_logger("VariabilitySweep")


if __name__ == "__main__":
    BASE_CONFIG = Path("results/ref_S1P_fixed/simulation_config.json")
    OUTPUT_DIR = Path("results/population_variability_sweep")
    
    cfg = SimulationConfig(str(BASE_CONFIG))

    base_config_dict = json.loads(BASE_CONFIG.read_text())
    
    params = {
        "DATASET_MODE": ["full", "shape_only", "material_only"]
    }

    
    sweep = ParameterSweep(params, OUTPUT_DIR)
    parametrizer = Parametrizer(sweep, lambda p, path: run_sweep_simulation(p, path, base_config_dict, full_dataset=True), verbose=True)
    
    result = parametrizer.run()
    
    logger.info(f"Complete: {len(result.param_points)} runs")
    logger.info(f"Results: {OUTPUT_DIR}/sweep_results.csv")
