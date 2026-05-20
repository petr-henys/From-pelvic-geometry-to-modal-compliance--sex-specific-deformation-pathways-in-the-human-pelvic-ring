#!/usr/bin/env python3
"""Cartilage modulus sensitivity analysis using generic parameter sweep framework."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from parametrizer import ParameterSweep, Parametrizer
from config import SimulationConfig
from logging_config import get_logger
from utils.sweep_runner import run_sweep_simulation

logger = get_logger("CartilageSweep")


if __name__ == "__main__":
    BASE_CONFIG = Path("results/ref_S1P_fixed/simulation_config.json")
    OUTPUT_DIR = Path("results/cartilage_modulus_sweep")
    
    cfg = SimulationConfig(str(BASE_CONFIG))
    base_sij_modulus = cfg["SIJ_CARTILAGE_MODULUS"]
    base_symphysis_modulus = cfg["SYMPHYSIS_MODULUS"]
    base_config_dict = json.loads(BASE_CONFIG.read_text())
    
    params = {
        "SIJ_CARTILAGE_MODULUS": [0.5 * base_sij_modulus, 
                                   1.0 * base_sij_modulus, 
                                   1.5 * base_sij_modulus],
        "SYMPHYSIS_MODULUS": [0.5 * base_symphysis_modulus,
                              1.0 * base_symphysis_modulus,
                              1.5 * base_symphysis_modulus],
        "SIJ_CARTILAGE_POISSON_RATIO": [0.1, 0.3, 0.45],
        "SYMPHYSIS_POISSON_RATIO": [0.1, 0.3, 0.45]
    }
    
    logger.info(f"Base SIJ cartilage modulus: {base_sij_modulus} MPa")
    logger.info(f"Base symphysis modulus: {base_symphysis_modulus} MPa")
    
    sweep = ParameterSweep(params, OUTPUT_DIR)
    parametrizer = Parametrizer(sweep, lambda p, path: run_sweep_simulation(p, path, base_config_dict), verbose=True)
    
    result = parametrizer.run()
    
    logger.info(f"Complete: {len(result.param_points)} runs")
    logger.info(f"Results: {OUTPUT_DIR}/sweep_results.csv")
