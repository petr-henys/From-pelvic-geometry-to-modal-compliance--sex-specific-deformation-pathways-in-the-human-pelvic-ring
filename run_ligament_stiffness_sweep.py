#!/usr/bin/env python3
"""Ligament sensitivity analysis using generic parameter sweep framework."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from itertools import product

sys.path.insert(0, str(Path(__file__).resolve().parent))

from parametrizer import ParameterSweep, Parametrizer
from config import SimulationConfig
from logging_config import get_logger
from utils.sweep_runner import run_sweep_simulation

logger = get_logger("LigamentSweep")


if __name__ == "__main__":
    BASE_CONFIG = Path("results/ref_S1P_fixed/simulation_config.json")
    OUTPUT_DIR = Path("results/ligament_stiffness_sweep")
    
    cfg = SimulationConfig(str(BASE_CONFIG))
    baseline = list(cfg.DEFAULT_LIGAMENT_STIFFNESSES)
    base_config_dict = json.loads(BASE_CONFIG.read_text())
    
    # Full factorial: 6 ligaments × 3 levels = 729 runs
    multipliers = [0.5, 1.0, 1.5]
    variations = []
    
    for mult_combo in product(multipliers, repeat=6):
        variant = baseline.copy()
        for i in range(6):
            variant[i] = baseline[i] * mult_combo[i]
            variant[i + 6] = baseline[i + 6] * mult_combo[i]
        variations.append(variant)
    
    params = {"LIGAMENT_STIFFNESSES": variations}
    
    logger.info(f"Baseline: {baseline}")
    logger.info(f"Total runs: {len(variations)}")
    
    sweep = ParameterSweep(params, OUTPUT_DIR)
    parametrizer = Parametrizer(sweep, lambda p, path: run_sweep_simulation(p, path, base_config_dict), verbose=True)
    
    result = parametrizer.run()
    
    logger.info(f"Complete: {len(result.param_points)} runs")
    logger.info(f"Results: {OUTPUT_DIR}/sweep_results.csv")
