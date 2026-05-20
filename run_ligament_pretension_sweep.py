#!/usr/bin/env python3
"""Ligament pretension analysis using generic parameter sweep framework."""

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

logger = get_logger("PretensionSweep")


if __name__ == "__main__":
    BASE_CONFIG = Path("results/ref_S1P_fixed/simulation_config.json")
    OUTPUT_DIR = Path("results/ligament_pretension_sweep")
    
    cfg = SimulationConfig(str(BASE_CONFIG))
    base_config_dict = json.loads(BASE_CONFIG.read_text())
    
    # Pretension levels: 0, 50, 100
    pretension_levels = [0.0, 50.0, 100.0]
    variations = []
    
    # Full factorial: 6 ligament types × 3 levels = 729 runs
    for pretension_combo in product(pretension_levels, repeat=6):
        variant = [pretension_combo[i % 6] for i in range(12)]  # Duplicate for left/right
        variations.append(variant)
    
    params = {"LIGAMENT_PRETENSIONS": variations}
    
    logger.info(f"Pretension levels: {pretension_levels}")
    logger.info(f"Total runs: {len(variations)}")
    
    sweep = ParameterSweep(params, OUTPUT_DIR)
    parametrizer = Parametrizer(sweep, lambda p, path: run_sweep_simulation(p, path, base_config_dict), verbose=True)
    
    result = parametrizer.run()
    
    logger.info(f"Complete: {len(result.param_points)} runs")
    logger.info(f"Results: {OUTPUT_DIR}/sweep_results.csv")
