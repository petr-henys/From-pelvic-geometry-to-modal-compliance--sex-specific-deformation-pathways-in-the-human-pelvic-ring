"""Run the eigenstiffness simulation with the default configuration."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is discoverable for package imports.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from simulation.simulation_core import EigenstiffnessSimulation

if __name__ == "__main__":
    simulation = EigenstiffnessSimulation()
    simulation.run()
