"""Default configuration values and helpers for the eigenstiffness simulation."""
from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


DEFAULT_CONFIG = {
    # Paths
    "TEMPLATE_MESH_PATH": "data/pelvic.vtk",
    "DOMAIN_MESH_PATH": "anatomy_data/full_model_new.feb",
    "SHAPE_DATA_PATH": "data/X.npy",
    "DENSITY_DATA_PATH": "data/HA.npy",
    "RESULTS_DIR": "results/ref_S1P_fixed_new2_material_only",
    
    # Material properties
    "BONE_POISSON_RATIO": 0.3,
    "BONE_MODULUS_ALPHA": 10200.0,
    "BONE_MODULUS_BETA": 2.0,
    "BONE_MODULUS_THRESHOLD": 0.486,
    "SIJ_CARTILAGE_MODULUS": 1.0,
    "SIJ_CARTILAGE_POISSON_RATIO": 0.45,
    "SYMPHYSIS_MODULUS": 1.0,
    "SYMPHYSIS_POISSON_RATIO": 0.45,
    "MIN_DENSITY_VALUE": 0.1,
    
    # Names must match discrete sets in FEBio model
    "LIGAMENTS": {
        "symphisys": {"stiffness": 1000.0, "pretension": 20.0},
        "right_anterior_SIJ_ligaments": {"stiffness": 1400.0, "pretension": 20.0},
        "right_posterior_SIJ_ligaments": {"stiffness": 5600.0, "pretension": 20.0},
        "right_INL": {"stiffness": 200.0, "pretension": 20.0},
        "right_SS": {"stiffness": 3000.0, "pretension": 20.0},
        "right_ST": {"stiffness": 3000.0, "pretension": 20.0},
        "left_anterior_SIJ_ligaments": {"stiffness": 1400.0, "pretension": 20.0},
        "left_posterior_SIJ_ligaments": {"stiffness": 5600.0, "pretension": 20.0},
        "left_INL": {"stiffness": 200.0, "pretension": 20.0},
        "left_SS": {"stiffness": 3000.0, "pretension": 20.0},
        "left_ST": {"stiffness": 3000.0, "pretension": 20.0},
    },
    
    # Solver settings
    "NUM_EIGENVALUES": 15,
    "EIGENVALUE_TARGET": 1e-6,
    "SURFACE_PENALTY_GAMMA": 1e6,
    "BOUNDARY_FIXED": "S1_facet",
    "SOLVER_RTOL": 1e-10,
    "SOLVER_ATOL": 1e-12,
    "SOLVER_MAX_ITERS": 1000,
    "QUADRATURE_DEGREE": 6,
    "RIGID_MODE_TOLERANCE": 1e-8,
    "LIGAMENT_MAPPING_TOLERANCE": 0.2,
    
    # Dataset settings
    "FULL_DATASET_FLAG": True,
    "DATASET_MODE": "material_only", # options: material_only, shape_only, full
    "STATIC_ANAL": True,
    "COMPUTE_SENSITIVITIES": True,
    "SAVE_SIJ_STATS": True,
    "VERBOSE": True,
    
    # RBF interpolation - data mapping (shape/density)
    "RBF_DATA_SMOOTHING": 50.0,
    "RBF_DATA_NEIGHBORS": 10,
    
    # RBF interpolation - solver (phi interpolation in solvers)
    "RBF_SOLVER_SMOOTHING": 0.0,
    "RBF_SOLVER_NEIGHBORS": 10,
}


class SimulationConfig:
    """Manage simulation settings loaded from defaults, dicts, or JSON files."""

    def __init__(self, config_path: str | dict | None = None):
        """Merge defaults with overrides from a dict or JSON file."""
        self.config = DEFAULT_CONFIG.copy()
        
        if isinstance(config_path, dict):
            self.config_path = None
            self.config.update(config_path)
        elif config_path:
            self.config_path = Path(config_path)
            if self.config_path.exists():
                with open(self.config_path) as f:
                    user_config = json.load(f)
                self.config.update(user_config)
                logger.info("Loaded config from: %s", self.config_path)
        else:
            self.config_path = None

    def get(self, key: str, default=None):
        """Return config value for key or default if missing."""
        return self.config.get(key, default)

    def __getitem__(self, key: str):
        """Return config value for key."""
        return self.config[key]

    def __setitem__(self, key: str, value):
        """Set config value for key."""
        self.config[key] = value

    def save(self, path: str | None = None):
        """Write configuration to JSON at provided path or original source."""
        save_path = Path(path) if path else self.config_path
        if not save_path:
            raise ValueError("No path specified for saving config")
        
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        logger.info("Saved config to: %s", save_path)

    def validate(self) -> bool:
        """Return True when required input files exist on disk."""
        required_files = [
            "TEMPLATE_MESH_PATH",
            "DOMAIN_MESH_PATH", 
            "SHAPE_DATA_PATH",
            "DENSITY_DATA_PATH",
        ]
        
        for key in required_files:
            path = Path(self.config[key])
            if not path.exists():
                logger.error("Required file not found: %s = %s", key, path)
                return False
        
        logger.info("Configuration validation passed")
        return True

    def ensure_results_dir(self) -> Path:
        """Create results directory if needed and return its Path object."""
        results_dir = Path(self.config["RESULTS_DIR"])
        results_dir.mkdir(parents=True, exist_ok=True)
        return results_dir
