"""Configuration constants for spectral panel analysis."""
from __future__ import annotations

import sys
from pathlib import Path

# Project root = parent of this script's directory
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# ── Data paths ────────────────────────────────────────────────
DATA_DIR_FULL = REPO_ROOT / "results" / "ref_S1P_fixed_new2" / "data"
DATA_DIR_SHAPE = REPO_ROOT / "results" / "ref_S1P_fixed_new2_shape_only" / "data"
DATA_DIR_MATERIAL = REPO_ROOT / "results" / "ref_S1P_fixed_new2_material_only" / "data"

for _name, _path in (
    ("DATA_DIR_FULL", DATA_DIR_FULL),
    ("DATA_DIR_SHAPE", DATA_DIR_SHAPE),
    ("DATA_DIR_MATERIAL", DATA_DIR_MATERIAL),
):
    if not _path.exists():
        raise FileNotFoundError(f"{_name} not found: {_path}")
RAW_DATA_DIR = REPO_ROOT / "data"
DIMS_ZARR = RAW_DATA_DIR / "dimensions_ref.zarr"
DIMS_ZARR_PALPATION = RAW_DATA_DIR / "dimensions_michal.zarr"
LANDMARK_DIR = REPO_ROOT / "anatomy_data" / "birth_canal"
OUTLET_LANDMARK_DIR = REPO_ROOT / "anatomy_data" / "palpace"

# ── Output paths ──────────────────────────────────────────────
OUT_DIR = REPO_ROOT / "analysis_outputs"
FIG_DIR = OUT_DIR / "figures"
TAB_DIR = OUT_DIR / "tables"

# ── Analysis parameters ──────────────────────────────────────
N_MODES = 15
EPS_IN_PERCENTILE = 25  # percentile to set cluster threshold eps_in
EPS_DEG = 1e-6  # true degeneracy threshold

MORPHOLOGY_PROXY = "AnteriorPosteriorInletDiameter"

# Per-cluster morphology proxy override.  Clusters whose canonical
# coupling is outlet-dominated should be visualised against an outlet
# dimension rather than the default AP inlet diameter.
CLUSTER_MORPHOLOGY_PROXY: dict[str, str] = {
    "modes_5_6": "BispinousWidth",
}

MIN_CLUSTER_PREVALENCE = 0.03  # at least ~3 % of subjects
TOP_N_CLUSTERS = 5

# ── Figure settings ───────────────────────────────────────────
FIG_FORMATS = ("pdf",)          # vector only; PNG removed to save disk / time
FIG_FMT = "pdf"                 # primary format returned by _save_fig

# Import publication layout constants from central plot_utils
from utils.plot_utils import (  # noqa: E402
    DEFAULT_DPI as FIG_DPI,
    FULL_WIDTH as COL2_WIDTH,
    HALF_WIDTH as COL_WIDTH,
    COL15_WIDTH,
    GOLDEN,
    ROW_H,
    ROW_H_SMALL,
    PANEL_PAD,
)

SUBJECT_SUBSET = None  # None = all; int = first N
