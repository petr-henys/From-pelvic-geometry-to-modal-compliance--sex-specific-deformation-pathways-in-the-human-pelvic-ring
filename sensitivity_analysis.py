#!/usr/bin/env python3
"""Multi-parameter sensitivity dashboard.

The dashboard focuses on dimensionless modal sensitivities so that values
represent ``d log(lambda) / d log(parameter)``. This makes ligament stiffness,
pretension, cartilage, and bone material parameters directly comparable and
clarifies whether the reported magnitudes are logarithmic (they are).

Main panels:
- A–E: heatmaps for ligament stiffness, ligament pretension, cartilage, bone
  alpha, and bone beta sensitivities across modes.
- F: split violins (male vs female) showing total ligament stiffness response
  per patient and ligament.
- G: sex-stratified symmetry violins (left vs right stiffness) to expose
  asymmetry.
- H–I: age and allometry trends for ligament stiffness totals.

Exports include per-patient totals, statistical summaries, symmetry metrics,
and heatmap medians for each parameter family.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy import stats
import zarr

from utils.plot_utils import (
    ANNOT_SIZE,
    FULL_WIDTH,
    ROW_H,
    setup_plot_style,
    MALE_COLOR,
    FEMALE_COLOR,
    format_mode_label_short,
)
from utils.data_utils import (
    load_patient_metadata,
    standardize_sex_column,
    add_age_groups,
)


# ---------------------------------------------------------------------------
# Paths and global constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

SIMULATION_DIR = PROJECT_ROOT / "results" / "ref_S1P_fixed"
DATA_DIR = SIMULATION_DIR / "data"
DEMOGRAPHY_PATH = DATA_DIR / "demography.xlsx"
EIGEN_DATA_PATH = DATA_DIR / "eigen_data.xlsx"
LIGAMENT_PATH = DATA_DIR / "ligament_sensitivity.xlsx"
PRETENSION_PATH = DATA_DIR / "pretension_sensitivity.xlsx"
CARTILAGE_PATH = DATA_DIR / "cartilage_sensitivity.xlsx"
BONE_ALPHA_PATH = DATA_DIR / "bone_alpha_sensitivities.zarr"
BONE_BETA_PATH = DATA_DIR / "bone_beta_sensitivities.zarr"
SIMULATION_METADATA_PATH = SIMULATION_DIR / "simulation_metadata.json"

OUTPUT_DIR = SIMULATION_DIR / "analysis" / "ligament_sensitivity_dashboard"
FIGURE_PATH = OUTPUT_DIR / "ligament_sensitivity_dashboard"

NUM_MODES = 0  # Will be set dynamically from data
MIN_GROUP_SIZE = 10
ALPHA = 0.05

DISPLAY_SCALE = 1e3  # convert dimensionless totals to x10^-3 units for violins/bars
DISPLAY_UNIT_LABEL = r"Total |d log $\lambda$/d log $k$| (×10$^{-3}$)"
HEATMAP_COLORBAR_LABEL = r"log10(|d log $\lambda$/d log param|)"

AGE_BINS = [0, 30, 45, 60, 80, 110]
AGE_LABELS = ["<30", "30-45", "45-60", "60-80", "80+"]

# ---------------------------------------------------------------------------
# Sensitivity metadata helpers
# ---------------------------------------------------------------------------

BONE_ALPHA_LABEL = "Bone modulus α"
BONE_BETA_LABEL = "Bone modulus β"

LIGAMENT_ORDER: list[str] = []
LIGAMENT_STIFFNESS_BASELINE: dict[object, float] = {}
LIGAMENT_PRETENSION_BASELINE: dict[object, float] = {}
CARTILAGE_BASELINES: dict[object, float] = {}
BONE_ALPHA_BASELINE: float = 0.0
BONE_BETA_BASELINE: float = 0.0


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SensitivityFrame:
    """Container for a long-form sensitivity table."""

    category: str
    frame: pd.DataFrame


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------


def _load_modes_table() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load eigenvalue and permutation tables from eigen_data.xlsx."""

    modes = pd.read_excel(EIGEN_DATA_PATH, sheet_name="modes")
    permutations = pd.read_excel(EIGEN_DATA_PATH, sheet_name="permutations")
    return modes, permutations


def _melt_modes_table(modes: pd.DataFrame) -> pd.DataFrame:
    mode_cols = [col for col in modes.columns if col.startswith("eig_")]
    long = modes.melt(
        id_vars="patient_id",
        value_vars=mode_cols,
        var_name="mode",
        value_name="eigenvalue",
    )
    long["mode"] = long["mode"].str.extract(r"(\d+)").astype(int)
    return long


def load_simulation_metadata() -> dict:
    """Load simulation metadata JSON."""

    if not SIMULATION_METADATA_PATH.exists():
        raise FileNotFoundError(f"Missing simulation metadata: {SIMULATION_METADATA_PATH}")
    return json.loads(SIMULATION_METADATA_PATH.read_text())


_LIGAMENT_NAME_MAP = {
    "symphisys": "Symphysis",
    "anterior_sij_ligaments": "Anterior_SIJ",
    "posterior_sij_ligaments": "Posterior_SIJ",
    "inl": "Interosseous",
    "ss": "Sacrospinous",
    "st": "Sacrotuberous",
}


_CARTILAGE_NAME_MAP = {
    "sijcartilage": "SIJ_Cartilage",
    "pubicsymphysis": "Pubic_Symphysis",
}


def _structure_display_name(structure: str) -> str:
    mapping = {
        "SIJ_Cartilage": "SIJ cartilage",
        "Pubic_Symphysis": "Pubic symphysis",
    }
    return mapping.get(structure, structure.replace("_", " "))


def _structure_side_display(structure: str, side: str) -> str:
    base = _structure_display_name(structure)
    if structure == "Symphysis":
        base = "Symphysis"
    if side in {"midline", "global"}:
        return base
    return f"{base} ({side[0].upper()})"


def _record_baseline(store: dict[object, float], structure: str, side: str, value: float | None) -> None:
    if value is None:
        return
    store[(structure, side)] = float(value)
    store.setdefault(structure, float(value))


def _parse_ligament_sheet_name(sheet_name: str) -> tuple[str, str]:
    name = sheet_name
    lower = name.lower()
    mirror = False
    if lower.startswith("mirror_"):
        mirror = True
        lower = lower[len("mirror_") :]
        name = name[len("mirror_") :]
    side = "midline"
    base = lower
    if lower.startswith("right_"):
        side = "left" if mirror else "right"
        base = lower[len("right_") :]
    elif lower.startswith("left_"):
        side = "left"
        base = lower[len("left_") :]
    structure = _LIGAMENT_NAME_MAP.get(base, base.replace("_", " ").title().replace(" ", "_"))
    return structure, side


def _parse_cartilage_sheet_name(sheet_name: str) -> tuple[str, str]:
    lower = sheet_name.lower()
    side = "midline"
    base = lower
    if lower.endswith("left"):
        side = "left"
        base = lower[:-4]
    elif lower.endswith("right"):
        side = "right"
        base = lower[:-5]
    structure = _CARTILAGE_NAME_MAP.get(base, base.replace("_", " ").title().replace(" ", "_"))
    return structure, side


def _load_workbook_sensitivities(
    path: Path,
    category: str,
    parser: Callable[[str], tuple[str, str]],
) -> tuple[pd.DataFrame, int]:
    """Load a multi-sheet workbook and return long-form sensitivities and mode count."""

    if not path.exists():
        raise FileNotFoundError(f"Sensitivity workbook not found: {path}")

    excel = pd.ExcelFile(path)
    frames: list[pd.DataFrame] = []
    num_modes: int | None = None

    for sheet_name in excel.sheet_names:
        frame = excel.parse(sheet_name)
        mode_cols = sorted(
            [col for col in frame.columns if col.startswith("sens_mode_")],
            key=lambda col: int(col.split("_")[-1]),
        )
        if not mode_cols:
            continue
        if num_modes is None:
            num_modes = len(mode_cols)
        elif len(mode_cols) != num_modes:
            raise ValueError(
                f"Sheet '{sheet_name}' in {path.name} has {len(mode_cols)} modes; expected {num_modes}."
            )
        structure, side = parser(sheet_name)
        long = frame.melt(
            id_vars="patient_id",
            value_vars=mode_cols,
            var_name="mode",
            value_name="sensitivity",
        )
        long["mode"] = long["mode"].str.extract(r"(\d+)").astype(int)
        long["structure"] = structure
        long["side"] = side
        long["category"] = category
        frames.append(long)

    if not frames:
        raise ValueError(f"No sensitivity data found in workbook {path}")

    return pd.concat(frames, ignore_index=True), int(num_modes)


def _align_zarr_sensitivities(
    path: Path,
    patient_ids: list[str],
    permutations: pd.DataFrame,
    category: str,
    label: str,
    num_modes: int,
) -> pd.DataFrame:
    """Load zarr sensitivities and align modes using permutation indices."""

    store = zarr.open(str(path))
    data = np.asarray(store["data"][:], dtype=float)
    perm_cols = [col for col in permutations.columns if col.startswith("perm_")]
    perm_matrix = permutations[perm_cols].to_numpy(dtype=int)
    if num_modes > perm_matrix.shape[1]:
        raise ValueError(
            f"Requested {num_modes} modes but permutation table provides only {perm_matrix.shape[1]}."
        )

    if data.shape[1] < perm_matrix.max(initial=0) + 1:
        raise ValueError(
            f"Zarr dataset {path.name} has fewer modes than required by permutations"
        )

    aligned = np.full((data.shape[0], num_modes), np.nan, dtype=float)
    for i in range(perm_matrix.shape[0]):
        for j in range(num_modes):
            idx = perm_matrix[i, j]
            if idx >= 0:
                aligned[i, j] = data[i, idx]

    columns = [f"mode_{idx}" for idx in range(1, num_modes + 1)]
    aligned_df = pd.DataFrame(aligned, columns=columns)
    aligned_df.insert(0, "patient_id", patient_ids)

    long = aligned_df.melt(
        id_vars="patient_id",
        value_vars=columns,
        var_name="mode",
        value_name="sensitivity",
    )
    long["mode"] = long["mode"].str.extract(r"(\d+)").astype(int)
    long["structure"] = label
    long["side"] = "global"
    long["category"] = category
    return long


def load_all_sensitivities() -> tuple[dict[str, SensitivityFrame], int]:
    """Load all sensitivity datasets in long form."""

    modes_table, perm_table = _load_modes_table()
    patient_ids = modes_table["patient_id"].astype(str).tolist()

    ligament_df, num_modes = _load_workbook_sensitivities(
        LIGAMENT_PATH,
        "ligament_stiffness",
        _parse_ligament_sheet_name,
    )
    pretension_df, num_modes_pret = _load_workbook_sensitivities(
        PRETENSION_PATH,
        "ligament_pretension",
        _parse_ligament_sheet_name,
    )
    if num_modes_pret != num_modes:
        raise ValueError("Ligament pretension workbook mode count does not match ligament stiffness.")
    cartilage_df, num_modes_cart = _load_workbook_sensitivities(
        CARTILAGE_PATH,
        "cartilage",
        _parse_cartilage_sheet_name,
    )
    if num_modes_cart != num_modes:
        raise ValueError("Cartilage workbook mode count does not match ligament stiffness.")

    bone_alpha_df = _align_zarr_sensitivities(
        BONE_ALPHA_PATH,
        patient_ids,
        perm_table,
        "bone_alpha",
        BONE_ALPHA_LABEL,
        num_modes,
    )
    bone_beta_df = _align_zarr_sensitivities(
        BONE_BETA_PATH,
        patient_ids,
        perm_table,
        "bone_beta",
        BONE_BETA_LABEL,
        num_modes,
    )

    return (
        {
            "ligament_stiffness": SensitivityFrame("ligament_stiffness", ligament_df),
            "ligament_pretension": SensitivityFrame("ligament_pretension", pretension_df),
            "cartilage": SensitivityFrame("cartilage", cartilage_df),
            "bone_alpha": SensitivityFrame("bone_alpha", bone_alpha_df),
            "bone_beta": SensitivityFrame("bone_beta", bone_beta_df),
        },
        num_modes,
    )


# ---------------------------------------------------------------------------
# Dimensionless conversion and aggregation
# ---------------------------------------------------------------------------


def _baseline_for_row(category: str, structure: str, side: str) -> float:
    if category == "ligament_stiffness":
        return LIGAMENT_STIFFNESS_BASELINE.get((structure, side), LIGAMENT_STIFFNESS_BASELINE.get(structure, np.nan))
    if category == "ligament_pretension":
        return LIGAMENT_PRETENSION_BASELINE.get((structure, side), LIGAMENT_PRETENSION_BASELINE.get(structure, np.nan))
    if category == "cartilage":
        return CARTILAGE_BASELINES.get((structure, side), CARTILAGE_BASELINES.get(structure, np.nan))
    if category == "bone_alpha":
        return BONE_ALPHA_BASELINE
    if category == "bone_beta":
        return BONE_BETA_BASELINE
    return np.nan


def attach_dimensionless_values(
    sensitivities: dict[str, SensitivityFrame],
    eigen_long: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Merge eigenvalues and compute dimensionless sensitivities for each set."""

    result: dict[str, pd.DataFrame] = {}
    for key, container in sensitivities.items():
        frame = container.frame.copy()
        merged = frame.merge(
            eigen_long,
            on=["patient_id", "mode"],
            how="left",
            validate="many_to_one",
        )
        if merged["eigenvalue"].isna().any():
            raise ValueError(f"Missing eigenvalues when merging category '{key}'")

        merged["baseline_value"] = merged.apply(
            lambda row: _baseline_for_row(key, str(row["structure"]), str(row["side"])),
            axis=1,
        )
        if merged["baseline_value"].isna().any():
            raise ValueError(f"Baseline not defined for some rows in category '{key}'")

        baseline = merged["baseline_value"].to_numpy(dtype=float)
        sens = merged["sensitivity"].to_numpy(dtype=float)
        eigen = merged["eigenvalue"].to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            dimensionless = np.divide(
                baseline * sens,
                eigen,
                out=np.full_like(sens, np.nan, dtype=float),
                where=eigen != 0.0,
            )

        merged["dimensionless"] = dimensionless
        merged["abs_dimensionless"] = np.abs(dimensionless)
        result[key] = merged

    return result


def summarize_ligament_totals(
    ligament_df: pd.DataFrame,
) -> pd.DataFrame:
    """Per-patient ligament totals (absolute dimensionless response)."""

    totals = (
        ligament_df.groupby(["patient_id", "structure"], as_index=False)["abs_dimensionless"]
        .sum()
        .rename(columns={"abs_dimensionless": "total_response"})
    )
    totals["total_response_scaled"] = totals["total_response"] * DISPLAY_SCALE
    return totals


def summarize_ligament_by_side(ligament_df: pd.DataFrame) -> pd.DataFrame:
    """Per-patient ligament totals for symmetry analysis (keep side)."""

    return (
        ligament_df.groupby(["patient_id", "structure", "side"], as_index=False)["abs_dimensionless"]
        .sum()
        .rename(columns={"abs_dimensionless": "total_response"})
    )


def compute_symmetry_ratios(
    by_side: pd.DataFrame,
    patient_meta: pd.DataFrame,
) -> pd.DataFrame:
    """Compute asymmetry ratios (left-right)/(left+right) per ligament."""

    usable = by_side[by_side["side"].isin(["left", "right"])]
    pivot = (
        usable.pivot_table(
            index=["patient_id", "structure"],
            columns="side",
            values="total_response",
            aggfunc="first",
        )
        .reset_index()
    )
    if pivot.empty:
        return pd.DataFrame(columns=["patient_id", "structure", "ratio", "diff", "sex"])

    pivot["left"] = pivot.get("left", np.nan)
    pivot["right"] = pivot.get("right", np.nan)
    pivot.dropna(subset=["left", "right"], inplace=True)

    denom = pivot["left"] + pivot["right"]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.divide(pivot["left"] - pivot["right"], denom)

    symmetry = pivot.assign(
        ratio=ratio,
        diff=pivot["left"] - pivot["right"],
    )
    merged = symmetry.merge(patient_meta[["patient_id", "sex"]], on="patient_id", how="left")
    return merged


# ---------------------------------------------------------------------------
# Statistical summaries
# ---------------------------------------------------------------------------


def compute_sex_statistics(per_patient: pd.DataFrame) -> pd.DataFrame:
    """Mann–Whitney comparison of male vs female ligament totals."""

    records: list[dict[str, object]] = []
    for ligament in LIGAMENT_ORDER:
        subset = per_patient[per_patient["structure"] == ligament]
        subset = subset.dropna(subset=["sex", "total_response"])

        male_vals = subset[subset["sex"] == "M"]["total_response"].to_numpy()
        female_vals = subset[subset["sex"] == "F"]["total_response"].to_numpy()

        record: dict[str, object] = {
            "structure": ligament,
            "n_male": int(male_vals.size),
            "n_female": int(female_vals.size),
            "median_male_scaled": float(np.median(male_vals) * DISPLAY_SCALE)
            if male_vals.size
            else np.nan,
            "median_female_scaled": float(np.median(female_vals) * DISPLAY_SCALE)
            if female_vals.size
            else np.nan,
        }

        if male_vals.size >= MIN_GROUP_SIZE and female_vals.size >= MIN_GROUP_SIZE:
            _, p_value = stats.mannwhitneyu(
                male_vals,
                female_vals,
                alternative="two-sided",
                method="auto",
            )
            record["p_value"] = float(p_value)
            record["median_difference_scaled"] = (
                record["median_male_scaled"] - record["median_female_scaled"]
            )
        else:
            record["p_value"] = np.nan
            record["median_difference_scaled"] = np.nan

        records.append(record)

    return pd.DataFrame(records)


def compute_age_statistics(per_patient: pd.DataFrame) -> pd.DataFrame:
    """Spearman correlation and slope per decade for ligament totals."""

    records: list[dict[str, object]] = []
    for ligament in LIGAMENT_ORDER:
        subset = per_patient[per_patient["structure"] == ligament]
        subset = subset.dropna(subset=["age", "total_response"])
        values = subset["total_response"].to_numpy(dtype=float)
        tages = subset["age"].to_numpy(dtype=float)

        record: dict[str, object] = {
            "structure": ligament,
            "n_samples": int(values.size),
        }

        if values.size >= MIN_GROUP_SIZE:
            rho, p_value = stats.spearmanr(tages, values)
            lin = stats.linregress(tages, values)
            record["spearman_rho"] = float(rho)
            record["spearman_p_value"] = float(p_value)
            record["slope_per_decade_scaled"] = float(lin.slope * 10.0 * DISPLAY_SCALE)
        else:
            record["spearman_rho"] = np.nan
            record["spearman_p_value"] = np.nan
            record["slope_per_decade_scaled"] = np.nan

        records.append(record)

    return pd.DataFrame(records)


def compute_scale_statistics(per_patient: pd.DataFrame) -> pd.DataFrame:
    """Allometry (patient scale) associations for ligament totals."""

    records: list[dict[str, object]] = []
    for ligament in LIGAMENT_ORDER:
        subset = per_patient[per_patient["structure"] == ligament]
        subset = subset.dropna(subset=["patient_scale", "total_response"])
        values = subset["total_response"].to_numpy(dtype=float)
        scales = subset["patient_scale"].to_numpy(dtype=float)

        record: dict[str, object] = {
            "structure": ligament,
            "n_samples": int(values.size),
        }

        if values.size >= MIN_GROUP_SIZE:
            rho, p_value = stats.spearmanr(scales, values)
            lin = stats.linregress(scales, values)
            record["spearman_rho"] = float(rho)
            record["spearman_p_value"] = float(p_value)
            record["slope_per_10pct_scaled"] = float(lin.slope * 0.1 * DISPLAY_SCALE)
        else:
            record["spearman_rho"] = np.nan
            record["spearman_p_value"] = np.nan
            record["slope_per_10pct_scaled"] = np.nan

        records.append(record)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _compute_heatmap_matrix(
    frame: pd.DataFrame,
    row_labels: list[str],
    *,
    group_by_side: bool,
    label_func: Callable[[str, str], str] | None = None,
) -> pd.DataFrame:
    data = frame.copy()
    labels: list[str] = []
    if group_by_side:
        if label_func is None:
            label_func = lambda struct, side: f"{struct} ({side[0].upper()})"
        labels = [label_func(struct, side) for struct, side in zip(data["structure"], data["side"])]
    else:
        labels = list(map(str, data["structure"]))

    data["row_label"] = labels
    pivot = (
        data.groupby(["row_label", "mode"], as_index=False)["abs_dimensionless"]
        .median()
        .pivot(index="row_label", columns="mode", values="abs_dimensionless")
    )
    pivot = pivot.reindex(row_labels)
    pivot.index.name = "row_label"
    return pivot


def cartilage_display_order(cartilage_df: pd.DataFrame) -> list[str]:
    labels: list[str] = []
    if cartilage_df.empty:
        return labels
    for structure, side in cartilage_df[["structure", "side"]].drop_duplicates().itertuples(index=False):
        label = _structure_side_display(structure, side)
        if label not in labels:
            labels.append(label)
    return labels


def plot_heatmap(ax: plt.Axes, heatmap_df: pd.DataFrame, title: str) -> None:
    values = heatmap_df.to_numpy(dtype=float)
    if values.size == 0:
        ax.set_title(title, fontweight="bold")
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])
        return
    values[values <= 0] = np.nan
    with np.errstate(divide="ignore"):
        log_values = np.log10(values)

    ax.imshow(
        log_values,
        aspect="auto",
        interpolation="nearest",
        cmap="magma",
    )

    ax.set_xticks(np.arange(NUM_MODES))
    ax.set_xticklabels(
        [format_mode_label_short(idx, one_based=True) for idx in range(NUM_MODES)],
        rotation=45,
        ha="right",
    )
    if heatmap_df.index.size:
        ax.set_yticks(np.arange(heatmap_df.index.size))
        ax.set_yticklabels(heatmap_df.index)
    else:
        ax.set_yticks([])
    ax.set_xlabel("Mode")
    ax.set_title(title, fontweight="bold")

    finite = np.isfinite(log_values)
    if finite.any():
        norm = Normalize(vmin=np.nanmin(log_values), vmax=np.nanmax(log_values))
    else:
        norm = Normalize(vmin=0.0, vmax=1.0)
    plt.colorbar(ScalarMappable(norm=norm, cmap="magma"), ax=ax, label=HEATMAP_COLORBAR_LABEL)


def _draw_split_violin(
    ax: plt.Axes,
    position: float,
    left_values: np.ndarray,
    right_values: np.ndarray,
    color_left: str,
    color_right: str,
    width: float = 0.8,
) -> None:
    for values, color, clip_side in [
        (left_values, color_left, lambda verts: np.clip(verts, -np.inf, position)),
        (right_values, color_right, lambda verts: np.clip(verts, position, np.inf)),
    ]:
        if values.size == 0:
            continue
        parts = ax.violinplot(
            [values],
            positions=[position],
            widths=width,
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )
        body = parts["bodies"][0]
        verts = body.get_paths()[0].vertices[:, 0]
        body.get_paths()[0].vertices[:, 0] = clip_side(verts)
        body.set_facecolor(color)
        body.set_edgecolor("black")
        body.set_linewidth(0.6)
        body.set_alpha(0.75)

    for offset_idx, (values, offset) in enumerate([(left_values, -0.35), (right_values, 0.35)]):
        if values.size == 0:
            continue
        rng = np.random.default_rng(123 + int(position * 100) + offset_idx)
        low, high = sorted((offset * 0.2, offset * 0.8))
        jitter = rng.uniform(low, high, size=values.size)
        scatter_x = np.full(values.size, position) + jitter
        ax.scatter(scatter_x, values, s=10, color="white", alpha=0.45, linewidths=0)

    if left_values.size:
        ax.hlines(np.median(left_values), position - 0.35, position - 0.02, colors=color_left, linewidth=1.6)
    if right_values.size:
        ax.hlines(np.median(right_values), position + 0.02, position + 0.35, colors=color_right, linewidth=1.6)


def plot_sex_violins(ax: plt.Axes, per_patient: pd.DataFrame) -> None:
    positions = np.arange(len(LIGAMENT_ORDER), dtype=float)
    counts: list[tuple[int, int]] = []
    for idx, ligament in enumerate(LIGAMENT_ORDER):
        subset = per_patient[per_patient["structure"] == ligament]
        male = subset[subset["sex"] == "M"]["total_response_scaled"].to_numpy()
        female = subset[subset["sex"] == "F"]["total_response_scaled"].to_numpy()
        _draw_split_violin(ax, float(idx), male, female, MALE_COLOR, FEMALE_COLOR)
        counts.append((male.size, female.size))

    ax.relim()
    ax.autoscale_view()
    ymin, ymax = ax.get_ylim()
    for position, (n_male, n_female) in zip(positions, counts):
        label = f"M{n_male}/F{n_female}"
        ax.text(
            position,
            ymin + 0.05 * (ymax - ymin),
            label,
            ha="center",
            fontsize=ANNOT_SIZE,
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(LIGAMENT_ORDER, rotation=20, ha="right")
    ax.set_ylabel(DISPLAY_UNIT_LABEL)
    ax.set_title("Sex differences (ligament stiffness)", fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    legend_handles = [
        plt.Line2D([0], [0], marker="s", color="w", label="Male", markerfacecolor=MALE_COLOR, markersize=8),
        plt.Line2D([0], [0], marker="s", color="w", label="Female", markerfacecolor=FEMALE_COLOR, markersize=8),
    ]
    ax.legend(handles=legend_handles, loc="upper left", frameon=True)


def plot_symmetry_violins(ax: plt.Axes, symmetry_df: pd.DataFrame) -> None:
    if symmetry_df.empty:
        ax.text(0.5, 0.5, "No symmetry data", ha="center", va="center")
        return

    ligaments = [lig for lig in LIGAMENT_ORDER if lig != "Symphysis"]
    positions = np.arange(len(ligaments))
    for idx, ligament in enumerate(ligaments):
        subset = symmetry_df[symmetry_df["structure"] == ligament]
        male = subset[subset["sex"] == "M"]["ratio"].to_numpy()
        female = subset[subset["sex"] == "F"]["ratio"].to_numpy()
        _draw_split_violin(ax, float(idx), male, female, MALE_COLOR, FEMALE_COLOR, width=0.7)

    ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(positions)
    ax.set_xticklabels(ligaments, rotation=20, ha="right")
    ax.set_ylabel("Asymmetry ratio (L-R)/(L+R)")
    ax.set_title("Ligament stiffness symmetry by sex", fontweight="bold")
    ax.set_ylim(-1.05, 1.05)
    ax.grid(axis="y", alpha=0.25)


def _annotation_offset(values: list[float]) -> float:
    finite = [abs(v) for v in values if np.isfinite(v)]
    if not finite:
        return 1.0
    return max(finite) * 0.08 if max(finite) > 0 else 1.0


def plot_slope_panel(ax: plt.Axes, stats_df: pd.DataFrame, ylabel: str, title: str) -> None:
    stats_map = stats_df.set_index("structure")
    values = [stats_map.at[lig, stats_df.columns[-1]] if lig in stats_map.index else np.nan for lig in LIGAMENT_ORDER]
    bars = ax.bar(np.arange(len(LIGAMENT_ORDER)), values, color="#6A9FB5", edgecolor="black", linewidth=0.8)
    ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--")

    ax.set_xticks(np.arange(len(LIGAMENT_ORDER)))
    ax.set_xticklabels(LIGAMENT_ORDER, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    if stats_df.empty:
        return

    offset = _annotation_offset(values)
    for bar, ligament in zip(bars, LIGAMENT_ORDER):
        if ligament not in stats_map.index:
            continue
        row = stats_map.loc[ligament]
        if not np.isfinite(row.get("spearman_rho", np.nan)):
            continue
        y_val = bar.get_height()
        va = "bottom" if y_val >= 0 else "top"
        adjust = offset if y_val >= 0 else -offset
        txt = f"rho={row['spearman_rho']:.2f}"
        p_val = row.get("spearman_p_value", np.nan)
        if np.isfinite(p_val):
            txt += f"\np={p_val:.3f}"
        ax.text(bar.get_x() + bar.get_width() / 2.0, y_val + adjust, txt, ha="center", va=va, fontsize=ANNOT_SIZE)


# ---------------------------------------------------------------------------
# Figure assembly
# ---------------------------------------------------------------------------


def build_figures(
    dimensionless_sets: dict[str, pd.DataFrame],
    per_patient: pd.DataFrame,
    symmetry_df: pd.DataFrame,
    sex_stats: pd.DataFrame,
    age_stats: pd.DataFrame,
    scale_stats: pd.DataFrame,
    cartilage_labels: list[str],
) -> None:
    setup_plot_style()
    fig = plt.figure(figsize=(FULL_WIDTH, 3 * 2 * ROW_H), constrained_layout=True)
    mosaic = [
        ["A", "B", "C"],
        ["D", "E", "F"],
        ["G", "H", "I"],
    ]
    axes = fig.subplot_mosaic(mosaic)

    ligament_heatmap = _compute_heatmap_matrix(
        dimensionless_sets["ligament_stiffness"],
        LIGAMENT_ORDER,
        group_by_side=False,
    )
    pretension_heatmap = _compute_heatmap_matrix(
        dimensionless_sets["ligament_pretension"],
        LIGAMENT_ORDER,
        group_by_side=False,
    )
    cartilage_heatmap = _compute_heatmap_matrix(
        dimensionless_sets["cartilage"],
        cartilage_labels,
        group_by_side=True,
        label_func=_structure_side_display,
    )
    bone_alpha_heatmap = _compute_heatmap_matrix(
        dimensionless_sets["bone_alpha"],
        [BONE_ALPHA_LABEL],
        group_by_side=False,
    )
    bone_beta_heatmap = _compute_heatmap_matrix(
        dimensionless_sets["bone_beta"],
        [BONE_BETA_LABEL],
        group_by_side=False,
    )

    plot_heatmap(axes["A"], ligament_heatmap, "Ligament stiffness sensitivities")
    plot_heatmap(axes["B"], pretension_heatmap, "Ligament pretension sensitivities")
    plot_heatmap(axes["C"], cartilage_heatmap, "Cartilage sensitivities")
    plot_heatmap(axes["D"], bone_alpha_heatmap, "Bone modulus α sensitivities")
    plot_heatmap(axes["E"], bone_beta_heatmap, "Bone modulus β sensitivities")
    plot_sex_violins(axes["F"], per_patient)
    plot_symmetry_violins(axes["G"], symmetry_df)
    plot_slope_panel(axes["H"], age_stats, "Slope per decade (×10$^{-3}$)", "Age effect")
    plot_slope_panel(axes["I"], scale_stats, "Slope per 10% scale (×10$^{-3}$)", "Allometry effect")

    axes["A"].text(-0.1, 1.05, "A", transform=axes["A"].transAxes, fontweight="bold")
    axes["B"].text(-0.1, 1.05, "B", transform=axes["B"].transAxes, fontweight="bold")
    axes["C"].text(-0.1, 1.05, "C", transform=axes["C"].transAxes, fontweight="bold")
    axes["D"].text(-0.1, 1.05, "D", transform=axes["D"].transAxes, fontweight="bold")
    axes["E"].text(-0.1, 1.05, "E", transform=axes["E"].transAxes, fontweight="bold")
    axes["F"].text(-0.1, 1.05, "F", transform=axes["F"].transAxes, fontweight="bold")
    axes["G"].text(-0.1, 1.05, "G", transform=axes["G"].transAxes, fontweight="bold")
    axes["H"].text(-0.1, 1.05, "H", transform=axes["H"].transAxes, fontweight="bold")
    axes["I"].text(-0.1, 1.05, "I", transform=axes["I"].transAxes, fontweight="bold")

    fig.savefig(FIGURE_PATH.with_suffix(".png"))
    fig.savefig(FIGURE_PATH.with_suffix(".pdf"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------


def export_tables(
    per_patient: pd.DataFrame,
    symmetry_df: pd.DataFrame,
    sex_stats: pd.DataFrame,
    age_stats: pd.DataFrame,
    scale_stats: pd.DataFrame,
    heatmaps: dict[str, pd.DataFrame],
) -> None:
    per_patient.to_csv(OUTPUT_DIR / "ligament_totals_dimensionless.csv", index=False)
    symmetry_df.to_csv(OUTPUT_DIR / "ligament_symmetry.csv", index=False)
    sex_stats.to_csv(OUTPUT_DIR / "ligament_sex_stats.csv", index=False)
    age_stats.to_csv(OUTPUT_DIR / "ligament_age_stats.csv", index=False)
    scale_stats.to_csv(OUTPUT_DIR / "ligament_scale_stats.csv", index=False)

    for name, matrix in heatmaps.items():
        out = matrix.reset_index().melt(id_vars="row_label", var_name="mode", value_name="median_abs_dimensionless")
        out.to_csv(OUTPUT_DIR / f"{name}_heatmap_medians.csv", index=False)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    metadata_json = load_simulation_metadata()
    config = metadata_json.get("config", {})
    ligament_config = config.get("LIGAMENTS", {})

    global LIGAMENT_ORDER
    global LIGAMENT_STIFFNESS_BASELINE
    global LIGAMENT_PRETENSION_BASELINE
    global CARTILAGE_BASELINES
    global BONE_ALPHA_BASELINE
    global BONE_BETA_BASELINE
    global NUM_MODES

    ligament_order: list[str] = []
    stiffness_baseline: dict[object, float] = {}
    pretension_baseline: dict[object, float] = {}
    for raw_name, params in ligament_config.items():
        structure, side = _parse_ligament_sheet_name(raw_name)
        if structure not in ligament_order:
            ligament_order.append(structure)
        _record_baseline(stiffness_baseline, structure, side, params.get("stiffness"))
        _record_baseline(pretension_baseline, structure, side, params.get("pretension"))
    if not ligament_order:
        ligament_order = ["Symphysis", "Anterior_SIJ", "Posterior_SIJ", "Interosseous", "Sacrospinous", "Sacrotuberous"]
    LIGAMENT_ORDER = ligament_order
    LIGAMENT_STIFFNESS_BASELINE = stiffness_baseline or {structure: 1.0 for structure in ligament_order}
    LIGAMENT_PRETENSION_BASELINE = pretension_baseline or {structure: 1.0 for structure in ligament_order}

    cartilage_baselines: dict[object, float] = {}
    _record_baseline(cartilage_baselines, "SIJ_Cartilage", "left", config.get("SIJ_CARTILAGE_MODULUS"))
    _record_baseline(cartilage_baselines, "SIJ_Cartilage", "right", config.get("SIJ_CARTILAGE_MODULUS"))
    _record_baseline(cartilage_baselines, "Pubic_Symphysis", "midline", config.get("SYMPHYSIS_MODULUS"))
    if not cartilage_baselines:
        _record_baseline(cartilage_baselines, "SIJ_Cartilage", "left", 1.0)
        _record_baseline(cartilage_baselines, "SIJ_Cartilage", "right", 1.0)
        _record_baseline(cartilage_baselines, "Pubic_Symphysis", "midline", 1.0)
    CARTILAGE_BASELINES = cartilage_baselines

    BONE_ALPHA_BASELINE = float(config.get("BONE_MODULUS_ALPHA", 0.0))
    BONE_BETA_BASELINE = float(config.get("BONE_MODULUS_BETA", 0.0))

    sensitivities, num_modes = load_all_sensitivities()
    NUM_MODES = num_modes

    modes_table, _ = _load_modes_table()
    eigen_long = _melt_modes_table(modes_table)
    dimensionless_sets = attach_dimensionless_values(sensitivities, eigen_long)

    if not LIGAMENT_ORDER:
        LIGAMENT_ORDER = list(dimensionless_sets["ligament_stiffness"]["structure"].unique())

    ligament_df = dimensionless_sets["ligament_stiffness"]
    per_patient = summarize_ligament_totals(ligament_df)
    per_patient_side = summarize_ligament_by_side(ligament_df)

    metadata = load_patient_metadata(DATA_DIR)
    if "scale" in metadata.columns:
        metadata = metadata.rename(columns={"scale": "patient_scale"})
    else:
        metadata["patient_scale"] = np.nan

    demography = pd.read_excel(DEMOGRAPHY_PATH)
    demography = standardize_sex_column(demography, col="sex")
    demography = add_age_groups(demography, age_col="age", bins=AGE_BINS, labels=AGE_LABELS)

    patient_meta = (
        metadata[["patient_id", "patient_scale"]]
        .merge(demography[["patient_id", "sex", "age", "age_group"]], on="patient_id", how="left")
    )

    per_patient = per_patient.merge(patient_meta, on="patient_id", how="left")
    symmetry_df = compute_symmetry_ratios(per_patient_side, patient_meta)

    sex_stats = compute_sex_statistics(per_patient)
    age_stats = compute_age_statistics(per_patient)
    scale_stats = compute_scale_statistics(per_patient)
    cartilage_labels = cartilage_display_order(dimensionless_sets["cartilage"])

    build_figures(
        dimensionless_sets,
        per_patient,
        symmetry_df,
        sex_stats,
        age_stats,
        scale_stats,
        cartilage_labels,
    )

    heatmaps = {
        "ligament_stiffness": _compute_heatmap_matrix(ligament_df, LIGAMENT_ORDER, group_by_side=False),
        "ligament_pretension": _compute_heatmap_matrix(dimensionless_sets["ligament_pretension"], LIGAMENT_ORDER, group_by_side=False),
        "cartilage": _compute_heatmap_matrix(
            dimensionless_sets["cartilage"],
            cartilage_labels,
            group_by_side=True,
            label_func=_structure_side_display,
        ),
        "bone_alpha": _compute_heatmap_matrix(dimensionless_sets["bone_alpha"], [BONE_ALPHA_LABEL], group_by_side=False),
        "bone_beta": _compute_heatmap_matrix(dimensionless_sets["bone_beta"], [BONE_BETA_LABEL], group_by_side=False),
    }

    export_tables(per_patient, symmetry_df, sex_stats, age_stats, scale_stats, heatmaps)

    print(f"Dashboard saved to {FIGURE_PATH.with_suffix('.png')} and {FIGURE_PATH.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
