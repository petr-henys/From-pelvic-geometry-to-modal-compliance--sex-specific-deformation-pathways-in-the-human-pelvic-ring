#!/usr/bin/env python3
"""Reference ligament sensitivity analysis with normalized metrics."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.plot_utils import FULL_WIDTH, ROW_H, setup_plot_style, format_mode_label_short

# Paths and constants
RESULTS_DIR = PROJECT_ROOT / "results" / "ref_S1P_fixed_test"
REFERENCE_PATH = RESULTS_DIR / "reference_solution.npz"
METADATA_PATH = RESULTS_DIR / "simulation_metadata.json"
OUTPUT_DIR = RESULTS_DIR / "analysis" / "reference_sensitivity"

MIN_EIGENVALUE = 1.0e-12
HEATMAP_CMAP = "coolwarm"
FIG_SIZE = (FULL_WIDTH, 2 * ROW_H)


@dataclass(frozen=True)
class ReferenceSensitivityData:
    """Container for reference sensitivity inputs."""

    eigenvalues: np.ndarray
    stiffness_sensitivities: np.ndarray
    pretension_sensitivities: np.ndarray
    ligament_names: list[str]
    stiffness_values: np.ndarray
    pretension_values: np.ndarray


def load_reference_data(reference_path: Path, metadata_path: Path) -> ReferenceSensitivityData:
    """Load eigenvalues, sensitivities, and baseline parameters for the reference model."""

    if not reference_path.exists():
        raise FileNotFoundError(f"Missing reference solution: {reference_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing simulation metadata: {metadata_path}")

    with np.load(reference_path) as archive:
        eigenvalues = np.asarray(archive["eigenvalues"], dtype=float)
        stiffness_sens = np.asarray(archive["ligament_sensitivities"], dtype=float)
        pretension_sens = np.asarray(archive["ligament_pretension_sensitivities"], dtype=float)
        ligament_names = [str(name) for name in np.asarray(archive["ligament_names"], dtype=str)]
        pretension_values = np.asarray(archive["ligament_pretensions"], dtype=float)

    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    config = metadata.get("config", {})
    
    # Extract stiffness values from config
    if "LIGAMENT_STIFFNESSES" in config:
        stiffness_values = np.asarray(config["LIGAMENT_STIFFNESSES"], dtype=float)
    elif "LIGAMENTS" in config:
        # Extract from nested dict format
        ligaments_dict = config["LIGAMENTS"]
        stiffness_values = np.array([ligaments_dict[name]["stiffness"] for name in ligament_names], dtype=float)
    else:
        raise ValueError("Cannot find ligament stiffness values in metadata config")

    if stiffness_sens.shape[0] != eigenvalues.size:
        raise ValueError("Stiffness sensitivities must match the number of eigenvalues")
    if pretension_sens.shape[0] != eigenvalues.size:
        raise ValueError("Pretension sensitivities must match the number of eigenvalues")

    num_ligaments = len(ligament_names)
    if stiffness_sens.shape[1] != num_ligaments:
        raise ValueError("Mismatch between ligament names and stiffness sensitivity columns")
    if pretension_sens.shape[1] != num_ligaments:
        raise ValueError("Mismatch between ligament names and pretension sensitivity columns")
    if stiffness_values.size != num_ligaments:
        raise ValueError("Baseline stiffness values must match ligament count")
    if pretension_values.size != num_ligaments:
        raise ValueError("Baseline pretension values must match ligament count")

    return ReferenceSensitivityData(
        eigenvalues=eigenvalues,
        stiffness_sensitivities=stiffness_sens,
        pretension_sensitivities=pretension_sens,
        ligament_names=ligament_names,
        stiffness_values=stiffness_values,
        pretension_values=pretension_values,
    )


def normalize_sensitivities(
    sensitivities: np.ndarray,
    parameter_values: np.ndarray,
    eigenvalues: np.ndarray,
) -> np.ndarray:
    """Return dimensionless sensitivities using parameter and eigenvalue scaling."""

    if sensitivities.shape != (eigenvalues.size, parameter_values.size):
        raise ValueError("Sensitivity matrix shape mismatch")

    eigen_column = eigenvalues.reshape(-1, 1)
    safe_denominator = np.where(np.abs(eigen_column) < MIN_EIGENVALUE, np.nan, eigen_column)
    normalized = sensitivities * (parameter_values.reshape(1, -1) / safe_denominator)
    return normalized


def build_long_table(
    stiffness_norm: np.ndarray,
    pretension_norm: np.ndarray,
    ligament_names: list[str],
) -> pd.DataFrame:
    """Construct tidy DataFrame with per-mode normalized sensitivities."""

    records: list[dict[str, float | int | str]] = []
    num_modes = stiffness_norm.shape[0]
    for mode_idx in range(num_modes):
        for lig_idx, ligament in enumerate(ligament_names):
            stiff_val = float(stiffness_norm[mode_idx, lig_idx])
            pret_val = float(pretension_norm[mode_idx, lig_idx])
            records.append(
                {
                    "mode": mode_idx + 1,
                    "ligament": ligament,
                    "normalized_stiffness": stiff_val,
                    "abs_normalized_stiffness": abs(stiff_val),
                    "normalized_pretension": pret_val,
                    "abs_normalized_pretension": abs(pret_val),
                }
            )
    return pd.DataFrame.from_records(records)


def summarize_by_ligament(long_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate normalized sensitivities per ligament."""

    grouped = (
        long_df.groupby("ligament", as_index=False)
        .agg(
            total_abs_stiffness=("abs_normalized_stiffness", "sum"),
            max_abs_stiffness=("abs_normalized_stiffness", "max"),
            total_abs_pretension=("abs_normalized_pretension", "sum"),
            max_abs_pretension=("abs_normalized_pretension", "max"),
        )
        .reset_index(drop=True)
    )

    stiff_idx = long_df.groupby("ligament")["abs_normalized_stiffness"].idxmax()
    pret_idx = long_df.groupby("ligament")["abs_normalized_pretension"].idxmax()

    dominant_stiff = (
        long_df.loc[stiff_idx, ["ligament", "mode", "normalized_stiffness"]]
        .rename(
            columns={
                "mode": "dominant_mode_stiffness",
                "normalized_stiffness": "dominant_value_stiffness",
            }
        )
        .reset_index(drop=True)
    )
    dominant_pret = (
        long_df.loc[pret_idx, ["ligament", "mode", "normalized_pretension"]]
        .rename(
            columns={
                "mode": "dominant_mode_pretension",
                "normalized_pretension": "dominant_value_pretension",
            }
        )
        .reset_index(drop=True)
    )

    merged = grouped.merge(dominant_stiff, on="ligament", how="left")
    merged = merged.merge(dominant_pret, on="ligament", how="left")
    merged["dominant_mode_stiffness"] = merged["dominant_mode_stiffness"].astype(int)
    merged["dominant_mode_pretension"] = merged["dominant_mode_pretension"].astype(int)
    return merged


def summarize_by_mode(long_df: pd.DataFrame, eigenvalues: np.ndarray) -> pd.DataFrame:
    """Aggregate normalized sensitivities per mode."""

    summary = (
        long_df.groupby("mode", as_index=False)
        .agg(
            total_abs_stiffness=("abs_normalized_stiffness", "sum"),
            total_abs_pretension=("abs_normalized_pretension", "sum"),
        )
        .reset_index(drop=True)
    )
    summary["mode_label"] = summary["mode"].apply(lambda m: format_mode_label_short(m - 1, one_based=True))
    summary["eigenvalue"] = summary["mode"].apply(lambda m: float(eigenvalues[m - 1]))
    return summary


def plot_heatmaps(
    stiffness_norm: np.ndarray,
    pretension_norm: np.ndarray,
    ligament_names: list[str],
    output_path: Path,
) -> None:
    """Render side-by-side heatmaps for normalized sensitivities."""

    setup_plot_style()

    # Independent color scales for each matrix
    vmax_stiff = float(np.nanmax(np.abs(stiffness_norm))) if np.isfinite(stiffness_norm).any() else 1.0
    vmax_pret = float(np.nanmax(np.abs(pretension_norm))) if np.isfinite(pretension_norm).any() else 1.0
    if vmax_stiff == 0.0:
        vmax_stiff = 1.0
    if vmax_pret == 0.0:
        vmax_pret = 1.0

    fig, (ax_stiff, ax_pret) = plt.subplots(1, 2, figsize=FIG_SIZE, sharey=True)

    def _plot(ax: plt.Axes, matrix: np.ndarray, title: str, vmax: float) -> plt.AxesImage:
        data = matrix.T
        im = ax.imshow(data, aspect="auto", cmap=HEATMAP_CMAP, vmin=-vmax, vmax=vmax)
        ax.set_xticks(np.arange(matrix.shape[0]))
        ax.set_xticklabels(
            [format_mode_label_short(idx, one_based=True) for idx in range(matrix.shape[0])],
            rotation=45,
        )
        ax.set_yticks(np.arange(len(ligament_names)))
        ax.set_yticklabels(ligament_names)
        ax.set_xlabel("Mode")
        ax.set_title(title, fontweight="bold")
        return im

    img_stiff = _plot(ax_stiff, stiffness_norm, "Normalized stiffness sensitivity", vmax_stiff)
    ax_stiff.set_ylabel("Ligament bundle")
    img_pret = _plot(ax_pret, pretension_norm, "Normalized pretension sensitivity", vmax_pret)

    cbar_stiff = fig.colorbar(img_stiff, ax=ax_stiff, shrink=0.85, pad=0.02)
    cbar_stiff.set_label("EA/λ·∂λ/∂EA")
    
    cbar_pret = fig.colorbar(img_pret, ax=ax_pret, shrink=0.85, pad=0.02)
    cbar_pret.set_label("T/λ·∂λ/∂T")

    fig.savefig(output_path.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def export_tables(
    long_df: pd.DataFrame,
    ligament_summary: pd.DataFrame,
    mode_summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Write analysis tables to CSV files."""

    long_df.to_csv(output_dir / "reference_ligament_sensitivities_long.csv", index=False)
    ligament_summary.to_csv(output_dir / "reference_ligament_sensitivities_per_ligament.csv", index=False)
    mode_summary.to_csv(output_dir / "reference_ligament_sensitivities_per_mode.csv", index=False)


def main() -> None:
    """Execute reference sensitivity analysis."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = load_reference_data(REFERENCE_PATH, METADATA_PATH)

    stiffness_normalized = normalize_sensitivities(
        data.stiffness_sensitivities,
        data.stiffness_values,
        data.eigenvalues,
    )
    pretension_normalized = normalize_sensitivities(
        data.pretension_sensitivities,
        data.pretension_values,
        data.eigenvalues,
    )

    long_df = build_long_table(stiffness_normalized, pretension_normalized, data.ligament_names)
    ligament_summary = summarize_by_ligament(long_df)
    mode_summary = summarize_by_mode(long_df, data.eigenvalues)

    plot_heatmaps(
        stiffness_normalized,
        pretension_normalized,
        data.ligament_names,
        OUTPUT_DIR / "reference_ligament_sensitivity_heatmaps",
    )
    export_tables(long_df, ligament_summary, mode_summary, OUTPUT_DIR)

    print(f"Reference sensitivity outputs written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
