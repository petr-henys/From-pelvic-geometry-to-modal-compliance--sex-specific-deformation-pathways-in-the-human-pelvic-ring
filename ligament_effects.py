#!/usr/bin/env python3
"""Ligament effects analysis on pelvic eigenstiffness.

Analyzes parametric sweep data to quantify how ligament stiffness variations
affect elastic eigenmodes. Uses simplified MAC for mode pairing.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import spearmanr
from scipy.optimize import linear_sum_assignment

from utils.plot_utils import setup_plot_style, format_mode_label, ANNOT_SIZE, FULL_WIDTH, ROW_H, SMALL_ANNOT_SIZE, MALE_COLOR, FEMALE_COLOR
from logging_config import get_logger

logger = get_logger("LigamentEffects")


def mac_simple(x: np.ndarray, y: np.ndarray) -> float:
    """Compute simplified Modal Assurance Criterion without mass matrix.
    
    MAC = |x^T y|^2 / [(x^T x)(y^T y)]
    
    Parameters
    ----------
    x, y : np.ndarray
        Flattened mode vectors
        
    Returns
    -------
    float
        MAC value in [0, 1]
    """
    x_flat = x.ravel()
    y_flat = y.ravel()
    
    numerator = np.abs(np.dot(x_flat, y_flat)) ** 2
    denominator = np.dot(x_flat, x_flat) * np.dot(y_flat, y_flat)
    
    return float(numerator / denominator) if denominator > 1e-12 else 0.0


def pair_modes_simple(
    ref_vecs: np.ndarray,
    ref_vals: np.ndarray,
    samp_vecs: np.ndarray,
    samp_vals: np.ndarray,
    mac_cut: float = 0.05,
) -> np.ndarray:
    """Simplified mode pairing using MAC without mass matrix."""
    n_ref = len(ref_vecs)
    n_samp = len(samp_vecs)
    
    # Compute MAC matrix
    mac_matrix = np.zeros((n_ref, n_samp))
    for i in range(n_ref):
        for j in range(n_samp):
            mac_matrix[i, j] = mac_simple(ref_vecs[i], samp_vecs[j])
    
    # Debug: Show MAC matrix for first simulation
    if logger.level <= 10:  # DEBUG level
        logger.debug(f"MAC matrix diagonal: {np.diag(mac_matrix)}")
        logger.debug(f"MAC matrix max off-diagonal: {np.max(mac_matrix - np.diag(np.diag(mac_matrix)))}")
    
    # Solve assignment problem
    row_ind, col_ind = linear_sum_assignment(-mac_matrix)
    
    # Build permutation with threshold
    perm = np.full(n_ref, -1, dtype=np.int32)
    for i_ref, j_samp in zip(row_ind, col_ind):
        if mac_matrix[i_ref, j_samp] >= mac_cut:
            perm[i_ref] = j_samp
    
    return perm

# Configuration
SIMULATION_DIR = PROJECT_ROOT / "results" / "ref_S1P_fixed"
SWEEP_DIR = PROJECT_ROOT / "results" / "ligament_sweep"
BASELINE_DIR = SIMULATION_DIR
OUTPUT_DIR = BASELINE_DIR / "ligament_sweep_results"
NUM_MODES = 12

# Ligament names (simplified - no side designation)
LIGAMENT_NAMES = [
    "Symphysis",
    "Anterior_SIJ",
    "Posterior_SIJ", 
    "Iliolumbar",
    "Sacrospinous",
    "Sacrotuberous",
]

# Expected multipliers from sweep
MULTIPLIERS = [0.5, 1.0, 1.5]


def load_baseline_reference():
    """Load baseline simulation as reference for mode pairing."""
    ref_path = BASELINE_DIR / "data" / "reference_solution.npz"
    if not ref_path.exists():
        ref_path = BASELINE_DIR / "reference_solution.npz"
    
    logger.info(f"Loading baseline reference: {ref_path}")
    data = np.load(ref_path)
    
    ref_vals = data["eigenvalues"][:NUM_MODES]
    ref_vecs = data["eigenvectors"][:NUM_MODES]
    
    logger.info(f"Baseline: {len(ref_vals)} modes, eigenvalue range [{ref_vals.min():.2e}, {ref_vals.max():.2e}]")
    
    return ref_vals, ref_vecs


def parse_ligament_config(metadata_path: Path):
    """Parse ligament configuration from simulation metadata."""
    with open(metadata_path) as f:
        data = json.load(f)
    
    # Extract stiffnesses from config (standard location)
    if "config" in data and "LIGAMENT_STIFFNESSES" in data["config"]:
        stiffnesses = np.array(data["config"]["LIGAMENT_STIFFNESSES"], dtype=float)
    elif "LIGAMENT_STIFFNESSES" in data:
        # Old format: direct key (backward compatibility)
        stiffnesses = np.array(data["LIGAMENT_STIFFNESSES"], dtype=float)
    else:
        raise ValueError(f"Cannot find LIGAMENT_STIFFNESSES in {metadata_path}")
    
    # Load baseline for comparison
    baseline_meta_path = BASELINE_DIR / "simulation_metadata.json"
    
    if not baseline_meta_path.exists():
        raise FileNotFoundError(f"Baseline metadata not found: {baseline_meta_path}")
    
    baseline_data = json.loads(baseline_meta_path.read_text())
    
    if "config" in baseline_data and "LIGAMENT_STIFFNESSES" in baseline_data["config"]:
        baseline = np.array(baseline_data["config"]["LIGAMENT_STIFFNESSES"], dtype=float)
    elif "LIGAMENT_STIFFNESSES" in baseline_data:
        baseline = np.array(baseline_data["LIGAMENT_STIFFNESSES"], dtype=float)
    else:
        raise ValueError(f"Cannot find baseline LIGAMENT_STIFFNESSES in {baseline_meta_path}")
    
    # Use only first 6 (right side) since left is identical
    multipliers = stiffnesses[:6] / baseline[:6]
    

    
    return stiffnesses, multipliers


def load_sweep_results(ref_vals: np.ndarray, ref_vecs: np.ndarray) -> pd.DataFrame:
    """Load all sweep simulations and pair modes to reference."""
    sim_dirs = sorted([d for d in SWEEP_DIR.iterdir() if d.is_dir()])
    
    logger.info(f"Loading {len(sim_dirs)} sweep simulations...")
    
    records = []
    failed = []
    
    for idx, sim_dir in enumerate(sim_dirs):
        if (idx + 1) % 50 == 0:
            logger.info(f"Progress: {idx+1}/{len(sim_dirs)}")
        
        try:
            # Load metadata
            metadata_path = sim_dir / "simulation_metadata.json"
            
            if not metadata_path.exists():
                failed.append((sim_dir.name, "Missing simulation_metadata.json"))
                continue
            
            _, multipliers = parse_ligament_config(metadata_path)
            
            # Load eigenvalues
            npz_path = sim_dir / "reference_solution.npz"
            if not npz_path.exists():
                failed.append((sim_dir.name, "Missing reference_solution.npz"))
                continue
            
            data = np.load(npz_path)
            samp_vals = data["eigenvalues"][:NUM_MODES]
            samp_vecs = data["eigenvectors"][:NUM_MODES]
            
            # Pair modes to reference (simplified MAC)
            perm = pair_modes_simple(ref_vecs, ref_vals, samp_vecs, samp_vals, mac_cut=0.05)
            
            # Build record
            record = {"sim_id": sim_dir.name}
            
            # Ligament multipliers (6 unique types)
            for i, mult in enumerate(multipliers):
                record[f"lig_{i}"] = mult
            
            # Paired eigenvalues and MAC
            for i in range(NUM_MODES):
                if perm[i] >= 0:
                    record[f"mode_{i}"] = samp_vals[perm[i]]
                    mac = mac_simple(ref_vecs[i], samp_vecs[perm[i]])
                    record[f"mac_{i}"] = mac
                    record[f"pairing_{i}"] = perm[i]
                else:
                    record[f"mode_{i}"] = np.nan
                    record[f"mac_{i}"] = np.nan
                    record[f"pairing_{i}"] = -1
            
            records.append(record)
            
        except (ValueError, KeyError, IndexError) as e:
            failed.append((sim_dir.name, str(e)))
            logger.warning(f"Failed to load {sim_dir.name}: {e}")
    
    logger.info(f"Successfully loaded {len(records)} simulations")
    if failed:
        logger.info(f"Failed to load {len(failed)} simulations:")
        for sim_id, reason in failed[:5]:
            logger.info(f"  {sim_id}: {reason}")
    
    return pd.DataFrame(records)
def compute_ligament_effects(df: pd.DataFrame, ref_vals: np.ndarray) -> pd.DataFrame:
    """Compute per-ligament, per-mode sensitivity metrics."""
    results = []
    
    for lig_idx in range(6):  # 6 unique ligament types
        lig_name = LIGAMENT_NAMES[lig_idx]
        lig_col = f"lig_{lig_idx}"
        
        for mode_idx in range(NUM_MODES):
            mode_col = f"mode_{mode_idx}"
            
            # Filter valid pairings
            valid = df[[lig_col, mode_col]].dropna()
            if len(valid) < 3:
                logger.warning(f"{lig_name} mode {mode_idx}: insufficient data ({len(valid)} samples)")
                continue
            
            x = valid[lig_col].values
            y = valid[mode_col].values
            
            # Spearman correlation
            rho, p_val = spearmanr(x, y)
            
            # Absolute and relative changes
            ref_val = ref_vals[mode_idx]
            abs_changes = np.abs(y - ref_val)
            rel_changes = 100.0 * (y - ref_val) / ref_val
            
            results.append({
                "ligament": lig_name,
                "mode": mode_idx,
                "mode_label": format_mode_label(mode_idx, one_based=True),
                "spearman_rho": rho,
                "spearman_p": p_val,
                "mean_abs_change": abs_changes.mean(),
                "mean_rel_change": rel_changes.mean(),
                "max_abs_change": abs_changes.max(),
                "max_rel_change": np.abs(rel_changes).max(),
                "n_samples": len(valid),
            })
    
    return pd.DataFrame(results)


def rank_ligament_importance(effects: pd.DataFrame) -> pd.DataFrame:
    """Rank ligaments by overall importance across all modes."""
    ranking = effects.groupby("ligament").agg({
        "spearman_rho": lambda x: np.abs(x).mean(),
        "mean_rel_change": lambda x: np.abs(x).mean(),
        "max_rel_change": "max",
        "spearman_p": lambda x: (x < 0.05).sum(),
    }).reset_index()
    
    ranking.columns = [
        "ligament",
        "mean_abs_corr",
        "mean_rel_change",
        "max_rel_change",
        "n_significant_modes",
    ]
    
    # Composite importance score (weighted combination)
    ranking["importance_score"] = (
        0.4 * ranking["mean_abs_corr"] +
        0.3 * ranking["mean_rel_change"] / 100.0 +
        0.3 * ranking["n_significant_modes"] / NUM_MODES
    )
    
    ranking = ranking.sort_values("importance_score", ascending=False)
    
    return ranking


def analyze_ligament_mode_swapping(df_sweep: pd.DataFrame) -> pd.DataFrame:
    """Compute which ligaments are associated with mode swapping.
    
    For each ligament and mode, count how often non-diagonal pairing occurs
    when that ligament is at non-baseline values.
    
    Returns DataFrame with columns: ligament, mode, swap_frequency, total_samples
    """
    results = []
    
    for lig_idx in range(6):
        lig_name = LIGAMENT_NAMES[lig_idx]
        lig_col = f"lig_{lig_idx}"
        
        # Identify non-baseline samples (multiplier != 1.0)
        non_baseline = df_sweep[np.abs(df_sweep[lig_col] - 1.0) > 0.01]
        
        if len(non_baseline) == 0:
            continue
        
        for mode_idx in range(NUM_MODES):
            pairing_col = f"pairing_{mode_idx}"
            
            # Count swaps: pairing != mode_idx (off-diagonal)
            valid_pairs = non_baseline[pairing_col].dropna()
            swaps = (valid_pairs != mode_idx).sum()
            total = len(valid_pairs)
            
            if total > 0:
                results.append({
                    "ligament": lig_name,
                    "mode": mode_idx,
                    "mode_label": format_mode_label(mode_idx, one_based=True),
                    "swaps": swaps,
                    "total": total,
                    "swap_frequency": swaps / total,
                })
    
    return pd.DataFrame(results)


def plot_dashboard(effects: pd.DataFrame, ranking: pd.DataFrame, df_sweep: pd.DataFrame, ref_vals: np.ndarray, save_path: Path):
    """Create comprehensive dashboard visualizing ligament effects."""
    setup_plot_style()
    
    logger.info("Analyzing mode pairing patterns...")
    pairing_cols = [f"pairing_{i}" for i in range(NUM_MODES)]
    for col in pairing_cols[:3]:  # Check first 3 columns
        valid_pairs = df_sweep[col].dropna()
        unique_pairs = valid_pairs.unique()
        logger.info(f"{col}: {len(valid_pairs)} valid pairs, unique values: {sorted(unique_pairs)[:10]}")
    
    # Analyze ligament-swapping associations
    swap_analysis = analyze_ligament_mode_swapping(df_sweep)
    
    fig = plt.figure(figsize=(FULL_WIDTH, 5 * ROW_H), constrained_layout=True)
    gs = GridSpec(3, 4, figure=fig)
    fig.patch.set_facecolor("white")
    
    # Panel 1: Correlation heatmap
    ax1 = fig.add_subplot(gs[0, :2])
    pivot = effects.pivot(index="ligament", columns="mode", values="spearman_rho")
    im = ax1.imshow(pivot.values, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    ax1.set_xticks(range(NUM_MODES))
    ax1.set_xticklabels([format_mode_label(i, one_based=True) for i in range(NUM_MODES)],
                        rotation=35, ha='right', fontsize=ANNOT_SIZE)
    ax1.set_yticks(range(len(LIGAMENT_NAMES)))
    ax1.set_yticklabels(pivot.index, fontsize=ANNOT_SIZE)
    ax1.set_title("Spearman Correlation: Ligament vs Eigenvalue", fontweight="bold", pad=15)
    ax1.set_xlabel("Elastic Mode", fontweight="bold")
    ax1.set_ylabel("Ligament", fontweight="bold")
    
    # Add correlation values
    for i in range(len(pivot.index)):
        for j in range(NUM_MODES):
            val = pivot.values[i, j]
            if not np.isnan(val):
                color = "white" if abs(val) > 0.5 else "black"
                ax1.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=SMALL_ANNOT_SIZE)
    
    cbar1 = plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label("Spearman ρ", rotation=270, labelpad=15, fontweight="bold")
    
    # Panel 2: Importance ranking  
    ax2 = fig.add_subplot(gs[0, 2])
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(ranking)))
    bars = ax2.barh(range(len(ranking)), ranking["importance_score"], color=colors)
    ax2.set_yticks(range(len(ranking)))
    ax2.set_yticklabels(ranking["ligament"])
    ax2.set_xlabel("Importance Score", fontweight="bold")
    ax2.set_title("Ligament Importance", fontweight="bold", pad=15)
    ax2.invert_yaxis()
    ax2.grid(axis="x", alpha=0.3)
    max_score = float(ranking["importance_score"].max()) if not ranking.empty else 1.0
    ax2.set_xlim(0, max(0.2, max_score * 1.15))
    
    # Add score values on bars
    for i, (bar, score) in enumerate(zip(bars, ranking["importance_score"])):
        width = bar.get_width()
        x_max = ax2.get_xlim()[1]
        padding = max_score * 0.03 if max_score > 0 else 0.02
        text_x = min(width + padding, x_max - padding)
        ax2.text(text_x, bar.get_y() + bar.get_height()/2,
                f'{score:.3f}', ha='left', va='center', fontsize=ANNOT_SIZE)
    
    # Panel 3: Additional ranking details
    ax3 = fig.add_subplot(gs[0, 3])
    
    # Show number of significant modes per ligament
    sig_modes = ranking["n_significant_modes"]
    colors_sig = plt.cm.Reds(np.linspace(0.3, 0.9, len(sig_modes)))
    bars3 = ax3.barh(range(len(ranking)), sig_modes, color=colors_sig)
    ax3.set_yticks(range(len(ranking)))
    ax3.set_yticklabels(ranking["ligament"])
    ax3.set_xlabel("Significant Modes", fontweight="bold")
    ax3.set_title("Significant Effects Count", fontweight="bold", pad=15)
    ax3.invert_yaxis()
    ax3.grid(axis="x", alpha=0.3)
    max_sig = float(sig_modes.max()) if len(sig_modes) else 1.0
    ax3.set_xlim(0, max(1.0, max_sig * 1.2))
    
    # Add count values
    for i, (bar, count) in enumerate(zip(bars3, sig_modes)):
        width = bar.get_width()
        if width > 0:
            x_max = ax3.get_xlim()[1]
            padding = max_sig * 0.05 if max_sig > 0 else 0.1
            text_x = min(width + padding, x_max - padding)
            ax3.text(text_x, bar.get_y() + bar.get_height()/2,
                    f'{int(count)}', ha='left', va='center', fontsize=ANNOT_SIZE)
    # Panel 4: Ligament-induced swapping heatmap (NEW)
    ax4 = fig.add_subplot(gs[1, :2])
    
    if not swap_analysis.empty:
        pivot_swaps = swap_analysis.pivot(index="ligament", columns="mode", values="swap_frequency")
        
        # Fill missing values with 0
        pivot_swaps = pivot_swaps.fillna(0)
        
        im4 = ax4.imshow(pivot_swaps.values, aspect="auto", cmap="Reds", vmin=0, vmax=0.5)
        ax4.set_xticks(range(NUM_MODES))
        ax4.set_xticklabels([format_mode_label(i, one_based=True) for i in range(NUM_MODES)],
                            rotation=35, ha='right', fontsize=ANNOT_SIZE)
        ax4.set_yticks(range(len(pivot_swaps.index)))
        ax4.set_yticklabels(pivot_swaps.index, fontsize=ANNOT_SIZE)
        ax4.set_title("Ligament-Induced Mode Swapping", fontweight="bold", pad=15)
        ax4.set_xlabel("Reference Mode", fontweight="bold")
        ax4.set_ylabel("Ligament", fontweight="bold")
        
        # Add swap frequency values
        for i in range(len(pivot_swaps.index)):
            for j in range(NUM_MODES):
                val = pivot_swaps.values[i, j]
                if val > 0.05:  # Show frequencies > 5%
                    color = "white" if val > 0.25 else "black"
                    ax4.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=SMALL_ANNOT_SIZE)
        
        cbar4 = plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
        cbar4.set_label("Swap Frequency", rotation=270, labelpad=15, fontweight="bold")
    else:
        ax4.text(0.5, 0.5, "No swapping data available", ha="center", va="center", 
                transform=ax4.transAxes)
        ax4.set_title("Ligament-Induced Mode Swapping", fontweight="bold", pad=15)
    
    # Panel 5: Mode stability analysis - compute swap_matrix for this
    ax5 = fig.add_subplot(gs[1, 2:])
    
    # Analyze mode swapping patterns for stability
    swap_matrix = np.zeros((NUM_MODES, NUM_MODES))
    total_valid_pairings = 0
    
    for idx, row in df_sweep.iterrows():
        for ref_mode in range(NUM_MODES):
            pairing_col = f"pairing_{ref_mode}"
            paired_mode = row[pairing_col]
            
            # Check if mode was successfully paired
            if not pd.isna(paired_mode) and paired_mode >= 0:
                paired_idx = int(paired_mode)
                if paired_idx < NUM_MODES:  # Safety check
                    swap_matrix[ref_mode, paired_idx] += 1
                    total_valid_pairings += 1
    
    # Log swap matrix statistics
    logger.info(f"Total valid pairings: {total_valid_pairings}")
    logger.info(f"Diagonal sum: {np.trace(swap_matrix)}")
    logger.info(f"Off-diagonal sum: {np.sum(swap_matrix) - np.trace(swap_matrix)}")
    
    # Normalize by total simulations
    total_sims = len(df_sweep)
    swap_matrix_freq = swap_matrix / total_sims
    
    # Calculate diagonal dominance (how often modes pair with themselves)
    diagonal_freq = np.diag(swap_matrix_freq)
    mode_labels = [format_mode_label(i, one_based=True) for i in range(NUM_MODES)]
    
    # Color by stability 
    colors_diag = []
    for freq in diagonal_freq:
        if freq > 0.8:
            colors_diag.append('#2E7D32')  # Dark green
        elif freq > 0.5:
            colors_diag.append('#FDD835')  # Yellow
        else:
            colors_diag.append('#C62828')  # Dark red
    
    bars = ax5.bar(range(NUM_MODES), diagonal_freq, color=colors_diag)
    ax5.set_xticks(range(NUM_MODES))
    ax5.set_xticklabels(mode_labels, rotation=35, ha='right', fontsize=ANNOT_SIZE)
    ax5.set_ylabel("Self-Pairing Frequency", fontweight="bold")
    ax5.set_title("Mode Stability (Self-Pairing Rate)", fontweight="bold", pad=15)
    ax5.grid(axis="y", alpha=0.3)
    ax5.set_ylim(0, 1.05)
    
    # Add value labels on bars
    for _, (bar, freq) in enumerate(zip(bars, diagonal_freq)):
        height = bar.get_height()
        text_y = min(height + 0.015, 1.02)
        ax5.text(bar.get_x() + bar.get_width()/2., text_y,
                 f'{freq:.2f}', ha='center', va='bottom', fontsize=SMALL_ANNOT_SIZE)
    
    # Add threshold lines
    ax5.axhline(0.8, color="green", linestyle="--", linewidth=1, alpha=0.7, label="Stable (>0.8)")
    ax5.axhline(0.5, color="orange", linestyle="--", linewidth=1, alpha=0.7, label="Moderate (>0.5)")
    ax5.legend(loc='upper right')

    # Panel 6: MAC quality distribution
    ax6 = fig.add_subplot(gs[2, 0])
    mac_cols = [f"mac_{i}" for i in range(NUM_MODES)]
    mac_values = df_sweep[mac_cols].values.ravel()
    mac_values = mac_values[~np.isnan(mac_values)]
    
    ax6.hist(mac_values, bins=30, alpha=0.7, color=MALE_COLOR, edgecolor="black")
    ax6.axvline(0.9, color="red", linestyle="--", linewidth=2, label="Excellent (>0.9)")
    ax6.axvline(0.5, color="orange", linestyle="--", linewidth=2, label="Good (>0.5)")
    ax6.set_xlabel("MAC Correlation", fontweight="bold")
    ax6.set_ylabel("Frequency", fontweight="bold")
    ax6.set_title("Mode Pairing Quality", fontweight="bold", pad=15)
    ax6.legend()
    ax6.grid(alpha=0.3)
    
    # Panel 7: Relative change distribution
    ax7 = fig.add_subplot(gs[2, 1])
    
    rel_changes = []
    for mode_idx in range(NUM_MODES):
        mode_col = f"mode_{mode_idx}"
        valid = df_sweep[mode_col].dropna()
        ref_val = ref_vals[mode_idx]
        rel_change = 100.0 * np.abs(valid - ref_val) / ref_val
        rel_changes.extend(rel_change.tolist())
    
    ax7.hist(rel_changes, bins=30, alpha=0.7, color=FEMALE_COLOR, edgecolor="black")
    ax7.set_xlabel("Relative Change (%)", fontweight="bold")
    ax7.set_ylabel("Frequency", fontweight="bold")
    ax7.set_title("Eigenvalue Sensitivity", fontweight="bold", pad=15)
    ax7.grid(alpha=0.3)
    
    # Panel 8: Eigenvalue ranges by mode
    ax8 = fig.add_subplot(gs[2, 2])
    
    # Show eigenvalue ranges for each mode
    mode_ranges = []
    mode_labels_plot = []
    
    for mode_idx in range(min(8, NUM_MODES)):  # Show first 8 modes
        mode_col = f"mode_{mode_idx}"
        valid_vals = df_sweep[mode_col].dropna()
        if len(valid_vals) > 10:
            mode_ranges.append(valid_vals.values)
            mode_labels_plot.append(format_mode_label(mode_idx, one_based=True))
    
    if mode_ranges:
        box_plot = ax8.boxplot(mode_ranges, tick_labels=mode_labels_plot, patch_artist=True)
        
        # Color boxes by median eigenvalue
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(box_plot['boxes'])))
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
    
    ax8.set_xlabel("Elastic Mode", fontweight="bold")
    ax8.set_ylabel("Eigenvalue", fontweight="bold")
    ax8.set_title("Eigenvalue Distributions Across Sweep", fontweight="bold", pad=15)
    ax8.tick_params(axis='x', rotation=30)
    ax8.grid(alpha=0.3)
    ax8.set_yscale('log')
    
    # Panel 9: Ligament swapping summary (NEW - replaces significant effects)
    ax9 = fig.add_subplot(gs[2, 3])
    
    if not swap_analysis.empty:
        # Total swapping impact per ligament
        swap_summary = swap_analysis.groupby("ligament").agg({
            "swaps": "sum",
            "total": "sum",
        }).reset_index()
        swap_summary["overall_swap_rate"] = swap_summary["swaps"] / swap_summary["total"]
        swap_summary = swap_summary.sort_values("overall_swap_rate", ascending=False)
        
        # Ensure all ligaments are present
        swap_summary = swap_summary.set_index("ligament").reindex(LIGAMENT_NAMES, fill_value=0).reset_index()
        
        colors_swap = plt.cm.Oranges(np.linspace(0.3, 0.9, len(swap_summary)))
        bars9 = ax9.barh(range(len(swap_summary)), swap_summary["overall_swap_rate"], color=colors_swap)
        ax9.set_yticks(range(len(swap_summary)))
        ax9.set_yticklabels(swap_summary["ligament"], fontsize=ANNOT_SIZE)
        ax9.set_xlabel("Swapping Rate", fontweight="bold")
        ax9.set_title("Mode Swapping by Ligament", fontweight="bold", pad=15)
        ax9.invert_yaxis()
        ax9.grid(axis="x", alpha=0.3)
        max_rate = float(swap_summary["overall_swap_rate"].max()) if len(swap_summary) else 0.5
        ax9.set_xlim(0, max(0.1, max_rate * 1.2))
        
        # Add value labels
        for bar, rate in zip(bars9, swap_summary["overall_swap_rate"]):
            width = bar.get_width()
            if width > 0:
                padding = max_rate * 0.05 if max_rate > 0 else 0.02
                text_x = min(width + padding, ax9.get_xlim()[1] - padding)
                ax9.text(text_x, bar.get_y() + bar.get_height()/2,
                         f"{rate:.2f}", ha="left", va="center", fontsize=ANNOT_SIZE)
    else:
        # Fallback: Show significant effects count
        sig_threshold = 0.3
        sig_counts = effects[np.abs(effects["spearman_rho"]) > sig_threshold].groupby("ligament").size()
        sig_counts = sig_counts.reindex(pd.Index(LIGAMENT_NAMES), fill_value=0)
        colors_sig = plt.cm.RdYlGn(np.linspace(0.2, 0.85, len(sig_counts)))
        ax9.barh(range(len(sig_counts)), sig_counts, color=colors_sig)
        ax9.set_yticks(range(len(sig_counts)))
        ax9.set_yticklabels(sig_counts.index, fontsize=ANNOT_SIZE)
        ax9.set_xlabel("Number of Modes", fontweight="bold")
        ax9.set_title(f"Significant Effects (|ρ| > {sig_threshold})", fontweight="bold", pad=15)
        ax9.invert_yaxis()
        ax9.grid(axis="x", alpha=0.3)

    fig.suptitle(
        "Ligament Parametric Sweep Analysis: Effects on Pelvic Eigenstiffness",
        fontweight="bold",
        y=0.995
    )

    plt.savefig(save_path, bbox_inches="tight", facecolor='white')
    plt.close()

    logger.info(f"Dashboard saved: {save_path}")


def save_summary_tables(effects: pd.DataFrame, ranking: pd.DataFrame, df_sweep: pd.DataFrame, swap_analysis: pd.DataFrame, output_dir: Path):
    """Save analysis results to Excel files."""
    # Convert mode indices to 1-based for Excel
    effects_export = effects.copy()
    effects_export["mode"] = effects_export["mode"] + 1
    
    # Summary table
    summary_path = output_dir / "ligament_effects_summary.xlsx"
    with pd.ExcelWriter(summary_path) as writer:
        effects_export.to_excel(writer, sheet_name="Effects", index=False)
        
        # Pivot tables
        pivot_rho = effects.pivot(index="ligament", columns="mode", values="spearman_rho")
        pivot_rho.columns = [f"Mode_{i+1}" for i in pivot_rho.columns]
        pivot_rho.to_excel(writer, sheet_name="Correlation_Matrix")
        
        pivot_rel = effects.pivot(index="ligament", columns="mode", values="mean_rel_change")
        pivot_rel.columns = [f"Mode_{i+1}" for i in pivot_rel.columns]
        pivot_rel.to_excel(writer, sheet_name="RelativeChange_Matrix")
        
        # Mode swapping analysis
        if not swap_analysis.empty:
            swap_export = swap_analysis.copy()
            swap_export["mode"] = swap_export["mode"] + 1
            swap_export.to_excel(writer, sheet_name="Mode_Swapping", index=False)
            
            # Pivot swapping by ligament
            pivot_swaps = swap_analysis.pivot(index="ligament", columns="mode", values="swap_frequency")
            pivot_swaps.columns = [f"Mode_{i+1}" for i in pivot_swaps.columns]
            pivot_swaps.to_excel(writer, sheet_name="Swapping_Matrix")
    
    logger.info(f"Effects summary saved: {summary_path}")
    
    # Ranking table
    ranking_path = output_dir / "ligament_ranking.xlsx"
    ranking.to_excel(ranking_path, index=False)
    logger.info(f"Ligament ranking saved: {ranking_path}")
    
    # Raw sweep data (convert mode columns to 1-based naming)
    sweep_export = df_sweep.copy()
    rename_map = {}
    for i in range(NUM_MODES):
        rename_map[f"mode_{i}"] = f"eig_{i+1}"
        rename_map[f"mac_{i}"] = f"mac_{i+1}"
        rename_map[f"pairing_{i}"] = f"pairing_{i+1}"
    sweep_export = sweep_export.rename(columns=rename_map)
    
    sweep_path = output_dir / "sweep_data_paired.xlsx"
    sweep_export.to_excel(sweep_path, index=False)
    logger.info(f"Sweep data saved: {sweep_path}")


def main():
    """Execute complete ligament effects analysis."""
    logger.info("="*80)
    logger.info("LIGAMENT PARAMETRIC SWEEP ANALYSIS")
    logger.info("="*80)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Load baseline reference
    logger.info("\n[1/5] Loading baseline reference...")
    ref_vals, ref_vecs = load_baseline_reference()
    
    # Step 2: Load sweep results with mode pairing
    logger.info("\n[2/5] Loading sweep simulations and pairing modes...")
    df_sweep = load_sweep_results(ref_vals, ref_vecs)
    logger.info(f"Loaded {len(df_sweep)} sweep configurations")
    
    # Step 3: Compute ligament effects
    logger.info("\n[3/5] Computing per-ligament, per-mode effects...")
    effects = compute_ligament_effects(df_sweep, ref_vals)
    logger.info(f"Analyzed {len(effects)} ligament-mode pairs")
    
    # Step 4: Rank ligaments
    logger.info("\n[4/5] Ranking ligament importance...")
    ranking = rank_ligament_importance(effects)
    
    logger.info("\nLigament Importance Ranking:")
    for idx, row in ranking.iterrows():
        logger.info(
            f"  #{idx+1:2d} {row['ligament']:20s} "
            f"score={row['importance_score']:.3f} "
            f"(|ρ|={row['mean_abs_corr']:.3f}, "
            f"Δλ={row['mean_rel_change']:.1f}%, "
            f"n_sig={int(row['n_significant_modes'])})"
        )
    
    # Step 5: Generate visualizations
    logger.info("\n[5/5] Generating dashboard and tables...")
    
    # Compute ligament-swapping associations
    swap_analysis = analyze_ligament_mode_swapping(df_sweep)
    logger.info(f"Analyzed mode swapping for {len(swap_analysis)} ligament-mode pairs")
    
    plot_dashboard(effects, ranking, df_sweep, ref_vals, OUTPUT_DIR / "ligament_effects_dashboard.png")
    save_summary_tables(effects, ranking, df_sweep, swap_analysis, OUTPUT_DIR)
    
    logger.info("\n" + "="*80)
    logger.info(f"Analysis complete. Results in: {OUTPUT_DIR}")
    logger.info("="*80)
    
    logger.info("\nKey files:")
    logger.info(f"  - Dashboard:     {OUTPUT_DIR / 'ligament_effects_dashboard.png'}")
    logger.info(f"  - Effects:       {OUTPUT_DIR / 'ligament_effects_summary.xlsx'}")
    logger.info(f"  - Ranking:       {OUTPUT_DIR / 'ligament_ranking.xlsx'}")
    logger.info(f"  - Sweep data:    {OUTPUT_DIR / 'sweep_data_paired.xlsx'}")


if __name__ == "__main__":
    main()
