#!/usr/bin/env python3
"""Population-level ligament uncertainty propagation.

Applies the first-order hierarchical uncertainty model to all 278 subjects
to evaluate the robustness of the spectral gaps and variance decomposition
across the population, stratified by sex.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_ligament_uncertainty_sweep import (
    UncertaintySpec,
    first_order_propagation,
    build_input_covariance,
)
from analysis.spectral_data import load_metadata

import matplotlib.pyplot as plt
from utils.plot_utils import (
    ANNOT_SIZE,
    FULL_WIDTH,
    ROW_H,
    setup_plot_style,
    MALE_COLOR,
    FEMALE_COLOR,
    panel_label,
)

# Paths
RESULTS_DIR = Path("results/ref_S1P_fixed_new2")
DATA_DIR = RESULTS_DIR / "data"
METADATA_PATH = RESULTS_DIR / "simulation_metadata.json"
OUTPUT_DIR = Path("analysis_outputs/ligament_uncertainty")


def load_population_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Load sensitivities, eigenvalues, permutations, and ligament names."""
    z_lig = zarr.open(str(DATA_DIR / "lig_sensitivities.zarr"), mode="r")
    z_pret = zarr.open(str(DATA_DIR / "pretension_sensitivities.zarr"), mode="r")
    z_eig = zarr.open(str(DATA_DIR / "eigenvalues.zarr"), mode="r")
    z_perm = zarr.open(str(DATA_DIR / "eig_permutations.zarr"), mode="r")
    
    lig_sens = np.asarray(z_lig["data"][:])
    pret_sens = np.asarray(z_pret["data"][:])
    eigenvalues = np.asarray(z_eig["data"][:])
    permutations = np.asarray(z_perm["data"][:])
    
    # Load ligament names from metadata
    with METADATA_PATH.open() as f:
        metadata = json.load(f)
    bundle_names = metadata.get("config", {}).get("LIGAMENT_NAMES", [])
    
    return lig_sens, pret_sens, eigenvalues, permutations, bundle_names


def get_baselines(bundle_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Extract baseline stiffness and pretension from metadata."""
    with METADATA_PATH.open() as f:
        metadata = json.load(f)
    ligaments_dict = metadata.get("config", {}).get("LIGAMENTS", {})
    
    stiffness = np.array([ligaments_dict[name]["stiffness"] for name in bundle_names], dtype=float)
    pretension = np.array([ligaments_dict[name]["pretension"] for name in bundle_names], dtype=float)
    return stiffness, pretension


def process_population() -> pd.DataFrame:
    """Run first-order propagation for all subjects."""
    lig_sens, pret_sens, eigenvalues, permutations, bundle_names = load_population_data()
    stiffness_baseline, pretension_baseline = get_baselines(bundle_names)
    sex, _ = load_metadata()
    
    spec = UncertaintySpec()
    n_subjects, n_modes, n_lig = lig_sens.shape
    
    all_records = []
    
    for i in range(n_subjects):
        # Reorder modes according to MAC permutations to match reference mode identities
        perm = permutations[i]
        
        # Filter out unmapped modes (-1)
        valid_mask = perm >= 0
        valid_ref_modes = np.where(valid_mask)[0]
        valid_subj_modes = perm[valid_mask]
        
        subj_eig = eigenvalues[i, valid_subj_modes]
        subj_lig_sens = lig_sens[i, valid_subj_modes, :]
        subj_pret_sens = pret_sens[i, valid_subj_modes, :]
        
        # Run propagation for this subject
        df_subj = first_order_propagation(
            subj_lig_sens, subj_pret_sens, subj_eig,
            stiffness_baseline, pretension_baseline,
            bundle_names, spec
        )
        
        # Compute individual ligament contributions
        Sigma = build_input_covariance(spec, bundle_names)
        n_lig = len(bundle_names)
        
        MIN_EIG = 1e-12
        safe_eig = np.where(np.abs(subj_eig) < MIN_EIG, np.nan, subj_eig)
        J_k = subj_lig_sens * (stiffness_baseline[np.newaxis, :] / safe_eig[:, np.newaxis])
        J_T = subj_pret_sens * (pretension_baseline[np.newaxis, :] / safe_eig[:, np.newaxis])
        
        for m in range(len(subj_eig)):
            if np.isnan(safe_eig[m]):
                continue
            
            # For each ligament, compute its marginal variance contribution
            for l_idx, bname in enumerate(bundle_names):
                jk = J_k[m, l_idx]
                jt = J_T[m, l_idx]
                
                var_k = Sigma[l_idx, l_idx]
                var_t = Sigma[n_lig + l_idx, n_lig + l_idx]
                cov_kt = Sigma[l_idx, n_lig + l_idx]
                
                # Marginal variance if only this ligament was uncertain
                marg_var = (jk**2 * var_k) + (jt**2 * var_t) + (2 * jk * jt * cov_kt)
                
                # Add to df_subj
                df_subj.loc[df_subj["mode"] == (m + 1), f"marg_var_{bname}"] = marg_var
        
        # Add subject metadata
        df_subj["subject_id"] = i
        df_subj["sex"] = "Male" if sex[i] == "M" else "Female"
        
        # The mode column in df_subj is 1-based index of the input arrays.
        # We need to map it back to the reference mode index (1-based).
        df_subj["mode"] = valid_ref_modes + 1
        
        all_records.append(df_subj)
        
    return pd.concat(all_records, ignore_index=True)


def compute_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean statistics grouped by mode and sex."""
    # Calculate gap Z-scores per subject first
    df = df.sort_values(["subject_id", "mode"]).copy()

    # Shift to get next mode's eigenvalue/std, then keep only truly adjacent
    # mode transitions m -> m+1.
    df["next_eig"] = df.groupby("subject_id")["eigenvalue"].shift(-1)
    df["next_mode"] = df.groupby("subject_id")["mode"].shift(-1)
    adjacent_mask = df["next_mode"] == (df["mode"] + 1)
    df.loc[~adjacent_mask, "next_eig"] = np.nan
    df["next_std"] = df.groupby("subject_id")["std_log_lambda"].shift(-1) * df["next_eig"]

    # Current mode's linear std
    df["curr_std"] = df["std_log_lambda"] * df["eigenvalue"]

    # Gap and Z-score
    df["gap"] = df["next_eig"] - df["eigenvalue"]
    df["gap_std"] = np.sqrt(df["curr_std"] ** 2 + df["next_std"] ** 2)
    df["gap_z_score"] = np.where(
        np.isfinite(df["gap"]) & np.isfinite(df["gap_std"]) & (df["gap_std"] > 0.0),
        df["gap"] / df["gap_std"],
        np.nan,
    )

    def _p10_or_nan(series: pd.Series) -> float:
        finite = np.asarray(series[np.isfinite(series)], dtype=float)
        if finite.size == 0:
            return float("nan")
        return float(np.percentile(finite, 10))

    # Aggregate
    agg_dict = {
        "mean_CoV": ("CoV_lambda", "mean"),
        "mean_frac_global": ("frac_global", "mean"),
        "mean_frac_type": ("frac_type", "mean"),
        "mean_frac_asym": ("frac_asym", "mean"),
        "mean_gap_z": ("gap_z_score", "mean"),
        "min_gap_z": ("gap_z_score", "min"),
        "p10_gap_z": ("gap_z_score", _p10_or_nan),
    }
    
    # Add marginal variances
    marg_cols = [c for c in df.columns if c.startswith("marg_var_")]
    for c in marg_cols:
        agg_dict[f"mean_{c}"] = (c, "mean")
        
    summary = df.groupby(["mode", "sex"]).agg(**agg_dict).reset_index()
    
    # Normalize marginal variances to get relative importance
    for sex in ["Male", "Female"]:
        mask = summary["sex"] == sex
        total_marg = summary.loc[mask, [f"mean_{c}" for c in marg_cols]].sum(axis=1)
        denom = total_marg.where(total_marg > 0.0, np.nan)
        for c in marg_cols:
            summary.loc[mask, f"rel_imp_{c.replace('marg_var_', '')}"] = (
                summary.loc[mask, f"mean_{c}"] / denom
            )

    return summary


def generate_latex_table(summary: pd.DataFrame, output_path: Path) -> None:
    """Generate a LaTeX table for the manuscript."""
    # Pivot to have Male/Female side by side
    pivot = summary.pivot(index="mode", columns="sex", values=[
        "mean_CoV", "mean_frac_global", "mean_gap_z", "p10_gap_z"
    ])
    
    # Flatten columns
    pivot.columns = [f"{col[0]}_{col[1]}" if len(col) > 1 and col[1] else col[0] for col in pivot.columns]
    pivot = pivot.reset_index()
    
    # Filter to first 15 modes
    pivot = pivot[pivot["mode"] <= 15]
    
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Population-level ligament uncertainty propagation stratified by sex. CoV is the coefficient of variation of the eigenvalue. Global fraction is the percentage of variance explained by the shared laxity factor. Gap $Z$-score measures the robustness of the spectral gap to the next mode ($Z > 2$ indicates 95\,\% confidence against mode crossing due to soft-tissue noise).}",
        r"\label{tab:ligament_uncertainty}",
        r"\begin{tabular}{l ccc ccc}",
        r"\toprule",
        r"& \multicolumn{3}{c}{Female ($N=150$)} & \multicolumn{3}{c}{Male ($N=128$)} \\",
        r"\cmidrule(lr){2-4} \cmidrule(lr){5-7}",
        r"Mode & CoV (\%) & Global (\%) & Gap $Z$ (p10) & CoV (\%) & Global (\%) & Gap $Z$ (p10) \\",
        r"\midrule"
    ]
    
    for _, row in pivot.iterrows():
        m = int(row["mode"])
        
        # Female
        cov_f = row["mean_CoV_Female"] * 100
        glob_f = row["mean_frac_global_Female"] * 100
        z_f = row["mean_gap_z_Female"]
        p10_z_f = row["p10_gap_z_Female"]
        
        # Male
        cov_m = row["mean_CoV_Male"] * 100
        glob_m = row["mean_frac_global_Male"] * 100
        z_m = row["mean_gap_z_Male"]
        p10_z_m = row["p10_gap_z_Male"]
        
        # Format gap Z-score (handle NaN for mode 15)
        z_str_f = f"{z_f:.1f} ({p10_z_f:.1f})" if pd.notna(z_f) else "--"
        z_str_m = f"{z_m:.1f} ({p10_z_m:.1f})" if pd.notna(z_m) else "--"
        
        line = f"{m} & {cov_f:.1f} & {glob_f:.1f} & {z_str_f} & {cov_m:.1f} & {glob_m:.1f} & {z_str_m} \\\\"
        lines.append(line)
        
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}"
    ])
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")
    print(f"Saved LaTeX table to {output_path}")


def plot_ligament_importance(summary: pd.DataFrame, output_path: Path) -> None:
    """Plot stacked bar chart of relative ligament importance per mode."""
    setup_plot_style()

    # Group bilateral ligaments into types for cleaner plotting
    types = ["symphisys", "anterior_SIJ", "posterior_SIJ", "INL", "SS", "ST"]

    # Create a new dataframe with grouped types
    grouped = pd.DataFrame()
    grouped["mode"] = summary["mode"]
    grouped["sex"] = summary["sex"]

    for t in types:
        cols = [c for c in summary.columns if c.startswith("rel_imp_") and t in c]
        grouped[t] = summary[cols].sum(axis=1)

    # Plot Female and Male side by side
    fig, axes = plt.subplots(
        2, 1, figsize=(FULL_WIDTH, 2 * ROW_H + 0.6),
        sharex=True, layout="constrained",
    )

    from utils.plot_utils import _COLOR_CYCLE
    colors = _COLOR_CYCLE[:len(types)]

    for i, sex in enumerate(["Female", "Male"]):
        ax = axes[i]
        data = grouped[grouped["sex"] == sex].set_index("mode")[types]

        # Plot stacked bars
        bottom = np.zeros(len(data))
        for j, t in enumerate(types):
            ax.bar(data.index, data[t] * 100, bottom=bottom,
                   label=t.replace("_", " ").title(), color=colors[j])
            bottom += data[t] * 100

        ax.set_ylabel("Relative Importance (%)")
        panel_label(ax, f"{'A' if i == 0 else 'B'}. {sex} Ligament Contribution")
        ax.set_ylim(0, 100)

        if i == 0:
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.2),
                      ncol=len(types), frameon=False, fontsize=ANNOT_SIZE)

    axes[1].set_xlabel("Mode Index")
    axes[1].set_xticks(range(1, 16))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("Processing population UQ...")
    df = process_population()
    df.to_csv(OUTPUT_DIR / "population_variance_decomposition.csv", index=False)
    
    print("Computing summary statistics...")
    summary = compute_summary_stats(df)
    summary.to_csv(OUTPUT_DIR / "population_uq_summary.csv", index=False)
    
    print("Generating LaTeX table...")
    generate_latex_table(summary, OUTPUT_DIR / "ligament_uncertainty_table.tex")
    
    print("Plotting ligament importance...")
    plot_ligament_importance(summary, OUTPUT_DIR / "ligament_importance.pdf")
    
    print("Done.")


if __name__ == "__main__":
    main()
