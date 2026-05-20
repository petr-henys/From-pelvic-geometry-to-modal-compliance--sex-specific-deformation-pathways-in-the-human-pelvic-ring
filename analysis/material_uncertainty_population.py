#!/usr/bin/env python3
"""Population-level uncertainty propagation for all material parameter groups.

Extends ``ligament_uncertainty_population.py`` by adding cartilage and
bone density-law parameters, producing a unified variance decomposition
across all three independent parameter groups (ligament, cartilage, bone).
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
    build_input_covariance,
    first_order_propagation,
)
from run_material_uncertainty import (
    CARTILAGE_PARAMS,
    BoneUncertaintySpec,
    CartilageUncertaintySpec,
    bone_first_order_propagation,
    build_bone_covariance,
    build_cartilage_covariance,
    cov_from_log_variance,
    cartilage_first_order_propagation,
)
from analysis.spectral_data import load_metadata

# ── Paths ────────────────────────────────────────────────────────────────
RESULTS_DIR = Path("results/ref_S1P_fixed_new2")
DATA_DIR = RESULTS_DIR / "data"
METADATA_PATH = RESULTS_DIR / "simulation_metadata.json"
OUTPUT_DIR = Path("analysis_outputs/material_uncertainty")


# ── Data loading ─────────────────────────────────────────────────────────

def load_population_data() -> dict[str, np.ndarray]:
    """Load all five sensitivity arrays, eigenvalues and permutations."""
    arrays: dict[str, np.ndarray] = {}
    for name in [
        "lig_sensitivities",
        "pretension_sensitivities",
        "cartilage_sensitivities",
        "bone_alpha_sensitivities",
        "bone_beta_sensitivities",
        "eigenvalues",
        "eig_permutations",
    ]:
        z = zarr.open_group(str(DATA_DIR / f"{name}.zarr"), mode="r")
        arrays[name] = np.asarray(z["data"][:])
    return arrays


def load_baselines() -> dict:
    """Load baseline parameter values from simulation metadata."""
    with METADATA_PATH.open() as f:
        metadata = json.load(f)
    cfg = metadata.get("config", {})

    ligaments_dict = cfg.get("LIGAMENTS", {})
    bundle_names = cfg.get("LIGAMENT_NAMES", list(ligaments_dict.keys()))

    return {
        "bundle_names": bundle_names,
        "stiffness_baseline": np.array(
            [ligaments_dict[n]["stiffness"] for n in bundle_names], dtype=float
        ),
        "pretension_baseline": np.array(
            [ligaments_dict[n]["pretension"] for n in bundle_names], dtype=float
        ),
        "cartilage_moduli": np.array(
            [cfg["SIJ_CARTILAGE_MODULUS"]] * 2 + [cfg["SYMPHYSIS_MODULUS"]],
            dtype=float,
        ),
        "bone_alpha": float(cfg["BONE_MODULUS_ALPHA"]),
        "bone_beta": float(cfg["BONE_MODULUS_BETA"]),
    }


# ── Per-subject propagation ─────────────────────────────────────────────

def process_population(
    lig_spec: UncertaintySpec | None = None,
    cart_spec: CartilageUncertaintySpec | None = None,
    bone_spec: BoneUncertaintySpec | None = None,
) -> pd.DataFrame:
    """Run first-order propagation for all subjects, all three groups."""
    if lig_spec is None:
        lig_spec = UncertaintySpec()
    if cart_spec is None:
        cart_spec = CartilageUncertaintySpec()
    if bone_spec is None:
        bone_spec = BoneUncertaintySpec()

    data = load_population_data()
    baselines = load_baselines()
    sex, _ = load_metadata()

    lig_sens = data["lig_sensitivities"]
    pret_sens = data["pretension_sensitivities"]
    cart_sens = data["cartilage_sensitivities"]
    alpha_sens = data["bone_alpha_sensitivities"]
    beta_sens = data["bone_beta_sensitivities"]
    eigenvalues = data["eigenvalues"]
    permutations = data["eig_permutations"]

    bundle_names = baselines["bundle_names"]
    stiffness_bl = baselines["stiffness_baseline"]
    pretension_bl = baselines["pretension_baseline"]
    cart_moduli = baselines["cartilage_moduli"]
    alpha_bl = baselines["bone_alpha"]
    beta_bl = baselines["bone_beta"]

    n_subjects = lig_sens.shape[0]

    # Precompute covariance matrices (shared across subjects)
    Sigma_lig = build_input_covariance(lig_spec, bundle_names)
    Sigma_cart = build_cartilage_covariance(cart_spec)
    Sigma_bone = build_bone_covariance(bone_spec)
    n_lig = len(bundle_names)

    all_records: list[dict] = []

    for i in range(n_subjects):
        perm = permutations[i]
        valid_mask = perm >= 0
        valid_ref_modes = np.where(valid_mask)[0]
        valid_subj_modes = perm[valid_mask].astype(int)

        subj_eig = eigenvalues[i, valid_subj_modes]
        subj_lig = lig_sens[i, valid_subj_modes, :]
        subj_pret = pret_sens[i, valid_subj_modes, :]
        subj_cart = cart_sens[i, valid_subj_modes, :]
        subj_alpha = alpha_sens[i, valid_subj_modes]
        subj_beta = beta_sens[i, valid_subj_modes]

        # ── Ligament propagation ──
        df_lig = first_order_propagation(
            subj_lig, subj_pret, subj_eig,
            stiffness_bl, pretension_bl, bundle_names, lig_spec,
        )

        # ── Cartilage propagation ──
        df_cart = cartilage_first_order_propagation(
            subj_cart, subj_eig, cart_moduli, cart_spec,
        )

        # ── Bone propagation ──
        df_bone = bone_first_order_propagation(
            subj_alpha, subj_beta, subj_eig,
            alpha_bl, beta_bl, bone_spec,
        )

        # ── Merge per-mode ──
        n_valid = len(subj_eig)
        for m_idx in range(n_valid):
            mode_1based = m_idx + 1  # mode index in df_lig/df_cart/df_bone

            lig_row = df_lig[df_lig["mode"] == mode_1based]
            cart_row = df_cart[df_cart["mode"] == mode_1based]
            bone_row = df_bone[df_bone["mode"] == mode_1based]

            if lig_row.empty or cart_row.empty or bone_row.empty:
                continue

            lr = lig_row.iloc[0]
            cr = cart_row.iloc[0]
            br = bone_row.iloc[0]

            var_lig = lr["var_log_lambda"]
            var_cart = cr["var_log_lambda_cart"]
            var_bone = br["var_log_lambda_bone"]
            var_total = var_lig + var_cart + var_bone

            rec = {
                "subject_id": i,
                "sex": "Male" if sex[i] == "M" else "Female",
                "mode": int(valid_ref_modes[m_idx] + 1),
                "eigenvalue": float(subj_eig[m_idx]),
                # ── Total ──
                "var_total": var_total,
                "std_total": float(np.sqrt(max(var_total, 0.0))),
                "CoV_total": cov_from_log_variance(var_total),
                # ── Ligament ──
                "var_lig": var_lig,
                "CoV_lig": lr["CoV_lambda"],
                "frac_lig": var_lig / var_total if var_total > 0 else np.nan,
                "lig_frac_global": lr["frac_global"],
                "lig_frac_type": lr["frac_type"],
                "lig_frac_asym": lr["frac_asym"],
                # ── Cartilage ──
                "var_cart": var_cart,
                "CoV_cart": cr["CoV_lambda_cart"],
                "frac_cart": var_cart / var_total if var_total > 0 else np.nan,
                "cart_frac_global": cr["cart_frac_global"],
                "cart_frac_type": cr["cart_frac_type"],
                "cart_frac_asym": cr["cart_frac_asym"],
                # ── Bone ──
                "var_bone": var_bone,
                "CoV_bone": br["CoV_lambda_bone"],
                "frac_bone": var_bone / var_total if var_total > 0 else np.nan,
                "bone_frac_alpha": br["bone_frac_alpha"],
                "bone_frac_beta": br["bone_frac_beta"],
                "bone_frac_cross": br["bone_frac_cross"],
            }

            # Ligament per-bundle marginal variances
            MIN_EIG = 1e-12
            safe_eig_m = subj_eig[m_idx] if abs(subj_eig[m_idx]) > MIN_EIG else np.nan
            if not np.isnan(safe_eig_m):
                jk = subj_lig[m_idx] * (stiffness_bl / safe_eig_m)
                jt = subj_pret[m_idx] * (pretension_bl / safe_eig_m)
                for l_idx, bname in enumerate(bundle_names):
                    var_k = Sigma_lig[l_idx, l_idx]
                    var_t = Sigma_lig[n_lig + l_idx, n_lig + l_idx]
                    cov_kt = Sigma_lig[l_idx, n_lig + l_idx]
                    rec[f"marg_var_{bname}"] = float(
                        jk[l_idx] ** 2 * var_k
                        + jt[l_idx] ** 2 * var_t
                        + 2 * jk[l_idx] * jt[l_idx] * cov_kt
                    )

                # Cartilage per-region marginal
                jc = subj_cart[m_idx] * (cart_moduli / safe_eig_m)
                for j, cname in enumerate(CARTILAGE_PARAMS):
                    rec[f"marg_var_{cname}"] = float(jc[j] ** 2 * Sigma_cart[j, j])

                # Bone per-param marginal
                ja = subj_alpha[m_idx] * (alpha_bl / safe_eig_m)
                jb = subj_beta[m_idx] * (beta_bl / safe_eig_m)
                rec["marg_var_bone_alpha"] = float(ja ** 2 * Sigma_bone[0, 0])
                rec["marg_var_bone_beta"] = float(jb ** 2 * Sigma_bone[1, 1])

            all_records.append(rec)

    return pd.DataFrame.from_records(all_records)


# ── Summary statistics ───────────────────────────────────────────────────

def compute_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean statistics grouped by mode and sex."""
    df = df.sort_values(["subject_id", "mode"]).copy()

    # Gap Z-scores (using total uncertainty), only for truly adjacent modes m -> m+1.
    df["next_eig"] = df.groupby("subject_id")["eigenvalue"].shift(-1)
    df["next_mode"] = df.groupby("subject_id")["mode"].shift(-1)
    adjacent_mask = df["next_mode"] == (df["mode"] + 1)

    df.loc[~adjacent_mask, "next_eig"] = np.nan
    df["next_std"] = df.groupby("subject_id")["std_total"].shift(-1) * df["next_eig"]
    df["curr_std"] = df["std_total"] * df["eigenvalue"]
    df["gap"] = df["next_eig"] - df["eigenvalue"]
    df["gap_std"] = np.sqrt(df["curr_std"] ** 2 + df["next_std"] ** 2)
    df["gap_z_score"] = np.where(
        np.isfinite(df["gap"]) & np.isfinite(df["gap_std"]) & (df["gap_std"] > 0),
        df["gap"] / df["gap_std"],
        np.nan,
    )

    def _p10_or_nan(series: pd.Series) -> float:
        finite = np.asarray(series[np.isfinite(series)], dtype=float)
        if finite.size == 0:
            return float("nan")
        return float(np.percentile(finite, 10))

    agg_dict = {
        # Total
        "mean_CoV_total": ("CoV_total", "mean"),
        "mean_var_total": ("var_total", "mean"),
        # Group fractions
        "mean_frac_lig": ("frac_lig", "mean"),
        "mean_frac_cart": ("frac_cart", "mean"),
        "mean_frac_bone": ("frac_bone", "mean"),
        # Group CoVs
        "mean_CoV_lig": ("CoV_lig", "mean"),
        "mean_CoV_cart": ("CoV_cart", "mean"),
        "mean_CoV_bone": ("CoV_bone", "mean"),
        # Ligament internal decomposition
        "mean_lig_frac_global": ("lig_frac_global", "mean"),
        "mean_lig_frac_type": ("lig_frac_type", "mean"),
        "mean_lig_frac_asym": ("lig_frac_asym", "mean"),
        # Bone decomposition
        "mean_bone_frac_alpha": ("bone_frac_alpha", "mean"),
        "mean_bone_frac_beta": ("bone_frac_beta", "mean"),
        # Gap Z
        "mean_gap_z": ("gap_z_score", "mean"),
        "min_gap_z": ("gap_z_score", "min"),
        "p10_gap_z": ("gap_z_score", _p10_or_nan),
    }

    # Marginal variances
    marg_cols = [c for c in df.columns if c.startswith("marg_var_")]
    for c in marg_cols:
        agg_dict[f"mean_{c}"] = (c, "mean")

    summary = df.groupby(["mode", "sex"]).agg(**agg_dict).reset_index()

    # Relative importance of every marginal
    for s in ["Male", "Female"]:
        mask = summary["sex"] == s
        total_marg = summary.loc[mask, [f"mean_{c}" for c in marg_cols]].sum(axis=1)
        denom = total_marg.where(total_marg > 0.0, np.nan)
        for c in marg_cols:
            summary.loc[mask, f"rel_imp_{c.replace('marg_var_', '')}"] = (
                summary.loc[mask, f"mean_{c}"] / denom
            )

    return summary


# ── LaTeX table ──────────────────────────────────────────────────────────

def generate_latex_table(summary: pd.DataFrame, output_path: Path) -> None:
    """Generate a LaTeX table of unified material uncertainty."""
    pivot = summary.pivot(index="mode", columns="sex", values=[
        "mean_CoV_total", "mean_frac_lig", "mean_frac_cart", "mean_frac_bone",
        "mean_gap_z", "p10_gap_z",
    ])
    pivot.columns = [f"{v}_{s}" for v, s in pivot.columns]
    pivot = pivot.reset_index()
    pivot = pivot[pivot["mode"] <= 15]

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\caption{Population-level material uncertainty propagation stratified by sex."
        r" CoV is the total coefficient of variation (ligament + cartilage + bone)."
        r" Lig/Cart/Bone columns show each group's fraction of total variance."
        r" Gap $Z$-score measures robustness of each spectral gap to combined"
        r" material noise ($Z > 2$: 95\,\% confidence).}",
        r"\label{tab:material_uncertainty}",
        r"\begin{tabular}{l cccc cccc}",
        r"\toprule",
        r"& \multicolumn{4}{c}{Female ($N=150$)} & \multicolumn{4}{c}{Male ($N=128$)} \\",
        r"\cmidrule(lr){2-5} \cmidrule(lr){6-9}",
        r"Mode & CoV & Lig & Cart & Bone & CoV & Lig & Cart & Bone \\",
        r"     & (\%) & (\%) & (\%) & (\%) & (\%) & (\%) & (\%) & (\%) \\",
        r"\midrule",
    ]

    for _, row in pivot.iterrows():
        m = int(row["mode"])
        # Female
        cov_f = row["mean_CoV_total_Female"] * 100
        lig_f = row["mean_frac_lig_Female"] * 100
        cart_f = row["mean_frac_cart_Female"] * 100
        bone_f = row["mean_frac_bone_Female"] * 100
        # Male
        cov_m = row["mean_CoV_total_Male"] * 100
        lig_m = row["mean_frac_lig_Male"] * 100
        cart_m = row["mean_frac_cart_Male"] * 100
        bone_m = row["mean_frac_bone_Male"] * 100

        line = (
            f"{m} & {cov_f:.1f} & {lig_f:.0f} & {cart_f:.0f} & {bone_f:.0f}"
            f" & {cov_m:.1f} & {lig_m:.0f} & {cart_m:.0f} & {bone_m:.0f} \\\\"
        )
        lines.append(line)

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")
    print(f"Saved LaTeX table → {output_path}")


# ── Gap Z-score table ────────────────────────────────────────────────────

def generate_gap_table(summary: pd.DataFrame, output_path: Path) -> None:
    """Generate LaTeX table of gap Z-scores under combined uncertainty."""
    pivot = summary.pivot(index="mode", columns="sex", values=[
        "mean_gap_z", "p10_gap_z",
    ])
    pivot.columns = [f"{v}_{s}" for v, s in pivot.columns]
    pivot = pivot.reset_index()
    pivot = pivot[pivot["mode"] <= 14]  # gap between m and m+1

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Spectral gap $Z$-scores under combined material uncertainty."
        r" Mean (p10) is the population mean and 10th-percentile of"
        r" $Z_m = (\lambda_{m+1} - \lambda_m) / \sigma_{\Delta\lambda}$.}",
        r"\label{tab:gap_z_material}",
        r"\begin{tabular}{l cc}",
        r"\toprule",
        r"Gap & Female & Male \\",
        r"\midrule",
    ]

    for _, row in pivot.iterrows():
        m = int(row["mode"])
        z_f = row["mean_gap_z_Female"]
        p10_f = row["p10_gap_z_Female"]
        z_m = row["mean_gap_z_Male"]
        p10_m = row["p10_gap_z_Male"]

        zf_str = f"{z_f:.1f} ({p10_f:.1f})" if pd.notna(z_f) else "--"
        zm_str = f"{z_m:.1f} ({p10_m:.1f})" if pd.notna(z_m) else "--"

        lines.append(f"${m}$--${m + 1}$ & {zf_str} & {zm_str} \\\\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")
    print(f"Saved gap Z table → {output_path}")


# ── Main ─────────────────────────────────────────────────────────────────

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Processing population UQ (all material groups) …")
    df = process_population()
    df.to_csv(OUTPUT_DIR / "population_material_uq.csv", index=False)
    print(f"  Saved {len(df)} records → population_material_uq.csv")

    print("Computing summary statistics …")
    summary = compute_summary_stats(df)
    summary.to_csv(OUTPUT_DIR / "population_material_summary.csv", index=False)

    print("Generating LaTeX tables …")
    generate_latex_table(summary, OUTPUT_DIR / "material_uncertainty_table.tex")
    generate_gap_table(summary, OUTPUT_DIR / "gap_z_material_table.tex")

    # ── Quick summary to stdout ──
    for s in ["Female", "Male"]:
        sub = summary[summary["sex"] == s]
        print(f"\n{'─' * 40}")
        print(f"  {s}")
        print(f"{'─' * 40}")
        for _, row in sub.iterrows():
            m = int(row["mode"])
            print(
                f"  Mode {m:2d}: CoV_total={row['mean_CoV_total'] * 100:.1f}%"
                f"  (L {row['mean_frac_lig'] * 100:.0f}%"
                f"  C {row['mean_frac_cart'] * 100:.0f}%"
                f"  B {row['mean_frac_bone'] * 100:.0f}%)"
            )

    print("\nDone.")


if __name__ == "__main__":
    main()
