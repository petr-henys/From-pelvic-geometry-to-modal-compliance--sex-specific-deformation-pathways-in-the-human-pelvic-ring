#!/usr/bin/env python3
"""Format clean publication LaTeX tables 4A and 4B."""

from pathlib import Path
import pandas as pd

PAPER_ROOT = Path("manuscripts/paper_JAnat_SIJ_micromotion")
GEN_DIR = PAPER_ROOT / "tables" / "generated"

p = pd.read_csv(GEN_DIR / "primary_models_cluster_robust.csv")
a = pd.read_csv(GEN_DIR / "allometry_models.csv")

term_map = {
    "C(load_case, Treatment(reference='SP2leg'))[T.SP1leg]": "SP1leg (vs SP2leg)",
    "C(load_case, Treatment(reference='SP2leg'))[T.LAB_phase1]": "LAB1 (vs SP2leg)",
    "C(load_case, Treatment(reference='SP2leg'))[T.LAB_phase2]": "LAB2 (vs SP2leg)",
    "C(load_case, Treatment(reference='SP2leg'))[T.LAB_phase3]": "LAB3 (vs SP2leg)",
    "sex_F": "Female sex",
    "age_z": "Age ($z$)",
    "log_total_volume_z": "Log pelvic volume ($z$)",
    "C(load_case, Treatment(reference='SP2leg'))[T.SP1leg]:log_total_volume_z": "SP1leg $\\times$ Log volume ($z$)",
    "C(load_case, Treatment(reference='SP2leg'))[T.LAB_phase1]:log_total_volume_z": "LAB1 $\\times$ Log volume ($z$)",
    "C(load_case, Treatment(reference='SP2leg'))[T.LAB_phase2]:log_total_volume_z": "LAB2 $\\times$ Log volume ($z$)",
    "C(load_case, Treatment(reference='SP2leg'))[T.LAB_phase3]:log_total_volume_z": "LAB3 $\\times$ Log volume ($z$)",
    "AP_z": "Inlet AP diameter ($z$)",
    "BiischiadicWidth_z": "Biischiadic width ($z$)",
    "SubpubicAngle_z": "Subpubic angle ($z$)",
    "sex_F:log_total_volume_z": "Female $\\times$ Log volume ($z$)",
}

outcome_map = {
    "rot_mag_deg": "Rotation magnitude ($|\\boldsymbol{\\theta}|$, deg)",
    "trans_mag_mm": "Translation magnitude ($|\\mathbf{d}|$, mm)",
}

# ----------------- Table 4A -----------------
lines_4a = [
    r"\begin{tabular}{llccc}",
    r"\toprule",
    r"Outcome & Predictor & $\beta$ [95\% CI] & Outcome-SD $\beta$ & $q_{\text{FDR}}$ \\",
    r"\midrule",
]

for outcome in ["rot_mag_deg", "trans_mag_mm"]:
    sub = p[p["outcome"] == outcome]
    first = True
    for _, row in sub.iterrows():
        t = row["term"]
        if t not in term_map:
            continue
        term_clean = term_map[t]
        b = row["beta"]
        ci_l = row["ci95_low"]
        ci_h = row["ci95_high"]
        b_std = row["beta_std"]
        q = row["p_fdr_bh"]

        q_str = "< 0.001" if q < 0.001 else f"{q:.3f}"
        b_ci = f"{b:.3f} [{ci_l:.3f}, {ci_h:.3f}]"

        out_col = outcome_map[outcome] if first else ""
        first = False
        lines_4a.append(
            f"{out_col} & {term_clean} & {b_ci} & {b_std:.3f} & {q_str} \\\\"
        )
    if outcome == "rot_mag_deg":
        lines_4a.append(r"\midrule")

lines_4a.extend([r"\bottomrule", r"\end{tabular}"])

with open(GEN_DIR / "table4a_primary_models.tex", "w") as f:
    f.write("\n".join(lines_4a) + "\n")

# ----------------- Table 4B -----------------
lines_4b = [
    r"\begin{tabular}{llccccc}",
    r"\toprule",
    r"Load case & Outcome & Male slope [95\% CI] & Female slope [95\% CI] & Interaction $\beta$ & $q_{\text{FDR}}$ & Adj.\ $R^2$ \\",
    r"\midrule",
]

load_map = {
    "SP2leg": "SP2leg",
    "SP1leg": "SP1leg",
    "LAB_phase1": "LAB1",
    "LAB_phase2": "LAB2",
    "LAB_phase3": "LAB3",
}

for lc in ["SP2leg", "SP1leg", "LAB_phase1", "LAB_phase2", "LAB_phase3"]:
    sub = a[a["load_case"] == lc]
    first = True
    for _, row in sub.iterrows():
        out_name = (
            r"Rotation ($|\boldsymbol{\theta}|$)"
            if row["outcome"] == "rot_mag_deg"
            else r"Translation ($|\mathbf{d}|$)"
        )
        m_exp = f"{row['male_exponent']:.3f} [{row['male_ci95_low']:.3f}, {row['male_ci95_high']:.3f}]"
        f_exp = f"{row['female_exponent']:.3f} [{row['female_ci95_low']:.3f}, {row['female_ci95_high']:.3f}]"
        b_int = f"{row['sex_slope_interaction_beta']:.3f}"
        q_int = (
            "< 0.001"
            if row["interaction_p_fdr_bh"] < 0.001
            else f"{row['interaction_p_fdr_bh']:.3f}"
        )
        r2 = f"{row['adj_r2']:.3f}"

        lc_col = load_map[lc] if first else ""
        first = False
        lines_4b.append(
            f"{lc_col} & {out_name} & {m_exp} & {f_exp} & {b_int} & {q_int} & {r2} \\\\"
        )
    if lc != "LAB_phase3":
        lines_4b.append(r"\midrule")

lines_4b.extend([r"\bottomrule", r"\end{tabular}"])

with open(GEN_DIR / "table4b_allometry.tex", "w") as f:
    f.write("\n".join(lines_4b) + "\n")

print("Tables 4A and 4B successfully formatted.")

