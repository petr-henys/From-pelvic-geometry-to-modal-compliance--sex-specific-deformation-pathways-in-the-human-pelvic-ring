import pandas as pd
from pathlib import Path

out_dir = Path("analysis_outputs/mode_mechanisms")
df = pd.read_csv(out_dir / "geometry_audit.csv")
inv_df = pd.read_csv(out_dir / "inverted_cells_details.csv")

n_total = len(df)
n_clean = int(df["is_strictly_valid"].sum())
n_with_inv = n_total - n_clean

print(f"Total subjects: {n_total}")
print(f"Subjects without inverted cells (is_strictly_valid): {n_clean} ({n_clean/n_total*100:.1f}%)")
print(f"Subjects with inverted cells: {n_with_inv} ({n_with_inv/n_total*100:.1f}%)")
print(f"Total inverted cells across all subjects: {len(inv_df)}")
print(f"Max inverted cells in any single subject: {df['n_cells_inverted'].max()}")
mean_inv_pct = df.loc[df['n_cells_inverted'] > 0, 'inv_vol_pct'].mean()
print(f"Mean inverted volume fraction among affected subjects: {mean_inv_pct:.6f}%")
max_inv_pct = df['inv_vol_pct'].max()
print(f"Max inverted volume fraction in any subject: {max_inv_pct:.6f}%")

print("\nTissues where inverted cells appear:")
print(inv_df["material"].value_counts())

bone_inv = int(df["inv_PelvisBone"].sum())
sij_l_inv = int(df["inv_SIJCartilageLeft"].sum())
sij_r_inv = int(df["inv_SIJCartilageRight"].sum())
sym_inv = int(df["inv_PubicSymphysis"].sum())

report_content = f"""# Population Geometry and Jacobian Audit Report

**Date**: 2026-09-18  
**Subjects Audited**: {n_total}  
**Mesh Discretization**: 166,808 tetrahedral cells, 48,291 nodes  

---

## 1. Executive Summary

- **Subjects with 100% Strictly Positive Jacobians ($J > 0$ in all 166,808 cells)**:
  {n_clean} / {n_total} ({n_clean/n_total*100:.1f}%)
- **Subjects with Inverted Cells ($J \\le 0$)**:
  {n_with_inv} / {n_total} ({n_with_inv/n_total*100:.1f}%)
- **Total Inverted Cells Across Entire Cohort (out of {n_total * 166808:,} cell evaluations)**:
  {len(inv_df)} ({len(inv_df) / (n_total * 166808) * 100:.6f}%)
- **Maximum Inverted Cells in Any Single Subject**:
  {df['n_cells_inverted'].max()} cells (out of 166,808)
- **Maximum Inverted Volume Fraction in Any Single Subject**:
  {max_inv_pct:.6f}% of subject volume
- **Mean Inverted Volume Fraction in Affected Subjects**:
  {mean_inv_pct:.6f}% of subject volume

---

## 2. Inversion Distribution by Tissue

| Tissue | Inverted Cells Total | Affected Cell Percentage |
|---|---|---|
| PelvisBone (162,551 cells/subject) | {bone_inv} | {bone_inv / (n_total * 162551) * 100:.6f}% |
| SIJCartilageLeft (1,867 cells/subject) | {sij_l_inv} | {sij_l_inv / (n_total * 1867) * 100:.6f}% |
| SIJCartilageRight (1,817 cells/subject) | {sij_r_inv} | {sij_r_inv / (n_total * 1817) * 100:.6f}% |
| PubicSymphysis (573 cells/subject) | {sym_inv} | {sym_inv / (n_total * 573) * 100:.6f}% |

---

## 3. Methodological Decision for Population Mode Classification

1. **Admissible Integration Domain $\\Omega_{{\\rm adm}}$**:
   Inverted cells ($J \\le 0$) represent localized RBF boundary fitting artifacts on thin cortical boundaries or cartilage junctions. Because the volume fraction is negligible ($< 0.003\\%$ in the worst case), and following the project specification (Fáze A):
   - Inverted cells are strictly excluded from the volume-weighted integration:
     $$\\Omega_{{\\rm adm}} = \\{{c \\in \\Omega : J_c > 0\\}}$$
   - Any cell with $J_c \\le 0$ receives weight $w_c = 0$.
2. **Cohort Stratification**:
   - Primary reporting covers all 278 subjects with $\\Omega_{{\\rm adm}}$ cell-level filtering.
   - Sensitivity benchmark compares the {n_clean} clean subjects ($100\\%$ inversion-free) with the full $N=278$ cohort to verify that localized RBF boundary artifacts do not bias modal deformation percentages.
"""

(out_dir / "geometry_audit_report.md").write_text(report_content, encoding="utf-8")
print("Saved geometry_audit_report.md successfully!")
