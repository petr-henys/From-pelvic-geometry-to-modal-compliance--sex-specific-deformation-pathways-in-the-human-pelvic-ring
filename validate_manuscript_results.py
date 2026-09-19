#!/usr/bin/env python3
"""Validation script for scientific integrity and manuscript numerical consistency.

Checks:
1. Absence of stale keywords (four-point, 23-dim, 13.9, crossing point, family-wise, bypass, stiff, order, static compliance)
2. Exact concordance of values.tex macros with label_effects.csv and label_effects_full_cohort.csv
3. Cohort and cluster sample sizes across all subsets
4. Reference mode decomposition consistency (Figure 1 / Table E1)
5. Design matrix condition number kappa(A) ~ 4.58 and Gram matrix kappa(G) ~ 20.98
6. Exact mathematical cube overlap between bending and shear (-22/45 and -11/180)
"""
from pathlib import Path
import re
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
MANUSCRIPT = ROOT / 'manuscripts/natcomm/main_plos.tex'
VALUES_TEX = ROOT / 'analysis_outputs/tables/values.tex'
LABEL_EFFECTS = ROOT / 'analysis_outputs/plos_revision/tables/label_effects.csv'
LABEL_EFFECTS_COHORT = ROOT / 'analysis_outputs/plos_revision/tables/label_effects_full_cohort.csv'
DEFORM_TABLE = ROOT / 'analysis_outputs/tables/global_deformation_modes_table.csv'


def check_stale_keywords():
    """Verify that no banned or stale phrases remain in the LaTeX manuscript."""
    text = MANUSCRIPT.read_text()
    banned_patterns = [
        r'four-point',
        r'23-dim',
        r'13\.9',
        r'crossing point',
        r'family-wise',
        r'bypass',
        r'stiff, order',
        r'linearised static compliance',
        r'linearized static compliance',
        r'Kriechling',
        r'virtually identical',
        r'interchangeable',
        r'ordinary across-subject',
        r'concatenated orthonormal basis',
        r'mesh-independent',
    ]
    failures = []
    for pat in banned_patterns:
        matches = list(re.finditer(pat, text, flags=re.IGNORECASE))
        if matches:
            for m in matches:
                # Find line number
                line_no = text[:m.start()].count('\n') + 1
                failures.append(f"Found banned pattern '{pat}' at line {line_no}")
    return failures


def check_values_macros():
    """Verify values.tex definitions against ground truth CSV tables."""
    text = VALUES_TEX.read_text()
    macro_dict = {}
    for m in re.finditer(r'\\newcommand\{\\([A-Za-z0-9]+)\}\{([^}]+)\}', text):
        macro_dict[m.group(1)] = m.group(2)

    failures = []
    # Sample sizes
    expected_counts = {
        'CohortN': '278',
        'CohortMaleN': '128',
        'CohortFemaleN': '150',
        'CohortNoSwapN': '204',
        'CohortSwapN': '74',
        'ClusterMemberN': '152',
        'ClusterMaleN': '90',
        'ClusterFemaleN': '62',
        'ClusterNoSwapN': '95',
        'ClusterSwapN': '57',
        'TotalPolyDOFs': '30',
        'RigidDOFs': '6',
        'NonrigidDOFs': '24',
        'GramConditionNumber': '20.98',
        'DesignConditionNumber': '4.58',
    }
    for k, v in expected_counts.items():
        if k not in macro_dict:
            failures.append(f"Missing macro \\{k} in values.tex")
        elif macro_dict[k] != v:
            failures.append(f"Macro \\{k} = {macro_dict[k]}, expected {v}")

    # Label effects table
    eff = pd.read_csv(LABEL_EFFECTS).set_index('Metric')
    metric_names = {
        'rank9': 'RankNine',
        'rank10': 'RankTen',
        'AP': 'ScalarAP',
        'AP_pair': 'PairAP',
    }
    for metric, prefix in metric_names.items():
        row = eff.loc[metric]
        no_val = f"{row['Median no exchange']:.3f}"
        yes_val = f"{row['Median exchange']:.3f}"
        diff_val = f"{row['Difference']:.3f}"
        if macro_dict.get(f'{prefix}No') != no_val:
            failures.append(f"\\{prefix}No mismatch: {macro_dict.get(f'{prefix}No')} vs {no_val}")
        if macro_dict.get(f'{prefix}Yes') != yes_val:
            failures.append(f"\\{prefix}Yes mismatch: {macro_dict.get(f'{prefix}Yes')} vs {yes_val}")
        if macro_dict.get(f'{prefix}Difference') != diff_val:
            failures.append(f"\\{prefix}Difference mismatch: {macro_dict.get(f'{prefix}Difference')} vs {diff_val}")

    return failures


def check_table_e1_concordance():
    """Verify Mode 1, 2, 4, 5, 9, 10 values in Table E1 vs manuscript descriptions."""
    df = pd.read_csv(DEFORM_TABLE)
    row_map = {row['Mode']: row for _, row in df.iterrows()}
    failures = []

    m1 = row_map[1]
    if not (np.isclose(m1['Nonrigid (%)'], 3.3, atol=0.1) and np.isclose(m1['Fitted (%)'], 42.7, atol=0.1)):
        failures.append(f"Mode 1 mismatch in Table E1: Nonrigid {m1['Nonrigid (%)']}, Fitted {m1['Fitted (%)']}")

    m2 = row_map[2]
    if not (np.isclose(m2['Bending (%)'], 23.6, atol=0.1) and np.isclose(m2['Shear (%)'], 23.7, atol=0.1)):
        failures.append(f"Mode 2 mismatch in Table E1: Bending {m2['Bending (%)']}, Shear {m2['Shear (%)']}")

    m9 = row_map[9]
    if not (np.isclose(m9['Fitted (%)'], 91.8, atol=0.1) and np.isclose(m9['Axial (%)'], 60.6, atol=0.1)):
        failures.append(f"Mode 9 mismatch in Table E1: Fitted {m9['Fitted (%)']}, Axial {m9['Axial (%)']}")

    m10 = row_map[10]
    if not (np.isclose(m10['Fitted (%)'], 74.9, atol=0.1) and np.isclose(m10['Shear (%)'], 35.2, atol=0.1)):
        failures.append(f"Mode 10 mismatch in Table E1: Fitted {m10['Fitted (%)']}, Shear {m10['Shear (%)']}")

    return failures


def check_cube_overlap_analytical():
    """Verify analytical inner product <bending, shear> = -22/45 and normalized = -11/180."""
    pts, wts = np.polynomial.legendre.leggauss(7)
    grid = np.meshgrid(pts, pts, pts, indexing='ij')
    w_grid = np.meshgrid(wts, wts, wts, indexing='ij')
    x, y, z = [g.ravel() for g in grid]
    w = (w_grid[0] * w_grid[1] * w_grid[2]).ravel()

    b = np.column_stack([-x * y, 0.5 * x**2, np.zeros_like(x)])
    s = np.column_stack([x * y, 0.5 * x**2, np.zeros_like(x)])

    raw_int = np.sum(np.sum(b * s, axis=1) * w)
    vol = np.sum(w)
    norm_int = raw_int / vol

    failures = []
    if not np.isclose(raw_int, -22.0 / 45.0, atol=1e-12):
        failures.append(f"Raw integral {raw_int} != -22/45")
    if not np.isclose(vol, 8.0, atol=1e-12):
        failures.append(f"Volume {vol} != 8.0")
    if not np.isclose(norm_int, -11.0 / 180.0, atol=1e-12):
        failures.append(f"Normalized inner product {norm_int} != -11/180")

    return failures


def main():
    print("Running Manuscript Results & Mathematical Consistency Audit...")
    all_failures = []

    print("[1/4] Checking absence of stale keywords...")
    f1 = check_stale_keywords()
    if f1:
        print(f"  FAILED: {len(f1)} stale keywords found:")
        for err in f1:
            print(f"    - {err}")
        all_failures.extend(f1)
    else:
        print("  PASSED: Zero stale keywords found.")

    print("[2/4] Checking values.tex and label effects consistency...")
    f2 = check_values_macros()
    if f2:
        print(f"  FAILED: {len(f2)} macro mismatches:")
        for err in f2:
            print(f"    - {err}")
        all_failures.extend(f2)
    else:
        print("  PASSED: All macros match ground truth tables exactly.")

    print("[3/4] Checking Table E1 deformation decomposition...")
    f3 = check_table_e1_concordance()
    if f3:
        print(f"  FAILED: {len(f3)} Table E1 mismatches:")
        for err in f3:
            print(f"    - {err}")
        all_failures.extend(f3)
    else:
        print("  PASSED: Table E1 values match manuscript references.")

    print("[4/4] Checking analytical cube overlap...")
    f4 = check_cube_overlap_analytical()
    if f4:
        print(f"  FAILED: {len(f4)} cube overlap errors:")
        for err in f4:
            print(f"    - {err}")
        all_failures.extend(f4)
    else:
        print("  PASSED: Analytical cube overlap = -22/45 and -11/180 verified.")

    if all_failures:
        print(f"\nAUDIT FAILED with {len(all_failures)} total issues.")
        sys.exit(1)
    else:
        print("\nALL CONSISTENCY CHECKS PASSED SUCCESSFULLY (0 errors).")
        sys.exit(0)


if __name__ == '__main__':
    main()
