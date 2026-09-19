from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import validate_manuscript_results as vmr


def test_manuscript_no_stale_keywords():
    failures = vmr.check_stale_keywords()
    assert not failures, f"Stale keywords found in manuscript: {failures}"


def test_manuscript_values_macros():
    failures = vmr.check_values_macros()
    assert not failures, f"Macro mismatches in values.tex: {failures}"


def test_manuscript_table_e1_concordance():
    failures = vmr.check_table_e1_concordance()
    assert not failures, f"Table E1 mismatches: {failures}"


def test_cube_overlap_analytical():
    failures = vmr.check_cube_overlap_analytical()
    assert not failures, f"Analytical cube overlap failures: {failures}"
