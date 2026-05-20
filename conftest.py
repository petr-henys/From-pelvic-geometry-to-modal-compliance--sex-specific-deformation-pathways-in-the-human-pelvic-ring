#!/usr/bin/env python3
"""Project-level pytest configuration for root-level test runs.

Keeps collection focused on the active codebase and avoids legacy snapshots in
``old/`` that contain duplicate test module names.
"""

collect_ignore = ["old"]


def pytest_configure(config) -> None:
    """Register project markers when pytest is invoked from repository root."""
    markers = [
        "slow: marks tests as slow (deselect with '-m \"not slow\"')",
        "simulation: finite-element simulation tests (single-rank only)",
        "materials: constitutive and material parameter tests",
        "storage: persistence and I/O validation tests",
        "analysis: analysis and post-processing utilities tests",
        "configuration: configuration and parametrization tests",
        "regression: stability and regression tests",
    ]
    for marker in markers:
        config.addinivalue_line("markers", marker)
