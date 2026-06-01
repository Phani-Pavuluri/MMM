"""Phase 5C — exact recovery investigation (INV-056) smoke tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mmm.validation.synthetic.exact_recovery_investigation import (
    analyze_world,
    list_exact_recovery_bundles,
    regularization_sweep,
    run_full_investigation,
)

REPO = Path(__file__).resolve().parents[1]
WORLD_008 = REPO / "validation" / "worlds" / "WORLD-008-exact-recovery"


def test_list_exact_recovery_includes_world_008() -> None:
    bundles = list_exact_recovery_bundles(REPO)
    assert any("WORLD-008" in b.name for b in bundles)


def test_analyze_world_008_structure() -> None:
    row = analyze_world(WORLD_008)
    assert row["world_id"] == "WORLD-008-exact-recovery"
    assert "fitted_transforms" in row
    assert "truth_pinned_transforms" in row
    bo = row["fitted_transforms"]["coefficient_recovery"]
    assert bo["pass"] is False
    assert bo["max_abs_error"] > 0.5


def test_truth_pinned_better_than_bo_on_world_008() -> None:
    row = analyze_world(WORLD_008)
    bo_err = row["fitted_transforms"]["coefficient_recovery"]["max_abs_error"]
    tp_err = row["truth_pinned_transforms"]["coefficient_recovery"]["max_abs_error"]
    assert tp_err < bo_err


def test_regularization_sweep_runs() -> None:
    out = regularization_sweep(WORLD_008, log_alphas=(-6, -4))
    assert len(out["sweep"]) == 2


@pytest.mark.slow
def test_full_investigation_keys() -> None:
    findings = run_full_investigation(REPO)
    assert findings["investigation_id"] == "INV-056"
    assert findings["recovery_decomposition"]
    assert findings["recovery_taxonomy"]


def test_report_artifact_exists() -> None:
    report = REPO / "docs" / "05_validation" / "exact_recovery_investigation_report.md"
    if not report.is_file():
        pytest.skip("run write_investigation_report to generate")
    assert "Executive summary" in report.read_text(encoding="utf-8")
    inv = REPO / "docs" / "05_validation" / "investigations" / "exact_recovery_findings.json"
    assert inv.is_file()
    data = json.loads(inv.read_text(encoding="utf-8"))
    assert data["recovery_decomposition"]
