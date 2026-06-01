"""Phase 5E — VAL-012 drift_detection_runner."""

from __future__ import annotations

from pathlib import Path

import pytest

from mmm.validation.synthetic.certification_runner import run_world_certification
from mmm.validation.synthetic.dgp_materializer import materialize_dgp_world
from mmm.validation.synthetic.drift_detection_runner import (
    RUNNER_VERSION,
    run_drift_detection_for_bundle,
    run_val_012_drift_detection,
)
from mmm.validation.synthetic.reliability_truth import build_world_011_truth
from mmm.validation.synthetic.trust_report_semantics import build_trust_report_interpretation

REPO_ROOT = Path(__file__).resolve().parents[1]
WORLD_011 = REPO_ROOT / "validation" / "worlds" / "WORLD-011-drift-recovery"


@pytest.fixture(scope="module")
def world_011_bundle(tmp_path_factory: pytest.TempPathFactory) -> Path:
    bundle = tmp_path_factory.mktemp("w011") / "WORLD-011-drift-recovery"
    bundle.mkdir(parents=True)
    truth = build_world_011_truth(seed=11011)
    (bundle / "world_truth.json").write_text(
        __import__("json").dumps(truth, indent=2) + "\n",
        encoding="utf-8",
    )
    src_cfg = REPO_ROOT / "validation" / "worlds" / "WORLD-011-drift-recovery" / "train_config.yaml"
    if src_cfg.is_file():
        (bundle / "train_config.yaml").write_text(src_cfg.read_text(encoding="utf-8"), encoding="utf-8")
    materialize_dgp_world(bundle, overwrite=True)
    return bundle


def test_runner_version() -> None:
    assert RUNNER_VERSION == "drift_detection_runner_v1.0.0"


def test_world_011_drift_detection(world_011_bundle: Path) -> None:
    truth = __import__("json").loads((world_011_bundle / "world_truth.json").read_text(encoding="utf-8"))
    result = run_val_012_drift_detection(world_011_bundle, truth)
    assert result.drift_expected is True
    assert result.val_012_outcome in ("pass", "warning")
    assert result.drift_severity_level in ("moderate", "severe", "minor")
    assert result.post_pre_mae_ratio is not None
    assert float(result.post_pre_mae_ratio) >= 1.15
    assert result.fit_degradation is True


def test_certification_wires_val_012(world_011_bundle: Path) -> None:
    report = run_world_certification(world_011_bundle, write_report=False, include_recovery=True).report
    drift_rows = [
        r
        for r in report.get("recovery_validation_results") or []
        if r.get("check_id") == "REC-4B5-DRIFT"
    ]
    assert drift_rows
    details = drift_rows[0].get("details") or {}
    assert details.get("runner_version") == RUNNER_VERSION
    assert details.get("registry_validation_id") == "VAL-012"


def test_trust_report_interpretation_from_scorecard() -> None:
    scorecard = {
        "decision_reliability_score": 0.95,
        "structural_reliability_score": 1.0,
        "attribution_diagnostic_score": 0.4,
        "trust_modifier_status": {"status": "caution", "drift_severity_max": "moderate"},
        "capability_summary": {},
    }
    interp = build_trust_report_interpretation(scorecard)
    assert interp["decision_usable"] is True
    assert interp["attribution_safe"] is False
    assert "interpretation_matrix" in interp


@pytest.mark.skipif(not WORLD_011.is_dir(), reason="WORLD-011 bundle not present")
def test_repo_world_011_runner() -> None:
    result = run_drift_detection_for_bundle(WORLD_011)
    assert result.world_id == "WORLD-011-drift-recovery"
