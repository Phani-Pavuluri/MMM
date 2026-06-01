"""Phase 4A — synthetic world certification runner."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from mmm.validation.synthetic.certification_registry import (
    CERTIFICATION_RUNNER_VERSION,
    DEFERRED_CHECK_IDS,
    PHASE_4A_CHECK_IDS,
    REPORT_ARTIFACT_NAME,
)
from mmm.validation.synthetic.certification_runner import run_world_certification
from mmm.validation.synthetic.materializer import materialize_world
from mmm.validation.synthetic.scenario_builder import WORLD_007_REPLAY_DRIFT, write_scenario_world
from mmm.validation.synthetic.validator import verify_checksums

REPO_ROOT = Path(__file__).resolve().parents[1]
WORLD_001 = REPO_ROOT / "validation" / "worlds" / "WORLD-001-baseline"
WORLD_002 = REPO_ROOT / "validation" / "worlds" / "WORLD-002-replay"
WORLD_005 = REPO_ROOT / "validation" / "worlds" / "WORLD-005-scenario-low-noise"
WORLD_006 = REPO_ROOT / "validation" / "worlds" / "WORLD-006-scenario-high-collinearity"
WORLD_007 = REPO_ROOT / "validation" / "worlds" / "WORLD-007-scenario-replay-drift"

REQUIRED_REPORT_KEYS = frozenset(
    {
        "world_id",
        "world_version",
        "world_contract_version",
        "generator_version",
        "materialization_version",
        "certification_runner_version",
        "executed_validations",
        "skipped_validations",
        "failed_validations",
        "validation_results",
        "overall_status",
        "warnings",
        "limitations",
        "contract_compatibility",
        "decision_surface_compatibility",
        "replay_compatibility",
        "trust_semantics_compatibility",
    }
)


def _materialize(src: Path, tmp_path: Path) -> Path:
    bundle = tmp_path / src.name
    shutil.copytree(src, bundle, dirs_exist_ok=True)
    materialize_world(bundle, overwrite=True)
    return bundle


@pytest.fixture(scope="module")
def baseline_bundle(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _materialize(WORLD_001, tmp_path_factory.mktemp("cert"))


@pytest.fixture(scope="module")
def replay_bundle(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _materialize(WORLD_002, tmp_path_factory.mktemp("cert"))


def test_baseline_world_certification_passes(baseline_bundle: Path) -> None:
    result = run_world_certification(baseline_bundle, write_report=True)
    assert result.passed
    assert result.report["overall_status"] == "pass"
    assert (baseline_bundle / REPORT_ARTIFACT_NAME).is_file()
    assert result.report["certification_runner_version"] == CERTIFICATION_RUNNER_VERSION


def test_replay_world_certification_passes(replay_bundle: Path) -> None:
    result = run_world_certification(replay_bundle)
    assert result.passed
    assert "CERT-4A-003" in result.report["executed_validations"]
    assert "CERT-4A-011" in result.report["executed_validations"]


def test_checksum_tamper_fails_certification(tmp_path: Path) -> None:
    bundle = _materialize(WORLD_001, tmp_path)
    panel = bundle / "panel.parquet"
    df = pd.read_parquet(panel)
    col = df.columns[-1]
    df.iloc[0, df.columns.get_loc(col)] = df.iloc[0][col] + 1.0
    df.to_parquet(panel, index=False)
    result = run_world_certification(bundle, write_report=False)
    assert not result.passed
    assert "CERT-4A-002" in result.report["failed_validations"]


def test_deferred_validations_explicitly_skipped(baseline_bundle: Path) -> None:
    result = run_world_certification(baseline_bundle, write_report=False)
    skipped_ids = {row["check_id"] for row in result.report["skipped_validations"]}
    assert skipped_ids == DEFERRED_CHECK_IDS
    for row in result.report["skipped_validations"]:
        assert row["skip_reason"] in (
            "requires_rich_dgp_worlds",
            "requires_train_decide_execution",
            "requires_thresholds",
        )
        vr = next(v for v in result.report["validation_results"] if v["check_id"] == row["check_id"])
        assert vr["status"] == "skipped"
        assert vr["status"] != "pass"


def test_contract_compatibility_rollups(baseline_bundle: Path) -> None:
    result = run_world_certification(baseline_bundle, write_report=False)
    assert result.report["contract_compatibility"]["passed"] is True
    assert result.report["decision_surface_compatibility"]["passed"] is True
    assert result.report["replay_compatibility"]["passed"] is True
    assert result.report["trust_semantics_compatibility"]["passed"] is True


def test_governance_warning_propagation(tmp_path: Path) -> None:
    bundle = _materialize(WORLD_006, tmp_path)
    result = run_world_certification(bundle, write_report=False)
    assert result.passed
    assert any("identifiability_collinearity" in w for w in result.report["warnings"])


def test_malformed_bundle_missing_truth(tmp_path: Path) -> None:
    bundle = tmp_path / "WORLD-empty"
    bundle.mkdir()
    result = run_world_certification(bundle)
    assert not result.passed
    assert result.report["overall_status"] == "error"


def test_replay_incompatibility_fails(tmp_path: Path) -> None:
    write_scenario_world(WORLD_007, WORLD_007_REPLAY_DRIFT)
    bundle = _materialize(WORLD_007, tmp_path)
    replay = json.loads((bundle / "replay_units.json").read_text(encoding="utf-8"))
    replay[0]["estimand"] = "unsupported_estimand_xyz"
    (bundle / "replay_units.json").write_text(json.dumps(replay, indent=2) + "\n", encoding="utf-8")
    result = run_world_certification(bundle, write_report=False)
    assert not result.passed
    assert "CERT-4A-010" in result.report["failed_validations"]


def test_certification_artifact_schema_completeness(baseline_bundle: Path) -> None:
    result = run_world_certification(baseline_bundle, write_report=False)
    missing = REQUIRED_REPORT_KEYS - set(result.report.keys())
    assert not missing, missing
    assert set(result.report["executed_validations"]) <= PHASE_4A_CHECK_IDS


def test_empty_decision_scenarios_still_valid(tmp_path: Path) -> None:
    """WORLD-007 ships with no decision scenarios — structure check must pass."""
    write_scenario_world(WORLD_007, WORLD_007_REPLAY_DRIFT)
    bundle = _materialize(WORLD_007, tmp_path)
    result = run_world_certification(bundle, write_report=False)
    cert_007 = next(v for v in result.report["validation_results"] if v["check_id"] == "CERT-4A-007")
    assert cert_007["status"] == "pass"


def test_malformed_replay_payload_fails(tmp_path: Path) -> None:
    bundle = _materialize(WORLD_002, tmp_path)
    (bundle / "replay_units.json").write_text("{not json", encoding="utf-8")
    result = run_world_certification(bundle, write_report=False)
    assert not result.passed
    assert any(
        v["check_id"] in ("CERT-4A-003", "CERT-4A-008", "CERT-4A-011") and v["status"] == "fail"
        for v in result.report["validation_results"]
    )


def test_transform_incompatibility_fails(tmp_path: Path) -> None:
    bundle = _materialize(WORLD_001, tmp_path)
    truth = json.loads((bundle / "world_truth.json").read_text(encoding="utf-8"))
    truth["transform_truth"]["adstock_family"] = "weibull"
    (bundle / "world_truth.json").write_text(json.dumps(truth, indent=2) + "\n", encoding="utf-8")
    result = run_world_certification(bundle, write_report=False)
    assert not result.passed
    assert "CERT-4A-004" in result.report["failed_validations"]


def test_negative_invalid_estimand_in_truth(tmp_path: Path) -> None:
    write_scenario_world(WORLD_007, WORLD_007_REPLAY_DRIFT)
    bundle = _materialize(WORLD_007, tmp_path)
    truth = json.loads((bundle / "world_truth.json").read_text(encoding="utf-8"))
    truth["experiment_truth"]["units"][0]["estimand"] = "invalid_estimand"
    (bundle / "world_truth.json").write_text(json.dumps(truth, indent=2) + "\n", encoding="utf-8")
    result = run_world_certification(bundle, write_report=False)
    assert "CERT-4A-010" in result.report["failed_validations"]


def test_negative_decision_surface_wrong_model_form(tmp_path: Path) -> None:
    bundle = _materialize(WORLD_001, tmp_path)
    truth = json.loads((bundle / "world_truth.json").read_text(encoding="utf-8"))
    truth["outcome_truth"]["model_form"] = "log_log"
    (bundle / "world_truth.json").write_text(json.dumps(truth, indent=2) + "\n", encoding="utf-8")
    result = run_world_certification(bundle, write_report=False)
    assert "CERT-4A-009" in result.report["failed_validations"]
    assert "CERT-4A-004" in result.report["failed_validations"]


def test_negative_missing_trust_report_gates(tmp_path: Path) -> None:
    bundle = _materialize(WORLD_001, tmp_path)
    truth = json.loads((bundle / "world_truth.json").read_text(encoding="utf-8"))
    truth["artifact_truth"]["expected_gates"] = []
    (bundle / "world_truth.json").write_text(json.dumps(truth, indent=2) + "\n", encoding="utf-8")
    result = run_world_certification(bundle, write_report=False)
    assert "CERT-4A-012" in result.report["failed_validations"]


def test_negative_malformed_calibration_signal(tmp_path: Path) -> None:
    bundle = _materialize(WORLD_002, tmp_path)
    replay = json.loads((bundle / "replay_units.json").read_text(encoding="utf-8"))
    replay[0]["lift_se"] = 0.0
    (bundle / "replay_units.json").write_text(json.dumps(replay, indent=2) + "\n", encoding="utf-8")
    result = run_world_certification(bundle, write_report=False)
    assert "CERT-4A-011" in result.report["failed_validations"]


def test_negative_release_gate_incompatible(tmp_path: Path) -> None:
    bundle = _materialize(WORLD_001, tmp_path)
    truth = json.loads((bundle / "world_truth.json").read_text(encoding="utf-8"))
    truth["governance_truth"]["model_release_state"] = "unknown_state"
    truth["governance_truth"]["replay_calibration_active"] = True
    truth["experiment_truth"]["units"] = []
    (bundle / "world_truth.json").write_text(json.dumps(truth, indent=2) + "\n", encoding="utf-8")
    result = run_world_certification(bundle, write_report=False)
    assert "CERT-4A-013" in result.report["failed_validations"]


def test_scenario_world_005_certifies(tmp_path: Path) -> None:
    bundle = _materialize(WORLD_005, tmp_path)
    result = run_world_certification(bundle, write_report=False)
    assert result.passed


def test_verify_checksums_aligns_with_cert_runner(baseline_bundle: Path) -> None:
    assert verify_checksums(baseline_bundle) == []
