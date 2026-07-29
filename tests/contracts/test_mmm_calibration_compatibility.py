"""Public calibration-compatibility contract and evaluator regressions."""

from __future__ import annotations

import inspect
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

import mmm.contracts as contract_package
from mmm.contracts.calibration_compatibility import (
    MMM_CALIBRATION_COMPATIBILITY_RULE_ORDER,
    MMMCalibrationCompatibilityReasonCode,
    MMMCalibrationCompatibilityRequest,
    MMMCalibrationCompatibilityResult,
    MMMCalibrationCompatibilityState,
    MMMCalibrationFreshnessDecision,
    MMMCalibrationMethodStatus,
    MMMCalibrationUncertaintyDecision,
    MMMCalibrationUncertaintyStatus,
    evaluate_mmm_calibration_compatibility,
    parse_mmm_calibration_compatibility_result,
)

NOW = datetime(2026, 7, 29, 12, 0, tzinfo=timezone.utc)
FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "mip_export" / "calibration_compatibility_v1"


def _request(**source_changes: Any) -> MMMCalibrationCompatibilityRequest:
    source: dict[str, Any] = {
        "source_readout_id": "readout-001",
        "source_readout_version": "v1",
        "source_producer_package": "experiment-producer",
        "source_producer_commit": "commit-001",
        "handoff_eligible": True,
        "kpi_id": "revenue",
        "unit": "USD",
        "estimand_id": "incremental_revenue",
        "effect_scale": "absolute_incremental",
        "channel_id": "search",
        "geography": "national",
        "grain": "weekly",
        "evidence_window_start": NOW - timedelta(days=28),
        "evidence_window_end": NOW,
        "observed_at": NOW - timedelta(days=2),
        "fresh_through": NOW + timedelta(days=14),
        "uncertainty_status": MMMCalibrationUncertaintyStatus.AVAILABLE,
        "method_family": "governed_experiment",
        "instrument_identity": "instrument-001",
        "method_status": MMMCalibrationMethodStatus.GOVERNED,
    }
    model: dict[str, Any] = {
        "model_id": "ridge-model-001",
        "run_id": "ridge-run-001",
        "model_family": "ridge",
        "kpi_id": "revenue",
        "unit": "USD",
        "estimand_id": "incremental_revenue",
        "effect_scale": "absolute_incremental",
        "channel_id": "search",
        "geography": "national",
        "grain": "weekly",
        "model_window_start": NOW - timedelta(days=28),
        "model_window_end": NOW,
        "evaluated_at": NOW,
        "uncertainty_status": MMMCalibrationUncertaintyStatus.AVAILABLE,
    }
    lineage: dict[str, Any] = {
        "evidence_artifact_id": "evidence-001",
        "model_artifact_id": "model-artifact-001",
        "configuration_hash": "config-001",
        "panel_id": "panel-001",
        "evaluation_id": "evaluation-001",
    }
    for key, value in source_changes.pop("model", {}).items():
        model[key] = value
    for key, value in source_changes.pop("lineage", {}).items():
        lineage[key] = value
    source.update(source_changes)
    return MMMCalibrationCompatibilityRequest.model_validate(
        {"source": source, "model_context": model, "lineage": lineage}
    )


def test_exact_rule_order_and_compatible_result_are_deterministic() -> None:
    request = _request()
    result = evaluate_mmm_calibration_compatibility(request)

    assert result.compatibility_state == MMMCalibrationCompatibilityState.COMPATIBLE
    assert result.evaluated_rule_order == MMM_CALIBRATION_COMPATIBILITY_RULE_ORDER
    assert result.reason_codes == []
    assert result.freshness_decision == MMMCalibrationFreshnessDecision.FRESH
    assert result.uncertainty_decision == MMMCalibrationUncertaintyDecision.COMPATIBLE
    assert result.to_json() == result.to_json()
    assert MMMCalibrationCompatibilityResult.from_json(result.to_json()) == result


@pytest.mark.parametrize(
    ("changes", "state", "reason"),
    [
        (
            {"kpi_id": "conversions"},
            MMMCalibrationCompatibilityState.INCOMPATIBLE,
            MMMCalibrationCompatibilityReasonCode.KPI_MISMATCH,
        ),
        (
            {"estimand_id": "incremental_conversions"},
            MMMCalibrationCompatibilityState.INCOMPATIBLE,
            MMMCalibrationCompatibilityReasonCode.ESTIMAND_MISMATCH,
        ),
        (
            {"channel_id": "social"},
            MMMCalibrationCompatibilityState.INCOMPATIBLE,
            MMMCalibrationCompatibilityReasonCode.CHANNEL_MISMATCH,
        ),
        (
            {"grain": "daily"},
            MMMCalibrationCompatibilityState.INCOMPATIBLE,
            MMMCalibrationCompatibilityReasonCode.GRAIN_MISMATCH,
        ),
        (
            {
                "evidence_window_start": NOW + timedelta(days=1),
                "evidence_window_end": NOW + timedelta(days=2),
            },
            MMMCalibrationCompatibilityState.INCOMPATIBLE,
            MMMCalibrationCompatibilityReasonCode.TIME_WINDOW_NO_OVERLAP,
        ),
        (
            {"fresh_through": NOW - timedelta(seconds=1)},
            MMMCalibrationCompatibilityState.STALE,
            MMMCalibrationCompatibilityReasonCode.EVIDENCE_STALE,
        ),
        (
            {
                "uncertainty_status": MMMCalibrationUncertaintyStatus.UNAVAILABLE,
                "model": {"uncertainty_status": MMMCalibrationUncertaintyStatus.UNAVAILABLE},
            },
            MMMCalibrationCompatibilityState.COMPATIBLE_WITH_WARNING,
            MMMCalibrationCompatibilityReasonCode.UNCERTAINTY_UNAVAILABLE,
        ),
        (
            {"method_status": MMMCalibrationMethodStatus.DIAGNOSTIC_ONLY},
            MMMCalibrationCompatibilityState.BLOCKED,
            MMMCalibrationCompatibilityReasonCode.METHOD_STATUS_NOT_GOVERNED,
        ),
    ],
)
def test_explicit_compatibility_rules_produce_all_required_state_classes(
    changes: dict[str, Any], state: MMMCalibrationCompatibilityState, reason: MMMCalibrationCompatibilityReasonCode
) -> None:
    result = evaluate_mmm_calibration_compatibility(_request(**changes))
    assert result.compatibility_state == state
    assert reason in result.reason_codes
    if state == MMMCalibrationCompatibilityState.BLOCKED:
        assert result.blockers and result.warnings == []
    elif state == MMMCalibrationCompatibilityState.COMPATIBLE_WITH_WARNING:
        assert result.warnings and result.blockers == []
    else:
        assert result.blockers == []


@pytest.mark.parametrize(
    ("source_field", "model_field", "reason"),
    [
        ("kpi_id", "kpi_id", MMMCalibrationCompatibilityReasonCode.KPI_MISMATCH),
        ("unit", "unit", MMMCalibrationCompatibilityReasonCode.UNIT_MISMATCH),
        ("estimand_id", "estimand_id", MMMCalibrationCompatibilityReasonCode.ESTIMAND_MISMATCH),
        ("effect_scale", "effect_scale", MMMCalibrationCompatibilityReasonCode.EFFECT_SCALE_MISMATCH),
        ("channel_id", "channel_id", MMMCalibrationCompatibilityReasonCode.CHANNEL_MISMATCH),
        ("geography", "geography", MMMCalibrationCompatibilityReasonCode.GEOGRAPHY_MISMATCH),
        ("grain", "grain", MMMCalibrationCompatibilityReasonCode.GRAIN_MISMATCH),
    ],
)
def test_identity_mismatches_are_incompatible(
    source_field: str, model_field: str, reason: MMMCalibrationCompatibilityReasonCode
) -> None:
    request = _request(**{source_field: f"different-{source_field}"})
    assert getattr(request.source, source_field) != getattr(request.model_context, model_field)
    result = evaluate_mmm_calibration_compatibility(request)
    assert result.compatibility_state == MMMCalibrationCompatibilityState.INCOMPATIBLE
    assert reason in result.reason_codes


def test_handoff_blocker_has_precedence_and_preserves_ordered_reason_codes() -> None:
    result = evaluate_mmm_calibration_compatibility(
        _request(
            handoff_eligible=False,
            method_status=MMMCalibrationMethodStatus.BLOCKED,
            kpi_id="conversions",
            fresh_through=NOW - timedelta(seconds=1),
            uncertainty_status=MMMCalibrationUncertaintyStatus.UNAVAILABLE,
        )
    )
    assert result.compatibility_state == MMMCalibrationCompatibilityState.BLOCKED
    assert result.reason_codes == [
        MMMCalibrationCompatibilityReasonCode.HANDOFF_NOT_ELIGIBLE,
        MMMCalibrationCompatibilityReasonCode.METHOD_STATUS_NOT_GOVERNED,
        MMMCalibrationCompatibilityReasonCode.KPI_MISMATCH,
        MMMCalibrationCompatibilityReasonCode.EVIDENCE_STALE,
        MMMCalibrationCompatibilityReasonCode.UNCERTAINTY_MISMATCH,
    ]


def test_missing_and_unknown_required_identities_fail_closed() -> None:
    payload = _request().model_dump(mode="json")
    del payload["source"]["source_readout_id"]
    with pytest.raises(ValidationError):
        MMMCalibrationCompatibilityRequest.model_validate(payload)
    with pytest.raises(ValidationError, match="safe known identity"):
        _request(model={"model_id": "unknown"})


def test_strict_parser_rejects_missing_unknown_and_inconsistent_result_fields() -> None:
    result = evaluate_mmm_calibration_compatibility(_request())
    payload = result.to_json_dict()
    parsed = parse_mmm_calibration_compatibility_result(payload)
    assert parsed == result
    payload.pop("lineage")
    with pytest.raises(ValueError, match="required envelope fields"):
        parse_mmm_calibration_compatibility_result(payload)
    with pytest.raises(ValueError, match="unsupported"):
        parse_mmm_calibration_compatibility_result({**result.to_json_dict(), "schema_version": "v2"})


def test_public_contract_is_exported_and_evaluator_has_no_model_side_effect_paths() -> None:
    request = _request()
    before = json.dumps(request.model_dump(mode="json"), sort_keys=True)
    result = evaluate_mmm_calibration_compatibility(request)
    after = json.dumps(request.model_dump(mode="json"), sort_keys=True)
    source = inspect.getsource(evaluate_mmm_calibration_compatibility).lower()

    assert before == after
    assert result.model_context is request.model_context
    assert "simulate(" not in source and "fit(" not in source and "recommend" in source
    assert contract_package.MMMCalibrationCompatibilityResult is MMMCalibrationCompatibilityResult
    assert contract_package.parse_mmm_calibration_compatibility_result is parse_mmm_calibration_compatibility_result


@pytest.mark.parametrize(
    "name",
    [
        "compatible.json",
        "compatible_with_warning.json",
        "stale.json",
        "incompatible.json",
        "blocked.json",
    ],
)
def test_deterministic_state_fixtures_parse_and_round_trip(name: str) -> None:
    payload = json.loads((FIXTURE_ROOT / name).read_text(encoding="utf-8"))
    result = parse_mmm_calibration_compatibility_result(payload)
    assert result.to_json() == json.dumps(payload, sort_keys=True, separators=(",", ":"))
