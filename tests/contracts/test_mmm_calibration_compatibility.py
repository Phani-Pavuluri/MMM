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


def test_research_only_method_status_is_blocked_by_name() -> None:
    result = evaluate_mmm_calibration_compatibility(
        _request(method_status=MMMCalibrationMethodStatus.RESEARCH_ONLY)
    )
    assert result.compatibility_state == MMMCalibrationCompatibilityState.BLOCKED
    assert MMMCalibrationCompatibilityReasonCode.METHOD_STATUS_NOT_GOVERNED in result.reason_codes
    assert result.blockers and result.warnings == []


def test_partial_window_overlap_warns_without_overriding_mismatch_precedence() -> None:
    warning_only = evaluate_mmm_calibration_compatibility(
        _request(
            model={
                "model_window_start": NOW - timedelta(days=14),
                "model_window_end": NOW + timedelta(days=14),
            }
        )
    )
    assert warning_only.compatibility_state == MMMCalibrationCompatibilityState.COMPATIBLE_WITH_WARNING
    assert warning_only.reason_codes == [MMMCalibrationCompatibilityReasonCode.TIME_WINDOW_PARTIAL_OVERLAP]

    with_mismatch = evaluate_mmm_calibration_compatibility(
        _request(
            kpi_id="conversions",
            model={
                "model_window_start": NOW - timedelta(days=14),
                "model_window_end": NOW + timedelta(days=14),
            },
        )
    )
    assert with_mismatch.compatibility_state == MMMCalibrationCompatibilityState.INCOMPATIBLE
    assert MMMCalibrationCompatibilityReasonCode.KPI_MISMATCH in with_mismatch.reason_codes
    assert MMMCalibrationCompatibilityReasonCode.TIME_WINDOW_PARTIAL_OVERLAP in [
        issue.reason_code for issue in with_mismatch.warnings
    ]


def test_naive_datetimes_and_unknown_source_identities_fail_closed() -> None:
    naive_start = NOW.replace(tzinfo=None) - timedelta(days=28)
    naive_end = NOW.replace(tzinfo=None)
    with pytest.raises(ValidationError):
        _request(evidence_window_start=naive_start, evidence_window_end=naive_end)
    with pytest.raises(ValidationError, match="safe known identity"):
        _request(source_readout_id="unknown")
    with pytest.raises(ValidationError, match="safe known identity"):
        _request(channel_id="n/a")


def test_tampered_rule_order_and_duplicate_reason_codes_are_rejected() -> None:
    ordered = evaluate_mmm_calibration_compatibility(_request()).to_json_dict()
    tampered_order = {
        **ordered,
        "evaluated_rule_order": list(reversed(ordered["evaluated_rule_order"])),
    }
    with pytest.raises(ValidationError):
        parse_mmm_calibration_compatibility_result(tampered_order)

    mismatched = evaluate_mmm_calibration_compatibility(_request(kpi_id="conversions")).to_json_dict()
    assert mismatched["reason_codes"] == [MMMCalibrationCompatibilityReasonCode.KPI_MISMATCH.value]
    duplicated = {**mismatched, "reason_codes": [mismatched["reason_codes"][0]] * 2}
    with pytest.raises(ValidationError):
        parse_mmm_calibration_compatibility_result(duplicated)


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
