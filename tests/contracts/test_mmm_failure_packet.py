"""Contract and producer-boundary tests for typed MMM failure packets."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from mmm.contracts.mip_export import parse_export_bundle
from mmm.contracts.mip_export_adapter import (
    MMMExportRuntimeContext,
    adapt_runtime_artifacts_to_export_outcome,
    emit_known_failure_outcome,
)
from mmm.contracts.mip_failure import (
    DEFAULT_FAILURE_POLICY,
    MMM_FAILURE_SCHEMA_VERSION,
    MMMExportOutcome,
    MMMFailureCode,
    MMMFailurePacket,
    MMMFailureStage,
    MMMRemediationAction,
    MMMRemediationActionCode,
    MMMRetryDisposition,
    build_mmm_failure_packet,
)

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "mip_export"
CREATED_AT = datetime(2026, 7, 13, 12, 0, tzinfo=timezone.utc)
REQUIRED_CODES = {
    "INSUFFICIENT_HISTORY",
    "INCOMPATIBLE_GRAIN",
    "KPI_NOT_SUPPORTED",
    "SPEND_VARIATION_INSUFFICIENT",
    "CONTROL_DATA_MISSING",
    "CALIBRATION_SCOPE_MISMATCH",
    "CALIBRATION_SIGNAL_EXPIRED",
    "MODEL_INSTABILITY",
    "HOLDOUT_FAILURE",
    "UNSUPPORTED_EXTRAPOLATION",
    "IDENTIFIABILITY_FAILURE",
    "MODEL_NOT_PROMOTED",
}


def _packet(**overrides: object) -> MMMFailurePacket:
    values: dict[str, object] = {
        "failure_id": "failure-001",
        "created_at": CREATED_AT,
        "code": MMMFailureCode.INSUFFICIENT_HISTORY,
        "stage": MMMFailureStage.DATA_VALIDATION,
        "source_component": "mmm.data.schema",
        "technical_summary": "Panel history does not meet the governed minimum.",
        "affected_resource": "panel.history_weeks",
    }
    values.update(overrides)
    return build_mmm_failure_packet(**values)  # type: ignore[arg-type]


def _context() -> MMMExportRuntimeContext:
    return MMMExportRuntimeContext(
        model_run_id="run-typed-failure-001",
        training_data_fingerprint="sha256:panel",
        model_artifact_fingerprint="sha256:model",
        generated_at="2026-07-13T12:00:00Z",
        package_version="0.1.0",
        git_commit="abc1234",
        model_form="semi_log",
        estimand="modeled_outcome",
        time_window="2026-01-01/2026-06-30",
        geo_scope="national",
        channel_scope=("search", "social"),
        outcome_metric="revenue",
        spend_metric="spend",
        currency="USD",
    )


def _fixture(name: str) -> dict[str, object]:
    return json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))


def test_complete_taxonomy_has_a_deterministic_retry_and_remediation_policy() -> None:
    assert {code.value for code in MMMFailureCode} == REQUIRED_CODES
    assert len({code.value for code in MMMFailureCode}) == len(MMMFailureCode)
    assert set(DEFAULT_FAILURE_POLICY) == set(MMMFailureCode)
    for retry, action in DEFAULT_FAILURE_POLICY.values():
        assert isinstance(retry, MMMRetryDisposition)
        assert isinstance(action, MMMRemediationActionCode)


def test_minimum_packet_is_versioned_and_allows_pre_run_failures() -> None:
    packet = _packet()
    assert packet.schema_version == MMM_FAILURE_SCHEMA_VERSION
    assert packet.run_id is None
    assert packet.model_id is None
    assert packet.retry_disposition == MMMRetryDisposition.RETRY_AFTER_INPUT_CHANGE
    assert packet.remediation_actions[0].action_code == MMMRemediationActionCode.INPUT_DATA


def test_model_calibration_promotion_and_plan_evidence_are_typed() -> None:
    calibration = _packet(
        code=MMMFailureCode.CALIBRATION_SCOPE_MISMATCH,
        stage=MMMFailureStage.CALIBRATION,
        calibration_signal_ids=["cal-signal-1"],
        affected_market="US",
    )
    assert calibration.retry_disposition == MMMRetryDisposition.RETRY_AFTER_CALIBRATION_CHANGE

    promotion = _packet(
        code=MMMFailureCode.MODEL_NOT_PROMOTED,
        stage=MMMFailureStage.PROMOTION_GATE,
        model_id="model-release-1",
        governance_blocker_ids=["promotion-not-approved"],
    )
    assert promotion.retry_disposition == MMMRetryDisposition.RETRY_AFTER_GOVERNANCE_CHANGE

    extrapolation = _packet(
        code=MMMFailureCode.UNSUPPORTED_EXTRAPOLATION,
        stage=MMMFailureStage.SIMULATION,
        supported_range_evidence=["spend_change_exceeds_validated_range"],
    )
    assert extrapolation.retry_disposition == MMMRetryDisposition.RETRY_AFTER_PLAN_CHANGE


def test_retry_policy_and_schema_validation_fail_closed() -> None:
    with pytest.raises(ValidationError, match="required remediation"):
        MMMFailurePacket(
            failure_id="failure-002",
            created_at=CREATED_AT,
            code=MMMFailureCode.INSUFFICIENT_HISTORY,
            stage=MMMFailureStage.DATA_VALIDATION,
            source_component="mmm.data.schema",
            technical_summary="Missing history.",
            retry_disposition=MMMRetryDisposition.RETRY_AFTER_INPUT_CHANGE,
            remediation_actions=[],
        )
    with pytest.raises(ValidationError, match="retry override"):
        _packet(retry_disposition=MMMRetryDisposition.RETRY_AFTER_PLAN_CHANGE)
    with pytest.raises(ValidationError):
        MMMFailurePacket.model_validate({**_packet().to_json_dict(), "schema_version": "unknown_failure_v9"})
    with pytest.raises(ValidationError, match="JSON-serializable"):
        _packet(technical_context={"captured": ValueError("do not serialize")})


def test_serialization_is_deterministic_and_round_trips_without_fabricating_optionals() -> None:
    packet = _packet(technical_context={"observed_history_weeks": 8})
    serialized = packet.to_json()
    assert serialized == packet.to_json()
    restored = MMMFailurePacket.from_json(serialized)
    assert restored == packet
    assert "run_id" not in packet.to_json_dict()
    assert "stack" not in serialized.lower()


def test_export_outcome_requires_exactly_one_payload_and_preserves_success() -> None:
    bundle = parse_export_bundle(_fixture("readiness_only_bundle.json"))
    success = MMMExportOutcome.success(bundle)
    failure = MMMExportOutcome.failure(_packet())
    assert success.export_bundle == bundle and success.failure_packet is None
    assert failure.failure_packet is not None and failure.export_bundle is None
    with pytest.raises(ValidationError, match="exactly one"):
        MMMExportOutcome(outcome_type="success")
    with pytest.raises(ValidationError, match="exactly one"):
        MMMExportOutcome(outcome_type="failure", export_bundle=bundle, failure_packet=_packet())


@pytest.mark.parametrize(
    ("code", "stage"),
    [
        (MMMFailureCode.INSUFFICIENT_HISTORY, MMMFailureStage.DATA_VALIDATION),
        (MMMFailureCode.CALIBRATION_SCOPE_MISMATCH, MMMFailureStage.CALIBRATION),
        (MMMFailureCode.HOLDOUT_FAILURE, MMMFailureStage.MODEL_VALIDATION),
        (MMMFailureCode.MODEL_NOT_PROMOTED, MMMFailureStage.PROMOTION_GATE),
        (MMMFailureCode.UNSUPPORTED_EXTRAPOLATION, MMMFailureStage.SIMULATION),
    ],
)
def test_producer_boundary_emits_explicitly_mapped_known_failures(
    code: MMMFailureCode, stage: MMMFailureStage
) -> None:
    outcome = emit_known_failure_outcome(
        failure_id=f"failure-{code.value.lower()}",
        created_at=CREATED_AT,
        code=code,
        stage=stage,
        source_component="mmm.contracts.mip_export_adapter",
        technical_summary=f"Known governed producer failure: {code.value}.",
        context=_context(),
        affected_resource="governed-input",
    )
    assert outcome.outcome_type == "failure"
    assert outcome.failure_packet is not None
    assert outcome.failure_packet.code == code
    assert outcome.failure_packet.run_id == "run-typed-failure-001"


def test_producer_boundary_wrapper_preserves_existing_success_behavior() -> None:
    outcome = adapt_runtime_artifacts_to_export_outcome(
        context=_context(), extension_report={"ridge_fit_summary": {"coef": [0.2]}}
    )
    assert outcome.outcome_type == "success"
    assert outcome.export_bundle is not None
    assert outcome.failure_packet is None


@pytest.mark.parametrize(
    "fixture_name",
    [
        "failure_insufficient_history.json",
        "failure_calibration_scope_mismatch.json",
        "failure_model_not_promoted.json",
        "failure_unsupported_extrapolation.json",
    ],
)
def test_shipped_failure_fixtures_round_trip(fixture_name: str) -> None:
    packet = MMMFailurePacket.model_validate(_fixture(fixture_name))
    assert MMMFailurePacket.from_json(packet.to_json()) == packet
