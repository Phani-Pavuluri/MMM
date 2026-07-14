"""Typed producer run-manifest contract and export-boundary regression tests."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from mmm.contracts.mip_export import CalibrationStatus, PromotionStatus
from mmm.contracts.mip_export_adapter import (
    MMMExportRuntimeContext,
    adapt_runtime_artifacts_to_export_manifest_outcome,
)
from mmm.contracts.mip_failure import (
    MMMFailureCode,
    MMMFailureStage,
    build_mmm_failure_packet,
)
from mmm.contracts.run_manifest import (
    MMM_RUN_MANIFEST_SCHEMA_VERSION,
    MMMArtifactReference,
    MMMExportManifestOutcome,
    MMMRunManifest,
    MMMRunStatus,
    MMMRunStep,
    MMMRunStepStatus,
)

CREATED_AT = datetime(2026, 7, 14, 12, 0, tzinfo=timezone.utc)
FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "mip_export"


def _artifact() -> MMMArtifactReference:
    return MMMArtifactReference(
        artifact_type="MMMExportBundle",
        artifact_id="run-manifest-001",
        contract_version="mmm_mip_export_v1",
        content_fingerprint="sha256:model",
        logical_name="mmm_producer_export_bundle",
    )


def _packet(*, code: MMMFailureCode = MMMFailureCode.INSUFFICIENT_HISTORY, stage: MMMFailureStage = MMMFailureStage.DATA_VALIDATION, status: str = "failed"):
    return build_mmm_failure_packet(
        failure_id="failure-manifest-001",
        created_at=CREATED_AT,
        code=code,
        stage=stage,
        source_component="mmm.contracts.run_manifest",
        technical_summary=f"Governed failure: {code.value}.",
        affected_resource="governed-input",
        failure_status=status,
    )


def _values(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "manifest_id": "manifest-001",
        "run_id": "run-manifest-001",
        "created_at": CREATED_AT,
        "producer_package_version": "0.1.0",
        "status": MMMRunStatus.RUNNING,
        "dataset_fingerprint": "sha256:panel",
    }
    values.update(overrides)
    return values


def _context() -> MMMExportRuntimeContext:
    return MMMExportRuntimeContext(
        model_run_id="run-manifest-001",
        training_data_fingerprint="sha256:panel",
        model_artifact_fingerprint="sha256:model",
        generated_at="2026-07-14T12:00:00Z",
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


def test_minimum_running_manifest_is_versioned_and_keeps_optional_lineage_absent() -> None:
    manifest = MMMRunManifest(**_values())
    assert manifest.schema_version == MMM_RUN_MANIFEST_SCHEMA_VERSION
    assert manifest.model_id is None
    assert "model_id" not in manifest.to_json_dict()


def test_success_failed_and_blocked_terminal_manifests_are_consistent() -> None:
    success = MMMRunManifest(**_values(status=MMMRunStatus.SUCCEEDED, successful_export=_artifact()))
    failed_packet = _packet()
    failed = MMMRunManifest(**_values(status=MMMRunStatus.FAILED, failure_packet=failed_packet))
    blocked_packet = _packet(code=MMMFailureCode.MODEL_NOT_PROMOTED, stage=MMMFailureStage.PROMOTION_GATE, status="blocked")
    blocked = MMMRunManifest(**_values(status=MMMRunStatus.BLOCKED, failure_packet=blocked_packet))
    assert success.successful_export is not None
    assert failed.failure_packet == failed_packet
    assert blocked.failure_packet == blocked_packet


def test_model_calibration_and_pre_model_failure_linkage_are_optional_and_safe() -> None:
    manifest = MMMRunManifest(
        **_values(
            status=MMMRunStatus.BLOCKED,
            failure_packet=_packet(
                code=MMMFailureCode.CALIBRATION_SCOPE_MISMATCH,
                stage=MMMFailureStage.CALIBRATION,
                status="blocked",
            ),
            model_id="model-001",
            model_family="ridge",
            promotion_status=PromotionStatus.DIAGNOSTIC_ONLY,
            calibration_signal_ids=["calibration-001"],
            calibration_status=CalibrationStatus.CONTEXT_ATTACHED,
        )
    )
    pre_model = MMMRunManifest(**_values(status=MMMRunStatus.FAILED, failure_packet=_packet()))
    assert manifest.calibration_signal_ids == ["calibration-001"]
    assert pre_model.model_id is None


def test_manifest_validation_rejects_terminal_conflicts_and_bad_ordering() -> None:
    packet = _packet()
    with pytest.raises(ValidationError, match="successful_export"):
        MMMRunManifest(**_values(status=MMMRunStatus.SUCCEEDED))
    with pytest.raises(ValidationError, match="cannot contain a terminal failure"):
        MMMRunManifest(**_values(status=MMMRunStatus.SUCCEEDED, successful_export=_artifact(), failure_packet=packet))
    with pytest.raises(ValidationError, match="typed failure_packet"):
        MMMRunManifest(**_values(status=MMMRunStatus.FAILED))
    with pytest.raises(ValidationError, match="cannot claim successful"):
        MMMRunManifest(**_values(status=MMMRunStatus.FAILED, failure_packet=packet, successful_export=_artifact()))
    with pytest.raises(ValidationError, match="running manifests"):
        MMMRunManifest(**_values(status=MMMRunStatus.RUNNING, successful_export=_artifact()))
    with pytest.raises(ValidationError, match="contiguous sequence"):
        MMMRunManifest(
            **_values(
                steps=[MMMRunStep(sequence=1, step_name="export", stage=MMMFailureStage.EXPORT, status=MMMRunStepStatus.PENDING)]
            )
        )


def test_manifest_validation_rejects_bad_timestamps_failed_step_and_unsafe_values() -> None:
    with pytest.raises(ValidationError, match="timezone-aware"):
        MMMRunManifest(**_values(created_at=datetime(2026, 7, 14, 12, 0)))
    with pytest.raises(ValidationError, match="completion time"):
        MMMRunStep(
            sequence=0,
            step_name="validate",
            stage=MMMFailureStage.DATA_VALIDATION,
            status=MMMRunStepStatus.SUCCEEDED,
            started_at=CREATED_AT,
            completed_at=CREATED_AT - timedelta(seconds=1),
        )
    with pytest.raises(ValidationError, match="manifest completion time"):
        MMMRunManifest(
            **_values(
                started_at=CREATED_AT,
                completed_at=CREATED_AT - timedelta(seconds=1),
            )
        )
    with pytest.raises(ValidationError, match="failure_packet_id"):
        MMMRunStep(sequence=0, step_name="validate", stage=MMMFailureStage.DATA_VALIDATION, status=MMMRunStepStatus.FAILED)
    with pytest.raises(ValidationError, match="paths"):
        MMMArtifactReference(artifact_type="MMMExportBundle", artifact_id="/tmp/model", contract_version="v1")
    with pytest.raises(ValidationError):
        MMMArtifactReference(artifact_type="MMMExportBundle", artifact_id=object(), contract_version="v1")
    with pytest.raises(ValidationError):
        MMMRunManifest.model_validate({**MMMRunManifest(**_values()).to_json_dict(), "schema_version": "unknown_v9"})


def test_serialization_is_deterministic_and_round_trips_without_python_leakage() -> None:
    manifest = MMMRunManifest(
        **_values(
            steps=[MMMRunStep(sequence=0, step_name="intake", stage=MMMFailureStage.DATA_INTAKE, status=MMMRunStepStatus.SUCCEEDED)]
        )
    )
    serialized = manifest.to_json()
    assert serialized == manifest.to_json()
    assert MMMRunManifest.from_json(serialized) == manifest
    assert "traceback" not in serialized.lower()
    assert "/users/" not in serialized.lower()


@pytest.mark.parametrize(
    ("code", "stage"),
    [
        (None, None),
        (MMMFailureCode.INSUFFICIENT_HISTORY, MMMFailureStage.DATA_VALIDATION),
        (MMMFailureCode.CALIBRATION_SCOPE_MISMATCH, MMMFailureStage.CALIBRATION),
        (MMMFailureCode.HOLDOUT_FAILURE, MMMFailureStage.MODEL_VALIDATION),
        (MMMFailureCode.MODEL_NOT_PROMOTED, MMMFailureStage.PROMOTION_GATE),
        (MMMFailureCode.UNSUPPORTED_EXTRAPOLATION, MMMFailureStage.SIMULATION),
    ],
)
def test_export_boundary_links_success_and_known_failures_to_manifests(code: MMMFailureCode | None, stage: MMMFailureStage | None) -> None:
    packet = None
    if code is not None:
        packet = _packet(code=code, stage=stage, status="blocked")
    result = adapt_runtime_artifacts_to_export_manifest_outcome(
        context=_context(),
        manifest_id=f"manifest-{code.value.lower() if code else 'success'}",
        created_at=CREATED_AT,
        known_failure=packet,
        extension_report={"ridge_fit_summary": {"coef": [0.2]}},
    )
    assert isinstance(result, MMMExportManifestOutcome)
    if packet is None:
        assert result.export_outcome.outcome_type == "success"
        assert result.run_manifest.status == MMMRunStatus.SUCCEEDED
        assert result.run_manifest.successful_export is not None
    else:
        assert result.export_outcome.failure_packet == packet
        assert result.run_manifest.failure_packet == packet
        assert result.run_manifest.steps[0].failure_packet_id == packet.failure_id


@pytest.mark.parametrize(
    "fixture_name",
    [
        "run_manifest_success.json",
        "run_manifest_input_validation_failure.json",
        "run_manifest_calibration_failure.json",
        "run_manifest_model_not_promoted.json",
        "run_manifest_unsupported_extrapolation.json",
    ],
)
def test_shipped_run_manifest_fixtures_round_trip(fixture_name: str) -> None:
    manifest = MMMRunManifest.model_validate(_fixture(fixture_name))
    assert MMMRunManifest.from_json(manifest.to_json()) == manifest
