"""Producer calibration-treatment lineage contract regressions."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from mmm.contracts.calibration_treatment import (
    MMMCalibrationApplicationRole,
    MMMCalibrationCompatibilityReason,
    MMMCalibrationCompatibilityStatus,
    MMMCalibrationFreshnessStatus,
    MMMCalibrationTransformationStep,
    MMMCalibrationTreatmentDisposition,
    MMMCalibrationTreatmentLineage,
    MMMCalibrationTreatmentRecord,
)
from mmm.contracts.mip_export_adapter import MMMExportRuntimeContext, adapt_runtime_artifacts_to_export_manifest_outcome
from mmm.contracts.mip_failure import MMMFailureCode, MMMFailureStage, build_mmm_failure_packet

NOW = datetime(2026, 7, 14, 12, 0, tzinfo=timezone.utc)
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "mip_export"


def _record(**changes: object) -> MMMCalibrationTreatmentRecord:
    values: dict[str, object] = {
        "sequence": 0,
        "signal_id": "signal-001",
        "source_type": "geox",
        "run_id": "run-cal-001",
        "freshness_status": MMMCalibrationFreshnessStatus.FRESH,
        "compatibility_status": MMMCalibrationCompatibilityStatus.COMPATIBLE,
        "disposition": MMMCalibrationTreatmentDisposition.ACCEPTED_NOT_APPLIED,
        "application_roles": [MMMCalibrationApplicationRole.EVIDENCE_CONTEXT_ONLY],
        "technical_detail": "Ridge diagnostic evidence context only; coefficients unchanged.",
    }
    values.update(changes)
    return MMMCalibrationTreatmentRecord(**values)


def _lineage(*records: MMMCalibrationTreatmentRecord) -> MMMCalibrationTreatmentLineage:
    return MMMCalibrationTreatmentLineage(
        lineage_id="lineage-001", run_id="run-cal-001", created_at=NOW, producer_package_version="0.1.0", records=list(records)
    )


def _packet(code: MMMFailureCode) -> object:
    return build_mmm_failure_packet(
        failure_id=f"failure-{code.value.lower()}", created_at=NOW, code=code,
        stage=MMMFailureStage.CALIBRATION, source_component="mmm.calibration",
        technical_summary="Known calibration treatment failure.", failure_status="blocked",
    )


def _context() -> MMMExportRuntimeContext:
    return MMMExportRuntimeContext(
        model_run_id="run-cal-001", training_data_fingerprint="sha256:panel", model_artifact_fingerprint="sha256:model",
        generated_at="2026-07-14T12:00:00Z", package_version="0.1.0", git_commit="abc1234",
        model_form="ridge", estimand="incremental_sales", time_window="2026-01-01/2026-06-30",
        geo_scope="national", channel_scope=("search",), outcome_metric="revenue", spend_metric="spend", currency="USD",
    )


def test_empty_and_context_only_lineage_are_versioned_and_deterministic() -> None:
    empty = _lineage()
    context = _lineage(_record())
    assert empty.summary_counts == {"disposition": {}, "freshness": {}, "compatibility": {}, "application_role": {}}
    assert MMMCalibrationTreatmentLineage.from_json(context.to_json()) == context
    assert context.to_json() == context.to_json()


def test_treatment_dispositions_freshness_compatibility_and_roles_remain_distinct() -> None:
    records = [
        _record(sequence=0, disposition=MMMCalibrationTreatmentDisposition.ACCEPTED_NOT_APPLIED),
        _record(sequence=1, signal_id="signal-rejected", disposition=MMMCalibrationTreatmentDisposition.REJECTED,
                compatibility_status=MMMCalibrationCompatibilityStatus.INCOMPATIBLE,
                compatibility_reasons=[MMMCalibrationCompatibilityReason.GEO_SCOPE_MISMATCH], application_roles=[MMMCalibrationApplicationRole.NONE]),
        _record(sequence=2, signal_id="signal-stale", freshness_status=MMMCalibrationFreshnessStatus.STALE,
                disposition=MMMCalibrationTreatmentDisposition.ACCEPTED_NOT_APPLIED),
        _record(sequence=3, signal_id="signal-blocked", disposition=MMMCalibrationTreatmentDisposition.BLOCKED,
                application_roles=[MMMCalibrationApplicationRole.NONE]),
    ]
    lineage = _lineage(*records)
    assert lineage.summary_counts["disposition"]["REJECTED"] == 1
    assert lineage.summary_counts["freshness"]["STALE"] == 1
    assert lineage.summary_counts["compatibility"]["INCOMPATIBLE"] == 1


def test_validation_rejects_invalid_application_and_stale_or_incompatible_application() -> None:
    with pytest.raises(ValidationError, match="non-NONE"):
        _record(disposition=MMMCalibrationTreatmentDisposition.APPLIED, application_roles=[MMMCalibrationApplicationRole.NONE])
    with pytest.raises(ValidationError, match="fit-affecting"):
        _record(disposition=MMMCalibrationTreatmentDisposition.REJECTED, application_roles=[MMMCalibrationApplicationRole.PRIOR], research_only=True)
    with pytest.raises(ValidationError, match="expired"):
        _record(freshness_status=MMMCalibrationFreshnessStatus.EXPIRED, disposition=MMMCalibrationTreatmentDisposition.APPLIED, application_roles=[MMMCalibrationApplicationRole.DIAGNOSTIC_ONLY])
    with pytest.raises(ValidationError, match="incompatible"):
        _record(compatibility_status=MMMCalibrationCompatibilityStatus.INCOMPATIBLE, disposition=MMMCalibrationTreatmentDisposition.APPLIED, application_roles=[MMMCalibrationApplicationRole.DIAGNOSTIC_ONLY])
    with pytest.raises(ValidationError, match="downweight"):
        _record(freshness_status=MMMCalibrationFreshnessStatus.STALE, disposition=MMMCalibrationTreatmentDisposition.APPLIED, application_roles=[MMMCalibrationApplicationRole.DIAGNOSTIC_ONLY])


def test_research_only_bayesian_role_and_ordered_transformations_are_explicit() -> None:
    record = _record(
        disposition=MMMCalibrationTreatmentDisposition.APPLIED,
        application_roles=[MMMCalibrationApplicationRole.LIKELIHOOD], model_family="bayesian", research_only=True,
        governing_policy_reference="bayes_h2_calibration_signal_mapping_v1:R5", applied_weight=0.25,
        transformations=[
            MMMCalibrationTransformationStep(sequence=0, transformation_code="lift_scale_conversion", transformation_version="v1"),
            MMMCalibrationTransformationStep(sequence=1, transformation_code="freshness_downweight", transformation_version="v1"),
        ],
    )
    assert record.research_only is True
    with pytest.raises(ValidationError, match="contiguous"):
        _record(transformations=[MMMCalibrationTransformationStep(sequence=1, transformation_code="x", transformation_version="v1")])


def test_failure_links_reuse_existing_calibration_failure_taxonomy() -> None:
    scope = _record(disposition=MMMCalibrationTreatmentDisposition.REJECTED, application_roles=[MMMCalibrationApplicationRole.NONE],
                    compatibility_status=MMMCalibrationCompatibilityStatus.INCOMPATIBLE,
                    failure_packet=_packet(MMMFailureCode.CALIBRATION_SCOPE_MISMATCH))
    expired = _record(signal_id="expired", freshness_status=MMMCalibrationFreshnessStatus.EXPIRED,
                      disposition=MMMCalibrationTreatmentDisposition.REJECTED, application_roles=[MMMCalibrationApplicationRole.NONE],
                      failure_packet=_packet(MMMFailureCode.CALIBRATION_SIGNAL_EXPIRED))
    assert scope.failure_packet is not None and expired.failure_packet is not None


def test_manifest_boundary_additively_links_lineage_without_changing_ridge_or_export() -> None:
    lineage = _lineage(_record())
    result = adapt_runtime_artifacts_to_export_manifest_outcome(
        context=_context(), manifest_id="manifest-cal-001", created_at=NOW, calibration_lineage=lineage,
        extension_report={"ridge_fit_summary": {"coef": [0.2]}},
    )
    assert result.export_outcome.outcome_type == "success"
    assert result.run_manifest.calibration_lineage_id == lineage.lineage_id
    assert result.run_manifest.calibration_signal_ids == ["signal-001"]


@pytest.mark.parametrize("name", [
    "calibration_lineage_context_only.json", "calibration_lineage_stale.json",
    "calibration_lineage_expired.json", "calibration_lineage_incompatible.json",
    "calibration_lineage_multi_signal.json", "calibration_lineage_blocked.json",
])
def test_calibration_lineage_fixtures_round_trip(name: str) -> None:
    lineage = MMMCalibrationTreatmentLineage.model_validate(json.loads((FIXTURES / name).read_text(encoding="utf-8")))
    assert MMMCalibrationTreatmentLineage.from_json(lineage.to_json()) == lineage
