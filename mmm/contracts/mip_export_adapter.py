"""Claim-gated runtime adapter for MMM -> MIP export bundles.

The adapter deliberately preserves the boundary established by MMM-EXPORT-002:
it describes package artifacts that already exist, but never promotes them to
production claims and never manufactures absent artifact families.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from mmm.contracts.mip_export import (
    ArtifactSafetyStatus,
    CalibrationStatus,
    DiagnosticStatus,
    ExportInventoryStatus,
    MMMExportBundle,
    PromotionStatus,
    UncertaintyStatus,
    validate_mmm_export_bundle,
)
from mmm.contracts.mip_failure import (
    MMMExportOutcome,
    MMMFailureCode,
    MMMFailurePacket,
    MMMFailureStage,
    MMMRemediationAction,
    MMMRetryDisposition,
    build_mmm_failure_packet,
)
from mmm.contracts.run_manifest import (
    MMMArtifactReference,
    MMMExportManifestOutcome,
    MMMRunStatus,
    MMMRunStep,
    MMMRunStepStatus,
    build_mmm_run_manifest,
)
from mmm.contracts.calibration_treatment import MMMCalibrationTreatmentLineage
from mmm.contracts.diagnostics_limitations import MMMDiagnosticsLimitations
from mmm.contracts.supported_range import MMMSupportedRangeEvidence


class MMMExportAdapterError(ValueError):
    """Raised when runtime artifacts cannot be adapted without guessing."""


@dataclass(frozen=True)
class MMMExportRuntimeContext:
    """Required, explicit metadata shared by every adapted artifact."""

    model_run_id: str
    training_data_fingerprint: str
    model_artifact_fingerprint: str
    generated_at: str
    package_version: str
    git_commit: str
    model_form: str
    estimand: str
    time_window: str
    geo_scope: str
    channel_scope: tuple[str, ...]
    outcome_metric: str
    spend_metric: str
    currency: str


_COMMON_FORBIDDEN_CLAIMS = (
    "channel_roi_ranking",
    "production_incrementality_claim",
    "budget_shift_recommendation",
)


def _require_context(context: MMMExportRuntimeContext) -> None:
    for name in (
        "model_run_id",
        "training_data_fingerprint",
        "model_artifact_fingerprint",
        "generated_at",
        "package_version",
        "git_commit",
        "model_form",
        "estimand",
        "time_window",
        "geo_scope",
        "outcome_metric",
        "spend_metric",
        "currency",
    ):
        if not str(getattr(context, name)).strip():
            raise MMMExportAdapterError(f"runtime export requires explicit {name}")


def _mapping(value: Any) -> dict[str, Any] | None:
    return dict(value) if isinstance(value, Mapping) else None


def _common(
    context: MMMExportRuntimeContext,
    *,
    artifact_type: str,
    source_artifacts: list[str],
    diagnostic_status: DiagnosticStatus,
    artifact_safety_status: ArtifactSafetyStatus,
    allowed_claims: list[str],
    forbidden_claims: list[str],
) -> dict[str, Any]:
    return {
        "artifact_type": artifact_type,
        "model_run_id": context.model_run_id,
        "training_data_fingerprint": context.training_data_fingerprint,
        "model_artifact_fingerprint": context.model_artifact_fingerprint,
        "source_artifacts": source_artifacts,
        "model_form": context.model_form,
        "estimand": context.estimand,
        "time_window": context.time_window,
        "geo_scope": context.geo_scope,
        "channel_scope": list(context.channel_scope),
        "outcome_metric": context.outcome_metric,
        "spend_metric": context.spend_metric,
        "currency": context.currency,
        "uncertainty_status": UncertaintyStatus.MISSING.value,
        "diagnostic_status": diagnostic_status.value,
        "promotion_status": PromotionStatus.DIAGNOSTIC_ONLY.value,
        "calibration_status": CalibrationStatus.UNKNOWN.value,
        "planning_allowed": False,
        "llm_exposure_allowed": False,
        "demo_fixture_allowed": False,
        "recommendation_allowed": False,
        "production_claim_allowed": False,
        "allowed_claims": allowed_claims,
        "forbidden_claims": sorted(set(forbidden_claims)),
        "generated_at": context.generated_at,
        "package_version": context.package_version,
        "git_commit": context.git_commit,
        "artifact_safety_status": artifact_safety_status.value,
    }


def adapt_runtime_artifacts_to_export_bundle(
    *,
    context: MMMExportRuntimeContext,
    extension_report: Mapping[str, Any] | None = None,
    simulation_result: Mapping[str, Any] | None = None,
    optimizer_result: Mapping[str, Any] | None = None,
    recommendation_contract: Mapping[str, Any] | None = None,
) -> MMMExportBundle:
    """Adapt existing runtime payloads into a conservative validated bundle.

    Payloads are retained under ``source_payload`` rather than interpreted into
    economic values whose semantics the source contract does not guarantee.
    A recommendation input is rejected: no current package runtime artifact is a
    governed ``MMMRecommendationContract``.
    """

    _require_context(context)
    report = dict(extension_report or {})
    artifacts: list[dict[str, Any]] = []
    forbidden = list(_COMMON_FORBIDDEN_CLAIMS)
    report_forbidden = report.get("forbidden_claims")
    if isinstance(report_forbidden, list):
        forbidden.extend(str(item) for item in report_forbidden if str(item).strip())

    fit = _mapping(report.get("ridge_fit_summary"))
    if fit is not None:
        artifacts.append(
            {
                **_common(
                    context,
                    artifact_type="MMMModelFitArtifact",
                    source_artifacts=["extension_report.ridge_fit_summary"],
                    diagnostic_status=DiagnosticStatus.UNKNOWN,
                    artifact_safety_status=ArtifactSafetyStatus.READINESS_ONLY,
                    allowed_claims=["readiness_explanation_allowed"],
                    forbidden_claims=forbidden,
                ),
                "framework": "ridge",
                "source_payload": fit,
            }
        )

    diagnostic_keys = (
        "ridge_production_diagnostics_report",
        "ridge_diagnostics",
        "identifiability",
        "feature_separability_report",
        "panel_qa",
    )
    diagnostics = {key: report[key] for key in diagnostic_keys if key in report}
    if diagnostics:
        artifacts.append(
            {
                **_common(
                    context,
                    artifact_type="MMMModelDiagnosticArtifact",
                    source_artifacts=[f"extension_report.{key}" for key in diagnostics],
                    diagnostic_status=DiagnosticStatus.UNKNOWN,
                    artifact_safety_status=ArtifactSafetyStatus.DIAGNOSTIC_ONLY,
                    allowed_claims=["diagnostic_explanation_allowed", "readiness_explanation_allowed"],
                    forbidden_claims=forbidden,
                ),
                "source_payload": diagnostics,
            }
        )

    optional_report_families = (
        ("contribution_summary", "MMMChannelContributionArtifact"),
        ("roi_summary", "MMMChannelROIArtifact"),
        ("curve_bundles", "MMMResponseCurveArtifact"),
        ("curve_bundle", "MMMResponseCurveArtifact"),
    )
    emitted_types: set[str] = set()
    for key, artifact_type in optional_report_families:
        if key not in report or artifact_type in emitted_types:
            continue
        emitted_types.add(artifact_type)
        child = _common(
            context,
            artifact_type=artifact_type,
            source_artifacts=[f"extension_report.{key}"],
            diagnostic_status=DiagnosticStatus.UNKNOWN,
            artifact_safety_status=ArtifactSafetyStatus.BLOCKED,
            allowed_claims=[],
            forbidden_claims=forbidden + ["blocked_until_uncertainty"],
        )
        child["source_payload"] = report[key]
        artifacts.append(child)

    if simulation_result is not None:
        artifacts.append(
            {
                **_common(
                    context,
                    artifact_type="MMMSimulationResultArtifact",
                    source_artifacts=["decide.simulation_result"],
                    diagnostic_status=DiagnosticStatus.UNKNOWN,
                    artifact_safety_status=ArtifactSafetyStatus.BLOCKED,
                    allowed_claims=[],
                    forbidden_claims=forbidden,
                ),
                "source_payload": dict(simulation_result),
            }
        )
    if optimizer_result is not None:
        artifacts.append(
            {
                **_common(
                    context,
                    artifact_type="MMMOptimizerResultArtifact",
                    source_artifacts=["decide.optimizer_result"],
                    diagnostic_status=DiagnosticStatus.UNKNOWN,
                    artifact_safety_status=ArtifactSafetyStatus.BLOCKED,
                    allowed_claims=[],
                    forbidden_claims=forbidden + ["blocked_until_recommendation_contract"],
                ),
                "has_recommendation_contract": False,
                "source_payload": dict(optimizer_result),
            }
        )
    if recommendation_contract is not None:
        raise MMMExportAdapterError(
            "runtime recommendation input is not governed; refusing to invent MMMRecommendationContract"
        )

    bundle = MMMExportBundle(
        model_run_id=context.model_run_id,
        training_data_fingerprint=context.training_data_fingerprint,
        model_artifact_fingerprint=context.model_artifact_fingerprint,
        source_artifacts=sorted({source for art in artifacts for source in art["source_artifacts"]}),
        generated_at=context.generated_at,
        package_version=context.package_version,
        git_commit=context.git_commit,
        inventory_status=ExportInventoryStatus.EXISTS_PARTIAL_NOT_CONSUMABLE,
        forbidden_claims=sorted(set(forbidden)),
        artifact_safety_status=ArtifactSafetyStatus.BLOCKED,
        artifacts=artifacts,
    )
    errors = validate_mmm_export_bundle(bundle)
    if errors:
        raise MMMExportAdapterError("invalid adapted export bundle: " + "; ".join(errors))
    return bundle


def emit_known_failure_outcome(
    *,
    failure_id: str,
    created_at: datetime,
    code: MMMFailureCode,
    stage: MMMFailureStage,
    source_component: str,
    technical_summary: str,
    context: MMMExportRuntimeContext | None = None,
    affected_resource: str | None = None,
    remediation_actions: list[MMMRemediationAction] | None = None,
    retry_disposition: MMMRetryDisposition | None = None,
    retry_override_evidence: str | None = None,
    **evidence: Any,
) -> MMMExportOutcome:
    """Emit one explicitly mapped known producer failure at the export boundary.

    Callers must supply a governed ``MMMFailureCode``. This helper intentionally
    does not catch arbitrary exceptions or infer codes from exception strings.
    """
    if context is not None:
        evidence.setdefault("run_id", context.model_run_id)
        evidence.setdefault("dataset_fingerprint", context.training_data_fingerprint)
        evidence.setdefault("configuration_hash", context.model_artifact_fingerprint)
        evidence.setdefault("producer_package_version", context.package_version)
        evidence.setdefault("producer_git_commit", context.git_commit)
    packet = build_mmm_failure_packet(
        failure_id=failure_id,
        created_at=created_at,
        code=code,
        stage=stage,
        source_component=source_component,
        technical_summary=technical_summary,
        affected_resource=affected_resource,
        remediation_actions=remediation_actions,
        retry_disposition=retry_disposition,
        retry_override_evidence=retry_override_evidence,
        **evidence,
    )
    return MMMExportOutcome.failure(packet)


def adapt_runtime_artifacts_to_export_outcome(
    *,
    context: MMMExportRuntimeContext,
    known_failure: MMMFailurePacket | None = None,
    extension_report: Mapping[str, Any] | None = None,
    simulation_result: Mapping[str, Any] | None = None,
    optimizer_result: Mapping[str, Any] | None = None,
    recommendation_contract: Mapping[str, Any] | None = None,
) -> MMMExportOutcome:
    """Non-breaking outcome wrapper around the existing successful export adapter."""
    if known_failure is not None:
        return MMMExportOutcome.failure(known_failure)
    return MMMExportOutcome.success(
        adapt_runtime_artifacts_to_export_bundle(
            context=context,
            extension_report=extension_report,
            simulation_result=simulation_result,
            optimizer_result=optimizer_result,
            recommendation_contract=recommendation_contract,
        )
    )


def adapt_runtime_artifacts_to_export_manifest_outcome(
    *,
    context: MMMExportRuntimeContext,
    manifest_id: str,
    created_at: datetime,
    known_failure: MMMFailurePacket | None = None,
    calibration_lineage: MMMCalibrationTreatmentLineage | None = None,
    diagnostics_limitations: MMMDiagnosticsLimitations | None = None,
    supported_range_evidence: MMMSupportedRangeEvidence | None = None,
    extension_report: Mapping[str, Any] | None = None,
    simulation_result: Mapping[str, Any] | None = None,
    optimizer_result: Mapping[str, Any] | None = None,
    recommendation_contract: Mapping[str, Any] | None = None,
) -> MMMExportManifestOutcome:
    """Return the existing export outcome with additive typed run evidence.

    This deliberately maps only an already-typed ``known_failure``.  It does not
    inspect exceptions or change the legacy success adapter's behaviour.
    """
    if calibration_lineage is not None and calibration_lineage.run_id != context.model_run_id:
        raise MMMExportAdapterError("calibration lineage run ID must match the runtime export context")
    if diagnostics_limitations is not None and diagnostics_limitations.run_id != context.model_run_id:
        raise MMMExportAdapterError("diagnostics aggregate run ID must match the runtime export context")
    if supported_range_evidence is not None and supported_range_evidence.run_id != context.model_run_id:
        raise MMMExportAdapterError("supported range evidence run ID must match the runtime export context")
    outcome = adapt_runtime_artifacts_to_export_outcome(
        context=context,
        known_failure=known_failure,
        extension_report=extension_report,
        simulation_result=simulation_result,
        optimizer_result=optimizer_result,
        recommendation_contract=recommendation_contract,
    )
    common: dict[str, Any] = {
        "manifest_id": manifest_id,
        "run_id": context.model_run_id,
        "created_at": created_at,
        "producer_package_version": context.package_version,
        "model_family": context.model_form,
        "estimator_identity": context.estimand,
        "configuration_hash": context.model_artifact_fingerprint,
        "dataset_fingerprint": context.training_data_fingerprint,
        "time_range": context.time_window,
        "market_scope": context.geo_scope,
        "channel_scope": list(context.channel_scope),
        "calibration_lineage_id": calibration_lineage.lineage_id if calibration_lineage else None,
        "calibration_signal_ids": [record.signal_id for record in calibration_lineage.records] if calibration_lineage else [],
        "diagnostics_limitations_id": diagnostics_limitations.aggregate_id if diagnostics_limitations else None,
        "supported_range_evidence_id": supported_range_evidence.evidence_id if supported_range_evidence else None,
    }
    if outcome.outcome_type == "success":
        bundle = outcome.export_bundle
        assert bundle is not None
        export_ref = MMMArtifactReference(
            artifact_type="MMMExportBundle",
            artifact_id=bundle.model_run_id,
            contract_version=bundle.schema_version,
            content_fingerprint=bundle.model_artifact_fingerprint,
            logical_name="mmm_producer_export_bundle",
        )
        manifest = build_mmm_run_manifest(
            **common,
            status=MMMRunStatus.SUCCEEDED,
            successful_export=export_ref,
            steps=[
                MMMRunStep(
                    sequence=0,
                    step_name="producer_export",
                    stage=MMMFailureStage.EXPORT,
                    status=MMMRunStepStatus.SUCCEEDED,
                    output_artifacts=[export_ref],
                    technical_detail="Validated producer export bundle emitted.",
                )
            ],
        )
    else:
        packet = outcome.failure_packet
        assert packet is not None
        manifest_status = MMMRunStatus.BLOCKED if packet.failure_status == "blocked" else MMMRunStatus.FAILED
        step_status = MMMRunStepStatus.BLOCKED if packet.failure_status == "blocked" else MMMRunStepStatus.FAILED
        manifest = build_mmm_run_manifest(
            **common,
            status=manifest_status,
            failure_packet=packet,
            calibration_signal_ids=packet.calibration_signal_ids,
            validation_result_ids=packet.validation_result_ids,
            diagnostic_ids=packet.diagnostic_ids,
            steps=[
                MMMRunStep(
                    sequence=0,
                    step_name=f"failure_{packet.stage.value.lower()}",
                    stage=packet.stage,
                    status=step_status,
                    failure_packet_id=packet.failure_id,
                    technical_detail=packet.technical_summary,
                )
            ],
        )
    return MMMExportManifestOutcome(
        export_outcome=outcome,
        run_manifest=manifest,
        supported_range_evidence_id=supported_range_evidence.evidence_id if supported_range_evidence else None,
    )


__all__ = [
    "MMMExportAdapterError",
    "MMMExportRuntimeContext",
    "adapt_runtime_artifacts_to_export_bundle",
    "adapt_runtime_artifacts_to_export_manifest_outcome",
    "adapt_runtime_artifacts_to_export_outcome",
    "emit_known_failure_outcome",
]
