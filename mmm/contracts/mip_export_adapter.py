"""Claim-gated runtime adapter for MMM -> MIP export bundles.

The adapter deliberately preserves the boundary established by MMM-EXPORT-002:
it describes package artifacts that already exist, but never promotes them to
production claims and never manufactures absent artifact families.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
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


__all__ = [
    "MMMExportAdapterError",
    "MMMExportRuntimeContext",
    "adapt_runtime_artifacts_to_export_bundle",
]
