"""Typed public full-panel Ridge simulation evidence; never an optimizer or recommendation."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from datetime import datetime
from enum import Enum
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from mmm.contracts.diagnostics_limitations import MMMTechnicalClaim, MMMTechnicalClaimDisposition
from mmm.contracts.mip_failure import MMMFailureCode, MMMFailurePacket, MMMFailureStage, build_mmm_failure_packet
from mmm.contracts.run_manifest import (
    MMMAnalyticalArtifactOutcome,
    MMMArtifactReference,
    MMMExportManifestOutcome,
    MMMRunManifest,
    MMMRunStatus,
    MMMRunStep,
    MMMRunStepStatus,
    build_mmm_run_manifest,
)
from mmm.contracts.supported_range import (
    MMMExtrapolationClassification,
    MMMRangeAvailabilityStatus,
    MMMRangeRelation,
    MMMSupportedRangeEvidence,
    MMMSupportedRangeSimulationEligibility,
)
from mmm.planning.baseline import BaselinePlan, BaselineType
from mmm.planning.context import RidgeFitContext
from mmm.planning.decision_simulate import simulate
from mmm.version import __version__ as MMM_PACKAGE_VERSION

MMM_PUBLIC_SIMULATION_SCHEMA_VERSION = "mmm_public_simulation_export_v1"
MMM_PUBLIC_SIMULATION_ARTIFACT_KIND: Final[Literal["MMMPublicSimulationExport"]] = "MMMPublicSimulationExport"


class MMMSimulationStatus(str, Enum):
    SUCCEEDED = "SUCCEEDED"
    SUCCEEDED_WITH_LIMITATIONS = "SUCCEEDED_WITH_LIMITATIONS"
    BLOCKED = "BLOCKED"
    FAILED = "FAILED"


class MMMSimulationPlanRole(str, Enum):
    BASELINE = "BASELINE"
    CANDIDATE = "CANDIDATE"


class MMMSimulationUncertaintyStatus(str, Enum):
    UNAVAILABLE = "UNAVAILABLE"
    AVAILABLE = "AVAILABLE"
    RESEARCH_ONLY = "RESEARCH_ONLY"


class MMMSimulationScope(BaseModel):
    model_config = ConfigDict(extra="forbid")
    metric_id: str
    model_id: str
    model_family: str
    panel_id: str
    evaluation_start: datetime
    evaluation_end: datetime
    panel_grain: str
    spend_unit: str
    spend_scale: str
    outcome_id: str | None = None
    estimand_id: str | None = None
    model_version: str | None = None
    configuration_hash: str | None = None
    geography: str | None = None
    segment: str | None = None
    transformation_id: str | None = None

    @field_validator(
        "metric_id",
        "model_id",
        "model_family",
        "panel_id",
        "panel_grain",
        "spend_unit",
        "spend_scale",
        "outcome_id",
        "estimand_id",
        "model_version",
        "configuration_hash",
        "geography",
        "segment",
        "transformation_id",
    )
    @classmethod
    def scope_text(cls, v, info):
        if v is None:
            return v
        if not isinstance(v, str) or not v.strip() or v.startswith(("/", "~")):
            raise ValueError(f"{info.field_name} must be safe non-empty text")
        return v.strip()

    @field_validator("evaluation_start", "evaluation_end")
    @classmethod
    def aware(cls, v):
        if v.tzinfo is None or v.utcoffset() is None:
            raise ValueError("evaluation timestamps must be timezone-aware")
        return v

    @model_validator(mode="after")
    def scope_valid(self):
        if self.evaluation_end <= self.evaluation_start:
            raise ValueError("evaluation end must follow start")
        if self.spend_scale.upper() == "TRANSFORMED" and not self.transformation_id:
            raise ValueError("transformed scale requires transformation_id")
        if self.spend_scale.upper() == "RAW" and self.transformation_id:
            raise ValueError("raw scale cannot carry transformation_id")
        return self


class MMMSimulationPlanItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    channel_id: str
    spend: float
    spend_unit: str
    geography: str | None = None
    segment: str | None = None
    time_bucket: str | None = None

    @field_validator("spend")
    @classmethod
    def finite(cls, v: float) -> float:
        if not math.isfinite(v) or v < 0:
            raise ValueError("spend must be finite and non-negative")
        return v


class MMMSimulationPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["mmm_simulation_plan_v1"] = "mmm_simulation_plan_v1"
    plan_id: str
    role: MMMSimulationPlanRole
    spend_unit: str
    evaluation_time_window: str
    items: list[MMMSimulationPlanItem]
    total_spend: float
    scope: MMMSimulationScope | None = None
    source_id: str | None = None
    technical_description: str | None = None

    @model_validator(mode="after")
    def valid(self):
        if not self.plan_id or not self.items:
            raise ValueError("plan ID and items are required")
        if [i.channel_id for i in self.items] != sorted(i.channel_id for i in self.items):
            raise ValueError("plan items must be deterministically ordered")
        if len({(i.channel_id, i.geography, i.segment, i.time_bucket) for i in self.items}) != len(self.items):
            raise ValueError("duplicate scoped plan item")
        if any(i.spend_unit != self.spend_unit for i in self.items) or not math.isclose(
            sum(i.spend for i in self.items), self.total_spend, rel_tol=0, abs_tol=1e-9
        ):
            raise ValueError("plan units and total spend must be consistent")
        return self


class MMMSimulationUncertainty(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: MMMSimulationUncertaintyStatus = MMMSimulationUncertaintyStatus.UNAVAILABLE
    artifact_reference: str | None = None
    semantics: str | None = None

    @model_validator(mode="after")
    def valid(self):
        if self.status == MMMSimulationUncertaintyStatus.AVAILABLE and not self.artifact_reference:
            raise ValueError("available uncertainty requires evidence")
        return self


class MMMSimulationRangeEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    range_record_id: str
    channel_id: str
    baseline_relation: MMMRangeRelation
    candidate_relation: MMMRangeRelation
    extrapolation: MMMExtrapolationClassification
    blocked: bool = False


class MMMSimulationMetricResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    metric_id: str
    estimand: str
    aggregation_scope: str
    baseline_mu: float
    candidate_mu: float
    delta_mu: float
    unit: str
    uncertainty: MMMSimulationUncertainty = Field(default_factory=MMMSimulationUncertainty)
    supported_range_references: list[str] = Field(default_factory=list)
    claim_dispositions: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def arithmetic(self):
        if not math.isclose(self.delta_mu, self.candidate_mu - self.baseline_mu, rel_tol=0, abs_tol=1e-9):
            raise ValueError("delta_mu must equal candidate_mu minus baseline_mu")
        return self


class MMMSimulationComparison(BaseModel):
    model_config = ConfigDict(extra="forbid")
    comparison_id: str
    run_id: str
    model_id: str | None = None
    baseline_plan_id: str
    candidate_plan_id: str
    status: MMMSimulationStatus
    scope: MMMSimulationScope | None = None
    metrics: list[MMMSimulationMetricResult] = Field(default_factory=list)
    range_evaluations: list[MMMSimulationRangeEvaluation] = Field(default_factory=list)
    technical_summary: str


class MMMPublicSimulationExport(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["mmm_public_simulation_export_v1"] = "mmm_public_simulation_export_v1"
    export_id: str
    run_id: str
    created_at: datetime
    baseline_plan: MMMSimulationPlan
    candidate_plan: MMMSimulationPlan
    comparison: MMMSimulationComparison | None = None
    supported_range_evidence_id: str
    status: MMMSimulationStatus = MMMSimulationStatus.SUCCEEDED
    limitation_references: list[str] = Field(default_factory=list)
    blocking_references: list[str] = Field(default_factory=list)
    failure_packet: MMMFailurePacket | None = None
    diagnostics_limitations_id: str | None = None
    run_manifest_id: str | None = None
    run_manifest: MMMRunManifest | None = None
    export_manifest_outcome: MMMExportManifestOutcome | None = None
    producer_package_name: Literal["mmm"] = "mmm"
    producer_package_version: str = MMM_PACKAGE_VERSION
    artifact_kind: Literal["MMMPublicSimulationExport"] = MMM_PUBLIC_SIMULATION_ARTIFACT_KIND
    artifact_schema_version: Literal["mmm_public_simulation_export_v1"] = "mmm_public_simulation_export_v1"
    artifact_id: str | None = None

    @model_validator(mode="after")
    def terminal(self):
        if not self.producer_package_version.strip() or self.producer_package_version.startswith(("/", "~")):
            raise ValueError("producer package version must be safe non-empty text")
        if self.artifact_id is None:
            self.artifact_id = f"mmm_public_simulation:{self.run_id}"
        if self.run_id not in self.artifact_id:
            raise ValueError("artifact ID must include run ID")
        analytical_outcome = (
            self.export_manifest_outcome.analytical_outcome if self.export_manifest_outcome is not None else None
        )
        if self.run_manifest is not None:
            if (
                self.run_manifest.run_id != self.run_id
                or self.run_manifest.producer_package_name != self.producer_package_name
                or self.run_manifest.producer_package_version != self.producer_package_version
            ):
                raise ValueError("run manifest identity must match simulation export")
            if self.run_manifest.model_family is not None and self.run_manifest.model_family.lower() == "bayesian":
                raise ValueError("public simulation manifests cannot identify Bayesian execution")
            if self.status != MMMSimulationStatus.FAILED and (self.run_manifest.model_family or "").lower() != "ridge":
                raise ValueError("public simulation manifests must identify deterministic Ridge")
            if self.run_manifest.status.value != self.status.value:
                raise ValueError("run manifest terminal status must match simulation export")
            if self.run_manifest.successful_export is not None and (
                self.run_manifest.successful_export.artifact_id != self.artifact_id
                or self.run_manifest.successful_export.artifact_type != self.artifact_kind
                or self.run_manifest.successful_export.contract_version != self.artifact_schema_version
            ):
                raise ValueError("run manifest output artifact must match simulation export identity")
        if self.export_manifest_outcome is not None:
            if (
                self.export_manifest_outcome.outcome_kind != "analytical_artifact"
                or self.export_manifest_outcome.export_outcome is not None
                or self.export_manifest_outcome.analytical_outcome is None
            ):
                raise ValueError("public simulation must use an analytical_artifact outcome")
            if (
                self.run_manifest is None
                or analytical_outcome is None
                or self.export_manifest_outcome.run_manifest != self.run_manifest
                or analytical_outcome.run_id != self.run_id
                or analytical_outcome.producer_package_name != self.producer_package_name
                or analytical_outcome.producer_package_version != self.producer_package_version
                or analytical_outcome.status.value != self.status.value
            ):
                raise ValueError("analytical outcome identity and status must match simulation export")
            if analytical_outcome.output_artifact is not None and (
                analytical_outcome.output_artifact.artifact_id != self.artifact_id
                or analytical_outcome.output_artifact.artifact_type != self.artifact_kind
                or analytical_outcome.output_artifact.contract_version != self.artifact_schema_version
            ):
                raise ValueError("analytical outcome artifact must match simulation export identity")
        if self.status == MMMSimulationStatus.SUCCEEDED:
            if (
                self.comparison is None
                or self.failure_packet
                or self.blocking_references
                or self.run_manifest is None
                or self.run_manifest.status != MMMRunStatus.SUCCEEDED
                or self.run_manifest.limitation_ids
                or self.export_manifest_outcome is None
            ):
                raise ValueError(
                    "succeeded simulation requires a full-success manifest, analytical outcome, "
                    "and comparison without failure or blockers"
                )
        elif self.status == MMMSimulationStatus.SUCCEEDED_WITH_LIMITATIONS:
            if (
                self.comparison is None
                or self.failure_packet
                or not self.limitation_references
                or self.run_manifest is None
                or self.run_manifest.status != MMMRunStatus.SUCCEEDED_WITH_LIMITATIONS
                or self.run_manifest.limitation_ids != self.limitation_references
                or self.export_manifest_outcome is None
                or analytical_outcome is None
                or analytical_outcome.limitation_ids != self.limitation_references
            ):
                raise ValueError(
                    "limited simulation requires matching limitations, analytical outcome, "
                    "and a limited-success manifest"
                )
        elif self.status == MMMSimulationStatus.BLOCKED:
            if (
                self.comparison is not None
                or not self.blocking_references
                or self.failure_packet is None
                or self.run_manifest is None
                or self.run_manifest.status != MMMRunStatus.BLOCKED
                or self.run_manifest.failure_packet != self.failure_packet
                or self.run_manifest.successful_export is not None
                or self.run_manifest.limitation_ids
                or self.export_manifest_outcome is None
                or analytical_outcome is None
                or analytical_outcome.failure_packet != self.failure_packet
                or analytical_outcome.blocker_references != self.blocking_references
            ):
                raise ValueError(
                    "blocked simulation requires matching manifest, analytical outcome, typed failure, "
                    "and blockers without comparison"
                )
            if self.failure_packet.run_id and self.failure_packet.run_id != self.run_id:
                raise ValueError("blocked failure run ID must match")
        else:
            if (
                self.comparison is not None
                or self.failure_packet is None
                or self.run_manifest is None
                or self.run_manifest.status != MMMRunStatus.FAILED
                or self.run_manifest.failure_packet != self.failure_packet
                or self.run_manifest.successful_export is not None
                or self.run_manifest.limitation_ids
                or self.export_manifest_outcome is None
                or analytical_outcome is None
                or analytical_outcome.failure_packet != self.failure_packet
                or analytical_outcome.blocker_references
                or analytical_outcome.limitation_ids
                or analytical_outcome.output_artifact is not None
            ):
                raise ValueError(
                    "failed simulation requires matching manifest, analytical outcome, "
                    "and typed failure without comparison"
                )
            if self.failure_packet.run_id and self.failure_packet.run_id != self.run_id:
                raise ValueError("failed failure run ID must match")
        return self

    def to_json(self) -> str:
        return json.dumps(self.model_dump(mode="json", exclude_none=True), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, payload: str) -> MMMPublicSimulationExport:
        data = json.loads(payload)
        if not isinstance(data, dict):
            raise ValueError("public simulation export payload must be an object")
        return parse_mmm_public_simulation_export(data)


_PUBLIC_SIMULATION_REQUIRED_ENVELOPE_FIELDS = frozenset(
    {
        "schema_version",
        "export_id",
        "run_id",
        "created_at",
        "baseline_plan",
        "candidate_plan",
        "supported_range_evidence_id",
        "status",
        "run_manifest",
        "export_manifest_outcome",
        "producer_package_name",
        "producer_package_version",
        "artifact_kind",
        "artifact_schema_version",
        "artifact_id",
    }
)


def parse_mmm_public_simulation_export(data: Mapping[str, Any]) -> MMMPublicSimulationExport:
    """Parse the public export envelope without defaulting absent identity or evidence."""
    missing = sorted(_PUBLIC_SIMULATION_REQUIRED_ENVELOPE_FIELDS - set(data))
    if missing:
        raise ValueError(f"public simulation export is missing required envelope fields: {', '.join(missing)}")
    if data["schema_version"] != MMM_PUBLIC_SIMULATION_SCHEMA_VERSION:
        raise ValueError("unsupported public simulation schema version")
    if data["artifact_kind"] != MMM_PUBLIC_SIMULATION_ARTIFACT_KIND:
        raise ValueError("unknown public simulation artifact kind")
    if data["artifact_schema_version"] != MMM_PUBLIC_SIMULATION_SCHEMA_VERSION:
        raise ValueError("unsupported public simulation artifact schema version")
    return MMMPublicSimulationExport.model_validate(dict(data))


def _blocked_manifest(
    *,
    export_id: str,
    run_id: str,
    created_at: datetime,
    baseline_plan: MMMSimulationPlan,
    candidate_plan: MMMSimulationPlan,
    evidence_id: str,
    packet: MMMFailurePacket,
) -> MMMRunManifest:
    baseline_ref = MMMArtifactReference(
        artifact_type="MMMSimulationPlan",
        artifact_id=baseline_plan.plan_id,
        contract_version=baseline_plan.schema_version,
        logical_name="baseline_plan",
    )
    candidate_ref = MMMArtifactReference(
        artifact_type="MMMSimulationPlan",
        artifact_id=candidate_plan.plan_id,
        contract_version=candidate_plan.schema_version,
        logical_name="candidate_plan",
    )
    step = MMMRunStep(
        sequence=0,
        step_name="ridge_public_simulation",
        stage=MMMFailureStage.SIMULATION,
        status=MMMRunStepStatus.BLOCKED,
        started_at=created_at,
        completed_at=created_at,
        input_artifacts=[baseline_ref, candidate_ref],
        failure_packet_id=packet.failure_id,
        technical_detail="Deterministic Ridge public simulation was blocked before runtime execution.",
    )
    scope = baseline_plan.scope
    return build_mmm_run_manifest(
        manifest_id=f"manifest:{export_id}",
        run_id=run_id,
        created_at=created_at,
        started_at=created_at,
        completed_at=created_at,
        producer_package_version=MMM_PACKAGE_VERSION,
        status=MMMRunStatus.BLOCKED,
        model_id=scope.model_id if scope else None,
        model_family="ridge",
        model_version=scope.model_version if scope else None,
        estimator_identity="ridge_full_panel_public_simulation",
        configuration_hash=scope.configuration_hash if scope else None,
        dataset_fingerprint=scope.panel_id if scope else None,
        data_grain=scope.panel_grain if scope else None,
        kpi_identity=scope.metric_id if scope else None,
        time_range=baseline_plan.evaluation_time_window,
        market_scope=scope.geography if scope else None,
        supported_range_evidence_id=evidence_id or None,
        channel_scope=[i.channel_id for i in candidate_plan.items],
        steps=[step],
        failure_packet=packet,
    )


def _blocked_export_manifest_outcome(
    *, manifest: MMMRunManifest, packet: MMMFailurePacket, blocker: str, evidence_id: str
) -> MMMExportManifestOutcome:
    analytical = MMMAnalyticalArtifactOutcome(
        status=MMMRunStatus.BLOCKED,
        run_id=manifest.run_id,
        producer_package_version=manifest.producer_package_version,
        failure_packet=packet,
        blocker_references=[blocker],
    )
    return MMMExportManifestOutcome(
        outcome_kind="analytical_artifact",
        analytical_outcome=analytical,
        run_manifest=manifest,
        supported_range_evidence_id=evidence_id or None,
    )


def _blocked(
    *,
    export_id: str,
    run_id: str,
    created_at: datetime,
    baseline_plan: MMMSimulationPlan,
    candidate_plan: MMMSimulationPlan,
    evidence_id: str,
    blocker: str,
) -> MMMPublicSimulationExport:
    packet = build_mmm_failure_packet(
        failure_id=f"failure:{export_id}",
        created_at=created_at,
        run_id=run_id,
        code=MMMFailureCode.UNSUPPORTED_EXTRAPOLATION,
        stage=MMMFailureStage.SIMULATION,
        source_component="mmm.contracts.public_simulation",
        technical_summary=blocker,
        affected_resource=candidate_plan.plan_id,
        blockers=[blocker],
        supported_range_evidence=[evidence_id] if evidence_id else [],
    )
    manifest = _blocked_manifest(
        export_id=export_id,
        run_id=run_id,
        created_at=created_at,
        baseline_plan=baseline_plan,
        candidate_plan=candidate_plan,
        evidence_id=evidence_id,
        packet=packet,
    )
    outcome = _blocked_export_manifest_outcome(
        manifest=manifest, packet=packet, blocker=blocker, evidence_id=evidence_id
    )
    return MMMPublicSimulationExport(
        export_id=export_id,
        run_id=run_id,
        created_at=created_at,
        baseline_plan=baseline_plan,
        candidate_plan=candidate_plan,
        comparison=None,
        supported_range_evidence_id=evidence_id or "range-evidence-unavailable",
        status=MMMSimulationStatus.BLOCKED,
        blocking_references=[blocker],
        failure_packet=packet,
        run_manifest=manifest,
        export_manifest_outcome=outcome,
    )


def _range_evidence_unusable(
    *,
    export_id: str,
    run_id: str,
    created_at: datetime,
    baseline_plan: MMMSimulationPlan,
    candidate_plan: MMMSimulationPlan,
    evidence_id: str,
    reason: str,
) -> MMMPublicSimulationExport:
    packet = build_mmm_failure_packet(
        failure_id=f"failure:{export_id}",
        created_at=created_at,
        run_id=run_id,
        code=MMMFailureCode.SUPPORTED_RANGE_EVIDENCE_UNUSABLE,
        stage=MMMFailureStage.SIMULATION,
        source_component="mmm.contracts.public_simulation",
        technical_summary="Supported-range evidence cannot validate the requested technical scope.",
        affected_resource="supported-range-evidence",
        blockers=[reason],
        technical_context={"reason": reason},
    )
    manifest = _blocked_manifest(
        export_id=export_id,
        run_id=run_id,
        created_at=created_at,
        baseline_plan=baseline_plan,
        candidate_plan=candidate_plan,
        evidence_id=evidence_id,
        packet=packet,
    )
    outcome = _blocked_export_manifest_outcome(
        manifest=manifest, packet=packet, blocker=reason, evidence_id=evidence_id
    )
    return MMMPublicSimulationExport(
        export_id=export_id,
        run_id=run_id,
        created_at=created_at,
        baseline_plan=baseline_plan,
        candidate_plan=candidate_plan,
        comparison=None,
        supported_range_evidence_id=evidence_id,
        status=MMMSimulationStatus.BLOCKED,
        blocking_references=[reason],
        failure_packet=packet,
        run_manifest=manifest,
        export_manifest_outcome=outcome,
    )


def _resolve_range_record(
    evidence: MMMSupportedRangeEvidence, item: MMMSimulationPlanItem, scope: MMMSimulationScope | None
):
    def matches(record):
        rs = record.scope
        if rs.channel != item.channel_id:
            return False
        if scope is None:
            return True
        checks = (
            (rs.kpi, scope.metric_id),
            (record.model_id, scope.model_id),
            (record.model_family, scope.model_family),
            (record.model_version, scope.model_version),
            (record.configuration_hash, scope.configuration_hash),
            (rs.geography, scope.geography),
            (rs.segment, scope.segment),
            (rs.data_grain, scope.panel_grain),
            (rs.transformation_id, scope.transformation_id),
        )
        if any(left is not None and left != right for left, right in checks):
            return False
        bounds = record.supported_lower or record.observed_lower
        return (
            bounds is not None and bounds.unit == item.spend_unit and bounds.scale.value == scope.spend_scale
            if scope
            else bounds is not None and bounds.unit == item.spend_unit
        )

    return [record for record in evidence.records if matches(record)]


def _failed_manifest(
    *,
    export_id: str,
    run_id: str,
    created_at: datetime,
    baseline_plan: MMMSimulationPlan,
    candidate_plan: MMMSimulationPlan,
    packet: MMMFailurePacket,
    inputs_valid: bool,
) -> MMMRunManifest:
    scope = (
        baseline_plan.scope
        if inputs_valid and baseline_plan.scope is not None
        else candidate_plan.scope
        if inputs_valid
        else None
    )
    inputs = []
    if inputs_valid:
        inputs = [
            MMMArtifactReference(
                artifact_type="MMMSimulationPlan",
                artifact_id=baseline_plan.plan_id,
                contract_version=baseline_plan.schema_version,
                logical_name="baseline_plan",
            ),
            MMMArtifactReference(
                artifact_type="MMMSimulationPlan",
                artifact_id=candidate_plan.plan_id,
                contract_version=candidate_plan.schema_version,
                logical_name="candidate_plan",
            ),
        ]
    step = MMMRunStep(
        sequence=0,
        step_name="public_simulation_request_validation",
        stage=MMMFailureStage.SIMULATION,
        status=MMMRunStepStatus.FAILED,
        started_at=created_at,
        completed_at=created_at,
        input_artifacts=inputs,
        failure_packet_id=packet.failure_id,
        technical_detail="Public simulation request validation failed before Ridge runtime execution.",
    )
    ridge_scope = scope if scope is not None and scope.model_family.lower() == "ridge" else None
    return build_mmm_run_manifest(
        manifest_id=f"manifest:{export_id}",
        run_id=run_id,
        created_at=created_at,
        started_at=created_at,
        completed_at=created_at,
        producer_package_version=MMM_PACKAGE_VERSION,
        status=MMMRunStatus.FAILED,
        model_id=ridge_scope.model_id if ridge_scope else None,
        model_family="ridge" if ridge_scope else None,
        model_version=ridge_scope.model_version if ridge_scope else None,
        estimator_identity="mmm_public_simulation_request_validation",
        configuration_hash=ridge_scope.configuration_hash if ridge_scope else None,
        dataset_fingerprint=ridge_scope.panel_id if ridge_scope else None,
        data_grain=ridge_scope.panel_grain if ridge_scope else None,
        kpi_identity=ridge_scope.metric_id if ridge_scope else None,
        time_range=baseline_plan.evaluation_time_window if inputs_valid else None,
        market_scope=ridge_scope.geography if ridge_scope else None,
        channel_scope=[i.channel_id for i in candidate_plan.items] if inputs_valid else [],
        steps=[step],
        failure_packet=packet,
    )


def _failed_export_manifest_outcome(*, manifest: MMMRunManifest, packet: MMMFailurePacket) -> MMMExportManifestOutcome:
    analytical = MMMAnalyticalArtifactOutcome(
        status=MMMRunStatus.FAILED,
        run_id=manifest.run_id,
        producer_package_version=manifest.producer_package_version,
        failure_packet=packet,
    )
    return MMMExportManifestOutcome(
        outcome_kind="analytical_artifact", analytical_outcome=analytical, run_manifest=manifest
    )


def _failed(
    *,
    export_id: str,
    run_id: str,
    created_at: datetime,
    baseline_plan: MMMSimulationPlan,
    candidate_plan: MMMSimulationPlan,
    inputs_valid: bool = True,
) -> MMMPublicSimulationExport:
    packet = build_mmm_failure_packet(
        failure_id=f"failure:{export_id}",
        created_at=created_at,
        run_id=run_id,
        code=MMMFailureCode.INVALID_PLAN_INPUT,
        stage=MMMFailureStage.SIMULATION,
        source_component="mmm.contracts.public_simulation",
        technical_summary="Caller-supplied simulation plan failed contract validation.",
        affected_resource="simulation-plan",
        failure_status="failed",
    )
    manifest = _failed_manifest(
        export_id=export_id,
        run_id=run_id,
        created_at=created_at,
        baseline_plan=baseline_plan,
        candidate_plan=candidate_plan,
        packet=packet,
        inputs_valid=inputs_valid,
    )
    outcome = _failed_export_manifest_outcome(manifest=manifest, packet=packet)
    return MMMPublicSimulationExport(
        export_id=export_id,
        run_id=run_id,
        created_at=created_at,
        baseline_plan=baseline_plan,
        candidate_plan=candidate_plan,
        comparison=None,
        supported_range_evidence_id="range-evidence-unavailable",
        status=MMMSimulationStatus.FAILED,
        failure_packet=packet,
        run_manifest=manifest,
        export_manifest_outcome=outcome,
    )


def _scopes_match(baseline: MMMSimulationScope | None, candidate: MMMSimulationScope | None) -> bool:
    if baseline is None or candidate is None:
        return baseline is candidate
    return baseline == candidate


def build_mmm_public_simulation_export_from_payloads(
    *,
    export_id: str,
    run_id: str,
    created_at: datetime,
    ctx: RidgeFitContext,
    baseline_payload: dict[str, Any],
    candidate_payload: dict[str, Any],
    supported_range_evidence: MMMSupportedRangeEvidence | None,
    model_id: str | None = None,
) -> MMMPublicSimulationExport:
    """Convert only caller plan-contract errors into a safe typed failed export."""
    fallback_b = MMMSimulationPlan(
        plan_id="invalid-baseline",
        role=MMMSimulationPlanRole.BASELINE,
        spend_unit="unknown",
        evaluation_time_window="unknown",
        items=[MMMSimulationPlanItem(channel_id="invalid", spend=0, spend_unit="unknown")],
        total_spend=0,
    )
    fallback_c = MMMSimulationPlan(
        plan_id="invalid-candidate",
        role=MMMSimulationPlanRole.CANDIDATE,
        spend_unit="unknown",
        evaluation_time_window="unknown",
        items=[MMMSimulationPlanItem(channel_id="invalid", spend=0, spend_unit="unknown")],
        total_spend=0,
    )
    try:
        baseline = MMMSimulationPlan.model_validate(baseline_payload)
        candidate = MMMSimulationPlan.model_validate(candidate_payload)
        if baseline.role != MMMSimulationPlanRole.BASELINE or candidate.role != MMMSimulationPlanRole.CANDIDATE:
            raise ValueError("plan roles are invalid")
    except (ValidationError, ValueError):
        return _failed(
            export_id=export_id,
            run_id=run_id,
            created_at=created_at,
            baseline_plan=fallback_b,
            candidate_plan=fallback_c,
            inputs_valid=False,
        )
    return build_mmm_public_simulation_export(
        export_id=export_id,
        run_id=run_id,
        created_at=created_at,
        ctx=ctx,
        baseline_plan=baseline,
        candidate_plan=candidate,
        supported_range_evidence=supported_range_evidence,
        model_id=model_id,
    )


def build_mmm_public_simulation_export(
    *,
    export_id: str,
    run_id: str,
    created_at: datetime,
    ctx: RidgeFitContext,
    baseline_plan: MMMSimulationPlan,
    candidate_plan: MMMSimulationPlan,
    supported_range_evidence: MMMSupportedRangeEvidence | None,
    model_id: str | None = None,
) -> MMMPublicSimulationExport:
    if baseline_plan.role != MMMSimulationPlanRole.BASELINE or candidate_plan.role != MMMSimulationPlanRole.CANDIDATE:
        raise ValueError("baseline and candidate roles are required")
    if not _scopes_match(baseline_plan.scope, candidate_plan.scope):
        return _failed(
            export_id=export_id,
            run_id=run_id,
            created_at=created_at,
            baseline_plan=baseline_plan,
            candidate_plan=candidate_plan,
        )
    if (
        baseline_plan.spend_unit != candidate_plan.spend_unit
        or baseline_plan.evaluation_time_window != candidate_plan.evaluation_time_window
    ):
        raise ValueError("plans must share unit and evaluation window")
    if supported_range_evidence is None:
        return _blocked(
            export_id=export_id,
            run_id=run_id,
            created_at=created_at,
            baseline_plan=baseline_plan,
            candidate_plan=candidate_plan,
            evidence_id="",
            blocker="Required supported-range evidence is unavailable.",
        )
    if supported_range_evidence.run_id != run_id:
        raise ValueError("supported range evidence must match run ID")
    ev = []
    for _baseline_item, c in zip(baseline_plan.items, candidate_plan.items, strict=True):
        matches = _resolve_range_record(supported_range_evidence, c, baseline_plan.scope)
        if len(matches) != 1:
            return _range_evidence_unusable(
                export_id=export_id,
                run_id=run_id,
                created_at=created_at,
                baseline_plan=baseline_plan,
                candidate_plan=candidate_plan,
                evidence_id=supported_range_evidence.evidence_id,
                reason="ambiguous" if len(matches) > 1 else "missing_or_scope_incompatible",
            )
        r = matches[0]
        status_reason = {
            MMMRangeAvailabilityStatus.PARTIALLY_AVAILABLE: "partially_available",
            MMMRangeAvailabilityStatus.UNAVAILABLE: "unavailable",
            MMMRangeAvailabilityStatus.BLOCKED: "blocked",
            MMMRangeAvailabilityStatus.RESEARCH_ONLY: "research_only",
        }
        if r.availability_status != MMMRangeAvailabilityStatus.AVAILABLE:
            return _range_evidence_unusable(
                export_id=export_id,
                run_id=run_id,
                created_at=created_at,
                baseline_plan=baseline_plan,
                candidate_plan=candidate_plan,
                evidence_id=supported_range_evidence.evidence_id,
                reason=status_reason.get(r.availability_status, "non_governed_status"),
            )
        if r.simulation_eligibility != MMMSupportedRangeSimulationEligibility.ELIGIBLE_FOR_TECHNICAL_SIMULATION:
            return _range_evidence_unusable(
                export_id=export_id,
                run_id=run_id,
                created_at=created_at,
                baseline_plan=baseline_plan,
                candidate_plan=candidate_plan,
                evidence_id=supported_range_evidence.evidence_id,
                reason="simulation_eligibility_not_assessed"
                if r.simulation_eligibility == MMMSupportedRangeSimulationEligibility.NOT_ASSESSED
                else "simulation_not_eligible",
            )
        outside = (
            r.supported_lower is None
            or r.supported_upper is None
            or c.spend < r.supported_lower.value
            or c.spend > r.supported_upper.value
        )
        ev.append(
            MMMSimulationRangeEvaluation(
                range_record_id=r.range_record_id,
                channel_id=c.channel_id,
                baseline_relation=MMMRangeRelation.WITHIN_SUPPORTED_RANGE,
                candidate_relation=MMMRangeRelation.OUTSIDE_SUPPORTED_RANGE
                if outside
                else MMMRangeRelation.WITHIN_SUPPORTED_RANGE,
                extrapolation=MMMExtrapolationClassification.UNSUPPORTED_EXTRAPOLATION
                if outside
                else MMMExtrapolationClassification.INTERPOLATION,
                blocked=outside,
            )
        )
    if any(x.blocked for x in ev):
        return _blocked(
            export_id=export_id,
            run_id=run_id,
            created_at=created_at,
            baseline_plan=baseline_plan,
            candidate_plan=candidate_plan,
            evidence_id=supported_range_evidence.evidence_id,
            blocker="Candidate spend exceeds the supported range and was not changed.",
        )
    channels = tuple(ctx.schema.channel_columns)
    if (
        tuple(i.channel_id for i in baseline_plan.items) != channels
        or tuple(i.channel_id for i in candidate_plan.items) != channels
    ):
        raise ValueError("public plans must completely match the fitted panel channels")
    baseline = BaselinePlan(
        BaselineType.LOCKED_PLAN,
        {i.channel_id: i.spend for i in baseline_plan.items},
        "Caller supplied baseline",
        "public_simulation",
        False,
    )
    result = simulate(
        {i.channel_id: i.spend for i in candidate_plan.items}, ctx, baseline_plan=baseline, uncertainty_mode="point"
    )
    metric = MMMSimulationMetricResult(
        metric_id=ctx.schema.target_column,
        estimand="full_panel_delta_mu",
        aggregation_scope=result.aggregation_semantics,
        baseline_mu=result.baseline_mu,
        candidate_mu=result.plan_mu,
        delta_mu=result.delta_mu,
        unit="modeling_scale",
        supported_range_references=[x.range_record_id for x in ev],
        claim_dispositions={
            MMMTechnicalClaim.IN_RANGE_SIMULATION.value: MMMTechnicalClaimDisposition.UNAVAILABLE.value
        },
    )
    comparison = MMMSimulationComparison(
        comparison_id=f"comparison:{export_id}",
        run_id=run_id,
        model_id=model_id,
        baseline_plan_id=baseline_plan.plan_id,
        candidate_plan_id=candidate_plan.plan_id,
        status=MMMSimulationStatus.SUCCEEDED,
        scope=baseline_plan.scope,
        metrics=[metric],
        range_evaluations=ev,
        technical_summary="Full-panel Ridge candidate minus baseline technical simulation; no recommendation.",
    )
    artifact_id = f"mmm_public_simulation:{run_id}"
    artifact = MMMArtifactReference(
        artifact_type="MMMPublicSimulationExport",
        artifact_id=artifact_id,
        contract_version=MMM_PUBLIC_SIMULATION_SCHEMA_VERSION,
        logical_name="mmm_public_simulation",
    )
    baseline_ref = MMMArtifactReference(
        artifact_type="MMMSimulationPlan",
        artifact_id=baseline_plan.plan_id,
        contract_version=baseline_plan.schema_version,
        logical_name="baseline_plan",
    )
    candidate_ref = MMMArtifactReference(
        artifact_type="MMMSimulationPlan",
        artifact_id=candidate_plan.plan_id,
        contract_version=candidate_plan.schema_version,
        logical_name="candidate_plan",
    )
    step = MMMRunStep(
        sequence=0,
        step_name="ridge_public_simulation",
        stage=MMMFailureStage.SIMULATION,
        status=MMMRunStepStatus.SUCCEEDED,
        started_at=created_at,
        completed_at=created_at,
        input_artifacts=[baseline_ref, candidate_ref],
        output_artifacts=[artifact],
        technical_detail="Deterministic Ridge full-panel candidate-minus-baseline simulation.",
    )
    manifest = build_mmm_run_manifest(
        manifest_id=f"manifest:{export_id}",
        run_id=run_id,
        created_at=created_at,
        started_at=created_at,
        completed_at=created_at,
        producer_package_version=MMM_PACKAGE_VERSION,
        status=MMMRunStatus.SUCCEEDED,
        model_id=model_id or (baseline_plan.scope.model_id if baseline_plan.scope else None),
        model_family="ridge",
        model_version=baseline_plan.scope.model_version if baseline_plan.scope else None,
        estimator_identity="ridge_full_panel_public_simulation",
        configuration_hash=baseline_plan.scope.configuration_hash if baseline_plan.scope else None,
        dataset_fingerprint=baseline_plan.scope.panel_id if baseline_plan.scope else None,
        data_grain=baseline_plan.scope.panel_grain if baseline_plan.scope else None,
        kpi_identity=baseline_plan.scope.metric_id if baseline_plan.scope else None,
        time_range=baseline_plan.evaluation_time_window,
        market_scope=baseline_plan.scope.geography if baseline_plan.scope else None,
        supported_range_evidence_id=supported_range_evidence.evidence_id,
        channel_scope=[i.channel_id for i in candidate_plan.items],
        steps=[step],
        successful_export=artifact,
    )
    analytical = MMMAnalyticalArtifactOutcome(
        status=MMMRunStatus.SUCCEEDED,
        run_id=run_id,
        producer_package_version=MMM_PACKAGE_VERSION,
        output_artifact=artifact,
    )
    outcome = MMMExportManifestOutcome(
        outcome_kind="analytical_artifact",
        analytical_outcome=analytical,
        run_manifest=manifest,
        supported_range_evidence_id=supported_range_evidence.evidence_id,
    )
    return MMMPublicSimulationExport(
        export_id=export_id,
        run_id=run_id,
        created_at=created_at,
        baseline_plan=baseline_plan,
        candidate_plan=candidate_plan,
        comparison=comparison,
        supported_range_evidence_id=supported_range_evidence.evidence_id,
        run_manifest=manifest,
        export_manifest_outcome=outcome,
    )
