"""Versioned producer run-manifest contracts for the MMM-to-MIP boundary.

The legacy :func:`build_run_manifest` remains below for existing internal callers.
``MMMRunManifest`` is the governed, typed handoff contract; it is technical
producer evidence and never a recommendation, TrustReport, or user response.
"""

from __future__ import annotations

import json
from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from mmm.contracts.mip_export import CalibrationStatus, PromotionStatus
from mmm.contracts.mip_failure import MMMExportOutcome, MMMFailurePacket, MMMFailureStage

RUN_MANIFEST_VERSION = "mmm_run_manifest_v1"
MMM_RUN_MANIFEST_SCHEMA_VERSION = "mmm_mip_run_manifest_v1"


class MMMRunStatus(str, Enum):
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"


class MMMRunStepStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"
    SKIPPED = "SKIPPED"


class MMMArtifactAvailability(str, Enum):
    AVAILABLE = "AVAILABLE"
    MISSING = "MISSING"
    BLOCKED = "BLOCKED"


_UNSAFE_TEXT_MARKERS = (
    "traceback (most recent call last)",
    "stack trace",
    "password=",
    "secret=",
    "api_key=",
    "token=",
)


def _safe_text(value: str, *, field_name: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} must be non-empty when supplied")
    normalized = cleaned.lower()
    if cleaned.startswith(("/", "~")) or any(marker in normalized for marker in _UNSAFE_TEXT_MARKERS):
        raise ValueError(f"{field_name} must not contain paths, stack traces, or secrets")
    return cleaned


class MMMArtifactReference(BaseModel):
    """A safe, stable identifier for a producer artifact, never its payload."""

    model_config = ConfigDict(extra="forbid")

    artifact_type: str = Field(min_length=1, max_length=120)
    artifact_id: str = Field(min_length=1, max_length=200)
    contract_version: str = Field(min_length=1, max_length=100)
    content_fingerprint: str | None = Field(default=None, max_length=200)
    logical_name: str | None = Field(default=None, max_length=200)
    availability: MMMArtifactAvailability = MMMArtifactAvailability.AVAILABLE

    @field_validator("artifact_type", "artifact_id", "contract_version", "content_fingerprint", "logical_name")
    @classmethod
    def _safe_reference_text(cls, value: str | None, info: Any) -> str | None:
        return None if value is None else _safe_text(value, field_name=info.field_name)


class MMMRunStep(BaseModel):
    """One deterministically ordered, machine-readable producer lifecycle step."""

    model_config = ConfigDict(extra="forbid")

    sequence: int = Field(ge=0)
    step_name: str = Field(min_length=1, max_length=120)
    stage: MMMFailureStage
    status: MMMRunStepStatus
    started_at: datetime | None = None
    completed_at: datetime | None = None
    input_artifacts: list[MMMArtifactReference] = Field(default_factory=list)
    output_artifacts: list[MMMArtifactReference] = Field(default_factory=list)
    validation_result_ids: list[str] = Field(default_factory=list)
    diagnostic_ids: list[str] = Field(default_factory=list)
    failure_packet_id: str | None = Field(default=None, max_length=200)
    technical_detail: str | None = Field(default=None, max_length=500)

    @field_validator("step_name", "failure_packet_id", "technical_detail")
    @classmethod
    def _safe_optional_text(cls, value: str | None, info: Any) -> str | None:
        return None if value is None else _safe_text(value, field_name=info.field_name)

    @field_validator("validation_result_ids", "diagnostic_ids")
    @classmethod
    def _safe_identifiers(cls, values: list[str], info: Any) -> list[str]:
        return [_safe_text(value, field_name=info.field_name) for value in values]

    @field_validator("started_at", "completed_at")
    @classmethod
    def _aware_timestamps(cls, value: datetime | None, info: Any) -> datetime | None:
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError(f"{info.field_name} must be timezone-aware")
        return value

    @model_validator(mode="after")
    def _consistent_step_state(self) -> MMMRunStep:
        if self.started_at and self.completed_at and self.completed_at < self.started_at:
            raise ValueError("step completion time cannot precede step start time")
        if self.status == MMMRunStepStatus.PENDING and (self.started_at or self.completed_at):
            raise ValueError("pending steps cannot have timestamps")
        if self.status == MMMRunStepStatus.RUNNING and self.completed_at:
            raise ValueError("running steps cannot have completion timestamps")
        failed = self.status in {MMMRunStepStatus.FAILED, MMMRunStepStatus.BLOCKED}
        if failed and not self.failure_packet_id:
            raise ValueError("failed or blocked steps require a failure_packet_id")
        if not failed and self.failure_packet_id:
            raise ValueError("only failed or blocked steps may reference a failure_packet_id")
        return self


class MMMRunManifest(BaseModel):
    """Stable, versioned MMM execution evidence for the producer boundary."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[MMM_RUN_MANIFEST_SCHEMA_VERSION] = MMM_RUN_MANIFEST_SCHEMA_VERSION
    manifest_id: str = Field(min_length=1, max_length=200)
    run_id: str = Field(min_length=1, max_length=200)
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    producer_package_name: str = Field(default="mmm", min_length=1, max_length=100)
    producer_package_version: str = Field(min_length=1, max_length=100)
    producer_contract_version: str = Field(default=MMM_RUN_MANIFEST_SCHEMA_VERSION, min_length=1, max_length=100)
    status: MMMRunStatus
    model_id: str | None = Field(default=None, max_length=200)
    model_family: str | None = Field(default=None, max_length=100)
    model_version: str | None = Field(default=None, max_length=100)
    estimator_identity: str | None = Field(default=None, max_length=200)
    promotion_status: PromotionStatus | None = None
    configuration_hash: str | None = Field(default=None, max_length=200)
    dataset_fingerprint: str | None = Field(default=None, max_length=200)
    data_grain: str | None = Field(default=None, max_length=100)
    kpi_identity: str | None = Field(default=None, max_length=200)
    time_range: str | None = Field(default=None, max_length=200)
    market_scope: str | None = Field(default=None, max_length=200)
    channel_scope: list[str] = Field(default_factory=list)
    calibration_signal_ids: list[str] = Field(default_factory=list)
    calibration_status: CalibrationStatus | None = None
    calibration_lineage_id: str | None = Field(default=None, max_length=200)
    diagnostics_limitations_id: str | None = Field(default=None, max_length=200)
    supported_range_evidence_id: str | None = Field(default=None, max_length=200)
    validation_result_ids: list[str] = Field(default_factory=list)
    diagnostic_ids: list[str] = Field(default_factory=list)
    steps: list[MMMRunStep] = Field(default_factory=list)
    successful_export: MMMArtifactReference | None = None
    failure_packet: MMMFailurePacket | None = None

    @field_validator(
        "manifest_id", "run_id", "producer_package_name", "producer_package_version", "producer_contract_version",
        "model_id", "model_family", "model_version", "estimator_identity", "configuration_hash",
        "dataset_fingerprint", "data_grain", "kpi_identity", "time_range", "market_scope", "calibration_lineage_id", "diagnostics_limitations_id", "supported_range_evidence_id",
    )
    @classmethod
    def _safe_manifest_text(cls, value: str | None, info: Any) -> str | None:
        return None if value is None else _safe_text(value, field_name=info.field_name)

    @field_validator("channel_scope", "calibration_signal_ids", "validation_result_ids", "diagnostic_ids")
    @classmethod
    def _safe_manifest_identifiers(cls, values: list[str], info: Any) -> list[str]:
        return [_safe_text(value, field_name=info.field_name) for value in values]

    @field_validator("created_at", "started_at", "completed_at")
    @classmethod
    def _aware_manifest_timestamps(cls, value: datetime | None, info: Any) -> datetime | None:
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError(f"{info.field_name} must be timezone-aware")
        return value

    @model_validator(mode="after")
    def _consistent_terminal_state(self) -> MMMRunManifest:
        if self.started_at and self.completed_at and self.completed_at < self.started_at:
            raise ValueError("manifest completion time cannot precede manifest start time")
        if [step.sequence for step in self.steps] != list(range(len(self.steps))):
            raise ValueError("steps must be ordered deterministically with contiguous sequence values")
        if self.status == MMMRunStatus.SUCCEEDED:
            if self.successful_export is None:
                raise ValueError("succeeded manifests require a successful_export reference")
            if self.successful_export.availability != MMMArtifactAvailability.AVAILABLE:
                raise ValueError("succeeded manifests require an available successful_export")
            if self.failure_packet is not None:
                raise ValueError("succeeded manifests cannot contain a terminal failure packet")
            if any(step.status in {MMMRunStepStatus.FAILED, MMMRunStepStatus.BLOCKED} for step in self.steps):
                raise ValueError("succeeded manifests cannot contain failed or blocked steps")
        elif self.status in {MMMRunStatus.FAILED, MMMRunStatus.BLOCKED}:
            if self.failure_packet is None:
                raise ValueError("failed or blocked manifests require a typed failure_packet")
            if self.successful_export is not None:
                raise ValueError("failed or blocked manifests cannot claim successful export completion")
            if self.status.value.lower() != self.failure_packet.failure_status:
                raise ValueError("manifest terminal status must match failure_packet.failure_status")
            for step in self.steps:
                if step.failure_packet_id and step.failure_packet_id != self.failure_packet.failure_id:
                    raise ValueError("failed step must reference the manifest failure packet")
        else:
            if self.successful_export is not None or self.failure_packet is not None:
                raise ValueError("running manifests cannot claim a terminal success or failure")
            if any(step.status in {MMMRunStepStatus.FAILED, MMMRunStepStatus.BLOCKED} for step in self.steps):
                raise ValueError("running manifests cannot contain failed or blocked steps")
        return self

    def to_json_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=True)

    def to_json(self) -> str:
        return json.dumps(self.to_json_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, payload: str) -> MMMRunManifest:
        return cls.model_validate_json(payload)


class MMMExportManifestOutcome(BaseModel):
    """Additive producer-boundary result linking a typed outcome to its manifest."""

    model_config = ConfigDict(extra="forbid")

    export_outcome: MMMExportOutcome
    run_manifest: MMMRunManifest
    supported_range_evidence_id: str | None = Field(default=None, max_length=200)

    @model_validator(mode="after")
    def _outcome_and_manifest_agree(self) -> MMMExportManifestOutcome:
        if self.supported_range_evidence_id and self.supported_range_evidence_id != self.run_manifest.supported_range_evidence_id:
            raise ValueError("supported range evidence outcome reference must match the run manifest")
        if self.export_outcome.outcome_type == "success":
            if self.run_manifest.status != MMMRunStatus.SUCCEEDED or self.run_manifest.successful_export is None:
                raise ValueError("successful export outcomes require a succeeded run manifest")
            if self.run_manifest.successful_export.artifact_id != self.export_outcome.export_bundle.model_run_id:  # type: ignore[union-attr]
                raise ValueError("successful manifest export reference must match the export bundle run ID")
        else:
            packet = self.export_outcome.failure_packet
            if self.run_manifest.status not in {MMMRunStatus.FAILED, MMMRunStatus.BLOCKED}:
                raise ValueError("failed export outcomes require a failed or blocked run manifest")
            if self.run_manifest.failure_packet != packet:
                raise ValueError("failed export outcome and run manifest must carry the same failure packet")
        return self


def build_mmm_run_manifest(**fields: Any) -> MMMRunManifest:
    """Build the strict public manifest without inferring absent producer evidence."""
    return MMMRunManifest(**fields)


# Keys on extension_report / trainer outputs and their artifact tier for human + machine readers.
EXTENSION_REPORT_SURFACE_TIERS: dict[str, str] = {
    "decision_bundle": "research",
    "ridge_fit_summary": "decision_input",
    "governance": "decision_input",
    "model_release": "decision_input",
    "operational_health": "decision_input",
    "post_fit_validation": "diagnostic",
    "response_diagnostics": "diagnostic",
    "curve_bundle": "diagnostic",
    "curve_bundles": "diagnostic",
    "roi_summary": "diagnostic",
    "posterior_exploration_quantity": "research",
    "uncertainty_decomposition": "research",
    "uncertainty_propagation_report": "research",
    "robust_optimization_research": "research",
    "continuous_validation_report": "diagnostic",
    "decision_validation_report": "diagnostic",
    "falsification": "diagnostic",
    "identifiability": "diagnostic",
    "feature_separability_report": "diagnostic",
    "experiment_scheduler_report": "diagnostic",
    "baselines": "diagnostic",
    "panel_qa": "diagnostic",
    "calibration_summary": "diagnostic",
    "experiment_matching": "diagnostic",
    "experiment_compatibility_report": "diagnostic",
    "evidence_weighting_report": "diagnostic",
    "counterfactual_shock_plan": "diagnostic",
    "experiment_evidence_registry_coverage": "diagnostic",
    "governance_unsupported_claims": "diagnostic",
    "evidence_weighted_replay_summary": "diagnostic",
    "bayesian_experiment_likelihood_report": "research",
    "bayesian_hierarchy_report": "research",
    "hierarchy_diagnostics": "diagnostic",
    "hierarchy_effect_summary": "diagnostic",
    "hierarchy_validation_report": "diagnostic",
    "hierarchy_governance_warnings": "diagnostic",
}


def build_run_manifest(extension_report: dict[str, Any], *, run_id: str | None = None) -> dict[str, Any]:
    """Legacy untyped internal index retained for existing orchestration callers."""
    keys = sorted(str(k) for k in extension_report if not str(k).startswith("_"))
    tiers = {k: EXTENSION_REPORT_SURFACE_TIERS.get(k, "unknown") for k in keys}
    return {
        "manifest_version": RUN_MANIFEST_VERSION,
        "run_id": run_id,
        "extension_report_top_level_keys": keys,
        "surface_tier_by_key": tiers,
        "canonical_decision_inputs": ["ridge_fit_summary", "governance", "model_release", "operational_health", "panel_qa"],
        "notes": [
            "Curve/decomposition/ROI-like blocks remain diagnostic unless promoted by separate decision bundle policy.",
            "Use mmm.decision.service paths for prod decision JSON; do not treat this manifest as decision truth.",
        ],
    }


__all__ = [
    "MMM_RUN_MANIFEST_SCHEMA_VERSION",
    "MMMArtifactAvailability",
    "MMMArtifactReference",
    "MMMExportManifestOutcome",
    "MMMRunManifest",
    "MMMRunStatus",
    "MMMRunStep",
    "MMMRunStepStatus",
    "build_mmm_run_manifest",
    "build_run_manifest",
]
