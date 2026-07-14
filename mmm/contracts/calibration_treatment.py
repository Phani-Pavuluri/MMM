"""Typed, producer-owned calibration-treatment lineage for MMM handoff evidence."""

from __future__ import annotations

import json
import math
from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from mmm.contracts.mip_failure import MMMFailureCode, MMMFailurePacket
from mmm.contracts.run_manifest import MMMArtifactReference

MMM_CALIBRATION_TREATMENT_LINEAGE_SCHEMA_VERSION = "mmm_calibration_treatment_lineage_v1"


class MMMCalibrationTreatmentDisposition(str, Enum):
    APPLIED = "APPLIED"
    ACCEPTED_NOT_APPLIED = "ACCEPTED_NOT_APPLIED"
    REJECTED = "REJECTED"
    UNUSED = "UNUSED"
    BLOCKED = "BLOCKED"


class MMMCalibrationApplicationRole(str, Enum):
    PRIOR = "PRIOR"
    LIKELIHOOD = "LIKELIHOOD"
    CONSTRAINT = "CONSTRAINT"
    DIAGNOSTIC_ONLY = "DIAGNOSTIC_ONLY"
    VALIDATION_ONLY = "VALIDATION_ONLY"
    EVIDENCE_CONTEXT_ONLY = "EVIDENCE_CONTEXT_ONLY"
    NONE = "NONE"


class MMMCalibrationCompatibilityStatus(str, Enum):
    COMPATIBLE = "COMPATIBLE"
    PARTIALLY_COMPATIBLE = "PARTIALLY_COMPATIBLE"
    INCOMPATIBLE = "INCOMPATIBLE"
    UNKNOWN = "UNKNOWN"


class MMMCalibrationFreshnessStatus(str, Enum):
    FRESH = "FRESH"
    STALE = "STALE"
    EXPIRED = "EXPIRED"
    UNKNOWN = "UNKNOWN"


class MMMCalibrationCompatibilityReason(str, Enum):
    KPI_MISMATCH = "KPI_MISMATCH"
    ESTIMAND_MISMATCH = "ESTIMAND_MISMATCH"
    GEO_SCOPE_MISMATCH = "GEO_SCOPE_MISMATCH"
    MARKET_SCOPE_MISMATCH = "MARKET_SCOPE_MISMATCH"
    SEGMENT_SCOPE_MISMATCH = "SEGMENT_SCOPE_MISMATCH"
    CHANNEL_SCOPE_MISMATCH = "CHANNEL_SCOPE_MISMATCH"
    TIME_WINDOW_MISMATCH = "TIME_WINDOW_MISMATCH"
    TREATMENT_DEFINITION_MISMATCH = "TREATMENT_DEFINITION_MISMATCH"
    OUTCOME_SCALE_MISMATCH = "OUTCOME_SCALE_MISMATCH"
    LIFT_SCALE_MISMATCH = "LIFT_SCALE_MISMATCH"
    UNCERTAINTY_MISSING = "UNCERTAINTY_MISSING"
    CAUSAL_EVIDENCE_RESTRICTED = "CAUSAL_EVIDENCE_RESTRICTED"
    GOVERNANCE_BLOCKED = "GOVERNANCE_BLOCKED"


def _safe(value: str, field_name: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError(f"{field_name} must be non-empty")
    lowered = value.lower()
    if value.startswith(("/", "~")) or any(x in lowered for x in ("traceback", "stack trace", "password=", "secret=", "token=")):
        raise ValueError(f"{field_name} must not contain paths, stack traces, or secrets")
    return value


class MMMCalibrationTransformationStep(BaseModel):
    """One declared, deterministic transformation actually used for a signal."""

    model_config = ConfigDict(extra="forbid")
    sequence: int = Field(ge=0)
    transformation_code: str = Field(min_length=1, max_length=120)
    transformation_version: str = Field(min_length=1, max_length=100)
    input_estimand_or_scale: str | None = Field(default=None, max_length=200)
    output_estimand_or_scale: str | None = Field(default=None, max_length=200)
    input_unit: str | None = Field(default=None, max_length=100)
    output_unit: str | None = Field(default=None, max_length=100)
    parameter_summary: str | None = Field(default=None, max_length=300)
    evidence_reference: str | None = Field(default=None, max_length=200)

    @field_validator("transformation_code", "transformation_version", "input_estimand_or_scale", "output_estimand_or_scale", "input_unit", "output_unit", "parameter_summary", "evidence_reference")
    @classmethod
    def _text(cls, value: str | None, info: Any) -> str | None:
        return None if value is None else _safe(value, info.field_name)


class MMMCalibrationTreatmentRecord(BaseModel):
    """Explicit technical disposition for one considered calibration signal."""

    model_config = ConfigDict(extra="forbid")
    sequence: int = Field(ge=0)
    signal_id: str = Field(min_length=1, max_length=200)
    signal_schema_version: str | None = Field(default=None, max_length=100)
    source_type: str = Field(min_length=1, max_length=100)
    source_artifact_id: str | None = Field(default=None, max_length=200)
    source_package: str | None = Field(default=None, max_length=100)
    run_id: str | None = Field(default=None, max_length=200)
    model_id: str | None = Field(default=None, max_length=200)
    model_family: str | None = Field(default=None, max_length=100)
    model_version: str | None = Field(default=None, max_length=100)
    configuration_hash: str | None = Field(default=None, max_length=200)
    run_manifest_step: str | None = Field(default=None, max_length=120)
    kpi_identity: str | None = Field(default=None, max_length=200)
    estimand: str | None = Field(default=None, max_length=200)
    treatment_definition: str | None = Field(default=None, max_length=300)
    point_estimate: float | None = None
    uncertainty_type: str | None = Field(default=None, max_length=100)
    uncertainty_value: float | None = None
    uncertainty_interval: tuple[float, float] | None = None
    geo_scope: list[str] = Field(default_factory=list)
    market_scope: list[str] = Field(default_factory=list)
    segment_scope: list[str] = Field(default_factory=list)
    channel_scope: list[str] = Field(default_factory=list)
    effective_time_window: str | None = Field(default=None, max_length=200)
    observed_at: datetime | None = None
    freshness_status: MMMCalibrationFreshnessStatus
    compatibility_status: MMMCalibrationCompatibilityStatus
    compatibility_reasons: list[MMMCalibrationCompatibilityReason] = Field(default_factory=list)
    disposition: MMMCalibrationTreatmentDisposition
    application_roles: list[MMMCalibrationApplicationRole] = Field(default_factory=lambda: [MMMCalibrationApplicationRole.NONE])
    applied_weight: float | None = Field(default=None, ge=0.0, le=1.0)
    governing_policy_reference: str | None = Field(default=None, max_length=200)
    transformations: list[MMMCalibrationTransformationStep] = Field(default_factory=list)
    diagnostic_references: list[str] = Field(default_factory=list)
    validation_references: list[str] = Field(default_factory=list)
    governance_references: list[str] = Field(default_factory=list)
    failure_packet: MMMFailurePacket | None = None
    research_only: bool = False
    technical_detail: str = Field(min_length=1, max_length=500)

    @field_validator("signal_id", "signal_schema_version", "source_type", "source_artifact_id", "source_package", "run_id", "model_id", "model_family", "model_version", "configuration_hash", "run_manifest_step", "kpi_identity", "estimand", "treatment_definition", "uncertainty_type", "effective_time_window", "governing_policy_reference", "technical_detail")
    @classmethod
    def _safe_text(cls, value: str | None, info: Any) -> str | None:
        return None if value is None else _safe(value, info.field_name)

    @field_validator("geo_scope", "market_scope", "segment_scope", "channel_scope", "diagnostic_references", "validation_references", "governance_references")
    @classmethod
    def _safe_list(cls, values: list[str], info: Any) -> list[str]:
        return [_safe(value, info.field_name) for value in values]

    @field_validator("point_estimate", "uncertainty_value")
    @classmethod
    def _finite(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("numeric evidence must be finite")
        return value

    @field_validator("observed_at")
    @classmethod
    def _aware(cls, value: datetime | None) -> datetime | None:
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("observed_at must be timezone-aware")
        return value

    @model_validator(mode="after")
    def _consistent(self) -> MMMCalibrationTreatmentRecord:
        roles = set(self.application_roles)
        fit_roles = {MMMCalibrationApplicationRole.PRIOR, MMMCalibrationApplicationRole.LIKELIHOOD, MMMCalibrationApplicationRole.CONSTRAINT}
        if MMMCalibrationApplicationRole.NONE in roles and len(roles) != 1:
            raise ValueError("NONE cannot be combined with an application role")
        if self.disposition == MMMCalibrationTreatmentDisposition.APPLIED and roles == {MMMCalibrationApplicationRole.NONE}:
            raise ValueError("applied treatment requires a non-NONE application role")
        if self.disposition in {MMMCalibrationTreatmentDisposition.REJECTED, MMMCalibrationTreatmentDisposition.UNUSED, MMMCalibrationTreatmentDisposition.BLOCKED} and roles & fit_roles:
            raise ValueError("rejected, unused, and blocked treatment cannot claim a fit-affecting role")
        if self.freshness_status == MMMCalibrationFreshnessStatus.EXPIRED and self.disposition == MMMCalibrationTreatmentDisposition.APPLIED:
            raise ValueError("expired signals cannot be applied")
        if self.compatibility_status == MMMCalibrationCompatibilityStatus.INCOMPATIBLE and self.disposition == MMMCalibrationTreatmentDisposition.APPLIED:
            raise ValueError("incompatible signals cannot be applied")
        if self.compatibility_status == MMMCalibrationCompatibilityStatus.UNKNOWN and self.disposition == MMMCalibrationTreatmentDisposition.APPLIED and not self.governing_policy_reference:
            raise ValueError("unknown compatibility requires a governing policy before application")
        if self.freshness_status == MMMCalibrationFreshnessStatus.STALE and self.disposition == MMMCalibrationTreatmentDisposition.APPLIED:
            if not self.governing_policy_reference or self.applied_weight is None:
                raise ValueError("stale applied treatment requires governing policy and downweight evidence")
        if roles & {MMMCalibrationApplicationRole.PRIOR, MMMCalibrationApplicationRole.LIKELIHOOD} and not self.research_only:
            raise ValueError("prior or likelihood lineage requires explicit research_only evidence")
        if self.failure_packet is not None:
            if self.failure_packet.code == MMMFailureCode.CALIBRATION_SCOPE_MISMATCH and self.compatibility_status != MMMCalibrationCompatibilityStatus.INCOMPATIBLE:
                raise ValueError("scope mismatch failure requires incompatible treatment")
            if self.failure_packet.code == MMMFailureCode.CALIBRATION_SIGNAL_EXPIRED and self.freshness_status != MMMCalibrationFreshnessStatus.EXPIRED:
                raise ValueError("expired failure requires expired treatment")
        if [step.sequence for step in self.transformations] != list(range(len(self.transformations))):
            raise ValueError("transformations must have contiguous deterministic sequence values")
        return self


class MMMCalibrationTreatmentLineage(BaseModel):
    """Versioned aggregation of every calibration signal considered by one MMM run."""

    model_config = ConfigDict(extra="forbid")
    schema_version: Literal[MMM_CALIBRATION_TREATMENT_LINEAGE_SCHEMA_VERSION] = MMM_CALIBRATION_TREATMENT_LINEAGE_SCHEMA_VERSION
    lineage_id: str = Field(min_length=1, max_length=200)
    run_id: str = Field(min_length=1, max_length=200)
    created_at: datetime
    producer_package_name: str = Field(default="mmm", min_length=1, max_length=100)
    producer_package_version: str = Field(min_length=1, max_length=100)
    run_manifest_id: str | None = Field(default=None, max_length=200)
    export_artifact: MMMArtifactReference | None = None
    records: list[MMMCalibrationTreatmentRecord] = Field(default_factory=list)
    summary_counts: dict[str, dict[str, int]] | None = None

    @field_validator("lineage_id", "run_id", "producer_package_name", "producer_package_version", "run_manifest_id")
    @classmethod
    def _text(cls, value: str | None, info: Any) -> str | None:
        return None if value is None else _safe(value, info.field_name)

    @field_validator("created_at")
    @classmethod
    def _created_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at must be timezone-aware")
        return value

    @model_validator(mode="after")
    def _derived_summary(self) -> MMMCalibrationTreatmentLineage:
        if [record.sequence for record in self.records] != list(range(len(self.records))):
            raise ValueError("records must have contiguous deterministic sequence values")
        if any(record.run_id is not None and record.run_id != self.run_id for record in self.records):
            raise ValueError("record run IDs must match lineage run ID when supplied")
        summary = {
            "disposition": _counts(record.disposition.value for record in self.records),
            "freshness": _counts(record.freshness_status.value for record in self.records),
            "compatibility": _counts(record.compatibility_status.value for record in self.records),
            "application_role": _counts(role.value for record in self.records for role in record.application_roles),
        }
        if self.summary_counts is not None and self.summary_counts != summary:
            raise ValueError("summary_counts must be derived exactly from treatment records")
        self.summary_counts = summary
        return self

    def to_json_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=True)

    def to_json(self) -> str:
        return json.dumps(self.to_json_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, payload: str) -> MMMCalibrationTreatmentLineage:
        return cls.model_validate_json(payload)


def _counts(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def build_mmm_calibration_treatment_lineage(**fields: Any) -> MMMCalibrationTreatmentLineage:
    """Build lineage only from explicit, producer-governed treatment evidence."""
    return MMMCalibrationTreatmentLineage(**fields)


__all__ = [
    "MMM_CALIBRATION_TREATMENT_LINEAGE_SCHEMA_VERSION",
    "MMMCalibrationApplicationRole",
    "MMMCalibrationCompatibilityReason",
    "MMMCalibrationCompatibilityStatus",
    "MMMCalibrationFreshnessStatus",
    "MMMCalibrationTransformationStep",
    "MMMCalibrationTreatmentDisposition",
    "MMMCalibrationTreatmentLineage",
    "MMMCalibrationTreatmentRecord",
    "build_mmm_calibration_treatment_lineage",
]
