"""Versioned, producer-owned supported-range evidence for the MMM boundary.

This module records existing range evidence.  It deliberately does not derive
new statistical support, widen observed domains, or authorize simulation,
response-surface, recommendation, or optimization behaviour.
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from mmm.contracts.diagnostics_limitations import (
    MMMAffectedScope,
    MMMClaimEffect,
    MMMTechnicalClaim,
    MMMTechnicalClaimDisposition,
)

MMM_SUPPORTED_RANGE_SCHEMA_VERSION = "mmm_supported_range_evidence_v1"
MMM_SUPPORTED_RANGE_RECORD_SCHEMA_VERSION = "mmm_supported_range_record_v1"


class MMMRangeEvidenceBasis(str, Enum):
    OBSERVED_DATA = "OBSERVED_DATA"
    TRAINING_DOMAIN = "TRAINING_DOMAIN"
    HOLDOUT_VALIDATED = "HOLDOUT_VALIDATED"
    DIAGNOSTIC_VALIDATED = "DIAGNOSTIC_VALIDATED"
    CALIBRATION_RESTRICTED = "CALIBRATION_RESTRICTED"
    MODEL_STRUCTURAL = "MODEL_STRUCTURAL"
    GOVERNANCE_RESTRICTED = "GOVERNANCE_RESTRICTED"
    RESEARCH_ONLY = "RESEARCH_ONLY"
    UNKNOWN = "UNKNOWN"


class MMMRangeAvailabilityStatus(str, Enum):
    AVAILABLE = "AVAILABLE"
    PARTIALLY_AVAILABLE = "PARTIALLY_AVAILABLE"
    UNAVAILABLE = "UNAVAILABLE"
    BLOCKED = "BLOCKED"
    RESEARCH_ONLY = "RESEARCH_ONLY"


class MMMRangeRelation(str, Enum):
    WITHIN_OBSERVED_RANGE = "WITHIN_OBSERVED_RANGE"
    WITHIN_SUPPORTED_RANGE = "WITHIN_SUPPORTED_RANGE"
    AT_LOWER_BOUNDARY = "AT_LOWER_BOUNDARY"
    AT_UPPER_BOUNDARY = "AT_UPPER_BOUNDARY"
    OUTSIDE_OBSERVED_BUT_SUPPORTED = "OUTSIDE_OBSERVED_BUT_SUPPORTED"
    OUTSIDE_SUPPORTED_RANGE = "OUTSIDE_SUPPORTED_RANGE"
    UNKNOWN = "UNKNOWN"


class MMMExtrapolationClassification(str, Enum):
    NONE = "NONE"
    INTERPOLATION = "INTERPOLATION"
    BOUNDARY = "BOUNDARY"
    LIMITED_EXTRAPOLATION = "LIMITED_EXTRAPOLATION"
    UNSUPPORTED_EXTRAPOLATION = "UNSUPPORTED_EXTRAPOLATION"
    UNKNOWN = "UNKNOWN"


class MMMRangeScale(str, Enum):
    RAW = "RAW"
    TRANSFORMED = "TRANSFORMED"


_UNSAFE = ("traceback", "stack trace", "password=", "secret=", "token=", "api_key=")


def _text(value: str, name: str) -> str:
    value = value.strip()
    if not value or value.startswith(("/", "~")) or any(marker in value.lower() for marker in _UNSAFE):
        raise ValueError(f"{name} must be non-empty safe technical text")
    return value


class MMMRangeBound(BaseModel):
    """One explicit numeric boundary on raw or identified transformed scale."""

    model_config = ConfigDict(extra="forbid")

    value: float
    inclusive: bool = True
    unit: str
    scale: MMMRangeScale = MMMRangeScale.RAW
    transformation_id: str | None = None

    @field_validator("value")
    @classmethod
    def _finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("range bound value must be finite")
        return value

    @field_validator("unit", "transformation_id")
    @classmethod
    def _safe_text(cls, value: str | None, info: Any) -> str | None:
        return None if value is None else _text(value, info.field_name)

    @model_validator(mode="after")
    def _scale_is_unambiguous(self) -> MMMRangeBound:
        if self.scale == MMMRangeScale.TRANSFORMED and not self.transformation_id:
            raise ValueError("transformed range bounds require transformation_id")
        if self.scale == MMMRangeScale.RAW and self.transformation_id:
            raise ValueError("raw range bounds cannot carry a transformation_id")
        return self


class MMMRangeScope(BaseModel):
    """Typed scope preventing range evidence from being applied elsewhere."""

    model_config = ConfigDict(extra="forbid")

    channel: str | None = None
    kpi: str | None = None
    geography: str | None = None
    segment: str | None = None
    time_window: str | None = None
    outcome_or_estimand: str | None = None
    data_grain: str | None = None
    transformation_id: str | None = None

    @field_validator("channel", "kpi", "geography", "segment", "time_window", "outcome_or_estimand", "data_grain", "transformation_id")
    @classmethod
    def _text_values(cls, value: str | None, info: Any) -> str | None:
        return None if value is None else _text(value, info.field_name)

    def is_explicit(self) -> bool:
        return any((self.channel, self.kpi, self.geography, self.segment, self.time_window, self.outcome_or_estimand, self.data_grain))


class MMMRangeRestriction(BaseModel):
    """A machine-readable technical restriction; never user-facing advice."""

    model_config = ConfigDict(extra="forbid")

    restriction_code: str
    technical_summary: str
    affected_scope: MMMRangeScope = Field(default_factory=MMMRangeScope)
    evidence_references: list[str] = Field(default_factory=list)
    affected_technical_claims: list[MMMTechnicalClaim] = Field(default_factory=list)
    claim_effects: list[MMMClaimEffect] = Field(default_factory=list)
    required_condition: str | None = None

    @field_validator("restriction_code", "technical_summary", "required_condition")
    @classmethod
    def _safe_fields(cls, value: str | None, info: Any) -> str | None:
        return None if value is None else _text(value, info.field_name)

    @field_validator("evidence_references")
    @classmethod
    def _references(cls, values: list[str]) -> list[str]:
        return [_text(value, "evidence_references") for value in values]

    @model_validator(mode="after")
    def _effects_match_claims(self) -> MMMRangeRestriction:
        if not self.evidence_references:
            raise ValueError("range restrictions require evidence_references")
        if any(effect.claim not in self.affected_technical_claims for effect in self.claim_effects):
            raise ValueError("range restriction claim effects must be listed as affected claims")
        return self


def _pair_ok(lower: MMMRangeBound | None, upper: MMMRangeBound | None, label: str) -> None:
    if (lower is None) != (upper is None):
        raise ValueError(f"{label} lower and upper bounds must be supplied together")
    if lower is not None and upper is not None:
        if lower.value > upper.value:
            raise ValueError(f"{label} lower bound cannot exceed upper bound")
        if lower.unit != upper.unit or lower.scale != upper.scale or lower.transformation_id != upper.transformation_id:
            raise ValueError(f"{label} bounds must have compatible unit and scale semantics")


class MMMSupportedRangeRecord(BaseModel):
    """One immutable scoped evidence record, not an inferred support algorithm."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[MMM_SUPPORTED_RANGE_RECORD_SCHEMA_VERSION] = MMM_SUPPORTED_RANGE_RECORD_SCHEMA_VERSION
    range_record_id: str
    run_id: str
    model_id: str | None = None
    model_family: str | None = None
    model_version: str | None = None
    configuration_hash: str | None = None
    scope: MMMRangeScope
    observed_lower: MMMRangeBound | None = None
    observed_upper: MMMRangeBound | None = None
    supported_lower: MMMRangeBound | None = None
    supported_upper: MMMRangeBound | None = None
    validated_lower: MMMRangeBound | None = None
    validated_upper: MMMRangeBound | None = None
    evidence_basis: list[MMMRangeEvidenceBasis] = Field(default_factory=list)
    availability_status: MMMRangeAvailabilityStatus
    range_relation: MMMRangeRelation = MMMRangeRelation.UNKNOWN
    extrapolation_classification: MMMExtrapolationClassification = MMMExtrapolationClassification.UNKNOWN
    restrictions: list[MMMRangeRestriction] = Field(default_factory=list)
    diagnostic_references: list[str] = Field(default_factory=list)
    limitation_references: list[str] = Field(default_factory=list)
    calibration_lineage_references: list[str] = Field(default_factory=list)
    failure_packet_reference: str | None = None
    validation_evidence_references: list[str] = Field(default_factory=list)
    data_evidence_references: list[str] = Field(default_factory=list)
    governed_extrapolation_evidence: list[str] = Field(default_factory=list)
    uncertainty_available: bool = False
    uncertainty_artifact_reference: str | None = None
    uncertainty_semantics: str | None = None
    technical_detail: str | None = None

    @field_validator("range_record_id", "run_id", "model_id", "model_family", "model_version", "configuration_hash", "failure_packet_reference", "uncertainty_artifact_reference", "uncertainty_semantics", "technical_detail")
    @classmethod
    def _safe_optional_text(cls, value: str | None, info: Any) -> str | None:
        return None if value is None else _text(value, info.field_name)

    @field_validator("diagnostic_references", "limitation_references", "calibration_lineage_references", "validation_evidence_references", "data_evidence_references", "governed_extrapolation_evidence")
    @classmethod
    def _safe_references(cls, values: list[str], info: Any) -> list[str]:
        return [_text(value, info.field_name) for value in values]

    @model_validator(mode="after")
    def _consistent_evidence(self) -> MMMSupportedRangeRecord:
        _pair_ok(self.observed_lower, self.observed_upper, "observed")
        _pair_ok(self.supported_lower, self.supported_upper, "supported")
        _pair_ok(self.validated_lower, self.validated_upper, "validated")
        pairs = [(self.observed_lower, self.observed_upper), (self.supported_lower, self.supported_upper), (self.validated_lower, self.validated_upper)]
        bounds = [bound for pair in pairs for bound in pair if bound is not None]
        if len({(bound.unit, bound.scale, bound.transformation_id) for bound in bounds}) > 1:
            raise ValueError("observed, supported, and validated bounds must not mix units or scale semantics")
        supported = self.supported_lower is not None
        validated = self.validated_lower is not None
        if self.availability_status == MMMRangeAvailabilityStatus.AVAILABLE and (not supported or not self.scope.is_explicit()):
            raise ValueError("available range evidence requires supported bounds and explicit scope")
        if self.availability_status == MMMRangeAvailabilityStatus.PARTIALLY_AVAILABLE and not (self.observed_lower or self.data_evidence_references or self.validation_evidence_references):
            raise ValueError("partially available evidence must identify available evidence")
        if self.availability_status == MMMRangeAvailabilityStatus.UNAVAILABLE and supported:
            raise ValueError("unavailable range evidence cannot claim supported bounds")
        if self.availability_status == MMMRangeAvailabilityStatus.BLOCKED and not (self.diagnostic_references or self.limitation_references or self.failure_packet_reference):
            raise ValueError("blocked range evidence requires a blocking reference")
        if self.availability_status == MMMRangeAvailabilityStatus.RESEARCH_ONLY and MMMRangeEvidenceBasis.RESEARCH_ONLY not in self.evidence_basis:
            raise ValueError("research-only range evidence requires RESEARCH_ONLY basis")
        if validated and not self.validation_evidence_references:
            raise ValueError("validated ranges require validation evidence")
        if self.range_relation in {MMMRangeRelation.WITHIN_SUPPORTED_RANGE, MMMRangeRelation.AT_LOWER_BOUNDARY, MMMRangeRelation.AT_UPPER_BOUNDARY, MMMRangeRelation.OUTSIDE_OBSERVED_BUT_SUPPORTED} and not supported:
            raise ValueError("supported range relation requires explicit supported bounds")
        if self.extrapolation_classification == MMMExtrapolationClassification.INTERPOLATION and not supported:
            raise ValueError("interpolation requires explicit supported bounds")
        if self.extrapolation_classification == MMMExtrapolationClassification.BOUNDARY and self.range_relation not in {MMMRangeRelation.AT_LOWER_BOUNDARY, MMMRangeRelation.AT_UPPER_BOUNDARY}:
            raise ValueError("boundary extrapolation classification requires a boundary relation")
        if self.extrapolation_classification == MMMExtrapolationClassification.LIMITED_EXTRAPOLATION and not self.governed_extrapolation_evidence:
            raise ValueError("limited extrapolation requires explicit governed evidence")
        if self.extrapolation_classification == MMMExtrapolationClassification.UNSUPPORTED_EXTRAPOLATION:
            if not any(effect.disposition in {MMMTechnicalClaimDisposition.BLOCKED, MMMTechnicalClaimDisposition.RESTRICTED} for restriction in self.restrictions for effect in restriction.claim_effects):
                raise ValueError("unsupported extrapolation requires a restriction or blocked claim")
        if self.uncertainty_available and not self.uncertainty_artifact_reference:
            raise ValueError("available uncertainty requires an artifact reference")
        if not self.uncertainty_available and (self.uncertainty_artifact_reference or self.uncertainty_semantics):
            raise ValueError("unavailable uncertainty cannot claim an artifact or semantics")
        if self.availability_status == MMMRangeAvailabilityStatus.RESEARCH_ONLY:
            for restriction in self.restrictions:
                for effect in restriction.claim_effects:
                    if effect.claim == MMMTechnicalClaim.PRODUCTION_USE and effect.disposition in {MMMTechnicalClaimDisposition.SUPPORTED, MMMTechnicalClaimDisposition.SUPPORTED_WITH_WARNING}:
                        raise ValueError("research-only range evidence cannot support production use")
        return self


class MMMSupportedRangeEvidence(BaseModel):
    """Deterministic aggregate of scoped technical supported-range evidence."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[MMM_SUPPORTED_RANGE_SCHEMA_VERSION] = MMM_SUPPORTED_RANGE_SCHEMA_VERSION
    evidence_id: str
    run_id: str
    created_at: datetime
    producer_package_name: str = "mmm"
    producer_package_version: str
    records: list[MMMSupportedRangeRecord]
    summary_counts: dict[str, int] | None = None
    run_manifest_reference: str | None = None
    diagnostics_limitations_reference: str | None = None
    calibration_lineage_reference: str | None = None
    export_artifact_reference: str | None = None
    terminal_failure_reference: str | None = None

    @field_validator("evidence_id", "run_id", "producer_package_name", "producer_package_version", "run_manifest_reference", "diagnostics_limitations_reference", "calibration_lineage_reference", "export_artifact_reference", "terminal_failure_reference")
    @classmethod
    def _safe_aggregate_text(cls, value: str | None, info: Any) -> str | None:
        return None if value is None else _text(value, info.field_name)

    @field_validator("created_at")
    @classmethod
    def _time(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at must be timezone-aware")
        return value

    @model_validator(mode="after")
    def _aggregate_consistency(self) -> MMMSupportedRangeEvidence:
        if not self.records:
            raise ValueError("supported range evidence requires at least one record")
        if [record.range_record_id for record in self.records] != sorted(record.range_record_id for record in self.records):
            raise ValueError("range records must be deterministically ordered by range_record_id")
        if len({record.range_record_id for record in self.records}) != len(self.records):
            raise ValueError("range record IDs must be unique")
        if any(record.run_id != self.run_id for record in self.records):
            raise ValueError("range record run IDs must match aggregate run ID")
        counts = {
            "available": sum(record.availability_status == MMMRangeAvailabilityStatus.AVAILABLE for record in self.records),
            "partially_available": sum(record.availability_status == MMMRangeAvailabilityStatus.PARTIALLY_AVAILABLE for record in self.records),
            "unavailable": sum(record.availability_status == MMMRangeAvailabilityStatus.UNAVAILABLE for record in self.records),
            "blocked": sum(record.availability_status == MMMRangeAvailabilityStatus.BLOCKED for record in self.records),
            "research_only": sum(record.availability_status == MMMRangeAvailabilityStatus.RESEARCH_ONLY for record in self.records),
            "supported": sum(record.supported_lower is not None for record in self.records),
            "restricted": sum(bool(record.restrictions) for record in self.records),
            "unsupported_extrapolation": sum(record.extrapolation_classification == MMMExtrapolationClassification.UNSUPPORTED_EXTRAPOLATION for record in self.records),
        }
        if self.summary_counts is not None and self.summary_counts != counts:
            raise ValueError("summary_counts must be derived exactly from records")
        self.summary_counts = counts
        return self

    def to_json_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=True)

    def to_json(self) -> str:
        return json.dumps(self.to_json_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, payload: str) -> MMMSupportedRangeEvidence:
        return cls.model_validate_json(payload)


def build_mmm_supported_range_evidence(**fields: Any) -> MMMSupportedRangeEvidence:
    """Build public evidence without inferring a missing support methodology."""
    return MMMSupportedRangeEvidence(**fields)


__all__ = [
    "MMM_SUPPORTED_RANGE_RECORD_SCHEMA_VERSION", "MMM_SUPPORTED_RANGE_SCHEMA_VERSION",
    "MMMExtrapolationClassification", "MMMRangeAvailabilityStatus", "MMMRangeBound",
    "MMMRangeEvidenceBasis", "MMMRangeRelation", "MMMRangeRestriction", "MMMRangeScale",
    "MMMRangeScope", "MMMSupportedRangeEvidence", "MMMSupportedRangeRecord",
    "build_mmm_supported_range_evidence",
]
