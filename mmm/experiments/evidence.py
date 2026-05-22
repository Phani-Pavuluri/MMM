"""First-class experiment evidence contract (Phase 1 — storage/validation only)."""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


class ExperimentType(StrEnum):
    GEOX = "geox"
    HOLDOUT = "holdout"
    INCREMENTALITY = "incrementality"
    AB = "ab"
    SWITCHBACK = "switchback"
    SYNTHETIC_CONTROL = "synthetic_control"
    OTHER = "other"


class GeoGranularity(StrEnum):
    """Spatial claim granularity (coarse → fine for compatibility)."""

    NATIONAL = "national"
    REGION = "region"
    DMA = "dma"
    GEO = "geo"
    USER = "user"


class ApprovalStatus(StrEnum):
    DRAFT = "draft"
    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    EXPIRED = "expired"


class TimeWindow(BaseModel):
    """Inclusive experiment evaluation window (ISO week/date strings)."""

    start: str
    end: str


class ConfidenceInterval(BaseModel):
    lower: float
    upper: float
    level: float = 0.95


class ExperimentEvidence(BaseModel):
    """
    Canonical experiment evidence payload for registry + compatibility (not replay execution).

    The registry stores and validates evidence; it does **not** run experiments or auto-calibrate.
    """

    experiment_id: str
    experiment_type: ExperimentType
    channel: str
    kpi: str
    estimand: str
    lift_estimate: float
    standard_error: float | None = None
    confidence_interval: ConfidenceInterval | None = None
    spend_delta: float | None = None
    exposure_delta: float | None = None
    time_window: TimeWindow
    geo_scope: list[str] = Field(default_factory=list)
    geo_granularity: GeoGranularity
    population_scope: str = ""
    randomization_unit: str = ""
    treatment_definition: str = ""
    control_definition: str = ""
    campaign: str | None = None
    feature_group: str | None = None
    quality_flags: list[str] = Field(default_factory=list)
    approval_status: ApprovalStatus = ApprovalStatus.DRAFT
    signature: str | None = None
    lineage: dict[str, Any] = Field(default_factory=dict)
    source_system: str = ""
    freshness_date: date | str
    #: Optional metadata for power/MDE, contamination, etc.
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "forbid"}

    @field_validator("freshness_date", mode="before")
    @classmethod
    def _coerce_freshness(cls, v: object) -> date | str:
        if isinstance(v, date):
            return v
        if isinstance(v, datetime):
            return v.date()
        return v

    @model_validator(mode="after")
    def _lift_finite(self) -> ExperimentEvidence:
        if self.lift_estimate != self.lift_estimate:
            raise ValueError("lift_estimate must be finite")
        if self.standard_error is not None and (
            self.standard_error != self.standard_error or self.standard_error <= 0
        ):
            raise ValueError("standard_error must be positive and finite when set")
        return self

    def freshness_as_date(self) -> date:
        fd = self.freshness_date
        if isinstance(fd, date):
            return fd
        return date.fromisoformat(str(fd)[:10])

    def to_registry_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


# Required fields for registry registration (fail-closed validation).
REGISTRY_REQUIRED_FIELDS: tuple[str, ...] = (
    "experiment_id",
    "experiment_type",
    "channel",
    "kpi",
    "estimand",
    "lift_estimate",
    "time_window",
    "geo_granularity",
    "source_system",
    "freshness_date",
)


def validate_evidence_for_registry(evidence: ExperimentEvidence) -> list[str]:
    """Return violation codes; empty if registrable."""
    violations: list[str] = []
    if not str(evidence.experiment_id).strip():
        violations.append("missing_experiment_id")
    if not str(evidence.channel).strip():
        violations.append("missing_channel")
    if not str(evidence.kpi).strip():
        violations.append("missing_kpi")
    if not str(evidence.estimand).strip():
        violations.append("missing_estimand")
    if not str(evidence.source_system).strip():
        violations.append("missing_source_system")
    if not evidence.time_window.start or not evidence.time_window.end:
        violations.append("incomplete_time_window")
    if evidence.approval_status == ApprovalStatus.REJECTED:
        violations.append("approval_rejected")
    return violations
