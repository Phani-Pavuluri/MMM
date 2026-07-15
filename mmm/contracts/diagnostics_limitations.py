"""Typed producer diagnostics, limitations, and technical-claim evidence."""

from __future__ import annotations

import json
from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from mmm.contracts.mip_failure import MMMFailurePacket

MMM_DIAGNOSTICS_LIMITATIONS_SCHEMA_VERSION = "mmm_diagnostics_limitations_v1"


class MMMDiagnosticStatus(str, Enum):
    PASSED = "PASSED"
    WARNING = "WARNING"
    FAILED = "FAILED"
    UNAVAILABLE = "UNAVAILABLE"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class MMMDiagnosticSeverity(str, Enum):
    INFO = "INFO"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class MMMDiagnosticCategory(str, Enum):
    DATA_SUFFICIENCY = "DATA_SUFFICIENCY"
    DATA_QUALITY = "DATA_QUALITY"
    SPEND_VARIATION = "SPEND_VARIATION"
    CONTROL_COVERAGE = "CONTROL_COVERAGE"
    CALIBRATION = "CALIBRATION"
    MODEL_FIT = "MODEL_FIT"
    MODEL_STABILITY = "MODEL_STABILITY"
    HOLDOUT_VALIDATION = "HOLDOUT_VALIDATION"
    IDENTIFIABILITY = "IDENTIFIABILITY"
    EXTRAPOLATION = "EXTRAPOLATION"
    UNCERTAINTY = "UNCERTAINTY"
    PROMOTION = "PROMOTION"
    EXPORT_INTEGRITY = "EXPORT_INTEGRITY"


class MMMTechnicalClaim(str, Enum):
    MODEL_FIT = "MODEL_FIT"
    COEFFICIENT_INTERPRETATION = "COEFFICIENT_INTERPRETATION"
    CONTRIBUTION = "CONTRIBUTION"
    ELASTICITY = "ELASTICITY"
    ROI_ROAS = "ROI_ROAS"
    RESPONSE_CURVE = "RESPONSE_CURVE"
    IN_RANGE_SIMULATION = "IN_RANGE_SIMULATION"
    EXTRAPOLATIVE_SIMULATION = "EXTRAPOLATIVE_SIMULATION"
    CANDIDATE_PLAN_COMPARISON = "CANDIDATE_PLAN_COMPARISON"
    BUDGET_OPTIMIZATION_INPUT = "BUDGET_OPTIMIZATION_INPUT"
    CAUSAL_LIFT = "CAUSAL_LIFT"
    PRODUCTION_USE = "PRODUCTION_USE"


class MMMTechnicalClaimDisposition(str, Enum):
    SUPPORTED = "SUPPORTED"
    SUPPORTED_WITH_WARNING = "SUPPORTED_WITH_WARNING"
    RESTRICTED = "RESTRICTED"
    BLOCKED = "BLOCKED"
    UNAVAILABLE = "UNAVAILABLE"
    NOT_APPLICABLE = "NOT_APPLICABLE"


def _text(value: str, name: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError(f"{name} must be non-empty")
    if value.startswith(("/", "~")) or any(x in value.lower() for x in ("traceback", "stack trace", "secret=", "password=", "token=")):
        raise ValueError(f"{name} must not contain paths, stack traces, or secrets")
    return value


class MMMAffectedScope(BaseModel):
    model_config = ConfigDict(extra="forbid")
    run_id: str | None = None
    model_id: str | None = None
    model_family: str | None = None
    channels: list[str] = Field(default_factory=list)
    kpi: str | None = None
    markets: list[str] = Field(default_factory=list)
    segments: list[str] = Field(default_factory=list)
    time_window: str | None = None
    calibration_signal_ids: list[str] = Field(default_factory=list)
    candidate_plan_id: str | None = None
    export_artifact_id: str | None = None
    technical_claims: list[MMMTechnicalClaim] = Field(default_factory=list)

    @field_validator("run_id", "model_id", "model_family", "kpi", "time_window", "candidate_plan_id", "export_artifact_id")
    @classmethod
    def _optional_text(cls, value: str | None, info: Any) -> str | None:
        return None if value is None else _text(value, info.field_name)

    @field_validator("channels", "markets", "segments", "calibration_signal_ids")
    @classmethod
    def _text_list(cls, value: list[str], info: Any) -> list[str]:
        return [_text(item, info.field_name) for item in value]


class MMMClaimEffect(BaseModel):
    model_config = ConfigDict(extra="forbid")
    claim: MMMTechnicalClaim
    disposition: MMMTechnicalClaimDisposition
    evidence_references: list[str] = Field(default_factory=list)
    warning_references: list[str] = Field(default_factory=list)
    blocking_references: list[str] = Field(default_factory=list)
    restriction: str | None = None

    @field_validator("evidence_references", "warning_references", "blocking_references")
    @classmethod
    def _refs(cls, value: list[str], info: Any) -> list[str]:
        return [_text(item, info.field_name) for item in value]

    @field_validator("restriction")
    @classmethod
    def _restriction(cls, value: str | None) -> str | None:
        return None if value is None else _text(value, "restriction")

    @model_validator(mode="after")
    def _effects_are_evidenced(self) -> MMMClaimEffect:
        if self.disposition == MMMTechnicalClaimDisposition.SUPPORTED and not self.evidence_references:
            raise ValueError("supported claims require affirmative evidence")
        if self.disposition == MMMTechnicalClaimDisposition.SUPPORTED_WITH_WARNING and not self.warning_references:
            raise ValueError("supported-with-warning claims require warning references")
        if self.disposition == MMMTechnicalClaimDisposition.RESTRICTED and not self.restriction:
            raise ValueError("restricted claims require a restriction")
        if self.disposition == MMMTechnicalClaimDisposition.BLOCKED and not self.blocking_references:
            raise ValueError("blocked claims require blocking references")
        return self


class MMMDiagnosticRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sequence: int = Field(ge=0)
    diagnostic_id: str
    diagnostic_code: str
    category: MMMDiagnosticCategory
    contract_version: str
    producer_component: str
    created_at: datetime
    status: MMMDiagnosticStatus
    severity: MMMDiagnosticSeverity
    technical_summary: str
    affected_scope: MMMAffectedScope = Field(default_factory=MMMAffectedScope)
    policy_reference: str | None = None
    observed_reference: str | None = None
    validation_result_ids: list[str] = Field(default_factory=list)
    evidence_references: list[str] = Field(default_factory=list)
    calibration_lineage_id: str | None = None
    run_manifest_step: str | None = None
    failure_packet: MMMFailurePacket | None = None
    claim_effects: list[MMMClaimEffect] = Field(default_factory=list)

    @field_validator("diagnostic_id", "diagnostic_code", "contract_version", "producer_component", "technical_summary", "policy_reference", "observed_reference", "calibration_lineage_id", "run_manifest_step")
    @classmethod
    def _fields(cls, value: str | None, info: Any) -> str | None:
        return None if value is None else _text(value, info.field_name)

    @field_validator("validation_result_ids", "evidence_references")
    @classmethod
    def _refs(cls, value: list[str], info: Any) -> list[str]:
        return [_text(item, info.field_name) for item in value]

    @field_validator("created_at")
    @classmethod
    def _time(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at must be timezone-aware")
        return value

    @model_validator(mode="after")
    def _status_consistency(self) -> MMMDiagnosticRecord:
        if self.status == MMMDiagnosticStatus.PASSED and self.failure_packet is not None:
            raise ValueError("passed diagnostics cannot carry a failure packet")
        if self.status == MMMDiagnosticStatus.UNAVAILABLE and any(effect.disposition == MMMTechnicalClaimDisposition.SUPPORTED for effect in self.claim_effects):
            raise ValueError("unavailable diagnostics cannot support claims")
        if self.status == MMMDiagnosticStatus.NOT_APPLICABLE and any(effect.disposition == MMMTechnicalClaimDisposition.BLOCKED for effect in self.claim_effects):
            raise ValueError("not-applicable diagnostics cannot block claims")
        return self


class MMMLimitationRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sequence: int = Field(ge=0)
    limitation_id: str
    limitation_code: str
    category: MMMDiagnosticCategory
    severity: MMMDiagnosticSeverity
    technical_summary: str
    affected_scope: MMMAffectedScope = Field(default_factory=MMMAffectedScope)
    originating_diagnostic_ids: list[str] = Field(default_factory=list)
    claim_effects: list[MMMClaimEffect] = Field(default_factory=list)
    required_condition: str | None = None
    failure_packet: MMMFailurePacket | None = None
    run_manifest_id: str | None = None
    calibration_lineage_id: str | None = None
    research_only: bool = False

    @field_validator("limitation_id", "limitation_code", "technical_summary", "required_condition", "run_manifest_id", "calibration_lineage_id")
    @classmethod
    def _fields(cls, value: str | None, info: Any) -> str | None:
        return None if value is None else _text(value, info.field_name)

    @field_validator("originating_diagnostic_ids")
    @classmethod
    def _diagnostics(cls, value: list[str]) -> list[str]:
        return [_text(item, "originating_diagnostic_ids") for item in value]

    @model_validator(mode="after")
    def _production_guard(self) -> MMMLimitationRecord:
        if self.research_only:
            for effect in self.claim_effects:
                if effect.claim == MMMTechnicalClaim.PRODUCTION_USE and effect.disposition in {MMMTechnicalClaimDisposition.SUPPORTED, MMMTechnicalClaimDisposition.SUPPORTED_WITH_WARNING}:
                    raise ValueError("research-only limitations cannot imply production-supported claims")
        return self


class MMMDiagnosticsLimitations(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal[MMM_DIAGNOSTICS_LIMITATIONS_SCHEMA_VERSION] = MMM_DIAGNOSTICS_LIMITATIONS_SCHEMA_VERSION
    aggregate_id: str
    run_id: str
    created_at: datetime
    producer_package_name: str = "mmm"
    producer_package_version: str
    diagnostics: list[MMMDiagnosticRecord] = Field(default_factory=list)
    limitations: list[MMMLimitationRecord] = Field(default_factory=list)
    run_manifest_id: str | None = None
    export_artifact_id: str | None = None
    calibration_lineage_id: str | None = None
    terminal_failure: MMMFailurePacket | None = None
    claim_dispositions: dict[str, str] | None = None

    @field_validator("aggregate_id", "run_id", "producer_package_name", "producer_package_version", "run_manifest_id", "export_artifact_id", "calibration_lineage_id")
    @classmethod
    def _fields(cls, value: str | None, info: Any) -> str | None:
        return None if value is None else _text(value, info.field_name)

    @field_validator("created_at")
    @classmethod
    def _time(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at must be timezone-aware")
        return value

    @model_validator(mode="after")
    def _derived_claims(self) -> MMMDiagnosticsLimitations:
        if [item.sequence for item in self.diagnostics] != list(range(len(self.diagnostics))):
            raise ValueError("diagnostics must have contiguous deterministic sequence values")
        if [item.sequence for item in self.limitations] != list(range(len(self.limitations))):
            raise ValueError("limitations must have contiguous deterministic sequence values")
        effects = [effect for record in self.diagnostics for effect in record.claim_effects] + [effect for record in self.limitations for effect in record.claim_effects]
        derived = {effect.claim.value: effect.disposition.value for effect in effects}
        if self.claim_dispositions is not None and self.claim_dispositions != derived:
            raise ValueError("claim_dispositions must be derived exactly from records")
        self.claim_dispositions = dict(sorted(derived.items()))
        return self

    def to_json(self) -> str:
        return json.dumps(self.model_dump(mode="json", exclude_none=True), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, payload: str) -> MMMDiagnosticsLimitations:
        return cls.model_validate_json(payload)


def build_mmm_diagnostics_limitations(**fields: Any) -> MMMDiagnosticsLimitations:
    return MMMDiagnosticsLimitations(**fields)
