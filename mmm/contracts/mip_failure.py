"""Typed producer failures for the MMM-to-MIP handoff.

This contract carries technical failure evidence only.  It intentionally does
not parse platform inputs, route user intent, or produce conversational text.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from mmm.contracts.mip_export import MMMExportBundle, PromotionStatus, parse_export_artifact, validate_mmm_export_bundle

MMM_FAILURE_SCHEMA_VERSION = "mmm_mip_failure_v1"


class MMMFailureCode(str, Enum):
    INSUFFICIENT_HISTORY = "INSUFFICIENT_HISTORY"
    INCOMPATIBLE_GRAIN = "INCOMPATIBLE_GRAIN"
    KPI_NOT_SUPPORTED = "KPI_NOT_SUPPORTED"
    SPEND_VARIATION_INSUFFICIENT = "SPEND_VARIATION_INSUFFICIENT"
    CONTROL_DATA_MISSING = "CONTROL_DATA_MISSING"
    CALIBRATION_SCOPE_MISMATCH = "CALIBRATION_SCOPE_MISMATCH"
    CALIBRATION_SIGNAL_EXPIRED = "CALIBRATION_SIGNAL_EXPIRED"
    MODEL_INSTABILITY = "MODEL_INSTABILITY"
    HOLDOUT_FAILURE = "HOLDOUT_FAILURE"
    UNSUPPORTED_EXTRAPOLATION = "UNSUPPORTED_EXTRAPOLATION"
    IDENTIFIABILITY_FAILURE = "IDENTIFIABILITY_FAILURE"
    MODEL_NOT_PROMOTED = "MODEL_NOT_PROMOTED"


class MMMFailureStage(str, Enum):
    DATA_INTAKE = "DATA_INTAKE"
    DATA_VALIDATION = "DATA_VALIDATION"
    CALIBRATION = "CALIBRATION"
    MODEL_FIT = "MODEL_FIT"
    MODEL_VALIDATION = "MODEL_VALIDATION"
    PROMOTION_GATE = "PROMOTION_GATE"
    SIMULATION = "SIMULATION"
    EXPORT = "EXPORT"


class MMMRetryDisposition(str, Enum):
    NOT_RETRYABLE = "NOT_RETRYABLE"
    RETRY_AFTER_INPUT_CHANGE = "RETRY_AFTER_INPUT_CHANGE"
    RETRY_AFTER_CONFIGURATION_CHANGE = "RETRY_AFTER_CONFIGURATION_CHANGE"
    RETRY_AFTER_CALIBRATION_CHANGE = "RETRY_AFTER_CALIBRATION_CHANGE"
    RETRY_AFTER_MODEL_OR_METHOD_CHANGE = "RETRY_AFTER_MODEL_OR_METHOD_CHANGE"
    RETRY_AFTER_PLAN_CHANGE = "RETRY_AFTER_PLAN_CHANGE"
    RETRY_AFTER_GOVERNANCE_CHANGE = "RETRY_AFTER_GOVERNANCE_CHANGE"
    RETRY_AFTER_DEPENDENCY_RECOVERY = "RETRY_AFTER_DEPENDENCY_RECOVERY"


class MMMRemediationActionCode(str, Enum):
    INPUT_DATA = "INPUT_DATA"
    CONFIGURATION = "CONFIGURATION"
    CALIBRATION = "CALIBRATION"
    MODEL_OR_METHOD = "MODEL_OR_METHOD"
    CANDIDATE_PLAN = "CANDIDATE_PLAN"
    GOVERNANCE = "GOVERNANCE"
    DEPENDENCY = "DEPENDENCY"


class MMMRemediationAction(BaseModel):
    """One machine-readable technical requirement for a later producer retry."""

    model_config = ConfigDict(extra="forbid")

    action_code: MMMRemediationActionCode
    affected_resource: str | None = Field(default=None, max_length=200)
    required: bool = True
    technical_detail: str = Field(min_length=1, max_length=500)
    evidence_references: list[str] = Field(default_factory=list)

    @field_validator("affected_resource", "technical_detail")
    @classmethod
    def _strip_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("text fields must be non-empty when supplied")
        return cleaned


DEFAULT_FAILURE_POLICY: dict[
    MMMFailureCode, tuple[MMMRetryDisposition, MMMRemediationActionCode]
] = {
    MMMFailureCode.INSUFFICIENT_HISTORY: (
        MMMRetryDisposition.RETRY_AFTER_INPUT_CHANGE,
        MMMRemediationActionCode.INPUT_DATA,
    ),
    MMMFailureCode.INCOMPATIBLE_GRAIN: (
        MMMRetryDisposition.RETRY_AFTER_CONFIGURATION_CHANGE,
        MMMRemediationActionCode.CONFIGURATION,
    ),
    MMMFailureCode.KPI_NOT_SUPPORTED: (
        MMMRetryDisposition.RETRY_AFTER_MODEL_OR_METHOD_CHANGE,
        MMMRemediationActionCode.MODEL_OR_METHOD,
    ),
    MMMFailureCode.SPEND_VARIATION_INSUFFICIENT: (
        MMMRetryDisposition.RETRY_AFTER_INPUT_CHANGE,
        MMMRemediationActionCode.INPUT_DATA,
    ),
    MMMFailureCode.CONTROL_DATA_MISSING: (
        MMMRetryDisposition.RETRY_AFTER_INPUT_CHANGE,
        MMMRemediationActionCode.INPUT_DATA,
    ),
    MMMFailureCode.CALIBRATION_SCOPE_MISMATCH: (
        MMMRetryDisposition.RETRY_AFTER_CALIBRATION_CHANGE,
        MMMRemediationActionCode.CALIBRATION,
    ),
    MMMFailureCode.CALIBRATION_SIGNAL_EXPIRED: (
        MMMRetryDisposition.RETRY_AFTER_CALIBRATION_CHANGE,
        MMMRemediationActionCode.CALIBRATION,
    ),
    MMMFailureCode.MODEL_INSTABILITY: (
        MMMRetryDisposition.RETRY_AFTER_MODEL_OR_METHOD_CHANGE,
        MMMRemediationActionCode.MODEL_OR_METHOD,
    ),
    MMMFailureCode.HOLDOUT_FAILURE: (
        MMMRetryDisposition.RETRY_AFTER_CONFIGURATION_CHANGE,
        MMMRemediationActionCode.CONFIGURATION,
    ),
    MMMFailureCode.UNSUPPORTED_EXTRAPOLATION: (
        MMMRetryDisposition.RETRY_AFTER_PLAN_CHANGE,
        MMMRemediationActionCode.CANDIDATE_PLAN,
    ),
    MMMFailureCode.IDENTIFIABILITY_FAILURE: (
        MMMRetryDisposition.RETRY_AFTER_MODEL_OR_METHOD_CHANGE,
        MMMRemediationActionCode.MODEL_OR_METHOD,
    ),
    MMMFailureCode.MODEL_NOT_PROMOTED: (
        MMMRetryDisposition.RETRY_AFTER_GOVERNANCE_CHANGE,
        MMMRemediationActionCode.GOVERNANCE,
    ),
}

_FORBIDDEN_CONTEXT_KEY_PARTS = (
    "exception",
    "stack",
    "traceback",
    "dataframe",
    "model_object",
    "pickle",
    "secret",
    "password",
    "token",
    "path",
    "filename",
)


def _validate_safe_json(value: Any, *, location: str = "technical_context") -> None:
    if value is None or isinstance(value, (bool, str, int)):
        if isinstance(value, str) and (value.startswith("/") or value.startswith("~")):
            raise ValueError(f"{location} must not contain filesystem paths")
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{location} must contain finite JSON numbers")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_safe_json(item, location=f"{location}[{index}]")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{location} keys must be strings")
            normalized = key.lower()
            if any(part in normalized for part in _FORBIDDEN_CONTEXT_KEY_PARTS):
                raise ValueError(f"{location} contains forbidden key {key!r}")
            _validate_safe_json(item, location=f"{location}.{key}")
        return
    raise ValueError(f"{location} must contain JSON-serializable technical evidence only")


class MMMFailurePacket(BaseModel):
    """Versioned, technical, producer-owned failure packet for MIP consumption."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[MMM_FAILURE_SCHEMA_VERSION] = MMM_FAILURE_SCHEMA_VERSION
    failure_id: str = Field(min_length=1, max_length=200)
    created_at: datetime
    producer_package_version: str | None = Field(default=None, max_length=100)
    producer_git_commit: str | None = Field(default=None, max_length=100)
    run_id: str | None = Field(default=None, max_length=200)
    model_id: str | None = Field(default=None, max_length=200)
    model_family: str | None = Field(default=None, max_length=100)
    configuration_hash: str | None = Field(default=None, max_length=200)
    dataset_fingerprint: str | None = Field(default=None, max_length=200)
    code: MMMFailureCode
    stage: MMMFailureStage
    source_component: str = Field(min_length=1, max_length=200)
    technical_summary: str = Field(min_length=1, max_length=500)
    failure_status: Literal["blocked", "failed"] = "blocked"
    retry_disposition: MMMRetryDisposition
    remediation_actions: list[MMMRemediationAction] = Field(default_factory=list)
    blockers: list[str] = Field(default_factory=list)
    validation_result_ids: list[str] = Field(default_factory=list)
    diagnostic_ids: list[str] = Field(default_factory=list)
    calibration_signal_ids: list[str] = Field(default_factory=list)
    governance_blocker_ids: list[str] = Field(default_factory=list)
    affected_channels: list[str] = Field(default_factory=list)
    affected_kpi: str | None = Field(default=None, max_length=200)
    affected_grain: str | None = Field(default=None, max_length=100)
    affected_market: str | None = Field(default=None, max_length=200)
    affected_segment: str | None = Field(default=None, max_length=200)
    affected_time_range: str | None = Field(default=None, max_length=200)
    supported_range_evidence: list[str] = Field(default_factory=list)
    technical_context: dict[str, Any] = Field(default_factory=dict)
    retry_override_evidence: str | None = Field(default=None, max_length=500)

    @field_validator(
        "failure_id",
        "producer_package_version",
        "producer_git_commit",
        "run_id",
        "model_id",
        "model_family",
        "configuration_hash",
        "dataset_fingerprint",
        "source_component",
        "technical_summary",
        "affected_kpi",
        "affected_grain",
        "affected_market",
        "affected_segment",
        "affected_time_range",
        "retry_override_evidence",
    )
    @classmethod
    def _normalize_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("text fields must be non-empty when supplied")
        return cleaned

    @field_validator("created_at")
    @classmethod
    def _require_timezone_aware_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at must be timezone-aware")
        return value

    @field_validator("technical_context")
    @classmethod
    def _technical_context_is_safe_json(cls, value: dict[str, Any]) -> dict[str, Any]:
        _validate_safe_json(value)
        return value

    @model_validator(mode="after")
    def _validate_policy_consistency(self) -> MMMFailurePacket:
        default_retry, default_action = DEFAULT_FAILURE_POLICY[self.code]
        override = self.retry_disposition != default_retry
        if override and not self.retry_override_evidence:
            raise ValueError("retry override requires retry_override_evidence")
        if not override and self.retry_override_evidence:
            raise ValueError("retry_override_evidence is only valid for a retry override")
        if self.retry_disposition != MMMRetryDisposition.NOT_RETRYABLE and not any(
            action.required for action in self.remediation_actions
        ):
            raise ValueError("retryable failures require at least one required remediation action")
        if not override and not any(
            action.action_code == default_action for action in self.remediation_actions
        ):
            raise ValueError(f"{self.code.value} requires a {default_action.value} remediation action")
        return self

    def to_json_dict(self) -> dict[str, Any]:
        """Return a JSON-safe payload with optional fields omitted, deterministically ordered on encoding."""
        return self.model_dump(mode="json", exclude_none=True)

    def to_json(self) -> str:
        return json.dumps(self.to_json_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, payload: str) -> MMMFailurePacket:
        return cls.model_validate_json(payload)


def build_mmm_failure_packet(
    *,
    failure_id: str,
    created_at: datetime,
    code: MMMFailureCode,
    stage: MMMFailureStage,
    source_component: str,
    technical_summary: str,
    affected_resource: str | None = None,
    remediation_actions: list[MMMRemediationAction] | None = None,
    retry_disposition: MMMRetryDisposition | None = None,
    retry_override_evidence: str | None = None,
    **packet_fields: Any,
) -> MMMFailurePacket:
    """Build a packet from the deterministic policy without parsing exception strings."""
    default_retry, default_action = DEFAULT_FAILURE_POLICY[code]
    disposition = retry_disposition or default_retry
    actions = remediation_actions
    if actions is None:
        actions = [
            MMMRemediationAction(
                action_code=default_action,
                affected_resource=affected_resource,
                required=disposition != MMMRetryDisposition.NOT_RETRYABLE,
                technical_detail=f"Resolve {code.value} before rerunning this producer stage.",
            )
        ]
    return MMMFailurePacket(
        failure_id=failure_id,
        created_at=created_at,
        code=code,
        stage=stage,
        source_component=source_component,
        technical_summary=technical_summary,
        retry_disposition=disposition,
        remediation_actions=actions,
        retry_override_evidence=retry_override_evidence,
        **packet_fields,
    )


class MMMExportOutcome(BaseModel):
    """Discriminated producer boundary result: exactly one export or failure packet."""

    model_config = ConfigDict(extra="forbid")

    outcome_type: Literal["success", "failure"]
    export_bundle: MMMExportBundle | None = None
    failure_packet: MMMFailurePacket | None = None

    @model_validator(mode="after")
    def _one_payload_and_safe_success(self) -> MMMExportOutcome:
        has_export = self.export_bundle is not None
        has_failure = self.failure_packet is not None
        if has_export == has_failure:
            raise ValueError("MMMExportOutcome requires exactly one export_bundle or failure_packet")
        if self.outcome_type == "success" and not has_export:
            raise ValueError("success outcome requires export_bundle")
        if self.outcome_type == "failure" and not has_failure:
            raise ValueError("failure outcome requires failure_packet")
        if self.export_bundle and self.export_bundle.production_claim_allowed:
            errors = validate_mmm_export_bundle(self.export_bundle)
            if errors:
                raise ValueError("success export bundle is invalid: " + "; ".join(errors))
            for raw_artifact in self.export_bundle.artifacts:
                artifact = parse_export_artifact(raw_artifact)
                if artifact.promotion_status != PromotionStatus.APPROVED_FOR_PROD:
                    raise ValueError("production-authorized success requires approved_for_prod artifacts")
                if artifact.framework == "bayesian" or artifact.research_lane:
                    raise ValueError("research-only model artifacts cannot imply production authorization")
        return self

    @classmethod
    def success(cls, export_bundle: MMMExportBundle) -> MMMExportOutcome:
        return cls(outcome_type="success", export_bundle=export_bundle)

    @classmethod
    def failure(cls, failure_packet: MMMFailurePacket) -> MMMExportOutcome:
        return cls(outcome_type="failure", failure_packet=failure_packet)


__all__ = [
    "DEFAULT_FAILURE_POLICY",
    "MMM_FAILURE_SCHEMA_VERSION",
    "MMMExportOutcome",
    "MMMFailureCode",
    "MMMFailurePacket",
    "MMMFailureStage",
    "MMMRemediationAction",
    "MMMRemediationActionCode",
    "MMMRetryDisposition",
    "build_mmm_failure_packet",
]
