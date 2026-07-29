"""Deterministic MMM calibration-evidence compatibility boundary.

This contract evaluates a normalized calibration readout against one already
identified MMM model context.  It never imports a sibling-repository readout
schema, mutates a model, refits, simulates, or authorizes downstream planning.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

MMM_CALIBRATION_COMPATIBILITY_SCHEMA_VERSION = "mmm_calibration_compatibility_result_v1"


class MMMCalibrationCompatibilityState(str, Enum):
    COMPATIBLE = "compatible"
    COMPATIBLE_WITH_WARNING = "compatible_with_warning"
    STALE = "stale"
    INCOMPATIBLE = "incompatible"
    BLOCKED = "blocked"


class MMMCalibrationMethodStatus(str, Enum):
    GOVERNED = "governed"
    RESEARCH_ONLY = "research_only"
    DIAGNOSTIC_ONLY = "diagnostic_only"
    BLOCKED = "blocked"


class MMMCalibrationUncertaintyStatus(str, Enum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"


class MMMCalibrationTimeOverlapDecision(str, Enum):
    EXACT = "exact"
    PARTIAL = "partial"
    NONE = "none"


class MMMCalibrationFreshnessDecision(str, Enum):
    FRESH = "fresh"
    STALE = "stale"


class MMMCalibrationUncertaintyDecision(str, Enum):
    COMPATIBLE = "compatible"
    UNAVAILABLE = "unavailable"
    MISMATCH = "mismatch"


class MMMCalibrationCompatibilityReasonCode(str, Enum):
    HANDOFF_NOT_ELIGIBLE = "handoff_not_eligible"
    METHOD_STATUS_NOT_GOVERNED = "method_status_not_governed"
    KPI_MISMATCH = "kpi_mismatch"
    UNIT_MISMATCH = "unit_mismatch"
    ESTIMAND_MISMATCH = "estimand_mismatch"
    EFFECT_SCALE_MISMATCH = "effect_scale_mismatch"
    CHANNEL_MISMATCH = "channel_mismatch"
    GEOGRAPHY_MISMATCH = "geography_mismatch"
    GRAIN_MISMATCH = "grain_mismatch"
    TIME_WINDOW_NO_OVERLAP = "time_window_no_overlap"
    TIME_WINDOW_PARTIAL_OVERLAP = "time_window_partial_overlap"
    EVIDENCE_STALE = "evidence_stale"
    UNCERTAINTY_UNAVAILABLE = "uncertainty_unavailable"
    UNCERTAINTY_MISMATCH = "uncertainty_mismatch"


MMM_CALIBRATION_COMPATIBILITY_RULE_ORDER: tuple[str, ...] = (
    "required_identity_and_handoff_eligibility",
    "method_and_instrument_status",
    "kpi_and_units",
    "estimand_and_effect_scale",
    "channel",
    "geography_and_grain",
    "time_window_overlap",
    "freshness",
    "uncertainty_compatibility",
    "final_state_assembly",
)

_UNSAFE_TEXT = ("traceback", "stack trace", "password=", "secret=", "token=")
_UNKNOWN_TEXT = {"unknown", "unset", "n/a", "none", "null"}


def _identity(value: str, field_name: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} must be non-empty")
    lowered = cleaned.lower()
    if cleaned.startswith(("/", "~")) or lowered in _UNKNOWN_TEXT or any(
        marker in lowered for marker in _UNSAFE_TEXT
    ):
        raise ValueError(f"{field_name} must be a safe known identity")
    return cleaned


def _aware(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value


class MMMCalibrationCompatibilityLineage(BaseModel):
    """Stable logical identity for the evidence and model context being compared."""

    model_config = ConfigDict(extra="forbid")

    evidence_artifact_id: str
    model_artifact_id: str
    configuration_hash: str
    panel_id: str
    evaluation_id: str

    @field_validator(
        "evidence_artifact_id",
        "model_artifact_id",
        "configuration_hash",
        "panel_id",
        "evaluation_id",
    )
    @classmethod
    def _known_identity(cls, value: str, info: Any) -> str:
        return _identity(value, info.field_name)


class MMMNormalizedCalibrationReadout(BaseModel):
    """MMM-owned normalized input; it intentionally does not encode a GeoX schema."""

    model_config = ConfigDict(extra="forbid")

    source_readout_id: str
    source_readout_version: str
    source_producer_package: str | None = None
    source_producer_commit: str | None = None
    handoff_eligible: bool
    kpi_id: str
    unit: str
    estimand_id: str
    effect_scale: str
    channel_id: str
    geography: str
    grain: str
    evidence_window_start: datetime
    evidence_window_end: datetime
    observed_at: datetime
    fresh_through: datetime
    uncertainty_status: MMMCalibrationUncertaintyStatus
    method_family: str
    instrument_identity: str
    method_status: MMMCalibrationMethodStatus

    @field_validator(
        "source_readout_id",
        "source_readout_version",
        "source_producer_package",
        "source_producer_commit",
        "kpi_id",
        "unit",
        "estimand_id",
        "effect_scale",
        "channel_id",
        "geography",
        "grain",
        "method_family",
        "instrument_identity",
    )
    @classmethod
    def _known_text(cls, value: str | None, info: Any) -> str | None:
        return None if value is None else _identity(value, info.field_name)

    @field_validator("evidence_window_start", "evidence_window_end", "observed_at", "fresh_through")
    @classmethod
    def _aware_timestamp(cls, value: datetime, info: Any) -> datetime:
        return _aware(value, info.field_name)

    @model_validator(mode="after")
    def _valid_windows(self) -> MMMNormalizedCalibrationReadout:
        if self.evidence_window_end <= self.evidence_window_start:
            raise ValueError("evidence window end must follow start")
        if self.fresh_through < self.observed_at:
            raise ValueError("fresh_through cannot precede observed_at")
        return self


class MMMCalibrationModelContext(BaseModel):
    """Existing MMM model context, supplied without fitting or state mutation."""

    model_config = ConfigDict(extra="forbid")

    model_id: str
    run_id: str
    model_family: str
    kpi_id: str
    unit: str
    estimand_id: str
    effect_scale: str
    channel_id: str
    geography: str
    grain: str
    model_window_start: datetime
    model_window_end: datetime
    evaluated_at: datetime
    uncertainty_status: MMMCalibrationUncertaintyStatus

    @field_validator(
        "model_id",
        "run_id",
        "model_family",
        "kpi_id",
        "unit",
        "estimand_id",
        "effect_scale",
        "channel_id",
        "geography",
        "grain",
    )
    @classmethod
    def _known_text(cls, value: str, info: Any) -> str:
        return _identity(value, info.field_name)

    @field_validator("model_window_start", "model_window_end", "evaluated_at")
    @classmethod
    def _aware_timestamp(cls, value: datetime, info: Any) -> datetime:
        return _aware(value, info.field_name)

    @model_validator(mode="after")
    def _valid_window(self) -> MMMCalibrationModelContext:
        if self.model_window_end <= self.model_window_start:
            raise ValueError("model window end must follow start")
        return self


class MMMCalibrationCompatibilityRequest(BaseModel):
    """Complete, immutable input to the pure compatibility evaluator."""

    model_config = ConfigDict(extra="forbid")

    source: MMMNormalizedCalibrationReadout
    model_context: MMMCalibrationModelContext
    lineage: MMMCalibrationCompatibilityLineage


class MMMCalibrationCompatibilityIssue(BaseModel):
    """Typed, deterministic warning or blocker from one ordered compatibility check."""

    model_config = ConfigDict(extra="forbid")

    reason_code: MMMCalibrationCompatibilityReasonCode
    rationale: str

    @field_validator("rationale")
    @classmethod
    def _safe_rationale(cls, value: str) -> str:
        return _identity(value, "rationale")


class MMMCalibrationCompatibilityResult(BaseModel):
    """Authoritative MMM calibration compatibility result; not a calibration action."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["mmm_calibration_compatibility_result_v1"] = "mmm_calibration_compatibility_result_v1"
    compatibility_result_id: str
    source: MMMNormalizedCalibrationReadout
    model_context: MMMCalibrationModelContext
    lineage: MMMCalibrationCompatibilityLineage
    time_overlap_decision: MMMCalibrationTimeOverlapDecision
    freshness_decision: MMMCalibrationFreshnessDecision
    uncertainty_decision: MMMCalibrationUncertaintyDecision
    compatibility_state: MMMCalibrationCompatibilityState
    reason_codes: list[MMMCalibrationCompatibilityReasonCode] = Field(default_factory=list)
    warnings: list[MMMCalibrationCompatibilityIssue] = Field(default_factory=list)
    blockers: list[MMMCalibrationCompatibilityIssue] = Field(default_factory=list)
    evaluated_rule_order: tuple[str, ...] = MMM_CALIBRATION_COMPATIBILITY_RULE_ORDER

    @field_validator("compatibility_result_id")
    @classmethod
    def _result_id(cls, value: str) -> str:
        return _identity(value, "compatibility_result_id")

    @model_validator(mode="after")
    def _consistent_terminal_state(self) -> MMMCalibrationCompatibilityResult:
        if self.evaluated_rule_order != MMM_CALIBRATION_COMPATIBILITY_RULE_ORDER:
            raise ValueError("compatibility checks must use the canonical ordered rule sequence")
        issue_codes = [issue.reason_code for issue in [*self.warnings, *self.blockers]]
        if list(dict.fromkeys(self.reason_codes)) != self.reason_codes:
            raise ValueError("reason_codes must be deterministic and unique")
        if set(issue_codes) - set(self.reason_codes):
            raise ValueError("warnings and blockers must be represented by reason_codes")
        if self.compatibility_state == MMMCalibrationCompatibilityState.COMPATIBLE:
            if self.reason_codes or self.warnings or self.blockers:
                raise ValueError("compatible results cannot contain warnings or blockers")
        elif self.compatibility_state == MMMCalibrationCompatibilityState.COMPATIBLE_WITH_WARNING:
            if not self.warnings or self.blockers:
                raise ValueError("warning results require warnings without blockers")
        elif self.compatibility_state == MMMCalibrationCompatibilityState.STALE:
            if MMMCalibrationCompatibilityReasonCode.EVIDENCE_STALE not in self.reason_codes or self.blockers:
                raise ValueError("stale results require stale evidence without blockers")
        elif self.compatibility_state == MMMCalibrationCompatibilityState.INCOMPATIBLE:
            if not self.reason_codes or self.blockers:
                raise ValueError("incompatible results require mismatch reasons without blockers")
        elif self.compatibility_state == MMMCalibrationCompatibilityState.BLOCKED and not self.blockers:
            raise ValueError("blocked results require typed blockers")
        return self

    def to_json_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=True)

    def to_json(self) -> str:
        return json.dumps(self.to_json_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, payload: str) -> MMMCalibrationCompatibilityResult:
        decoded = json.loads(payload)
        if not isinstance(decoded, dict):
            raise ValueError("calibration compatibility result payload must be an object")
        return parse_mmm_calibration_compatibility_result(decoded)


_REQUIRED_RESULT_ENVELOPE_FIELDS = frozenset(
    {
        "schema_version",
        "compatibility_result_id",
        "source",
        "model_context",
        "lineage",
        "time_overlap_decision",
        "freshness_decision",
        "uncertainty_decision",
        "compatibility_state",
        "reason_codes",
        "warnings",
        "blockers",
        "evaluated_rule_order",
    }
)

_RATIONALES: dict[MMMCalibrationCompatibilityReasonCode, str] = {
    MMMCalibrationCompatibilityReasonCode.HANDOFF_NOT_ELIGIBLE: (
        "The source readout is not eligible for MMM calibration compatibility evaluation."
    ),
    MMMCalibrationCompatibilityReasonCode.METHOD_STATUS_NOT_GOVERNED: (
        "The source method or instrument is not governed for this compatibility boundary."
    ),
    MMMCalibrationCompatibilityReasonCode.KPI_MISMATCH: "The source and MMM KPI identities differ.",
    MMMCalibrationCompatibilityReasonCode.UNIT_MISMATCH: "The source and MMM KPI units differ.",
    MMMCalibrationCompatibilityReasonCode.ESTIMAND_MISMATCH: "The source and MMM estimand identities differ.",
    MMMCalibrationCompatibilityReasonCode.EFFECT_SCALE_MISMATCH: "The source and MMM effect scales differ.",
    MMMCalibrationCompatibilityReasonCode.CHANNEL_MISMATCH: "The source and MMM channel identities differ.",
    MMMCalibrationCompatibilityReasonCode.GEOGRAPHY_MISMATCH: "The source and MMM geography identities differ.",
    MMMCalibrationCompatibilityReasonCode.GRAIN_MISMATCH: "The source and MMM data grains differ.",
    MMMCalibrationCompatibilityReasonCode.TIME_WINDOW_NO_OVERLAP: (
        "The evidence and MMM model windows do not overlap."
    ),
    MMMCalibrationCompatibilityReasonCode.TIME_WINDOW_PARTIAL_OVERLAP: (
        "The evidence and MMM model windows overlap only partially."
    ),
    MMMCalibrationCompatibilityReasonCode.EVIDENCE_STALE: (
        "The source readout is stale at the declared evaluation time."
    ),
    MMMCalibrationCompatibilityReasonCode.UNCERTAINTY_UNAVAILABLE: (
        "Uncertainty is explicitly unavailable for at least one compared input."
    ),
    MMMCalibrationCompatibilityReasonCode.UNCERTAINTY_MISMATCH: (
        "The source and MMM uncertainty availability states differ."
    ),
}


def _issue(code: MMMCalibrationCompatibilityReasonCode) -> MMMCalibrationCompatibilityIssue:
    return MMMCalibrationCompatibilityIssue(reason_code=code, rationale=_RATIONALES[code])


def _overlap(
    source: MMMNormalizedCalibrationReadout, model: MMMCalibrationModelContext
) -> MMMCalibrationTimeOverlapDecision:
    if (
        source.evidence_window_start == model.model_window_start
        and source.evidence_window_end == model.model_window_end
    ):
        return MMMCalibrationTimeOverlapDecision.EXACT
    if source.evidence_window_start < model.model_window_end and model.model_window_start < source.evidence_window_end:
        return MMMCalibrationTimeOverlapDecision.PARTIAL
    return MMMCalibrationTimeOverlapDecision.NONE


def _uncertainty_decision(
    source: MMMNormalizedCalibrationReadout, model: MMMCalibrationModelContext
) -> MMMCalibrationUncertaintyDecision:
    if source.uncertainty_status == model.uncertainty_status == MMMCalibrationUncertaintyStatus.AVAILABLE:
        return MMMCalibrationUncertaintyDecision.COMPATIBLE
    if source.uncertainty_status != model.uncertainty_status:
        return MMMCalibrationUncertaintyDecision.MISMATCH
    return MMMCalibrationUncertaintyDecision.UNAVAILABLE


def evaluate_mmm_calibration_compatibility(
    request: MMMCalibrationCompatibilityRequest,
) -> MMMCalibrationCompatibilityResult:
    """Evaluate only explicit inputs in the documented canonical order.

    This function is pure: it performs no IO and does not mutate or fit any MMM
    object.  Its output reports compatibility evidence; it is not a refit,
    simulation, promotion, optimization, or recommendation path.
    """

    source = request.source
    model = request.model_context
    blockers: list[MMMCalibrationCompatibilityIssue] = []
    mismatches: list[MMMCalibrationCompatibilityReasonCode] = []
    warnings: list[MMMCalibrationCompatibilityIssue] = []

    # 1. Required identity is contract-validated before evaluation; only eligibility is dynamic.
    if not source.handoff_eligible:
        blockers.append(_issue(MMMCalibrationCompatibilityReasonCode.HANDOFF_NOT_ELIGIBLE))

    # 2. Method/instrument status.
    if source.method_status != MMMCalibrationMethodStatus.GOVERNED:
        blockers.append(_issue(MMMCalibrationCompatibilityReasonCode.METHOD_STATUS_NOT_GOVERNED))

    # 3. KPI and units.
    if source.kpi_id != model.kpi_id:
        mismatches.append(MMMCalibrationCompatibilityReasonCode.KPI_MISMATCH)
    if source.unit != model.unit:
        mismatches.append(MMMCalibrationCompatibilityReasonCode.UNIT_MISMATCH)

    # 4. Estimand and effect scale.
    if source.estimand_id != model.estimand_id:
        mismatches.append(MMMCalibrationCompatibilityReasonCode.ESTIMAND_MISMATCH)
    if source.effect_scale != model.effect_scale:
        mismatches.append(MMMCalibrationCompatibilityReasonCode.EFFECT_SCALE_MISMATCH)

    # 5. Channel.
    if source.channel_id != model.channel_id:
        mismatches.append(MMMCalibrationCompatibilityReasonCode.CHANNEL_MISMATCH)

    # 6. Geography and grain.
    if source.geography != model.geography:
        mismatches.append(MMMCalibrationCompatibilityReasonCode.GEOGRAPHY_MISMATCH)
    if source.grain != model.grain:
        mismatches.append(MMMCalibrationCompatibilityReasonCode.GRAIN_MISMATCH)

    # 7. Time-window overlap.
    overlap = _overlap(source, model)
    if overlap == MMMCalibrationTimeOverlapDecision.NONE:
        mismatches.append(MMMCalibrationCompatibilityReasonCode.TIME_WINDOW_NO_OVERLAP)
    elif overlap == MMMCalibrationTimeOverlapDecision.PARTIAL:
        warnings.append(_issue(MMMCalibrationCompatibilityReasonCode.TIME_WINDOW_PARTIAL_OVERLAP))

    # 8. Freshness.
    freshness = (
        MMMCalibrationFreshnessDecision.STALE
        if source.fresh_through < model.evaluated_at
        else MMMCalibrationFreshnessDecision.FRESH
    )

    # 9. Uncertainty compatibility.
    uncertainty = _uncertainty_decision(source, model)
    if uncertainty == MMMCalibrationUncertaintyDecision.UNAVAILABLE:
        warnings.append(_issue(MMMCalibrationCompatibilityReasonCode.UNCERTAINTY_UNAVAILABLE))
    elif uncertainty == MMMCalibrationUncertaintyDecision.MISMATCH:
        warnings.append(_issue(MMMCalibrationCompatibilityReasonCode.UNCERTAINTY_MISMATCH))

    # 10. Final state assembly.  Blocking and incompatibility take precedence over staleness.
    if blockers:
        state = MMMCalibrationCompatibilityState.BLOCKED
    elif mismatches:
        state = MMMCalibrationCompatibilityState.INCOMPATIBLE
    elif freshness == MMMCalibrationFreshnessDecision.STALE:
        state = MMMCalibrationCompatibilityState.STALE
    elif warnings:
        state = MMMCalibrationCompatibilityState.COMPATIBLE_WITH_WARNING
    else:
        state = MMMCalibrationCompatibilityState.COMPATIBLE

    reasons = [*(issue.reason_code for issue in blockers), *mismatches]
    if freshness == MMMCalibrationFreshnessDecision.STALE:
        reasons.append(MMMCalibrationCompatibilityReasonCode.EVIDENCE_STALE)
    reasons.extend(issue.reason_code for issue in warnings)
    # Preserve canonical rule ordering even where different checks produce the same code category.
    reasons = list(dict.fromkeys(reasons))
    return MMMCalibrationCompatibilityResult(
        compatibility_result_id=(
            f"mmm_calibration_compatibility:{model.run_id}:{source.source_readout_id}:"
            f"{source.source_readout_version}"
        ),
        source=source,
        model_context=model,
        lineage=request.lineage,
        time_overlap_decision=overlap,
        freshness_decision=freshness,
        uncertainty_decision=uncertainty,
        compatibility_state=state,
        reason_codes=reasons,
        warnings=warnings,
        blockers=blockers,
    )


def parse_mmm_calibration_compatibility_result(
    data: Mapping[str, Any],
) -> MMMCalibrationCompatibilityResult:
    """Strict, fail-closed parser for the registered public result contract."""

    missing = sorted(_REQUIRED_RESULT_ENVELOPE_FIELDS - set(data))
    if missing:
        raise ValueError(
            "calibration compatibility result is missing required envelope fields: " + ", ".join(missing)
        )
    if data["schema_version"] != MMM_CALIBRATION_COMPATIBILITY_SCHEMA_VERSION:
        raise ValueError("unsupported calibration compatibility schema version")
    return MMMCalibrationCompatibilityResult.model_validate(dict(data))


__all__ = [
    "MMM_CALIBRATION_COMPATIBILITY_RULE_ORDER",
    "MMM_CALIBRATION_COMPATIBILITY_SCHEMA_VERSION",
    "MMMCalibrationCompatibilityIssue",
    "MMMCalibrationCompatibilityLineage",
    "MMMCalibrationCompatibilityReasonCode",
    "MMMCalibrationCompatibilityRequest",
    "MMMCalibrationCompatibilityResult",
    "MMMCalibrationCompatibilityState",
    "MMMCalibrationFreshnessDecision",
    "MMMCalibrationMethodStatus",
    "MMMCalibrationModelContext",
    "MMMCalibrationTimeOverlapDecision",
    "MMMCalibrationUncertaintyDecision",
    "MMMCalibrationUncertaintyStatus",
    "MMMNormalizedCalibrationReadout",
    "evaluate_mmm_calibration_compatibility",
    "parse_mmm_calibration_compatibility_result",
]
