"""Typed MMM → MIP export artifact schemas and claim-safety validators (MMM-EXPORT-002).

Schema / fixture layer only. Does **not** adapt live train/decide outputs, emit
recommendations, or mark ROI as production-safe. See
``docs/05_validation/mmm_export_schema_and_fixture_contract.md``.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SCHEMA_VERSION = "mmm_mip_export_v1"

# Claim codes that must never appear in allowed_claims when recommendation/ROI are blocked.
ROI_CLAIM_CODES = frozenset(
    {
        "channel_roi_ranking",
        "channel_roas_claim",
        "incremental_roi_truth",
        "highest_roi_channel",
    }
)
RECOMMENDATION_CLAIM_CODES = frozenset(
    {
        "budget_shift_recommendation",
        "budget_reallocation",
        "optimize_budget_advice",
        "move_spend_between_channels",
    }
)
BUDGET_SHIFT_ALLOWED_CODES = frozenset(
    {
        "budget_shift_recommendation",
        "budget_reallocation",
        "move_spend_between_channels",
    }
)


class ExportInventoryStatus(str, Enum):
    EXISTS_GOVERNED_AND_CONSUMABLE_BY_MIP = "EXISTS_GOVERNED_AND_CONSUMABLE_BY_MIP"
    EXISTS_PARTIAL_NOT_CONSUMABLE = "EXISTS_PARTIAL_NOT_CONSUMABLE"
    EXISTS_RESEARCH_ONLY = "EXISTS_RESEARCH_ONLY"
    PLANNED_NOT_IMPLEMENTED = "PLANNED_NOT_IMPLEMENTED"
    MISSING = "MISSING"


class ClaimSafetyCode(str, Enum):
    PRODUCTION_CLAIM_ALLOWED = "production_claim_allowed"
    DIAGNOSTIC_EXPLANATION_ALLOWED = "diagnostic_explanation_allowed"
    READINESS_EXPLANATION_ALLOWED = "readiness_explanation_allowed"
    DEMO_FIXTURE_ONLY = "demo_fixture_only"
    BLOCKED_UNTIL_CONTRACT = "blocked_until_contract"
    BLOCKED_UNTIL_UNCERTAINTY = "blocked_until_uncertainty"
    BLOCKED_UNTIL_PROMOTION = "blocked_until_promotion"
    BLOCKED_UNTIL_RECOMMENDATION_CONTRACT = "blocked_until_recommendation_contract"


class ArtifactSafetyStatus(str, Enum):
    PRODUCTION_SAFE = "production_safe"
    DIAGNOSTIC_ONLY = "diagnostic_only"
    READINESS_ONLY = "readiness_only"
    DEMO_FIXTURE_ONLY = "demo_fixture_only"
    BLOCKED = "blocked"


class UncertaintyStatus(str, Enum):
    PRESENT = "present"
    PARTIAL = "partial"
    MISSING = "missing"
    NONE = "none"
    NOT_APPLICABLE = "not_applicable"


class DiagnosticStatus(str, Enum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"
    UNKNOWN = "unknown"
    NOT_APPLICABLE = "not_applicable"


class PromotionStatus(str, Enum):
    RESEARCH_ONLY = "research_only"
    DIAGNOSTIC_ONLY = "diagnostic_only"
    PLANNING_CANDIDATE = "planning_candidate"
    APPROVED_FOR_PROD = "approved_for_prod"
    DEMO_SYNTHETIC = "demo_synthetic"
    BLOCKED = "blocked"


class CalibrationStatus(str, Enum):
    NONE = "none"
    CONTEXT_ATTACHED = "context_attached"
    CONFLICT = "conflict"
    STALE = "stale"
    NOT_APPLICABLE = "not_applicable"
    UNKNOWN = "unknown"


class MMMClaimSafety(BaseModel):
    """Claim-safety rollup for an artifact or bundle."""

    model_config = ConfigDict(extra="forbid")

    production_claim_allowed: bool = False
    diagnostic_explanation_allowed: bool = False
    readiness_explanation_allowed: bool = False
    demo_fixture_only: bool = False
    claim_codes: list[str] = Field(default_factory=list)
    artifact_safety_status: ArtifactSafetyStatus = ArtifactSafetyStatus.BLOCKED


class MMMArtifactLineage(BaseModel):
    """Lineage fingerprint block (must be present on consumable fixtures)."""

    model_config = ConfigDict(extra="forbid")

    training_data_fingerprint: str
    model_artifact_fingerprint: str
    source_artifacts: list[str] = Field(default_factory=list)
    package_version: str
    git_commit: str


class MMMUncertaintySummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    uncertainty_status: UncertaintyStatus
    notes: list[str] = Field(default_factory=list)


class MMMDiagnosticGateSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    diagnostic_status: DiagnosticStatus
    severity_summary: str | None = None
    notes: list[str] = Field(default_factory=list)


class MMMExportArtifactBase(BaseModel):
    """Common fields required on every MMM → MIP export family."""

    model_config = ConfigDict(extra="allow")

    artifact_type: str
    schema_version: str = SCHEMA_VERSION
    model_run_id: str
    training_data_fingerprint: str
    model_artifact_fingerprint: str
    source_artifacts: list[str] = Field(default_factory=list)
    model_form: str
    estimand: str
    time_window: str
    geo_scope: str
    channel_scope: list[str] = Field(default_factory=list)
    outcome_metric: str
    spend_metric: str
    currency: str
    uncertainty_status: UncertaintyStatus
    diagnostic_status: DiagnosticStatus
    promotion_status: PromotionStatus
    calibration_status: CalibrationStatus
    planning_allowed: bool = False
    llm_exposure_allowed: bool = False
    demo_fixture_allowed: bool = False
    recommendation_allowed: bool = False
    production_claim_allowed: bool = False
    allowed_claims: list[str] = Field(default_factory=list)
    forbidden_claims: list[str] = Field(default_factory=list)
    generated_at: str
    package_version: str
    git_commit: str
    is_docs_planned_placeholder: bool = False
    artifact_safety_status: ArtifactSafetyStatus = ArtifactSafetyStatus.BLOCKED
    claim_safety: MMMClaimSafety | None = None
    synthetic_demo_label: str | None = None
    framework: str | None = None
    research_lane: str | None = None

    @field_validator("schema_version")
    @classmethod
    def _schema_version_nonempty(cls, v: str) -> str:
        if not str(v).strip():
            raise ValueError("schema_version is required")
        return v

    @field_validator("model_run_id")
    @classmethod
    def _run_id_nonempty(cls, v: str) -> str:
        if not str(v).strip():
            raise ValueError("model_run_id is required")
        return v


class MMMModelFitArtifact(MMMExportArtifactBase):
    artifact_type: Literal["MMMModelFitArtifact"] = "MMMModelFitArtifact"
    framework: str = "ridge"
    model_release_state: str | None = None


class MMMModelDiagnosticArtifact(MMMExportArtifactBase):
    artifact_type: Literal["MMMModelDiagnosticArtifact"] = "MMMModelDiagnosticArtifact"
    severity_summary: str | None = None
    calibration_evidence_present: bool = False


class MMMChannelContributionArtifact(MMMExportArtifactBase):
    artifact_type: Literal["MMMChannelContributionArtifact"] = "MMMChannelContributionArtifact"
    contributions: dict[str, float] = Field(default_factory=dict)


class MMMChannelROIArtifact(MMMExportArtifactBase):
    artifact_type: Literal["MMMChannelROIArtifact"] = "MMMChannelROIArtifact"
    roi_by_channel: dict[str, float] = Field(default_factory=dict)
    roas_by_channel: dict[str, float] = Field(default_factory=dict)
    has_roi_values: bool = False

    @model_validator(mode="after")
    def _sync_has_roi(self) -> MMMChannelROIArtifact:
        if self.roi_by_channel or self.roas_by_channel:
            object.__setattr__(self, "has_roi_values", True)
        return self


class MMMResponseCurveArtifact(MMMExportArtifactBase):
    artifact_type: Literal["MMMResponseCurveArtifact"] = "MMMResponseCurveArtifact"
    channels_with_curves: list[str] = Field(default_factory=list)
    note: str = "Curves explain; full-panel simulation decides."


class MMMSimulationResultArtifact(MMMExportArtifactBase):
    artifact_type: Literal["MMMSimulationResultArtifact"] = "MMMSimulationResultArtifact"
    scenario_id: str | None = None
    delta_mu_summary: dict[str, Any] = Field(default_factory=dict)


class MMMOptimizerResultArtifact(MMMExportArtifactBase):
    artifact_type: Literal["MMMOptimizerResultArtifact"] = "MMMOptimizerResultArtifact"
    objective: str | None = None
    solution_spend: dict[str, float] = Field(default_factory=dict)
    has_recommendation_contract: bool = False


class MMMRecommendationContract(MMMExportArtifactBase):
    artifact_type: Literal["MMMRecommendationContract"] = "MMMRecommendationContract"
    source_optimizer_artifact_id: str | None = None
    proposed_budget_shifts: list[dict[str, Any]] = Field(default_factory=list)
    trust_report_refs: list[str] = Field(default_factory=list)


ARTIFACT_TYPE_TO_MODEL: dict[str, type[MMMExportArtifactBase]] = {
    "MMMModelFitArtifact": MMMModelFitArtifact,
    "MMMModelDiagnosticArtifact": MMMModelDiagnosticArtifact,
    "MMMChannelContributionArtifact": MMMChannelContributionArtifact,
    "MMMChannelROIArtifact": MMMChannelROIArtifact,
    "MMMResponseCurveArtifact": MMMResponseCurveArtifact,
    "MMMSimulationResultArtifact": MMMSimulationResultArtifact,
    "MMMOptimizerResultArtifact": MMMOptimizerResultArtifact,
    "MMMRecommendationContract": MMMRecommendationContract,
}


class MMMExportBundle(BaseModel):
    """Envelope of export artifacts for one model_run_id."""

    model_config = ConfigDict(extra="allow")

    artifact_type: Literal["MMMExportBundle"] = "MMMExportBundle"
    schema_version: str = SCHEMA_VERSION
    model_run_id: str
    training_data_fingerprint: str
    model_artifact_fingerprint: str
    source_artifacts: list[str] = Field(default_factory=list)
    generated_at: str
    package_version: str
    git_commit: str
    inventory_status: ExportInventoryStatus = ExportInventoryStatus.EXISTS_PARTIAL_NOT_CONSUMABLE
    llm_exposure_allowed: bool = False
    demo_fixture_allowed: bool = False
    recommendation_allowed: bool = False
    production_claim_allowed: bool = False
    planning_allowed: bool = False
    allowed_claims: list[str] = Field(default_factory=list)
    forbidden_claims: list[str] = Field(default_factory=list)
    artifact_safety_status: ArtifactSafetyStatus = ArtifactSafetyStatus.BLOCKED
    synthetic_demo_label: str | None = None
    is_docs_planned_placeholder: bool = False
    artifacts: list[dict[str, Any]] = Field(default_factory=list)


def parse_export_artifact(data: dict[str, Any]) -> MMMExportArtifactBase:
    """Parse a dict into the typed family model."""
    at = str(data.get("artifact_type") or "")
    model = ARTIFACT_TYPE_TO_MODEL.get(at)
    if model is None:
        raise ValueError(f"unknown artifact_type: {at!r}")
    return model.model_validate(data)


def parse_export_bundle(data: dict[str, Any]) -> MMMExportBundle:
    return MMMExportBundle.model_validate(data)


def _lineage_missing(obj: MMMExportArtifactBase | MMMExportBundle) -> list[str]:
    errors: list[str] = []
    if obj.is_docs_planned_placeholder:
        return errors
    for field in (
        "training_data_fingerprint",
        "model_artifact_fingerprint",
        "package_version",
        "git_commit",
    ):
        val = getattr(obj, field, None)
        if val is None or not str(val).strip():
            errors.append(f"missing lineage field: {field}")
    if isinstance(obj, MMMExportArtifactBase) and not obj.source_artifacts:
        errors.append("missing lineage field: source_artifacts (empty)")
    return errors


def _uncertainty_incomplete(status: UncertaintyStatus) -> bool:
    return status in {UncertaintyStatus.MISSING, UncertaintyStatus.NONE}


def validate_claim_safety(artifact: MMMExportArtifactBase | dict[str, Any]) -> list[str]:
    """Return claim-safety violations (empty if consistent)."""
    art = artifact if isinstance(artifact, MMMExportArtifactBase) else parse_export_artifact(artifact)
    errors: list[str] = []

    if art.production_claim_allowed and art.promotion_status != PromotionStatus.APPROVED_FOR_PROD:
        errors.append(
            "production_claim_allowed=true requires promotion_status=approved_for_prod"
        )

    if art.demo_fixture_allowed and art.production_claim_allowed:
        errors.append(
            "demo_fixture_allowed=true requires production_claim_allowed=false "
            "(unless separately governed outside this schema)"
        )

    if art.llm_exposure_allowed:
        if art.diagnostic_status is None:
            errors.append("llm_exposure_allowed=true requires diagnostic_status")
        if not art.forbidden_claims:
            errors.append("llm_exposure_allowed=true requires non-empty forbidden_claims")

    is_researchish = (
        art.framework == "bayesian"
        or art.research_lane == "bayes_h5"
        or art.promotion_status == PromotionStatus.RESEARCH_ONLY
    )
    if is_researchish and (
        art.production_claim_allowed or art.artifact_safety_status == ArtifactSafetyStatus.PRODUCTION_SAFE
    ):
        errors.append("research-only / Bayes artifacts cannot be production-safe")

    if art.promotion_status == PromotionStatus.RESEARCH_ONLY and (
        art.production_claim_allowed or art.recommendation_allowed
    ):
        errors.append("research_only promotion_status blocks production and recommendation claims")

    allowed = set(art.allowed_claims)
    blockedish = (
        art.artifact_safety_status
        in {
            ArtifactSafetyStatus.BLOCKED,
            ArtifactSafetyStatus.DEMO_FIXTURE_ONLY,
            ArtifactSafetyStatus.READINESS_ONLY,
            ArtifactSafetyStatus.DIAGNOSTIC_ONLY,
        }
        or not art.production_claim_allowed
        or art.recommendation_allowed is False
    )
    if (
        blockedish
        and allowed & ROI_CLAIM_CODES
        and (
            art.artifact_type == "MMMChannelROIArtifact"
            or ClaimSafetyCode.BLOCKED_UNTIL_UNCERTAINTY.value in art.forbidden_claims
            or _uncertainty_incomplete(art.uncertainty_status)
        )
        and not art.production_claim_allowed
    ):
        errors.append(
            "allowed_claims cannot include ROI truth claims when production/ROI is blocked"
        )
    if blockedish and allowed & RECOMMENDATION_CLAIM_CODES and not art.recommendation_allowed:
        errors.append(
            "allowed_claims cannot include recommendation claims when recommendation_allowed=false"
        )

    if art.artifact_type == "MMMChannelROIArtifact":
        roi = art  # type: ignore[assignment]
        assert isinstance(roi, MMMChannelROIArtifact)
        if roi.has_roi_values or roi.roi_by_channel or roi.roas_by_channel:
            if art.uncertainty_status is None or art.diagnostic_status is None:
                errors.append("ChannelROIArtifact with ROI/ROAS values requires uncertainty and diagnostic status")
            if _uncertainty_incomplete(art.uncertainty_status):
                if art.production_claim_allowed or art.artifact_safety_status == ArtifactSafetyStatus.PRODUCTION_SAFE:
                    errors.append(
                        "ChannelROIArtifact cannot be production-safe when uncertainty_status is missing/none"
                    )
                if ClaimSafetyCode.BLOCKED_UNTIL_UNCERTAINTY.value not in art.forbidden_claims and (
                    art.llm_exposure_allowed and "channel_roi" in " ".join(art.allowed_claims)
                ):
                    errors.append("ROI without uncertainty must include blocked_until_uncertainty")

    if art.artifact_type == "MMMResponseCurveArtifact" and art.recommendation_allowed:
        errors.append("ResponseCurveArtifact cannot set recommendation_allowed=true by itself")

    if art.artifact_type == "MMMOptimizerResultArtifact" and art.recommendation_allowed:
        opt = art
        assert isinstance(opt, MMMOptimizerResultArtifact)
        if not opt.has_recommendation_contract:
            errors.append(
                "OptimizerResultArtifact cannot set recommendation_allowed=true without a valid RecommendationContract"
            )

    if art.artifact_type == "MMMRecommendationContract":
        errors.extend(validate_recommendation_contract(art))

    return errors


def validate_recommendation_contract(
    contract: MMMRecommendationContract | dict[str, Any],
) -> list[str]:
    art = (
        contract
        if isinstance(contract, MMMRecommendationContract)
        else MMMRecommendationContract.model_validate(contract)
    )
    errors: list[str] = []
    if not art.source_optimizer_artifact_id:
        errors.append("RecommendationContract requires source_optimizer_artifact_id")
    if not (set(art.allowed_claims) & BUDGET_SHIFT_ALLOWED_CODES) and art.recommendation_allowed:
        errors.append("RecommendationContract with recommendation_allowed requires budget-shift allowed_claims")
    if not art.forbidden_claims:
        errors.append("RecommendationContract requires forbidden_claims")
    if art.diagnostic_status is None:
        errors.append("RecommendationContract requires diagnostic_status")
    if art.promotion_status is None:
        errors.append("RecommendationContract requires promotion_status")
    if art.demo_fixture_allowed or art.promotion_status in {
        PromotionStatus.DEMO_SYNTHETIC,
        PromotionStatus.RESEARCH_ONLY,
        PromotionStatus.DIAGNOSTIC_ONLY,
        PromotionStatus.BLOCKED,
    }:
        if art.production_claim_allowed or art.artifact_safety_status == ArtifactSafetyStatus.PRODUCTION_SAFE:
            errors.append("RecommendationContract with demo/non-promoted source remains blocked from production")
        if art.recommendation_allowed and art.demo_fixture_allowed:
            errors.append(
                "RecommendationContract demo sources cannot set recommendation_allowed=true for production exposure"
            )
    return errors


def validate_mmm_export_artifact(artifact: MMMExportArtifactBase | dict[str, Any]) -> list[str]:
    """Structural + claim-safety validation. Empty list = pass."""
    try:
        art = artifact if isinstance(artifact, MMMExportArtifactBase) else parse_export_artifact(artifact)
    except Exception as exc:  # noqa: BLE001 — surface parse errors as validation failures
        return [f"parse_error: {exc}"]

    errors: list[str] = []
    if not art.schema_version or not str(art.schema_version).strip():
        errors.append("missing schema_version")
    if not art.model_run_id or not str(art.model_run_id).strip():
        errors.append("missing model_run_id")
    errors.extend(_lineage_missing(art))
    errors.extend(validate_claim_safety(art))
    return errors


def validate_mmm_export_bundle(bundle: MMMExportBundle | dict[str, Any]) -> list[str]:
    try:
        b = bundle if isinstance(bundle, MMMExportBundle) else parse_export_bundle(bundle)
    except Exception as exc:  # noqa: BLE001
        return [f"parse_error: {exc}"]

    errors: list[str] = []
    if not b.schema_version or not str(b.schema_version).strip():
        errors.append("missing schema_version")
    if not b.model_run_id or not str(b.model_run_id).strip():
        errors.append("missing model_run_id")
    errors.extend(_lineage_missing(b))

    if b.llm_exposure_allowed and not b.forbidden_claims:
        errors.append("bundle llm_exposure_allowed=true requires non-empty forbidden_claims")
    if b.demo_fixture_allowed and b.production_claim_allowed:
        errors.append("bundle demo_fixture_allowed=true requires production_claim_allowed=false")
    if b.production_claim_allowed and b.inventory_status != ExportInventoryStatus.EXISTS_GOVERNED_AND_CONSUMABLE_BY_MIP:
        errors.append("bundle production_claim_allowed requires EXISTS_GOVERNED_AND_CONSUMABLE_BY_MIP")

    child_arts: list[MMMExportArtifactBase] = []
    has_recommendation_contract = False
    for raw in b.artifacts:
        child_errors = validate_mmm_export_artifact(raw)
        if child_errors:
            at = raw.get("artifact_type", "?")
            errors.extend([f"artifact[{at}]: {e}" for e in child_errors])
        else:
            child = parse_export_artifact(raw)
            child_arts.append(child)
            if child.artifact_type == "MMMRecommendationContract":
                has_recommendation_contract = True

    for child in child_arts:
        if (
            child.artifact_type == "MMMOptimizerResultArtifact"
            and child.recommendation_allowed
            and not has_recommendation_contract
        ):
            errors.append(
                "bundle: OptimizerResultArtifact recommendation_allowed without RecommendationContract"
            )

    if any(validate_claim_safety(c) for c in child_arts):
        # already reported above via validate_mmm_export_artifact
        pass

    if b.recommendation_allowed and not has_recommendation_contract:
        errors.append("bundle recommendation_allowed=true requires a RecommendationContract child")

    if not artifact_is_mip_exposable_bundle_contents(b, child_arts) and b.inventory_status == (
        ExportInventoryStatus.EXISTS_GOVERNED_AND_CONSUMABLE_BY_MIP
    ):
        errors.append("bundle claims EXISTS_GOVERNED_AND_CONSUMABLE_BY_MIP but children are not MIP-exposable")

    return errors


def artifact_is_readiness_exposable(artifact: MMMExportArtifactBase | dict[str, Any]) -> bool:
    art = artifact if isinstance(artifact, MMMExportArtifactBase) else parse_export_artifact(artifact)
    if validate_mmm_export_artifact(art):
        return False
    return (
        art.artifact_safety_status == ArtifactSafetyStatus.READINESS_ONLY
        or ClaimSafetyCode.READINESS_EXPLANATION_ALLOWED.value in art.allowed_claims
        or art.claim_safety is not None
        and art.claim_safety.readiness_explanation_allowed
    ) and not art.production_claim_allowed and not art.recommendation_allowed


def artifact_is_demo_safe(artifact: MMMExportArtifactBase | dict[str, Any]) -> bool:
    art = artifact if isinstance(artifact, MMMExportArtifactBase) else parse_export_artifact(artifact)
    if validate_mmm_export_artifact(art):
        return False
    return (
        art.demo_fixture_allowed
        and not art.production_claim_allowed
        and not art.recommendation_allowed
        and art.artifact_safety_status
        in {ArtifactSafetyStatus.DEMO_FIXTURE_ONLY, ArtifactSafetyStatus.BLOCKED, ArtifactSafetyStatus.DIAGNOSTIC_ONLY}
    )


def artifact_is_mip_exposable(artifact: MMMExportArtifactBase | dict[str, Any]) -> bool:
    """True only for governed production-consumable exposure (none of today's fixtures)."""
    art = artifact if isinstance(artifact, MMMExportArtifactBase) else parse_export_artifact(artifact)
    if validate_mmm_export_artifact(art):
        return False
    return (
        art.artifact_safety_status == ArtifactSafetyStatus.PRODUCTION_SAFE
        and art.production_claim_allowed
        and art.promotion_status == PromotionStatus.APPROVED_FOR_PROD
        and not art.demo_fixture_allowed
        and art.promotion_status != PromotionStatus.RESEARCH_ONLY
    )


def artifact_is_mip_exposable_bundle_contents(
    bundle: MMMExportBundle,
    children: list[MMMExportArtifactBase] | None = None,
) -> bool:
    kids = children
    if kids is None:
        kids = []
        for raw in bundle.artifacts:
            try:
                kids.append(parse_export_artifact(raw))
            except Exception:  # noqa: BLE001
                return False
    if any(validate_mmm_export_artifact(c) for c in kids):
        return False
    if any(validate_claim_safety(c) for c in kids):
        return False
    if bundle.production_claim_allowed:
        return all(artifact_is_mip_exposable(c) for c in kids) if kids else False
    # Non-production bundles are not MIP-consumable for truth claims.
    return False


def bundle_is_mip_consumable(bundle: MMMExportBundle | dict[str, Any]) -> bool:
    b = bundle if isinstance(bundle, MMMExportBundle) else parse_export_bundle(bundle)
    if validate_mmm_export_bundle(b):
        return False
    return (
        b.inventory_status == ExportInventoryStatus.EXISTS_GOVERNED_AND_CONSUMABLE_BY_MIP
        and b.production_claim_allowed
        and artifact_is_mip_exposable_bundle_contents(b)
    )


def roi_is_blocked_until_uncertainty(artifact: MMMChannelROIArtifact | dict[str, Any]) -> bool:
    art = (
        artifact
        if isinstance(artifact, MMMChannelROIArtifact)
        else MMMChannelROIArtifact.model_validate(artifact)
    )
    return bool(art.has_roi_values or art.roi_by_channel) and _uncertainty_incomplete(art.uncertainty_status)


__all__ = [
    "SCHEMA_VERSION",
    "ExportInventoryStatus",
    "ClaimSafetyCode",
    "ArtifactSafetyStatus",
    "UncertaintyStatus",
    "DiagnosticStatus",
    "PromotionStatus",
    "CalibrationStatus",
    "MMMClaimSafety",
    "MMMArtifactLineage",
    "MMMUncertaintySummary",
    "MMMDiagnosticGateSummary",
    "MMMExportArtifactBase",
    "MMMModelFitArtifact",
    "MMMModelDiagnosticArtifact",
    "MMMChannelContributionArtifact",
    "MMMChannelROIArtifact",
    "MMMResponseCurveArtifact",
    "MMMSimulationResultArtifact",
    "MMMOptimizerResultArtifact",
    "MMMRecommendationContract",
    "MMMExportBundle",
    "parse_export_artifact",
    "parse_export_bundle",
    "validate_mmm_export_artifact",
    "validate_mmm_export_bundle",
    "validate_claim_safety",
    "validate_recommendation_contract",
    "artifact_is_mip_exposable",
    "artifact_is_demo_safe",
    "artifact_is_readiness_exposable",
    "bundle_is_mip_consumable",
    "roi_is_blocked_until_uncertainty",
]
