"""Fail-closed answerability policy for externally produced MMM bundles."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from mmm.contracts.mmm_export_bundle import ParsedMMMExportArtifact, ParsedMMMExportBundle


class MMMIntent(str, Enum):
    READINESS = "mmm_readiness"
    DIAGNOSTICS = "model_diagnostics"
    CONTRIBUTION = "channel_contribution"
    ROI = "channel_roi"
    RESPONSE_CURVE = "response_curve"
    SIMULATION = "simulation_result"
    BUDGET_RECOMMENDATION = "budget_recommendation"


@dataclass(frozen=True)
class AnswerabilityResult:
    intent: MMMIntent
    allowed: bool
    scope: str
    reason_code: str
    reasons: tuple[str, ...]
    required_disclosures: tuple[str, ...] = ()

    @property
    def cannot_say_reason(self) -> str | None:
        return None if self.allowed else "Cannot say: " + "; ".join(self.reasons)

    def as_dict(self) -> dict[str, object]:
        return {
            "intent": self.intent.value,
            "allowed": self.allowed,
            "scope": self.scope,
            "reason_code": self.reason_code,
            "reasons": list(self.reasons),
            "required_disclosures": list(self.required_disclosures),
            "cannot_say_reason": self.cannot_say_reason,
        }


_ARTIFACT_TYPES = {
    MMMIntent.READINESS: frozenset({"MMMModelFitArtifact", "MMMModelDiagnosticArtifact"}),
    MMMIntent.DIAGNOSTICS: frozenset({"MMMModelDiagnosticArtifact"}),
    MMMIntent.CONTRIBUTION: frozenset({"MMMChannelContributionArtifact"}),
    MMMIntent.ROI: frozenset({"MMMChannelROIArtifact"}),
    MMMIntent.RESPONSE_CURVE: frozenset({"MMMResponseCurveArtifact"}),
    MMMIntent.SIMULATION: frozenset({"MMMSimulationResultArtifact"}),
    MMMIntent.BUDGET_RECOMMENDATION: frozenset({"MMMRecommendationContract"}),
}

_ALLOWED_CODES = {
    MMMIntent.READINESS: frozenset({"readiness_explanation_allowed"}),
    MMMIntent.DIAGNOSTICS: frozenset({"diagnostic_explanation_allowed"}),
    MMMIntent.CONTRIBUTION: frozenset({"channel_contribution_claim", "channel_contribution_allowed"}),
    MMMIntent.ROI: frozenset({"channel_roi_claim", "channel_roas_claim", "channel_roi_ranking"}),
    MMMIntent.RESPONSE_CURVE: frozenset({"response_curve_explanation_allowed"}),
    MMMIntent.SIMULATION: frozenset({"simulation_result_explanation_allowed"}),
    MMMIntent.BUDGET_RECOMMENDATION: frozenset(
        {"budget_shift_recommendation", "budget_reallocation", "move_spend_between_channels"}
    ),
}

_FORBIDDEN_CODES = {
    MMMIntent.READINESS: frozenset({"readiness_explanation_allowed"}),
    MMMIntent.DIAGNOSTICS: frozenset({"diagnostic_explanation_allowed"}),
    MMMIntent.CONTRIBUTION: frozenset(
        {"channel_contribution_claim", "channel_contribution_allowed", "production_contribution_claim"}
    ),
    MMMIntent.ROI: frozenset(
        {
            "channel_roi_claim",
            "channel_roas_claim",
            "channel_roi_ranking",
            "highest_roi_channel",
            "incremental_roi_truth",
            "production_roi_claim",
        }
    ),
    MMMIntent.RESPONSE_CURVE: frozenset({"response_curve_explanation_allowed"}),
    MMMIntent.SIMULATION: frozenset({"simulation_result_explanation_allowed"}),
    MMMIntent.BUDGET_RECOMMENDATION: frozenset(
        {
            "budget_shift_recommendation",
            "budget_reallocation",
            "move_spend_between_channels",
            "optimize_budget_advice",
            "blocked_until_recommendation_contract",
        }
    ),
}


def _blocked(intent: MMMIntent, code: str, *reasons: str) -> AnswerabilityResult:
    return AnswerabilityResult(intent, False, "blocked", code, tuple(reasons))


def _claim_is_explicit(bundle: ParsedMMMExportBundle, artifact: ParsedMMMExportArtifact, intent: MMMIntent) -> bool:
    codes = _ALLOWED_CODES[intent]
    return bool(bundle.allowed_claims & codes) and bool(artifact.allowed_claims & codes)


def _forbidden_override(
    bundle: ParsedMMMExportBundle, artifact: ParsedMMMExportArtifact, intent: MMMIntent
) -> frozenset[str]:
    return (bundle.forbidden_claims | artifact.forbidden_claims) & _FORBIDDEN_CODES[intent]


def evaluate_mmm_export_answerability(
    bundle: ParsedMMMExportBundle,
    intent: MMMIntent | str,
    *,
    implies_recommendation: bool = False,
) -> AnswerabilityResult:
    """Classify what an LLM/verifier may say for one intent.

    ``implies_recommendation`` must be true when a curve or simulation answer is
    being used to suggest spend changes.  Such use is routed through the budget
    recommendation contract gate rather than inheriting explanatory access.
    """

    try:
        normalized_intent = intent if isinstance(intent, MMMIntent) else MMMIntent(intent)
    except ValueError:
        return _blocked(MMMIntent.READINESS, "unsupported_intent", f"unsupported MMM intent: {intent!r}")

    if implies_recommendation and normalized_intent in {MMMIntent.RESPONSE_CURVE, MMMIntent.SIMULATION}:
        return _blocked(
            normalized_intent,
            "recommendation_implication_blocked",
            "response curves and simulations do not grant recommendation authority",
            "a valid recommendation contract is required for spend advice",
        )

    artifacts = tuple(a for a in bundle.artifacts if a.artifact_type in _ARTIFACT_TYPES[normalized_intent])
    if not artifacts:
        return _blocked(
            normalized_intent,
            "required_artifact_missing",
            f"no {_ARTIFACT_TYPES[normalized_intent]} artifact is present",
        )

    failures: list[str] = []
    for artifact in artifacts:
        is_demo = bundle.demo_fixture_allowed and artifact.demo_fixture_allowed
        if normalized_intent is MMMIntent.ROI and is_demo:
            demo_forbidden = "demo_fixture_only" in (bundle.forbidden_claims | artifact.forbidden_claims)
            if (
                "demo_fixture_only" in bundle.allowed_claims
                and "demo_fixture_only" in artifact.allowed_claims
                and not demo_forbidden
            ):
                return AnswerabilityResult(
                    normalized_intent,
                    True,
                    "demo_only",
                    "demo_fixture_explanation_allowed",
                    ("synthetic ROI values may be explained only as a demo example",),
                    (
                        "Label every value as synthetic/demo.",
                        "Do not present the values as production or business truth.",
                        "Do not rank channels or recommend budget changes.",
                    ),
                )
            failures.append("demo fixture use was not explicitly allowed")
            continue

        forbidden = _forbidden_override(bundle, artifact, normalized_intent)
        if forbidden:
            failures.append("forbidden claims override allowed claims: " + ", ".join(sorted(forbidden)))
            continue

        if not bundle.llm_exposure_allowed or not artifact.llm_exposure_allowed:
            failures.append("LLM exposure was not explicitly allowed at bundle and artifact level")
            continue
        if not _claim_is_explicit(bundle, artifact, normalized_intent):
            failures.append("the requested claim was not explicitly allowed at bundle and artifact level")
            continue

        if normalized_intent is MMMIntent.READINESS:
            if artifact.artifact_safety_status not in {"readiness_only", "diagnostic_only", "production_safe"}:
                failures.append("artifact is not readiness/diagnostic safe")
                continue
            return AnswerabilityResult(
                normalized_intent,
                True,
                "explanation_only",
                "readiness_explanation_allowed",
                ("readiness may be explained from the governed readiness artifact",),
            )

        if normalized_intent is MMMIntent.DIAGNOSTICS:
            if artifact.artifact_safety_status not in {"diagnostic_only", "production_safe", "readiness_only"}:
                failures.append("artifact is not diagnostic safe")
                continue
            return AnswerabilityResult(
                normalized_intent,
                True,
                "explanation_only",
                "diagnostic_explanation_allowed",
                ("diagnostics may be explained without extending them into ROI or recommendations",),
            )

        if normalized_intent in {MMMIntent.RESPONSE_CURVE, MMMIntent.SIMULATION}:
            if not bundle.planning_allowed or not artifact.planning_allowed:
                failures.append("planning use was not explicitly allowed at bundle and artifact level")
                continue
            return AnswerabilityResult(
                normalized_intent,
                True,
                "explanation_only",
                "planning_explanation_allowed",
                ("the artifact may be explained but does not authorize a recommendation",),
                ("Do not imply an optimal allocation or budget shift.",),
            )

        if normalized_intent is MMMIntent.BUDGET_RECOMMENDATION:
            if not bundle.recommendation_allowed or not artifact.recommendation_allowed:
                failures.append("recommendation_allowed is not true at bundle and contract level")
                continue
            if not bundle.planning_allowed or not artifact.planning_allowed:
                failures.append("planning_allowed is not true at bundle and contract level")
                continue
            if not artifact.source_optimizer_artifact_id or not artifact.trust_report_refs:
                failures.append("recommendation contract lineage or TrustReport references are missing")
                continue
            if not artifact.proposed_budget_shifts:
                failures.append("recommendation contract contains no proposed budget shifts")
                continue

        if not bundle.production_claim_allowed or not artifact.production_claim_allowed:
            failures.append("production_claim_allowed is not true at bundle and artifact level")
            continue
        if bundle.artifact_safety_status != "production_safe" or artifact.artifact_safety_status != "production_safe":
            failures.append("bundle and artifact are not marked production_safe")
            continue
        if artifact.promotion_status != "approved_for_prod":
            failures.append("artifact is not approved_for_prod")
            continue
        if artifact.uncertainty_status != "present":
            failures.append("governed uncertainty is not present")
            continue
        return AnswerabilityResult(
            normalized_intent,
            True,
            "production",
            "governed_claim_allowed",
            ("all explicit claim, promotion, uncertainty, and exposure gates passed",),
        )

    return _blocked(
        normalized_intent,
        "safety_gate_blocked",
        *(failures or ["no artifact passed the fail-closed safety gates"]),
    )
