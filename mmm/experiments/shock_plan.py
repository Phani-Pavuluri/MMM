"""Counterfactual spend/exposure shock planner (computational bridge only)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from mmm.experiments.compatibility import ReplayCompatibilityDecision, ReplayMode
from mmm.experiments.evidence import ExperimentEvidence


class AllocationMethod(StrEnum):
    OBSERVED_WITHHELD_SPEND = "observed_withheld_spend"
    OBSERVED_WITHHELD_IMPRESSIONS = "observed_withheld_impressions"
    ELIGIBLE_USER_WEIGHTED = "eligible_user_weighted"
    IMPRESSION_OR_OPPORTUNITY_WEIGHTED = "impression_or_opportunity_weighted"
    OBSERVED_SPEND_WEIGHTED = "observed_spend_weighted"
    BASELINE_CONVERSION_WEIGHTED = "baseline_conversion_weighted"
    UNIFORM = "uniform"


# Ranked by quality (best first).
ALLOCATION_METHOD_RANK: tuple[AllocationMethod, ...] = (
    AllocationMethod.OBSERVED_WITHHELD_SPEND,
    AllocationMethod.OBSERVED_WITHHELD_IMPRESSIONS,
    AllocationMethod.ELIGIBLE_USER_WEIGHTED,
    AllocationMethod.IMPRESSION_OR_OPPORTUNITY_WEIGHTED,
    AllocationMethod.OBSERVED_SPEND_WEIGHTED,
    AllocationMethod.BASELINE_CONVERSION_WEIGHTED,
    AllocationMethod.UNIFORM,
)


class AllocationQuality(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    REJECTED = "rejected"


@dataclass
class CounterfactualShockPlan:
    allocation_method: str
    allocation_quality: str
    allocation_role: str = "computational_bridge_only"
    supports_subgeo_claims: bool = False
    spend_delta_source: str | None = None
    exposure_delta_source: str | None = None
    warnings: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "allocation_method": self.allocation_method,
            "allocation_quality": self.allocation_quality,
            "allocation_role": self.allocation_role,
            "supports_subgeo_claims": self.supports_subgeo_claims,
            "spend_delta_source": self.spend_delta_source,
            "exposure_delta_source": self.exposure_delta_source,
            "warnings": list(self.warnings),
        }


def _method_quality(method: AllocationMethod) -> AllocationQuality:
    if method in {
        AllocationMethod.OBSERVED_WITHHELD_SPEND,
        AllocationMethod.OBSERVED_WITHHELD_IMPRESSIONS,
    }:
        return AllocationQuality.HIGH
    if method in {
        AllocationMethod.ELIGIBLE_USER_WEIGHTED,
        AllocationMethod.IMPRESSION_OR_OPPORTUNITY_WEIGHTED,
    }:
        return AllocationQuality.MEDIUM
    if method == AllocationMethod.UNIFORM:
        return AllocationQuality.LOW
    return AllocationQuality.MEDIUM


class CounterfactualShockPlanner:
    """
    Plan observed vs counterfactual spend/exposure paths for replay.

    Never treats allocated DMA shock as experimental DMA truth.
    """

    def plan(
        self,
        evidence: ExperimentEvidence,
        compatibility: ReplayCompatibilityDecision,
        *,
        metadata_hints: dict[str, Any] | None = None,
    ) -> CounterfactualShockPlan:
        hints = metadata_hints or evidence.metadata or {}
        warnings: list[str] = []

        if compatibility.replay_mode == ReplayMode.REJECT_INCOMPATIBLE:
            return CounterfactualShockPlan(
                allocation_method="none",
                allocation_quality=AllocationQuality.REJECTED.value,
                warnings=["compatibility_rejected: cannot construct shock plan"],
            )

        has_spend = evidence.spend_delta is not None
        has_exposure = evidence.exposure_delta is not None
        if not has_spend and not has_exposure:
            warnings.append("no_credible_spend_or_exposure_delta: diagnostic_only_or_reject")
            return CounterfactualShockPlan(
                allocation_method="none",
                allocation_quality=AllocationQuality.REJECTED.value,
                warnings=warnings,
            )

        method = self._select_method(evidence, hints, compatibility)
        qual = _method_quality(method)
        if method == AllocationMethod.UNIFORM:
            warnings.append("uniform_allocation_last_resort: low_confidence_bridge")

        if compatibility.allocation_required and method in {
            AllocationMethod.OBSERVED_WITHHELD_SPEND,
            AllocationMethod.OBSERVED_WITHHELD_IMPRESSIONS,
        }:
            warnings.append(
                "national_or_coarse_delta_allocated_to_subgeo: computational_only; not_experimental_subgeo_truth"
            )

        subgeo_claims = bool(
            hints.get("subgeo_effect_identified")
            and compatibility.supports_subgeo_claims
            and not compatibility.allocation_required
        )

        return CounterfactualShockPlan(
            allocation_method=method.value,
            allocation_quality=qual.value,
            supports_subgeo_claims=subgeo_claims,
            spend_delta_source="evidence.spend_delta" if has_spend else None,
            exposure_delta_source="evidence.exposure_delta" if has_exposure else None,
            warnings=warnings,
        )

    def _select_method(
        self,
        evidence: ExperimentEvidence,
        hints: dict[str, Any],
        compatibility: ReplayCompatibilityDecision,
    ) -> AllocationMethod:
        if hints.get("withheld_spend_path"):
            return AllocationMethod.OBSERVED_WITHHELD_SPEND
        if hints.get("withheld_impressions_path"):
            return AllocationMethod.OBSERVED_WITHHELD_IMPRESSIONS
        if hints.get("eligible_user_weights"):
            return AllocationMethod.ELIGIBLE_USER_WEIGHTED
        if hints.get("impression_weights") or hints.get("opportunity_weights"):
            return AllocationMethod.IMPRESSION_OR_OPPORTUNITY_WEIGHTED
        if evidence.spend_delta is not None and compatibility.allocation_required:
            return AllocationMethod.OBSERVED_SPEND_WEIGHTED
        if hints.get("baseline_conversion_weights"):
            return AllocationMethod.BASELINE_CONVERSION_WEIGHTED
        allowed = {m for m in compatibility.allowed_allocation_methods}
        for ranked in ALLOCATION_METHOD_RANK:
            if ranked.value in allowed or not allowed:
                if ranked == AllocationMethod.OBSERVED_WITHHELD_SPEND and evidence.spend_delta is not None:
                    return AllocationMethod.OBSERVED_SPEND_WEIGHTED
                if ranked.value in allowed:
                    return ranked
        return AllocationMethod.UNIFORM
