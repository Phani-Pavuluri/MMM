"""Evidence quality scoring and weighting for replay / Bayesian likelihood (Phase 1)."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from datetime import date
from enum import StrEnum
from typing import Any

from mmm.experiments.compatibility import CompatibilityStatus, ReplayCompatibilityDecision
from mmm.experiments.evidence import ApprovalStatus, ExperimentEvidence, ExperimentType


class QualityTier(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    REJECTED = "rejected"


CONSERVATIVE_DEFAULT_SE = 1.0
MIN_EVIDENCE_WEIGHT = 0.05
MAX_EVIDENCE_WEIGHT = 1.0


@dataclass
class EvidenceQualityContext:
    """Optional context for overlap / compatibility scoring."""

    target_kpi: str | None = None
    channel_match: bool = True
    geo_scope_match: bool = True
    population_overlap: float = 1.0
    compatibility: ReplayCompatibilityDecision | None = None
    as_of: date | None = None
    stale_after_days: int = 365
    allow_missing_se: bool = False


@dataclass
class EvidenceQualityScore:
    evidence_weight: float
    quality_tier: QualityTier
    reasons: list[str] = field(default_factory=list)
    expiration_status: str = "active"
    freshness_weight: float = 1.0

    def to_json(self) -> dict[str, Any]:
        return {
            "evidence_weight": self.evidence_weight,
            "quality_tier": self.quality_tier.value,
            "reasons": list(self.reasons),
            "expiration_status": self.expiration_status,
            "freshness_weight": self.freshness_weight,
        }


def score_evidence_quality(
    evidence: ExperimentEvidence,
    ctx: EvidenceQualityContext | None = None,
) -> EvidenceQualityScore:
    """
    Compute evidence weight in (0, 1] for weighted replay / Bayesian likelihood.

    Low-quality experiments cannot dominate; expired → diagnostic-only tier.
    """
    c = ctx or EvidenceQualityContext()
    reasons: list[str] = []
    ref = c.as_of or date.today()

    if evidence.approval_status in {ApprovalStatus.REJECTED, ApprovalStatus.EXPIRED}:
        return EvidenceQualityScore(
            evidence_weight=0.0,
            quality_tier=QualityTier.REJECTED,
            reasons=[f"approval_status_{evidence.approval_status.value}"],
            expiration_status="expired" if evidence.approval_status == ApprovalStatus.EXPIRED else "rejected",
            freshness_weight=0.0,
        )

    fd = evidence.freshness_as_date()
    age_days = (ref - fd).days
    stale = age_days > c.stale_after_days
    freshness_w = max(0.1, 1.0 - age_days / max(c.stale_after_days, 1))
    if stale:
        reasons.append("stale_experiment_downweighted")
        freshness_w *= 0.5

    if evidence.standard_error is None or evidence.standard_error <= 0:
        if not c.allow_missing_se:
            return EvidenceQualityScore(
                evidence_weight=0.0,
                quality_tier=QualityTier.REJECTED,
                reasons=["missing_or_invalid_standard_error"],
                expiration_status="expired" if stale else "active",
                freshness_weight=freshness_w,
            )
        reasons.append("conservative_default_se_used")
        se = CONSERVATIVE_DEFAULT_SE
    else:
        se = float(evidence.standard_error)

    prec = 1.0 / (1.0 + se)
    weight = prec * freshness_w

    if evidence.experiment_type in {ExperimentType.OTHER}:
        weight *= 0.85
        reasons.append("experiment_type_other_discount")

    if "spillover" in [f.lower() for f in evidence.quality_flags]:
        weight *= 0.6
        reasons.append("spillover_flag_discount")
    if "contamination" in [f.lower() for f in evidence.quality_flags]:
        weight *= 0.7
        reasons.append("contamination_flag_discount")

    meta = evidence.metadata
    if meta.get("design_strength") is not None:
        with contextlib.suppress(TypeError, ValueError):
            weight *= 0.9 + 0.1 * min(1.0, float(meta["design_strength"]))

    if not c.channel_match:
        weight *= 0.5
        reasons.append("channel_mismatch_discount")
    if not c.geo_scope_match:
        weight *= 0.5
        reasons.append("geo_scope_mismatch_discount")
    weight *= max(0.0, min(1.0, float(c.population_overlap)))

    compat = c.compatibility
    if compat is not None:
        if compat.compatibility_status == CompatibilityStatus.REJECTED:
            return EvidenceQualityScore(
                evidence_weight=0.0,
                quality_tier=QualityTier.REJECTED,
                reasons=reasons + ["compatibility_rejected"],
                expiration_status="expired" if stale else "active",
                freshness_weight=freshness_w,
            )
        if compat.compatibility_status == CompatibilityStatus.DIAGNOSTIC_ONLY:
            weight *= 0.4
            reasons.append("diagnostic_only_compatibility")
        elif compat.compatibility_status == CompatibilityStatus.AGGREGATE_ONLY:
            weight *= 0.85
            reasons.append("aggregate_only_compatibility")

    if evidence.approval_status != ApprovalStatus.ACCEPTED:
        weight *= 0.75
        reasons.append(f"approval_not_accepted_{evidence.approval_status.value}")

    if not evidence.signature:
        weight *= 0.9
        reasons.append("missing_signature_lineage")

    weight = float(max(MIN_EVIDENCE_WEIGHT, min(MAX_EVIDENCE_WEIGHT, weight)))

    if weight >= 0.65 and not stale and evidence.approval_status == ApprovalStatus.ACCEPTED:
        tier = QualityTier.HIGH
    elif weight >= 0.35:
        tier = QualityTier.MEDIUM
    elif weight > 0:
        tier = QualityTier.LOW
    else:
        tier = QualityTier.REJECTED

    expiration = "stale_diagnostic_only" if stale else "active"
    return EvidenceQualityScore(
        evidence_weight=weight,
        quality_tier=tier,
        reasons=reasons,
        expiration_status=expiration,
        freshness_weight=freshness_w,
    )


def weighted_replay_loss_term(
    mmm_lift: float,
    experiment_lift: float,
    se: float,
    weight: float,
) -> float:
    """Single weighted standardized squared error term."""
    se_eff = max(se, 1e-12)
    return float(weight) * float(((mmm_lift - experiment_lift) / se_eff) ** 2)


def aggregate_weighted_replay_loss(
    terms: list[tuple[float, float, float, float]],
) -> tuple[float, dict[str, Any]]:
    """
    Aggregate weighted replay loss from (mmm_lift, experiment_lift, se, weight) tuples.

    Returns mean loss and metadata (for Ridge BO when evidence_weighting_enabled).
    """
    if not terms:
        return 0.0, {"n_terms": 0, "weighted_replay_loss": 0.0}
    losses = [weighted_replay_loss_term(m, e, s, w) for m, e, s, w in terms]
    import numpy as np

    arr = np.array(losses, dtype=float)
    return float(np.mean(arr)), {
        "n_terms": len(terms),
        "weighted_replay_loss": float(np.mean(arr)),
        "per_term_loss": losses,
    }
