"""Canonical schemas for decision-safe payloads and transparency reports."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from mmm.contracts.estimands import EstimandKind
from mmm.governance.semantics import ArtifactTier, DecisionSemantics, SafetyFlags, Surface

ARTIFACT_BUNDLE_SCHEMA_VERSION = "mmm_artifact_bundle_v3"


class UnsupportedQuestionReport(BaseModel):
    """Explicit reporting when an exact answer is not available (no silent approximation)."""

    supported: bool = True
    requested_question: str | None = None
    reason: str | None = None
    available_alternative: str | None = None


class SimulationDecisionResult(BaseModel):
    """
    Canonical decision-safe full-panel Δμ simulation slice.

    Only payloads equivalent to this (tier/surface/semantics + safety) may feed decision APIs
    and budget optimization in prod.
    """

    tier: ArtifactTier = ArtifactTier.DECISION
    surface: Surface = Surface.DECISION
    semantics: DecisionSemantics = DecisionSemantics.FULL_PANEL_DELTA_MU
    estimand_kind: EstimandKind = EstimandKind.FULL_PANEL_DELTA_MU
    safety: SafetyFlags = Field(
        default_factory=lambda: SafetyFlags(
            decision_safe=True,
            prod_safe=True,
            approximate=False,
            unsupported_for=[],
        )
    )
    baseline_mu: float
    candidate_mu: float
    delta_mu: float
    governance_refs: dict[str, Any] = Field(default_factory=dict)
    lineage_refs: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_simulation_json(cls, sim: dict[str, Any], *, governance_refs: dict, lineage_refs: dict) -> SimulationDecisionResult:
        return cls(
            baseline_mu=float(sim["baseline_mu"]),
            candidate_mu=float(sim["plan_mu"]),
            delta_mu=float(sim["delta_mu"]),
            governance_refs=governance_refs,
            lineage_refs=lineage_refs,
        )

    def as_result_dict(self) -> dict[str, Any]:
        """Flat dict for ``require_decision_safe_result`` + persistence."""
        return {
            "tier": self.tier.value,
            "artifact_tier": self.tier.value,
            "surface": self.surface.value,
            "semantics": self.semantics.value,
            "estimand_kind": self.estimand_kind.value,
            "decision_safe": self.safety.decision_safe,
            "prod_safe": self.safety.prod_safe,
            "approximate": self.safety.approximate,
            "unsupported_for": list(self.safety.unsupported_for),
            "baseline_mu": self.baseline_mu,
            "candidate_mu": self.candidate_mu,
            "plan_mu": self.candidate_mu,
            "delta_mu": self.delta_mu,
            "governance_refs": dict(self.governance_refs),
            "lineage_refs": dict(self.lineage_refs),
        }
