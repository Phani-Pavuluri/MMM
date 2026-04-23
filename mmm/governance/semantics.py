"""Canonical artifact tiers, surfaces, decision semantics, and safety flags for downstream payloads."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class ArtifactTier(str, Enum):
    """How strongly an artifact may drive budgeting / prod decisions."""

    DIAGNOSTIC = "diagnostic"
    RESEARCH = "research"
    DECISION = "decision"


class Surface(str, Enum):
    """Consumer context loading an artifact (reader-side enforcement)."""

    DIAGNOSTIC = "diagnostic"
    RESEARCH = "research"
    DECISION = "decision"


class DecisionSemantics(str, Enum):
    """What quantity or approximation the payload represents."""

    FULL_PANEL_DELTA_MU = "full_panel_delta_mu"
    APPROX_CURVE = "approx_curve"
    DECOMPOSITION = "decomposition"
    ROI_FIRST_ORDER = "roi_first_order"
    POSTERIOR_EXPLORATION = "posterior_exploration"
    DIAGNOSTIC_UNCERTAINTY_BUCKETS = "diagnostic_uncertainty_buckets"


class SafetyFlags(BaseModel):
    """Machine-readable safety posture for any result section."""

    decision_safe: bool = False
    prod_safe: bool = False
    approximate: bool = False
    unsupported_for: list[str] = Field(default_factory=list)


DECISION_TIER_VALUE = ArtifactTier.DECISION.value

# Downstream bans (populate ``unsupported_for`` on approximate / proxy surfaces)
BAN_BUDGETING = "budgeting"
BAN_FINANCIAL_COMMITMENT = "financial_commitment"
BAN_CHANNEL_ALLOCATION = "channel_allocation"
BAN_EXACT_PROFIT_FORECAST = "exact_profit_forecast"

__all__ = [
    "ArtifactTier",
    "BAN_BUDGETING",
    "BAN_CHANNEL_ALLOCATION",
    "BAN_EXACT_PROFIT_FORECAST",
    "BAN_FINANCIAL_COMMITMENT",
    "DECISION_TIER_VALUE",
    "DecisionSemantics",
    "SafetyFlags",
    "Surface",
]
