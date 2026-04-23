"""Typed estimand identifiers for decision vs approximate quantities (enforced in code, not only JSON)."""

from __future__ import annotations

from enum import Enum


class EstimandKind(str, Enum):
    """Structural quantity kinds — align with ``DecisionSemantics`` for decision-grade paths."""

    FULL_PANEL_DELTA_MU = "full_panel_delta_mu"
    APPROX_CURVE = "approx_curve"
    DECOMPOSITION = "decomposition"
    ROI_FIRST_ORDER = "roi_first_order"
    POSTERIOR_EXPLORATION = "posterior_exploration"
    DIAGNOSTIC_UNCERTAINTY_BUCKETS = "diagnostic_uncertainty_buckets"


DECISION_OPTIMIZER_ESTIMAND = EstimandKind.FULL_PANEL_DELTA_MU
