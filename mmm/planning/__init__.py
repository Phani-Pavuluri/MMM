"""Decision planning: baseline policy, full-model μ simulation, and optimization contracts."""

from mmm.planning.baseline import (
    BaselinePlan,
    BaselineType,
    bau_baseline_from_panel,
    bau_baseline_per_geo_from_panel,
    historical_average_baseline_from_panel,
    locked_geo_plan_baseline,
    zero_spend_baseline,
)
from mmm.planning.context import RidgeFitContext, ridge_context_from_fit
from mmm.planning.control_overlay import ControlOverlaySpec
from mmm.planning.decision_simulate import SimulationResult, simulate
from mmm.planning.posterior_planning import (
    PosteriorPlanningDisabled,
    PosteriorPlanResult,
    posterior_planning_gate,
    simulate_posterior,
)
from mmm.planning.spend_path import PiecewiseSpendPath, SpendSegment

__all__ = [
    "BaselinePlan",
    "BaselineType",
    "ControlOverlaySpec",
    "PosteriorPlanResult",
    "PosteriorPlanningDisabled",
    "RidgeFitContext",
    "SimulationResult",
    "bau_baseline_from_panel",
    "bau_baseline_per_geo_from_panel",
    "historical_average_baseline_from_panel",
    "locked_geo_plan_baseline",
    "posterior_planning_gate",
    "ridge_context_from_fit",
    "simulate",
    "simulate_posterior",
    "zero_spend_baseline",
    "PiecewiseSpendPath",
    "SpendSegment",
]
