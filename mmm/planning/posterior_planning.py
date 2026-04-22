"""
Posterior draw–based planning on the **Ridge design-matrix linear** μ path.

Requires **Bayesian decision diagnostics** (``posterior_diagnostics_ok`` and
``posterior_predictive_ok`` on ``bayesian_fit_meta``) plus caller-supplied
``linear_coef_draws`` aligned with :func:`mmm.models.ridge_bo.ridge.predict_ridge`
coefficients for the same ``X`` geometry as training (e.g. bootstrap or exported
posterior over linear coefficients — not automatic from Ridge BO point fit).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from mmm.config.schema import MMMConfig, RunEnvironment
from mmm.planning.baseline import BaselinePlan, bau_baseline_from_panel
from mmm.planning.context import RidgeFitContext
from mmm.planning.control_overlay import ControlOverlaySpec
from mmm.planning.mu_path import (
    DeltaMuAggregation,
    aggregate_mean_mu_draws,
    counterfactual_design_matrix,
)
from mmm.planning.spend_path import PiecewiseSpendPath


def _geos_from_panel(panel: Any, schema: Any) -> list[str]:
    return sorted({str(x) for x in panel[schema.geo_column].unique()})


def _bayesian_diagnostics_ok(fit_meta: dict[str, Any] | None) -> bool:
    if not fit_meta:
        return False
    return bool(fit_meta.get("posterior_diagnostics_ok")) and bool(fit_meta.get("posterior_predictive_ok"))


def _overlay_from_controls_plan(controls_plan: Any) -> ControlOverlaySpec | None:
    if controls_plan is None:
        return None
    if isinstance(controls_plan, ControlOverlaySpec):
        return controls_plan
    if isinstance(controls_plan, dict):
        return ControlOverlaySpec.from_dict(controls_plan)
    raise TypeError("controls_plan must be a dict, ControlOverlaySpec, or None")


def _resolve_plan_control_overlay(
    control_overlay_plan: ControlOverlaySpec | None,
    control_overlay: ControlOverlaySpec | None,
    controls_plan: Any,
) -> ControlOverlaySpec | None:
    if control_overlay_plan is not None:
        return control_overlay_plan
    if control_overlay is not None:
        return control_overlay
    return _overlay_from_controls_plan(controls_plan)


def _baseline_geo_and_scalar(
    base: BaselinePlan, geos: list[str]
) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    if base.spend_by_geo_channel:
        return dict(base.spend_by_geo_channel), base.spend_by_channel
    vec = dict(base.spend_by_channel)
    return {g: dict(vec) for g in geos}, vec


class PosteriorPlanningDisabled(RuntimeError):
    """Raised when posterior-scored planning is not permitted under current diagnostics / config."""

    def __init__(self, reasons: list[str]):
        self.reasons = reasons
        super().__init__("; ".join(reasons))


def posterior_planning_gate(
    config: MMMConfig,
    bayesian_fit_meta: dict[str, Any] | None,
    *,
    linear_coef_draws: np.ndarray | None,
) -> dict[str, Any]:
    """
    Decision-safe gating for draw-based objectives and ``simulate(..., uncertainty_mode="posterior")``.

    Production additionally requires ``extensions.product.posterior_planning_mode == "draws"``.
    """
    reasons: list[str] = []
    if not _bayesian_diagnostics_ok(bayesian_fit_meta):
        reasons.append("bayesian_decision_diagnostics_not_ok")
    draws = None if linear_coef_draws is None else np.asarray(linear_coef_draws, dtype=float)
    if draws is None or draws.size == 0:
        reasons.append("linear_coef_draws_missing_or_empty")
    elif draws.ndim != 2:
        reasons.append("linear_coef_draws_must_be_2d")
    allowed = len(reasons) == 0
    if (
        allowed
        and config.run_environment == RunEnvironment.PROD
        and config.extensions.product.posterior_planning_mode != "draws"
    ):
        allowed = False
        reasons.append("prod_requires_extensions.product.posterior_planning_mode_draws")
    return {"allowed": allowed, "reasons": reasons, "diagnostics_ok": _bayesian_diagnostics_ok(bayesian_fit_meta)}


def assert_posterior_planning_allowed(
    config: MMMConfig,
    bayesian_fit_meta: dict[str, Any] | None,
    linear_coef_draws: np.ndarray | None,
) -> None:
    g = posterior_planning_gate(config, bayesian_fit_meta, linear_coef_draws=linear_coef_draws)
    if not g["allowed"]:
        raise PosteriorPlanningDisabled(list(g["reasons"]))


def delta_mu_draws_linear_ridge(
    ctx: RidgeFitContext,
    *,
    baseline_plan: BaselinePlan | None,
    spend_plan: dict[str, float],
    linear_coef_draws: np.ndarray,
    intercept_draws: np.ndarray | None = None,
    spend_path_plan: PiecewiseSpendPath | None = None,
    spend_plan_geo: dict[str, dict[str, float]] | None = None,
    controls_plan: Any = None,
    control_overlay: ControlOverlaySpec | None = None,
    control_overlay_baseline: ControlOverlaySpec | None = None,
    control_overlay_plan: ControlOverlaySpec | None = None,
    delta_mu_aggregation: DeltaMuAggregation = "global_row_mean",
    max_draws: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return ``(baseline_mean_mu_draws, plan_mean_mu_draws, delta_mu_draws)`` on the modeling scale.

    Mirrors spend / overlay semantics of :func:`mmm.planning.decision_simulate.simulate` for Ridge BO.
    """
    if spend_path_plan is not None and spend_plan_geo is not None:
        raise ValueError("set at most one of spend_path_plan and spend_plan_geo")
    base = baseline_plan or bau_baseline_from_panel(ctx.panel, ctx.schema)
    if spend_path_plan is not None and base.spend_by_geo_channel is not None:
        raise ValueError(
            "piecewise spend_path_plan with per-geo baseline (spend_by_geo_channel) is not supported"
        )

    coef_draws = np.asarray(linear_coef_draws, dtype=float)
    if coef_draws.ndim != 2:
        raise ValueError("linear_coef_draws must have shape (n_draws, n_coef)")
    if max_draws is not None and coef_draws.shape[0] > int(max_draws):
        coef_draws = coef_draws[: int(max_draws), :]
    if coef_draws.shape[1] != int(ctx.coef.shape[0]):
        raise ValueError(
            f"linear_coef_draws second dim {coef_draws.shape[1]} != len(ctx.coef) {ctx.coef.shape[0]}"
        )

    geos = _geos_from_panel(ctx.panel, ctx.schema)
    base_geo, _ = _baseline_geo_and_scalar(base, geos)
    ob = control_overlay_baseline
    op = _resolve_plan_control_overlay(control_overlay_plan, control_overlay, controls_plan)

    if base.spend_by_geo_channel:
        Xb, dfb, _ = counterfactual_design_matrix(
            ctx.panel,
            ctx.schema,
            ctx.config,
            best_params=ctx.best_params,
            spend_by_geo_channel=base_geo,
            control_overlay=ob,
        )
    else:
        Xb, dfb, _ = counterfactual_design_matrix(
            ctx.panel,
            ctx.schema,
            ctx.config,
            best_params=ctx.best_params,
            spend_by_channel=base.spend_by_channel,
            control_overlay=ob,
        )

    if spend_path_plan is not None:
        Xp, dfp, _ = counterfactual_design_matrix(
            ctx.panel,
            ctx.schema,
            ctx.config,
            best_params=ctx.best_params,
            spend_path=spend_path_plan,
            control_overlay=op,
        )
    elif spend_plan_geo is not None:
        plan_geo = spend_plan_geo
        Xp, dfp, _ = counterfactual_design_matrix(
            ctx.panel,
            ctx.schema,
            ctx.config,
            best_params=ctx.best_params,
            spend_by_geo_channel=plan_geo,
            control_overlay=op,
        )
    else:
        Xp, dfp, _ = counterfactual_design_matrix(
            ctx.panel,
            ctx.schema,
            ctx.config,
            best_params=ctx.best_params,
            spend_by_channel=spend_plan,
            control_overlay=op,
        )

    int_draws = intercept_draws
    if int_draws is None:
        int_draws = np.full(coef_draws.shape[0], float(ctx.intercept.reshape(-1)[0]), dtype=float)

    mu_b = aggregate_mean_mu_draws(Xb, coef_draws, int_draws, dfb, ctx.schema, delta_mu_aggregation)
    mu_p = aggregate_mean_mu_draws(Xp, coef_draws, int_draws, dfp, ctx.schema, delta_mu_aggregation)
    return mu_b, mu_p, mu_p - mu_b


RiskObjective = Literal["p50", "cvar", "expected_minus_lambda_std"]


def risk_objective_scalar(
    delta_mu_draws: np.ndarray,
    objective: RiskObjective,
    *,
    cvar_alpha: float = 0.1,
    risk_lambda: float = 1.0,
) -> float:
    """Scalar objective to **maximize** (higher is better) from posterior Δμ draws."""
    x = np.sort(np.asarray(delta_mu_draws, dtype=float))
    if x.size == 0:
        return float("nan")
    if objective == "p50":
        return float(np.percentile(x, 50))
    if objective == "cvar":
        k = max(1, int(np.ceil(float(cvar_alpha) * x.size)))
        return float(np.mean(x[:k]))
    if objective == "expected_minus_lambda_std":
        return float(np.mean(x) - float(risk_lambda) * float(np.std(x)))
    raise ValueError(f"unknown risk objective {objective!r}")


@dataclass
class PosteriorPlanResult:
    """Output of :func:`simulate_posterior`."""

    baseline_mu_draws: list[float]
    plan_mu_draws: list[float]
    delta_mu_draws: list[float]
    p10: float
    p50: float
    p90: float
    n_draws: int
    gate: dict[str, Any] = field(default_factory=dict)
    disclosure: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "baseline_mu_draws": self.baseline_mu_draws,
            "plan_mu_draws": self.plan_mu_draws,
            "delta_mu_draws": self.delta_mu_draws,
            "p10": self.p10,
            "p50": self.p50,
            "p90": self.p90,
            "n_draws": self.n_draws,
            "gate": dict(self.gate),
            "disclosure": self.disclosure,
        }


def simulate_posterior(
    spend_plan: dict[str, float],
    ctx: RidgeFitContext,
    *,
    baseline_plan: BaselinePlan | None = None,
    bayesian_fit_meta: dict[str, Any] | None = None,
    linear_coef_draws: np.ndarray,
    intercept_draws: np.ndarray | None = None,
    draws: int | None = None,
    spend_path_plan: PiecewiseSpendPath | None = None,
    spend_plan_geo: dict[str, dict[str, float]] | None = None,
    controls_plan: Any = None,
    control_overlay: ControlOverlaySpec | None = None,
    control_overlay_baseline: ControlOverlaySpec | None = None,
    control_overlay_plan: ControlOverlaySpec | None = None,
    delta_mu_aggregation: DeltaMuAggregation | None = None,
) -> PosteriorPlanResult:
    """
    Score a spend plan across posterior draws of linear coefficients (same ``X`` construction as training).

    ``draws`` optionally truncates to the first ``draws`` rows of ``linear_coef_draws`` for runtime control.
    """
    assert_posterior_planning_allowed(ctx.config, bayesian_fit_meta, linear_coef_draws)
    agg: DeltaMuAggregation = delta_mu_aggregation or ctx.config.extensions.product.planning_delta_mu_aggregation
    mu_b, mu_p, dlt = delta_mu_draws_linear_ridge(
        ctx,
        baseline_plan=baseline_plan,
        spend_plan=spend_plan,
        linear_coef_draws=linear_coef_draws,
        intercept_draws=intercept_draws,
        spend_path_plan=spend_path_plan,
        spend_plan_geo=spend_plan_geo,
        controls_plan=controls_plan,
        control_overlay=control_overlay,
        control_overlay_baseline=control_overlay_baseline,
        control_overlay_plan=control_overlay_plan,
        delta_mu_aggregation=agg,
        max_draws=draws,
    )
    p10, p50, p90 = (float(x) for x in np.percentile(dlt, [10, 50, 90]))
    gate = posterior_planning_gate(ctx.config, bayesian_fit_meta, linear_coef_draws=linear_coef_draws)
    disc = ""
    if ctx.config.extensions.product.posterior_planning_mode != "draws":
        disc = (
            "posterior_planning_mode is not draws; quantiles are research-grade unless "
            "mode and governance are aligned."
        )
    return PosteriorPlanResult(
        baseline_mu_draws=[float(x) for x in mu_b],
        plan_mu_draws=[float(x) for x in mu_p],
        delta_mu_draws=[float(x) for x in dlt],
        p10=p10,
        p50=p50,
        p90=p90,
        n_draws=int(dlt.size),
        gate=gate,
        disclosure=disc,
    )
