"""Canonical decision simulation: Δμ = μ(candidate) − μ(baseline) on the modeling scale."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from mmm.config.schema import Framework, MMMConfig, RunEnvironment
from mmm.economics.canonical import economics_output_metadata
from mmm.planning.baseline import (
    BaselinePlan,
    BaselineType,
    bau_baseline_from_panel,
    channel_means_from_geo_plan,
    disclosure_for_non_bau_optimization,
    spend_delta_l1,
    spend_delta_l1_geo,
    total_spend_geo_plan,
    total_spend_vector,
)
from mmm.planning.context import RidgeFitContext
from mmm.planning.control_overlay import ControlOverlaySpec
from mmm.planning.mu_path import DeltaMuAggregation, mean_mu_and_kpi_summary
from mmm.planning.posterior_planning import delta_mu_draws_linear_ridge, posterior_planning_gate
from mmm.planning.spend_path import (
    PiecewiseSpendPath,
    counterfactual_piecewise_spend_panel,
    time_mean_spend_by_channel,
)

UncertaintyMode = Literal["point", "posterior"]


def _ridge_monetary_ci_forbidden(config: MMMConfig) -> bool:
    return config.framework == Framework.RIDGE_BO and config.run_environment == RunEnvironment.PROD


def _geos_from_panel(panel: Any, schema: Any) -> list[str]:
    return sorted({str(x) for x in panel[schema.geo_column].unique()})


def _baseline_geo_and_scalar(
    base: BaselinePlan, geos: list[str]
) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    if base.spend_by_geo_channel:
        return dict(base.spend_by_geo_channel), base.spend_by_channel
    vec = dict(base.spend_by_channel)
    return {g: dict(vec) for g in geos}, vec


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


@dataclass
class SimulationResult:
    """Decision-facing simulation output; all monetary KPI fields use the same μ construction."""

    baseline_mu: float
    plan_mu: float
    delta_mu: float
    delta_spend: float
    roi: float | None
    mroas: float | None
    baseline_type: str
    baseline_definition: str
    uncertainty_mode: UncertaintyMode
    decision_safe: bool
    economics_version: str
    planner_mode: str
    canonical_quantity: str = "delta_mu_mean_row_mu_modeling_scale"
    mean_kpi_level_baseline: float | None = None
    mean_kpi_level_plan: float | None = None
    delta_kpi_level: float | None = None
    disclosure: str = ""
    p10: float | None = None
    p50: float | None = None
    p90: float | None = None
    # Explicit counterfactual / planning semantics (decision contract)
    horizon_weeks: int | None = None
    candidate_plan_type: str = "constant_channel_levels"
    counterfactual_construction_method: str = "full_panel_recursive_adstock_constant_spend"
    spend_path_assumption: str = "constant_per_channel_across_time"
    aggregation_semantics: str = "mean_mu_over_all_panel_rows_equal_weight"
    kpi_column: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "baseline_mu": self.baseline_mu,
            "plan_mu": self.plan_mu,
            "delta_mu": self.delta_mu,
            "delta_spend": self.delta_spend,
            "roi": self.roi,
            "mroas": self.mroas,
            "baseline_type": self.baseline_type,
            "baseline_definition": self.baseline_definition,
            "uncertainty_mode": self.uncertainty_mode,
            "decision_safe": self.decision_safe,
            "economics_version": self.economics_version,
            "planner_mode": self.planner_mode,
            "canonical_quantity": self.canonical_quantity,
            "mean_kpi_level_baseline": self.mean_kpi_level_baseline,
            "mean_kpi_level_plan": self.mean_kpi_level_plan,
            "delta_kpi_level": self.delta_kpi_level,
            "disclosure": self.disclosure,
            "p10": self.p10,
            "p50": self.p50,
            "p90": self.p90,
            "horizon_weeks": self.horizon_weeks,
            "candidate_plan_type": self.candidate_plan_type,
            "counterfactual_construction_method": self.counterfactual_construction_method,
            "spend_path_assumption": self.spend_path_assumption,
            "aggregation_semantics": self.aggregation_semantics,
            "kpi_column": self.kpi_column,
        }
        d.update(self.extra)
        return d


def simulate(
    spend_plan: dict[str, float],
    ctx: RidgeFitContext,
    *,
    baseline_plan: BaselinePlan | None = None,
    controls_plan: Any = None,
    horizon: int | None = None,
    uncertainty_mode: UncertaintyMode = "point",
    economics_version: str = "mmm_economics_contract_v1",
    planner_mode: str | None = None,
    bayesian_fit_meta: dict[str, Any] | None = None,
    spend_path_plan: PiecewiseSpendPath | None = None,
    spend_plan_geo: dict[str, dict[str, float]] | None = None,
    control_overlay: ControlOverlaySpec | None = None,
    control_overlay_baseline: ControlOverlaySpec | None = None,
    control_overlay_plan: ControlOverlaySpec | None = None,
    delta_mu_aggregation: DeltaMuAggregation | None = None,
    linear_coef_draws: np.ndarray | None = None,
    intercept_draws: np.ndarray | None = None,
) -> SimulationResult:
    """
    Evaluate **Δμ = μ̂(plan) − μ̂(baseline)** on the modeling scale (aggregation configurable).

    **Spend (exactly one candidate construction):**

    - Default: ``spend_plan`` = constant channel levels replicated across weeks (and expanded per geo
      when geo-economics apply).
    - ``spend_path_plan``: piecewise calendar-week channel overwrites on the full panel.
    - ``spend_plan_geo``: per-geo channel levels (constant in time within each geo).

    ``spend_path_plan`` and ``spend_plan_geo`` cannot both be set.

    **Non-spend (controls / promos):** optional overlays applied before the design matrix.

    - ``control_overlay_baseline`` rewrites columns on the **baseline** counterfactual panel.
    - Plan path uses ``control_overlay_plan``, else ``control_overlay``, else ``controls_plan`` parsed as
      :class:`ControlOverlaySpec` (dict with ``overrides`` / ``rows``).

    **Posterior / P10–P90:** when ``uncertainty_mode="posterior"``, pass ``linear_coef_draws`` (and optional
    ``intercept_draws``) together with ``bayesian_fit_meta`` satisfying
    :func:`mmm.planning.posterior_planning.posterior_planning_gate` to populate ``p10`` / ``p50`` / ``p90``
    on **Δμ draws** (linear μ on the Ridge design matrix).

    **Economics totals:** when the baseline or candidate uses per-geo spend levels, totals and L1 deltas
    use pooled **Σ_geo Σ_channel spend** for consistency. Piecewise paths use time-mean **national** vector
    totals (same as pre-geo release) and cannot be combined with a per-geo baseline in this release.
    """
    _ = horizon  # reserved API surface
    if ctx.config.framework != Framework.RIDGE_BO:
        raise NotImplementedError("decision simulate() Ridge path only in this release")

    if spend_path_plan is not None and spend_plan_geo is not None:
        raise ValueError("set at most one of spend_path_plan and spend_plan_geo")

    base = baseline_plan or bau_baseline_from_panel(ctx.panel, ctx.schema)
    if spend_path_plan is not None and base.spend_by_geo_channel is not None:
        raise ValueError(
            "piecewise spend_path_plan with per-geo baseline (spend_by_geo_channel) is not supported; "
            "use a global baseline or a per-geo spend_plan_geo candidate."
        )

    pm = planner_mode or ctx.config.extensions.product.planner_mode
    agg: DeltaMuAggregation = delta_mu_aggregation or ctx.config.extensions.product.planning_delta_mu_aggregation
    geos = _geos_from_panel(ctx.panel, ctx.schema)
    base_geo, base_scalar = _baseline_geo_and_scalar(base, geos)
    use_geo_economics = spend_plan_geo is not None or base.spend_by_geo_channel is not None

    ob = control_overlay_baseline
    op = _resolve_plan_control_overlay(control_overlay_plan, control_overlay, controls_plan)

    if base.spend_by_geo_channel:
        bsum = mean_mu_and_kpi_summary(
            ctx.panel,
            ctx.schema,
            ctx.config,
            best_params=ctx.best_params,
            coef=ctx.coef,
            intercept=ctx.intercept,
            spend_by_geo_channel=base_geo,
            control_overlay=ob,
            delta_mu_aggregation=agg,
        )
    else:
        bsum = mean_mu_and_kpi_summary(
            ctx.panel,
            ctx.schema,
            ctx.config,
            best_params=ctx.best_params,
            coef=ctx.coef,
            intercept=ctx.intercept,
            spend_by_channel=base.spend_by_channel,
            control_overlay=ob,
            delta_mu_aggregation=agg,
        )

    if spend_path_plan is not None:
        p_df = counterfactual_piecewise_spend_panel(ctx.panel, ctx.schema, spend_path_plan)
        plan_spend_for_econ = time_mean_spend_by_channel(p_df, ctx.schema)
        plan_geo = base_geo
        psum = mean_mu_and_kpi_summary(
            ctx.panel,
            ctx.schema,
            ctx.config,
            best_params=ctx.best_params,
            coef=ctx.coef,
            intercept=ctx.intercept,
            spend_path=spend_path_plan,
            control_overlay=op,
            delta_mu_aggregation=agg,
        )
    elif spend_plan_geo is not None:
        plan_geo = spend_plan_geo
        plan_spend_for_econ = channel_means_from_geo_plan(plan_geo, ctx.schema, geos)
        psum = mean_mu_and_kpi_summary(
            ctx.panel,
            ctx.schema,
            ctx.config,
            best_params=ctx.best_params,
            coef=ctx.coef,
            intercept=ctx.intercept,
            spend_by_geo_channel=plan_geo,
            control_overlay=op,
            delta_mu_aggregation=agg,
        )
    else:
        plan_geo = {g: dict(spend_plan) for g in geos}
        plan_spend_for_econ = spend_plan
        psum = mean_mu_and_kpi_summary(
            ctx.panel,
            ctx.schema,
            ctx.config,
            best_params=ctx.best_params,
            coef=ctx.coef,
            intercept=ctx.intercept,
            spend_by_channel=spend_plan,
            control_overlay=op,
            delta_mu_aggregation=agg,
        )

    baseline_mu = float(bsum["mean_mu_modeling"])
    plan_mu = float(psum["mean_mu_modeling"])
    delta_mu = float(plan_mu - baseline_mu)

    if spend_path_plan is not None:
        d_spend = spend_delta_l1(ctx.schema, base_scalar, plan_spend_for_econ)
        tot0 = total_spend_vector(ctx.schema, base_scalar)
        tot1 = total_spend_vector(ctx.schema, plan_spend_for_econ)
    elif use_geo_economics:
        d_spend = spend_delta_l1_geo(ctx.schema, base_geo, plan_geo, geos)
        tot0 = total_spend_geo_plan(ctx.schema, base_geo)
        tot1 = total_spend_geo_plan(ctx.schema, plan_geo)
    else:
        d_spend = spend_delta_l1(ctx.schema, base_scalar, plan_spend_for_econ)
        tot0 = total_spend_vector(ctx.schema, base_scalar)
        tot1 = total_spend_vector(ctx.schema, plan_spend_for_econ)
    delta_total = float(tot1 - tot0)

    kb = bsum.get("mean_kpi_level")
    kp = psum.get("mean_kpi_level")
    d_kpi = float(kp - kb) if isinstance(kb, (int, float)) and isinstance(kp, (int, float)) else None
    roi = (d_kpi / delta_total) if d_kpi is not None and abs(delta_total) > 1e-12 else None
    mroas = None
    if d_spend > 1e-12 and abs(delta_total) > 1e-12:
        mroas = float(delta_mu / delta_total) if delta_total != 0 else None

    disc_parts = [base.disclosure, disclosure_for_non_bau_optimization(base)]
    disc = " ".join(s for s in disc_parts if s).strip()
    pp_mode = ctx.config.extensions.product.posterior_planning_mode
    if pp_mode == "disclosure_only":
        disc = (
            disc + " posterior_planning_mode=disclosure_only: set product.posterior_planning_mode=draws in prod "
            "for decision-grade posterior planning APIs."
        ).strip()

    decision_safe = bool(base.suitable_for_decisioning and base.baseline_type == BaselineType.BAU)
    if base.baseline_type != BaselineType.BAU:
        decision_safe = False

    p10 = p50 = p90 = None
    post_gate: dict[str, Any] | None = None
    if uncertainty_mode == "posterior":
        post_gate = posterior_planning_gate(ctx.config, bayesian_fit_meta, linear_coef_draws=linear_coef_draws)
        if linear_coef_draws is not None and post_gate.get("allowed"):
            _, _, d_draws = delta_mu_draws_linear_ridge(
                ctx,
                baseline_plan=base,
                spend_plan=spend_plan,
                linear_coef_draws=np.asarray(linear_coef_draws, dtype=float),
                intercept_draws=intercept_draws,
                spend_path_plan=spend_path_plan,
                spend_plan_geo=spend_plan_geo,
                controls_plan=controls_plan,
                control_overlay=control_overlay,
                control_overlay_baseline=ob,
                control_overlay_plan=op,
                delta_mu_aggregation=agg,
            )
            p10 = float(np.percentile(d_draws, 10))
            p50 = float(np.percentile(d_draws, 50))
            p90 = float(np.percentile(d_draws, 90))
        elif _ridge_monetary_ci_forbidden(ctx.config):
            disc = (
                disc + " posterior uncertainty requested but diagnostics not validated; no credible intervals."
            ).strip()
        else:
            reasons = "; ".join(post_gate.get("reasons", [])) if post_gate else ""
            disc = (
                disc + " posterior intervals unavailable (gate failed or no coef draws); downgraded to point μ."
                + (f" Gate: {reasons}" if reasons else "")
            ).strip()

    if spend_path_plan is not None:
        cand_type = "piecewise_calendar_week"
        ccm = "full_panel_recursive_adstock_piecewise_channel_overwrites"
        spa = "piecewise_constant_in_calendar_week_segments"
    elif spend_plan_geo is not None:
        cand_type = "per_geo_channel_levels"
        ccm = "full_panel_recursive_adstock_per_geo_constant_spend"
        spa = "constant_per_channel_per_geo_across_time"
    else:
        cand_type = "constant_channel_levels"
        ccm = "full_panel_recursive_adstock_constant_spend"
        spa = (
            "constant_per_channel_across_time"
            if horizon is None
            else "constant_per_channel_across_time_horizon_parameter_reserved"
        )
    agg_sem = f"{agg}:{psum.get('aggregation', '')}"
    geo_time = (
        "per_geo_mean_mu_then_mean_across_geos"
        if agg == "geo_mean_then_global_mean"
        else "row_level_then_mean_mu_global"
    )

    ctrl_sem = []
    if ob is not None and ob.rows:
        ctrl_sem.append("baseline_control_overlay")
    else:
        ctrl_sem.append("baseline_observed_controls")
    if op is not None and op.rows:
        ctrl_sem.append("plan_control_overlay")
    else:
        ctrl_sem.append("plan_observed_controls")

    return SimulationResult(
        baseline_mu=baseline_mu,
        plan_mu=plan_mu,
        delta_mu=delta_mu,
        delta_spend=float(abs(delta_total)),
        roi=roi,
        mroas=mroas,
        baseline_type=base.baseline_type.value,
        baseline_definition=base.baseline_definition,
        uncertainty_mode=uncertainty_mode,
        decision_safe=decision_safe,
        economics_version=economics_version,
        planner_mode=str(pm),
        mean_kpi_level_baseline=float(kb) if isinstance(kb, (int, float)) else None,
        mean_kpi_level_plan=float(kp) if isinstance(kp, (int, float)) else None,
        delta_kpi_level=d_kpi,
        disclosure=disc,
        p10=p10,
        p50=p50,
        p90=p90,
        horizon_weeks=horizon,
        candidate_plan_type=cand_type,
        counterfactual_construction_method=ccm,
        spend_path_assumption=spa,
        aggregation_semantics=agg_sem,
        kpi_column=ctx.schema.target_column,
        extra={
            "baseline_plan_source": base.baseline_plan_source,
            "baseline_suitable_for_decisioning": base.suitable_for_decisioning,
            "controls_path_semantics": "+".join(ctrl_sem),
            "seasonality_path": "observed_panel_seasonality",
            "promo_path": "observed_panel_promos_if_in_design",
            "geo_time_aggregation": geo_time,
            "posterior_planning_mode": pp_mode,
            "posterior_planning_gate": post_gate,
            "plan_spend_for_economics": plan_spend_for_econ,
            "spend_economics_mode": "pooled_geo_sum" if use_geo_economics else "national_vector_sum",
            "baseline_has_per_geo_spend": base.spend_by_geo_channel is not None,
            "recommended_spend_plan_by_geo": plan_geo if use_geo_economics or spend_plan_geo is not None else None,
            "n_piecewise_segments": len(spend_path_plan.segments) if spend_path_plan else 0,
            "economics_output_metadata": economics_output_metadata(
                ctx.config,
                uncertainty_mode=uncertainty_mode,
                surface="full_model_simulation",
                baseline_type=base.baseline_type.value,
                decision_safe=decision_safe,
            ),
        },
    )
