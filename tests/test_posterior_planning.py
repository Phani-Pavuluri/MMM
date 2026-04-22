"""Posterior draw–based Δμ planning (linear Ridge design matrix)."""

from __future__ import annotations

import numpy as np
import pytest

from mmm.config.extensions import ExtensionSuiteConfig, ProductScopeConfig
from mmm.config.schema import CVConfig, DataConfig, Framework, MMMConfig, ModelForm, RunEnvironment
from mmm.optimization.budget.risk_optimizer import optimize_budget_risk_aware
from mmm.planning import bau_baseline_from_panel, simulate, simulate_posterior
from mmm.planning.context import RidgeFitContext, ridge_context_from_fit
from mmm.planning.posterior_planning import (
    PosteriorPlanningDisabled,
    posterior_planning_gate,
    risk_objective_scalar,
)
from mmm.models.ridge_bo.trainer import RidgeBOMMMTrainer
from mmm.utils.synthetic import SyntheticGeoPanelSpec, generate_geo_panel


def _ctx():
    df, schema = generate_geo_panel(
        SyntheticGeoPanelSpec(n_geos=2, n_weeks=24, channels=("a", "b"), betas=(0.35, 0.35))
    )
    ext = ExtensionSuiteConfig(product=ProductScopeConfig(posterior_planning_mode="draws"))
    cfg = MMMConfig(
        framework=Framework.RIDGE_BO,
        model_form=ModelForm.SEMI_LOG,
        data=DataConfig(
            path=None,
            geo_column=schema.geo_column,
            week_column=schema.week_column,
            target_column=schema.target_column,
            channel_columns=list(schema.channel_columns),
            control_columns=[],
        ),
        cv=CVConfig(mode="rolling", n_splits=2, min_train_weeks=10, horizon_weeks=2),
        ridge_bo={"n_trials": 2},
        extensions=ext,
    )
    tr = RidgeBOMMMTrainer(cfg, schema)
    fit = tr.fit(df)
    return ridge_context_from_fit(df, schema, cfg, fit), df, schema, cfg


def test_posterior_planning_gate_requires_diagnostics_and_draws() -> None:
    ctx, _df, _schema, cfg = _ctx()
    g0 = posterior_planning_gate(cfg, None, linear_coef_draws=np.ones((5, len(ctx.coef))))
    assert g0["allowed"] is False
    meta_bad = {"posterior_diagnostics_ok": False, "posterior_predictive_ok": True}
    g1 = posterior_planning_gate(cfg, meta_bad, linear_coef_draws=np.ones((5, len(ctx.coef))))
    assert g1["allowed"] is False
    meta_ok = {"posterior_diagnostics_ok": True, "posterior_predictive_ok": True}
    g2 = posterior_planning_gate(cfg, meta_ok, linear_coef_draws=np.ones((5, len(ctx.coef))))
    assert g2["allowed"] is True


def test_simulate_posterior_p_quantiles() -> None:
    ctx, df, schema, cfg = _ctx()
    bau = bau_baseline_from_panel(df, schema)
    plan = {c: float(bau.spend_by_channel[c]) * 1.08 for c in schema.channel_columns}
    rng = np.random.default_rng(42)
    s = 80
    draws = rng.normal(loc=ctx.coef, scale=0.02, size=(s, len(ctx.coef)))
    meta = {"posterior_diagnostics_ok": True, "posterior_predictive_ok": True}
    out = simulate_posterior(plan, ctx, baseline_plan=bau, bayesian_fit_meta=meta, linear_coef_draws=draws)
    assert out.p10 <= out.p50 <= out.p90
    assert out.n_draws == s


def test_simulate_uncertainty_posterior_fills_p_when_gated() -> None:
    ctx, df, schema, cfg = _ctx()
    bau = bau_baseline_from_panel(df, schema)
    plan = {c: float(bau.spend_by_channel[c]) * 1.05 for c in schema.channel_columns}
    rng = np.random.default_rng(1)
    draws = rng.normal(loc=ctx.coef, scale=0.015, size=(60, len(ctx.coef)))
    meta = {"posterior_diagnostics_ok": True, "posterior_predictive_ok": True}
    sim = simulate(
        plan,
        ctx,
        baseline_plan=bau,
        uncertainty_mode="posterior",
        bayesian_fit_meta=meta,
        linear_coef_draws=draws,
    )
    assert sim.p10 is not None and sim.p50 is not None and sim.p90 is not None
    assert sim.p10 <= sim.p50 <= sim.p90


def test_prod_requires_posterior_planning_mode_draws() -> None:
    ctx, df, schema, cfg0 = _ctx()
    cfg = cfg0.model_copy(
        update={
            "run_environment": RunEnvironment.PROD,
            "extensions": cfg0.extensions.model_copy(
                update={"product": ProductScopeConfig(posterior_planning_mode="off")}
            ),
        }
    )
    ctx_prod = RidgeFitContext(
        panel=ctx.panel,
        schema=ctx.schema,
        config=cfg,
        best_params=ctx.best_params,
        coef=ctx.coef,
        intercept=ctx.intercept,
    )
    draws = np.tile(ctx.coef, (10, 1))
    meta = {"posterior_diagnostics_ok": True, "posterior_predictive_ok": True}
    with pytest.raises(PosteriorPlanningDisabled):
        simulate_posterior(
            {c: 1.0 for c in schema.channel_columns},
            ctx_prod,
            baseline_plan=bau_baseline_from_panel(df, schema),
            bayesian_fit_meta=meta,
            linear_coef_draws=draws,
        )


def test_risk_objective_cvar_below_mean_on_positive_skew() -> None:
    x = np.linspace(-2, 3, 200)
    cvar = risk_objective_scalar(x, "cvar", cvar_alpha=0.1)
    assert cvar < float(np.mean(x))


def test_optimize_budget_risk_aware_smoke() -> None:
    ctx, df, schema, cfg = _ctx()
    bau = bau_baseline_from_panel(df, schema)
    names = list(schema.channel_columns)
    n = len(names)
    rng = np.random.default_rng(7)
    draws = rng.normal(loc=ctx.coef, scale=0.02, size=(40, len(ctx.coef)))
    meta = {"posterior_diagnostics_ok": True, "posterior_predictive_ok": True}
    tot = float(sum(bau.spend_by_channel[c] for c in names))
    res = optimize_budget_risk_aware(
        ctx,
        baseline_plan=bau,
        current_spend=np.array([bau.spend_by_channel[c] for c in names], dtype=float),
        total_budget=tot,
        channel_min=np.zeros(n),
        channel_max=np.ones(n) * 1e6,
        linear_coef_draws=draws,
        bayesian_fit_meta=meta,
        objective="p50",
        max_draws_per_eval=40,
    )
    assert res["success"] in (True, False)
    assert "posterior_delta_mu_p50" in res
    assert "recommended_spend_plan" in res
    ppm = res.get("posterior_planning_metadata") or {}
    assert ppm.get("posterior_planning_mode") == "draw_based_approximation"
    assert ppm.get("draw_source_artifact") == "linear_coef_draws"
    assert ppm.get("decision_safe") is True  # non-prod ctx from _ctx()
    assert ppm.get("economics_contract_version")
