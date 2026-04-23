"""Posterior draw Δμ path aligns with point μ when draws collapse to the point estimate (Phase 5)."""

from __future__ import annotations

import numpy as np

from mmm.planning import bau_baseline_from_panel, simulate
from mmm.planning.posterior_planning import posterior_planning_gate


def test_linear_ridge_posterior_p50_matches_point_when_draws_identical() -> None:
    from mmm.config.extensions import ExtensionSuiteConfig, ProductScopeConfig
    from mmm.config.schema import CVConfig, DataConfig, Framework, MMMConfig, ModelForm, RunEnvironment
    from mmm.models.ridge_bo.trainer import RidgeBOMMMTrainer
    from mmm.utils.synthetic import SyntheticGeoPanelSpec, generate_geo_panel

    df, schema = generate_geo_panel(
        SyntheticGeoPanelSpec(n_geos=2, n_weeks=24, channels=("a", "b"), betas=(0.35, 0.35))
    )
    ext = ExtensionSuiteConfig(product=ProductScopeConfig(posterior_planning_mode="draws"))
    cfg = MMMConfig(
        framework=Framework.RIDGE_BO,
        model_form=ModelForm.SEMI_LOG,
        run_environment=RunEnvironment.RESEARCH,
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
    from mmm.planning.context import ridge_context_from_fit

    ctx = ridge_context_from_fit(df, schema, cfg, fit)
    bau = bau_baseline_from_panel(df, schema)
    plan = {c: float(bau.spend_by_channel[c]) * 1.05 for c in schema.channel_columns}
    meta = {"posterior_diagnostics_ok": True, "posterior_predictive_ok": True}
    g = posterior_planning_gate(cfg, meta, linear_coef_draws=np.ones((5, len(ctx.coef))))
    assert g["allowed"] is True
    s_point = simulate(plan, ctx, baseline_plan=bau, uncertainty_mode="point")
    n_draw = 80
    draws = np.tile(np.asarray(ctx.coef, dtype=float), (n_draw, 1))
    int_draws = np.full(n_draw, float(np.asarray(ctx.intercept).ravel()[0]), dtype=float)
    s_post = simulate(
        plan,
        ctx,
        baseline_plan=bau,
        uncertainty_mode="posterior",
        bayesian_fit_meta=meta,
        linear_coef_draws=draws,
        intercept_draws=int_draws,
    )
    assert s_post.p50 is not None
    assert abs(float(s_point.delta_mu) - float(s_post.p50)) < 1e-5
    assert abs(float(s_point.delta_mu) - float(s_post.p10)) < 1e-5
    assert abs(float(s_point.delta_mu) - float(s_post.p90)) < 1e-5
