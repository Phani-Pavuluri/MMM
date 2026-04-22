"""Piecewise spend paths and geo-aware Δμ aggregation."""

from __future__ import annotations

import numpy as np

from mmm.config.schema import CVConfig, DataConfig, Framework, MMMConfig, ModelForm
from mmm.data.schema import PanelSchema
from mmm.models.ridge_bo.trainer import RidgeBOMMMTrainer
from mmm.optimization.budget.simulation_optimizer import optimize_budget_via_simulation
from mmm.planning import (
    bau_baseline_from_panel,
    bau_baseline_per_geo_from_panel,
    simulate,
)
from mmm.planning.baseline import total_spend_geo_plan
from mmm.planning.context import RidgeFitContext, ridge_context_from_fit
from mmm.planning.control_overlay import ControlOverlaySpec
from mmm.planning.spend_path import PiecewiseSpendPath, SpendSegment
from mmm.utils.synthetic import SyntheticGeoPanelSpec, generate_geo_panel


def _fit_ctx(df, schema, cfg):
    tr = RidgeBOMMMTrainer(cfg, schema)
    fit = tr.fit(df)
    return ridge_context_from_fit(df, schema, cfg, fit)


def test_piecewise_spend_path_changes_mu_vs_constant() -> None:
    df, schema = generate_geo_panel(
        SyntheticGeoPanelSpec(n_geos=2, n_weeks=20, channels=("a",), betas=(0.5,))
    )
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
        cv=CVConfig(mode="rolling", n_splits=2, min_train_weeks=8, horizon_weeks=2),
        ridge_bo={"n_trials": 2},
    )
    ctx = _fit_ctx(df, schema, cfg)
    bau = bau_baseline_from_panel(df, schema)
    const = {c: float(bau.spend_by_channel[c]) * 1.1 for c in schema.channel_columns}
    s0 = simulate(const, ctx, baseline_plan=bau, uncertainty_mode="point")
    path = PiecewiseSpendPath(
        segments=(
            SpendSegment(week_start=1, week_end=10, spend_by_channel={"a": const["a"] * 0.5}),
            SpendSegment(week_start=11, week_end=20, spend_by_channel={"a": const["a"] * 1.5}),
        )
    )
    s1 = simulate(const, ctx, baseline_plan=bau, uncertainty_mode="point", spend_path_plan=path)
    assert s0.candidate_plan_type == "constant_channel_levels"
    assert s1.candidate_plan_type == "piecewise_calendar_week"
    assert s0.delta_mu != s1.delta_mu


def test_geo_mean_then_global_aggregation_semantics() -> None:
    df, schema = generate_geo_panel(
        SyntheticGeoPanelSpec(n_geos=3, n_weeks=15, channels=("a", "b"), betas=(0.4, 0.4))
    )
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
        cv=CVConfig(mode="rolling", n_splits=2, min_train_weeks=8, horizon_weeks=2),
        ridge_bo={"n_trials": 2},
    )
    ctx = _fit_ctx(df, schema, cfg)
    bau = bau_baseline_from_panel(df, schema)
    plan = {c: float(bau.spend_by_channel[c]) * 1.05 for c in schema.channel_columns}
    s = simulate(
        plan,
        ctx,
        baseline_plan=bau,
        uncertainty_mode="point",
        delta_mu_aggregation="geo_mean_then_global_mean",
    )
    assert "geo_mean_then_global_mean" in s.aggregation_semantics


def test_per_geo_plan_matches_per_geo_baseline_zero_delta() -> None:
    df, schema0 = generate_geo_panel(
        SyntheticGeoPanelSpec(n_geos=2, n_weeks=14, channels=("a", "b"), betas=(0.35, 0.35))
    )
    schema = schema0
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
        cv=CVConfig(mode="rolling", n_splits=2, min_train_weeks=8, horizon_weeks=2),
        ridge_bo={"n_trials": 2},
    )
    ctx = _fit_ctx(df, schema, cfg)
    bau_pg = bau_baseline_per_geo_from_panel(df, schema)
    assert bau_pg.spend_by_geo_channel is not None
    plan_geo = {g: dict(bau_pg.spend_by_geo_channel[g]) for g in bau_pg.spend_by_geo_channel}
    sim = simulate(
        dict(bau_pg.spend_by_channel),
        ctx,
        baseline_plan=bau_pg,
        uncertainty_mode="point",
        spend_plan_geo=plan_geo,
    )
    assert abs(sim.delta_mu) < 1e-5
    assert sim.extra.get("spend_economics_mode") == "pooled_geo_sum"
    assert sim.candidate_plan_type == "per_geo_channel_levels"


def test_control_overlay_plan_semantics_in_extra() -> None:
    df, schema0 = generate_geo_panel(
        SyntheticGeoPanelSpec(n_geos=2, n_weeks=12, channels=("a",), betas=(0.4,))
    )
    df = df.copy()
    df["promo"] = 0.0
    schema = PanelSchema(
        geo_column=schema0.geo_column,
        week_column=schema0.week_column,
        target_column=schema0.target_column,
        channel_columns=schema0.channel_columns,
        control_columns=("promo",),
    )
    cfg = MMMConfig(
        framework=Framework.RIDGE_BO,
        model_form=ModelForm.SEMI_LOG,
        data=DataConfig(
            path=None,
            geo_column=schema.geo_column,
            week_column=schema.week_column,
            target_column=schema.target_column,
            channel_columns=list(schema.channel_columns),
            control_columns=["promo"],
        ),
        cv=CVConfig(mode="rolling", n_splits=2, min_train_weeks=8, horizon_weeks=2),
        ridge_bo={"n_trials": 2},
    )
    ctx = _fit_ctx(df, schema, cfg)
    bau = bau_baseline_from_panel(df, schema)
    wk = df[schema.week_column].iloc[0]
    geo = str(df[schema.geo_column].iloc[0])
    ov = ControlOverlaySpec.from_dict(
        {"overrides": [{"geo": geo, "week": wk, "column": "promo", "value": 1.0}]}
    )
    sim = simulate(
        dict(bau.spend_by_channel),
        ctx,
        baseline_plan=bau,
        uncertainty_mode="point",
        control_overlay_plan=ov,
    )
    assert "plan_control_overlay" in sim.extra["controls_path_semantics"]


def test_geo_budget_optimizer_smoke() -> None:
    df, schema0 = generate_geo_panel(
        SyntheticGeoPanelSpec(n_geos=2, n_weeks=16, channels=("a",), betas=(0.45,))
    )
    schema = schema0
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
        cv=CVConfig(mode="rolling", n_splits=2, min_train_weeks=8, horizon_weeks=2),
        ridge_bo={"n_trials": 2},
    )
    ctx = _fit_ctx(df, schema, cfg)
    bau_pg = bau_baseline_per_geo_from_panel(df, schema)
    geos = sorted({str(x) for x in df[schema.geo_column].unique()})
    tot = total_spend_geo_plan(schema, bau_pg.spend_by_geo_channel or {})
    n = len(schema.channel_columns)
    channel_min = np.zeros(n, dtype=float)
    channel_max = np.ones(n, dtype=float) * 1e6
    nb = ctx.config.budget.model_copy(update={"geo_budget_enabled": True})
    nc = ctx.config.model_copy(update={"budget": nb})
    ctx_g = RidgeFitContext(
        panel=ctx.panel,
        schema=ctx.schema,
        config=nc,
        best_params=ctx.best_params,
        coef=ctx.coef,
        intercept=ctx.intercept,
    )
    res = optimize_budget_via_simulation(
        ctx_g,
        baseline_plan=bau_pg,
        current_spend=np.ones(n),
        total_budget=float(tot),
        channel_min=channel_min,
        channel_max=channel_max,
    )
    assert res.get("recommended_spend_plan_by_geo")
    opt_tot = sum(
        float(res["recommended_spend_plan_by_geo"][g][c])
        for g in geos
        for c in schema.channel_columns
    )
    assert abs(opt_tot - float(tot)) < 0.05
