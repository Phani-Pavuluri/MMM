"""Scenario overlay summary, linear ramp spend paths, hierarchical posterior draws."""

from __future__ import annotations

import numpy as np

from mmm.planning.control_overlay import ControlOverlaySpec, summarize_scenario_overlays
from mmm.planning import bau_baseline_from_panel, simulate
from mmm.planning.context import ridge_context_from_fit
from mmm.planning.posterior_planning import (
    delta_mu_draws_hierarchical_geo_beta,
    posterior_planning_gate,
    simulate_posterior,
)
from mmm.data.schema import PanelSchema
from mmm.planning.spend_path import PiecewiseSpendPath
from mmm.models.ridge_bo.trainer import RidgeBOMMMTrainer
from mmm.config.extensions import ExtensionSuiteConfig, ProductScopeConfig
from mmm.config.schema import CVConfig, DataConfig, Framework, MMMConfig, ModelForm
from mmm.utils.synthetic import SyntheticGeoPanelSpec, generate_geo_panel


def _ridge_ctx():
    df0, _schema0 = generate_geo_panel(
        SyntheticGeoPanelSpec(n_geos=2, n_weeks=24, channels=("a", "b"), betas=(0.35, 0.35))
    )
    schema = PanelSchema(
        geo_column=_schema0.geo_column,
        week_column=_schema0.week_column,
        target_column=_schema0.target_column,
        channel_columns=_schema0.channel_columns,
        control_columns=("promo_x",),
    )
    df = df0.copy()
    df["promo_x"] = 0.0
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
            control_columns=["promo_x"],
        ),
        cv=CVConfig(mode="rolling", n_splits=2, min_train_weeks=10, horizon_weeks=2),
        ridge_bo={"n_trials": 2},
        extensions=ext,
    )
    tr = RidgeBOMMMTrainer(cfg, schema)
    fit = tr.fit(df)
    return ridge_context_from_fit(df, schema, cfg, fit), df, schema, cfg


def test_summarize_scenario_overlays_lists_columns() -> None:
    import pandas as pd

    w0 = pd.Timestamp("2022-01-03")
    w1 = pd.Timestamp("2022-01-10")
    b = ControlOverlaySpec.from_dict(
        {
            "overrides": [
                {"geo": "G0", "week": w0, "column": "promo_x", "value": 1.0},
            ]
        }
    )
    p = ControlOverlaySpec.from_dict(
        {
            "overrides": [
                {"geo": "G0", "week": w1, "column": "promo_x", "value": 2.0},
            ]
        }
    )
    s = summarize_scenario_overlays(b, p)
    assert "promo_x" in s["all_overlay_columns"]


def test_piecewise_linear_ramp_non_empty_segments() -> None:
    path = PiecewiseSpendPath.from_channel_linear_ramp(
        week_start=1.0,
        week_end=10.0,
        spend_start_by_channel={"a": 1.0, "b": 1.0},
        spend_end_by_channel={"a": 2.0, "b": 3.0},
        n_steps=4,
    )
    assert len(path.segments) == 4
    assert path.segments[0].spend_by_channel["a"] < path.segments[-1].spend_by_channel["a"]


def test_posterior_planning_gate_rejects_both_draw_types() -> None:
    ctx, _df, _schema, cfg = _ridge_ctx()
    meta = {"posterior_diagnostics_ok": True, "posterior_predictive_ok": True}
    pack = {
        "kind": "hierarchical_geo_linear",
        "pooling": "partial",
        "alpha_draws": np.ones((3, 2)),
        "beta_draws": np.ones((3, 2, len(ctx.coef))),
        "n_draws": 3,
        "n_geo": 2,
        "n_coef": len(ctx.coef),
    }
    g = posterior_planning_gate(
        cfg,
        meta,
        linear_coef_draws=np.ones((3, len(ctx.coef))),
        hierarchical_draw_pack=pack,
    )
    assert g["allowed"] is False
    assert "ambiguous" in " ".join(g["reasons"])


def test_simulate_posterior_hierarchical_smoke() -> None:
    ctx, df, schema, cfg = _ridge_ctx()
    bau = bau_baseline_from_panel(df, schema)
    plan = {c: float(bau.spend_by_channel[c]) * 1.05 for c in schema.channel_columns}
    rng = np.random.default_rng(0)
    s = 25
    n_geo = 2
    p = len(ctx.coef)
    alpha = rng.normal(size=(s, n_geo))
    beta = rng.normal(loc=0.1, scale=0.05, size=(s, n_geo, p))
    pack = {
        "kind": "hierarchical_geo_linear",
        "pooling": "partial",
        "alpha_draws": alpha,
        "beta_draws": beta,
        "n_draws": s,
        "n_geo": n_geo,
        "n_coef": p,
    }
    meta = {"posterior_diagnostics_ok": True, "posterior_predictive_ok": True}
    out = simulate_posterior(
        plan,
        ctx,
        baseline_plan=bau,
        bayesian_fit_meta=meta,
        hierarchical_draw_pack=pack,
    )
    assert out.n_draws == s
    assert out.p10 <= out.p90


def test_delta_mu_hierarchical_matches_gate() -> None:
    ctx, df, schema, cfg = _ridge_ctx()
    bau = bau_baseline_from_panel(df, schema)
    plan = {c: float(bau.spend_by_channel[c]) for c in schema.channel_columns}
    rng = np.random.default_rng(1)
    s, n_geo, p = 15, 2, len(ctx.coef)
    pack = {
        "kind": "hierarchical_geo_linear",
        "pooling": "partial",
        "alpha_draws": rng.normal(size=(s, n_geo)),
        "beta_draws": rng.normal(size=(s, n_geo, p)),
        "n_draws": s,
        "n_geo": n_geo,
        "n_coef": p,
    }
    _, _, dlt = delta_mu_draws_hierarchical_geo_beta(
        ctx,
        baseline_plan=bau,
        spend_plan=plan,
        hierarchical_draw_pack=pack,
    )
    assert dlt.shape == (s,)


def test_simulate_includes_overlay_summary_in_extra() -> None:
    ctx, df, schema, cfg = _ridge_ctx()
    bau = bau_baseline_from_panel(df, schema)
    plan = {c: float(bau.spend_by_channel[c]) for c in schema.channel_columns}
    wk0 = df[schema.week_column].iloc[0]
    ov = ControlOverlaySpec.from_dict(
        {
            "overrides": [
                {
                    "geo": str(df[schema.geo_column].iloc[0]),
                    "week": wk0,
                    "column": "promo_x",
                    "value": 1.0,
                },
            ]
        }
    )
    sim = simulate(
        plan,
        ctx,
        baseline_plan=bau,
        control_overlay_baseline=ov,
        control_overlay_plan=ov,
    )
    assert "promo_x" in sim.extra["scenario_overlay_summary"]["all_overlay_columns"]
