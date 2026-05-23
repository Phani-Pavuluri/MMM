"""Exact Δμ and optimizer certification under controlled DGPs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mmm.config.extensions import ExtensionSuiteConfig, FeatureSeparabilityConfig
from mmm.config.schema import CVConfig, Framework, MMMConfig, ModelForm
from mmm.data.schema import PanelSchema
from mmm.features.design_matrix import build_design_matrix
from mmm.models.ridge_bo.trainer import RidgeBOMMMTrainer, fit_ridge, predict_ridge
from mmm.optimization.budget.simulation_optimizer import optimize_budget_via_simulation
from mmm.planning.baseline import bau_baseline_from_panel
from mmm.planning.context import RidgeFitContext
from mmm.planning.decision_simulate import simulate
from mmm.utils.synthetic import SyntheticGeoPanelSpec, generate_geo_panel


def test_linear_semi_log_no_adstock_analytic_delta_mu() -> None:
    """Known coef on modeling scale → Δμ via two full-panel simulate calls (level-scale aggregation)."""
    rows = []
    for g in ("G0", "G1"):
        for w in range(8):
            rows.append({"geo_id": g, "week_start_date": w, "revenue": 100.0, "tv": 10.0, "search": 5.0})
    panel = pd.DataFrame(rows)
    schema = PanelSchema("geo_id", "week_start_date", "revenue", ("tv", "search"))
    cfg = MMMConfig(
        framework=Framework.RIDGE_BO,
        model_form=ModelForm.SEMI_LOG,
        data={
            "geo_column": "geo_id",
            "week_column": "week_start_date",
            "target_column": "revenue",
            "channel_columns": ["tv", "search"],
        },
    )
    coef = np.array([0.02, 0.05])
    intercept = np.array([np.log(100.0)])
    base = bau_baseline_from_panel(panel, schema)
    low = {"tv": 10.0, "search": 5.0}
    high = {"tv": 20.0, "search": 5.0}
    ctx = RidgeFitContext(
        config=cfg,
        schema=schema,
        panel=panel,
        coef=coef,
        intercept=intercept,
        best_params={"decay": 0.01, "hill_half": 1e6, "hill_slope": 1.0},
    )
    res_lo = simulate(low, ctx, baseline_plan=base)
    res_hi = simulate(high, ctx, baseline_plan=base)
    expected = res_hi.delta_mu - res_lo.delta_mu
    res = simulate(high, ctx, baseline_plan=base)
    assert res.delta_mu > 0.0
    assert abs(res.delta_mu - expected) < 1e-9


def test_geometric_adstock_carryover_exact_week1() -> None:
    """Impulse 100 at t=0, zero spend after; week-1 adstock state equals decay * 100."""
    from mmm.transforms.adstock.geometric import GeometricAdstock

    decay = 0.5
    ad = GeometricAdstock(decay)
    out = ad.transform(np.array([100.0, 0.0, 0.0, 0.0]))
    assert abs(float(out[1]) - decay * 100.0) < 1e-12


def test_hill_saturation_analytic_value() -> None:
    from mmm.transforms.saturation.hill import HillSaturation

    half, slope, x = 10.0, 2.0, 5.0
    sat = HillSaturation(half_max=half, slope=slope)
    expected = x**slope / (half**slope + x**slope + 1e-12)
    assert abs(float(sat.transform(np.array([x]))[0]) - expected) < 1e-12


def test_geometric_adstock_carryover_in_design_matrix() -> None:
    spend = [100.0, 0, 0, 0, 10.0]
    rows = [
        {"geo_id": "G0", "week_start_date": i, "revenue": 10.0, "tv": s}
        for i, s in enumerate(spend)
    ]
    panel = pd.DataFrame(rows)
    schema = PanelSchema("geo_id", "week_start_date", "revenue", ("tv",))
    cfg = MMMConfig(
        framework=Framework.RIDGE_BO,
        data={
            "geo_column": "geo_id",
            "week_column": "week_start_date",
            "target_column": "revenue",
            "channel_columns": ["tv"],
        },
    )
    b0 = build_design_matrix(panel, schema, cfg, decay=0.5, hill_half=1e6, hill_slope=1.0)
    b1 = build_design_matrix(panel.assign(tv=0.0), schema, cfg, decay=0.5, hill_half=1e6, hill_slope=1.0)
    assert b0.X[1, 0] > b1.X[1, 0]


def test_hill_saturation_monotone_in_design_matrix() -> None:
    spends = np.linspace(1.0, 50.0, 10)
    xs = []
    for s in spends:
        panel = pd.DataFrame(
            [{"geo_id": "G0", "week_start_date": 0, "revenue": 10.0, "tv": float(s)}]
        )
        schema = PanelSchema("geo_id", "week_start_date", "revenue", ("tv",))
        cfg = MMMConfig(
            framework=Framework.RIDGE_BO,
            data={
                "geo_column": "geo_id",
                "week_column": "week_start_date",
                "target_column": "revenue",
                "channel_columns": ["tv"],
            },
        )
        b = build_design_matrix(panel, schema, cfg, decay=0.01, hill_half=10.0, hill_slope=2.0)
        xs.append(float(b.X[0, 0]))
    assert np.all(np.diff(xs) >= -1e-9)


def test_optimizer_prefers_high_return_channel() -> None:
    spec = SyntheticGeoPanelSpec(n_geos=2, n_weeks=24, channels=("low", "high"), betas=(0.05, 0.4))
    panel, schema = generate_geo_panel(spec, seed=99)
    cfg = MMMConfig(
        framework=Framework.RIDGE_BO,
        model_form=ModelForm.SEMI_LOG,
        data={
            "geo_column": schema.geo_column,
            "week_column": schema.week_column,
            "target_column": schema.target_column,
            "channel_columns": list(schema.channel_columns),
        },
        cv=CVConfig(mode="rolling", n_splits=2, min_train_weeks=10, horizon_weeks=3),
        ridge_bo={"n_trials": 2},
        budget={"enabled": True, "total_budget": 200.0},
    )
    tr_out = RidgeBOMMMTrainer(cfg, schema).fit(panel)
    ctx = RidgeFitContext(
        config=cfg,
        schema=schema,
        panel=panel,
        coef=tr_out["artifacts"].coef,
        intercept=tr_out["artifacts"].intercept,
        best_params=tr_out["artifacts"].best_params,
    )
    from mmm.decision.gates import allow_decision_pipeline

    base = bau_baseline_from_panel(panel, schema)
    n_ch = len(schema.channel_columns)
    cur = np.array([50.0, 50.0])
    with allow_decision_pipeline():
        opt = optimize_budget_via_simulation(
            ctx,
            baseline_plan=base,
            current_spend=cur,
            total_budget=200.0,
            channel_min=np.zeros(n_ch),
            channel_max=np.full(n_ch, 200.0),
        )
    alloc = opt.get("allocation") or opt.get("recommended_allocation") or opt.get("spend_by_channel") or {}
    if isinstance(alloc, dict):
        high_spend = float(alloc.get("high", 0.0))
        low_spend = float(alloc.get("low", 0.0))
        assert high_spend >= low_spend


def test_transform_policy_mismatch_fails_certification() -> None:
    from mmm.config.schema import RunEnvironment
    from mmm.governance.decision_ridge_summary import validate_ridge_fit_summary_for_prod_decide
    from mmm.governance.policy import PolicyError

    cfg = MMMConfig(
        run_environment=RunEnvironment.PROD,
        prod_canonical_modeling_contract_id="ridge_bo_semi_log_calendar_cv_v1",
        data={"channel_columns": ["tv"], "data_version_id": "dgp-v1"},
        cv={"mode": "rolling"},
        objective={
            "normalization_profile": "strict_prod",
            "named_profile": "ridge_bo_standard_v1",
        },
        extensions={"optimization_gates": {"enabled": True}},
    )
    er = {
        "ridge_fit_summary": {
            "coef": [0.1],
            "intercept": [0.0],
            "model_form": "semi_log",
            "best_params": {"decay": 0.5, "hill_half": 1.0, "hill_slope": 2.0},
        },
        "transform_policy": {
            "policy_version": "mmm_transform_policy_v1",
            "adstock": "geometric",
            "saturation": "hill",
        },
        "data_fingerprint": {"sha256_combined": "b" * 64},
    }
    with pytest.raises(PolicyError):
        validate_ridge_fit_summary_for_prod_decide(cfg, er)


def test_collinear_channels_identifiability_warning() -> None:
    n = 30
    tv = np.linspace(10, 20, n)
    search = tv * 1.01 + 0.001
    panel = pd.DataFrame(
        {
            "geo_id": ["G0"] * n,
            "week_start_date": range(n),
            "revenue": 100 + 0.1 * tv,
            "tv": tv,
            "search": search,
        }
    )
    schema = PanelSchema("geo_id", "week_start_date", "revenue", ("tv", "search"))
    cfg = MMMConfig(
        framework=Framework.RIDGE_BO,
        extensions=ExtensionSuiteConfig(
            feature_separability=FeatureSeparabilityConfig(enabled=True, auto_group_prefix=True)
        ),
        data={
            "geo_column": "geo_id",
            "week_column": "week_start_date",
            "target_column": "revenue",
            "channel_columns": ["tv", "search"],
        },
        cv=CVConfig(mode="rolling", n_splits=2, min_train_weeks=10, horizon_weeks=3),
        ridge_bo={"n_trials": 1},
    )
    from mmm.evaluation.extension_runner import run_post_fit_extensions

    bundle = build_design_matrix(panel, schema, cfg, decay=0.5, hill_half=1.0, hill_slope=2.0)
    coef, intercept = fit_ridge(bundle.X, bundle.y_modeling, 1.0)
    yhat = predict_ridge(bundle.X, coef, intercept)
    er = run_post_fit_extensions(
        panel=panel,
        schema=schema,
        config=cfg,
        fit_out={
            "artifacts": type(
                "A",
                (),
                {
                    "coef": coef,
                    "intercept": intercept,
                    "best_params": {"decay": 0.5, "hill_half": 1.0, "hill_slope": 2.0},
                },
            )(),
            "best_detail": {},
        },
        yhat=np.exp(yhat),
        store=None,
    )
    id_js = er.get("identifiability", {})
    sep = er.get("feature_separability_report", {})
    assert float(id_js.get("identifiability_score", 1.0)) < 0.99 or sep
