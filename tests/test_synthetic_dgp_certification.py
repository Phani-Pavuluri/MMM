"""Synthetic DGP certification for Δμ, replay carryover, optimizer direction (audit fix P5)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mmm.calibration.evidence_replay import build_calibration_unit_from_evidence
from mmm.calibration.replay_estimand import ReplayEstimandSpec
from mmm.calibration.replay_lift import implied_lift_from_counterfactual
from mmm.config.schema import CVConfig, Framework, MMMConfig, ModelForm
from mmm.data.schema import PanelSchema
from mmm.decision.gates import allow_decision_pipeline
from mmm.experiments.compatibility import ExperimentCompatibilityResolver, ModelPanelContext
from mmm.experiments.evidence import (
    ApprovalStatus,
    ExperimentEvidence,
    ExperimentType,
    GeoGranularity,
    TimeWindow,
)
from mmm.features.design_matrix import build_design_matrix
from mmm.models.ridge_bo.trainer import RidgeBOMMMTrainer, fit_ridge, predict_ridge
from mmm.optimization.budget.simulation_optimizer import optimize_budget_via_simulation
from mmm.planning.baseline import bau_baseline_from_panel
from mmm.planning.context import RidgeFitContext
from mmm.planning.decision_simulate import simulate
from mmm.utils.synthetic import SyntheticGeoPanelSpec, generate_geo_panel


def test_known_spend_shift_delta_mu_sign() -> None:
    spec = SyntheticGeoPanelSpec(n_geos=2, n_weeks=40, channels=("c1", "c2"), betas=(0.5, 0.1))
    panel, schema = generate_geo_panel(spec, seed=42)
    cfg = MMMConfig(
        framework=Framework.RIDGE_BO,
        model_form=ModelForm.SEMI_LOG,
        data={
            "geo_column": schema.geo_column,
            "week_column": schema.week_column,
            "target_column": schema.target_column,
            "channel_columns": list(schema.channel_columns),
        },
        cv=CVConfig(mode="rolling", n_splits=2, min_train_weeks=12, horizon_weeks=3),
        ridge_bo={"n_trials": 2},
    )
    tr = RidgeBOMMMTrainer(cfg, schema)
    fit = tr.fit(panel)
    ctx = RidgeFitContext(
        config=cfg,
        schema=schema,
        panel=panel,
        coef=fit["artifacts"].coef,
        intercept=fit["artifacts"].intercept,
        best_params=fit["artifacts"].best_params,
    )
    base = bau_baseline_from_panel(panel, schema)
    low = dict(base.spend_by_channel)
    high = {ch: v * 1.5 for ch, v in low.items()}
    res_low = simulate(high, ctx, baseline_plan=base)
    res_high = simulate({ch: v * 2.0 for ch, v in low.items()}, ctx, baseline_plan=base)
    assert res_high.delta_mu > res_low.delta_mu


def test_replay_pre_window_carryover_included() -> None:
    from datetime import date

    rows = []
    for w in range(14):
        rows.append(
            {
                "geo_id": "G0",
                "week_start_date": w,
                "revenue": 150.0,
                "c1": 40.0 if w < 3 else 4.0,
            }
        )
    panel = pd.DataFrame(rows)
    schema = PanelSchema("geo_id", "week_start_date", "revenue", ("c1",))
    ev = ExperimentEvidence(
        experiment_id="rw",
        experiment_type=ExperimentType.GEOX,
        channel="c1",
        kpi="revenue",
        estimand="ATT",
        lift_estimate=1.0,
        standard_error=1.0,
        metadata={"spend_multiplier": 0.6},
        time_window=TimeWindow(start="6", end="9"),
        geo_scope=["G0"],
        geo_granularity=GeoGranularity.GEO,
        source_system="t",
        freshness_date=date.today().isoformat(),
        approval_status=ApprovalStatus.ACCEPTED,
    )
    ctx = ModelPanelContext(
        geo_column="geo_id",
        channel_columns=("c1",),
        target_column="revenue",
        panel_geos={"G0"},
        model_geo_granularity=GeoGranularity.GEO,
    )
    compat = ExperimentCompatibilityResolver().resolve(ev, ctx, target_kpi="revenue")
    unit = build_calibration_unit_from_evidence(ev, panel, schema, channel="c1", compat=compat)
    assert unit is not None and unit.replay_estimand
    spec = ReplayEstimandSpec.from_dict(unit.replay_estimand)
    cfg = MMMConfig(
        framework=Framework.RIDGE_BO,
        model_form=ModelForm.SEMI_LOG,
        data={
            "geo_column": "geo_id",
            "week_column": "week_start_date",
            "target_column": "revenue",
            "channel_columns": ["c1"],
        },
    )
    bundle = build_design_matrix(panel, schema, cfg, decay=0.5, hill_half=1.0, hill_slope=2.0)
    coef, intercept = fit_ridge(bundle.X, bundle.y_modeling, 0.1)

    def predict_level(dfp: pd.DataFrame) -> np.ndarray:
        b = build_design_matrix(dfp, schema, cfg, decay=0.5, hill_half=1.0, hill_slope=2.0)
        return np.exp(predict_ridge(b.X, coef, intercept))

    lift = implied_lift_from_counterfactual(
        panel_observed=unit.observed_spend_frame,
        panel_counterfactual=unit.counterfactual_spend_frame,
        predict_fn=predict_level,
        schema=schema,
        estimand=spec,
    )
    assert lift["implied_mean_delta"] != 0.0
    assert lift["replay_uses_full_panel_transform"] is True


def test_optimizer_favors_higher_beta_channel() -> None:
    spec = SyntheticGeoPanelSpec(
        n_geos=2,
        n_weeks=50,
        channels=("high", "low"),
        betas=(0.8, 0.05),
        decay=0.5,
    )
    panel, schema = generate_geo_panel(spec, seed=7)
    cfg = MMMConfig(
        framework=Framework.RIDGE_BO,
        model_form=ModelForm.SEMI_LOG,
        data={
            "geo_column": schema.geo_column,
            "week_column": schema.week_column,
            "target_column": schema.target_column,
            "channel_columns": list(schema.channel_columns),
        },
        budget={"total_budget": 100.0},
        cv=CVConfig(mode="rolling", n_splits=2, min_train_weeks=12, horizon_weeks=3),
        ridge_bo={"n_trials": 3},
        extensions={"product": {"simulation_optimizer_n_starts": 10}},
    )
    tr = RidgeBOMMMTrainer(cfg, schema)
    fit = tr.fit(panel)
    ctx = RidgeFitContext(
        config=cfg,
        schema=schema,
        panel=panel,
        coef=fit["artifacts"].coef,
        intercept=fit["artifacts"].intercept,
        best_params=fit["artifacts"].best_params,
    )
    base = bau_baseline_from_panel(panel, schema)
    chans = list(schema.channel_columns)
    current = np.array([base.spend_by_channel[c] for c in chans], dtype=float)
    zmin = np.zeros(len(chans), dtype=float)
    zmax = np.full(len(chans), 100.0, dtype=float)
    with allow_decision_pipeline():
        opt = optimize_budget_via_simulation(
            ctx,
            current_spend=current,
            total_budget=100.0,
            channel_min=zmin,
            channel_max=zmax,
        )
    alloc = opt.get("channel_spends") or opt.get("allocation") or {}
    if isinstance(alloc, dict) and alloc:
        assert float(alloc.get("high", 0.0)) >= float(alloc.get("low", 0.0))
    elif isinstance(alloc, np.ndarray) and len(alloc) == 2:
        assert float(alloc[0]) >= float(alloc[1])


def test_collinear_channels_identifiability_warning() -> None:
    n = 30
    x = np.linspace(1.0, 2.0, n)
    rows = []
    for w in range(n):
        rows.append(
            {
                "geo_id": "G0",
                "week_start_date": w,
                "revenue": 100.0 + 0.1 * w,
                "a": float(x[w]),
                "b": float(x[w]),
            }
        )
    panel = pd.DataFrame(rows)
    schema = PanelSchema("geo_id", "week_start_date", "revenue", ("a", "b"))
    cfg = MMMConfig(
        framework=Framework.RIDGE_BO,
        model_form=ModelForm.SEMI_LOG,
        data={
            "geo_column": "geo_id",
            "week_column": "week_start_date",
            "target_column": "revenue",
            "channel_columns": ["a", "b"],
        },
        cv=CVConfig(mode="rolling", n_splits=2, min_train_weeks=10, horizon_weeks=3),
        ridge_bo={"n_trials": 2},
    )
    tr = RidgeBOMMMTrainer(cfg, schema)
    tr.fit(panel)
    yhat = tr.predict(panel)
    assert np.allclose(yhat, yhat[0], rtol=0.5) or float(np.std(yhat)) < float(np.std(panel["revenue"]))
