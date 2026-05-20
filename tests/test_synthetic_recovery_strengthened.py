"""Strengthened synthetic recovery: prediction, MMM structure, decision Δμ, null cases."""

from __future__ import annotations

import numpy as np

from mmm.config.schema import CVConfig, Framework, MMMConfig, ModelForm
from mmm.models.ridge_bo.trainer import RidgeBOMMMTrainer
from mmm.planning import bau_baseline_from_panel, simulate
from mmm.planning.context import ridge_context_from_fit
from mmm.utils.synthetic import SyntheticGeoPanelSpec, generate_geo_panel

# Documented screening thresholds (fail on implementation drift).
MAPE_NOISY_MAX = 0.20
COEF_POSITIVE_MIN = 0.0
DELTA_MU_SPEND_SHIFT_MIN = 0.0
PLACEBO_COEF_ABS_MAX = 0.08


def _ridge_cfg(schema, *, n_trials: int = 8) -> MMMConfig:
    return MMMConfig(
        framework=Framework.RIDGE_BO,
        model_form=ModelForm.SEMI_LOG,
        data={
            "path": None,
            "geo_column": schema.geo_column,
            "week_column": schema.week_column,
            "target_column": schema.target_column,
            "channel_columns": list(schema.channel_columns),
        },
        cv=CVConfig(mode="rolling", n_splits=2, min_train_weeks=20, horizon_weeks=4),
        ridge_bo={"n_trials": n_trials},
    )


def test_noisy_prediction_recovery() -> None:
    spec = SyntheticGeoPanelSpec(n_geos=4, n_weeks=70, channels=("search",), betas=(0.7,), noise=0.04)
    df, schema = generate_geo_panel(spec, seed=1)
    tr = RidgeBOMMMTrainer(_ridge_cfg(schema), schema)
    tr.fit(df)
    y = df[schema.target_column].to_numpy(float)
    yhat = tr.predict(df)
    mape = float(np.mean(np.abs(y - yhat) / (np.abs(y) + 1e-9)))
    assert mape < MAPE_NOISY_MAX


def test_known_positive_elasticity_sign() -> None:
    spec = SyntheticGeoPanelSpec(n_geos=4, n_weeks=80, channels=("search",), betas=(0.8,), noise=0.02)
    df, schema = generate_geo_panel(spec, seed=2)
    fit = RidgeBOMMMTrainer(_ridge_cfg(schema, n_trials=6), schema).fit(df)
    coef = float(np.asarray(fit["artifacts"].coef).ravel()[0])
    assert coef > COEF_POSITIVE_MIN, f"expected non-negative media coef, got {coef}"


def test_decision_delta_mu_positive_under_spend_increase() -> None:
    spec = SyntheticGeoPanelSpec(n_geos=3, n_weeks=50, channels=("search", "social"), betas=(0.5, 0.4))
    df, schema = generate_geo_panel(spec, seed=3)
    fit = RidgeBOMMMTrainer(_ridge_cfg(schema, n_trials=4), schema).fit(df)
    ctx = ridge_context_from_fit(df, schema, _ridge_cfg(schema, n_trials=4), fit)
    base = bau_baseline_from_panel(df, schema)
    med = float(df["search"].median())
    res = simulate({**base.spend_by_channel, "search": med * 1.25}, ctx, baseline_plan=base)
    assert res.delta_mu > DELTA_MU_SPEND_SHIFT_MIN


def test_placebo_null_near_zero_coef() -> None:
    spec = SyntheticGeoPanelSpec(n_geos=4, n_weeks=60, channels=("search",), betas=(0.0,), noise=0.05)
    df, schema = generate_geo_panel(spec, seed=4)
    fit = RidgeBOMMMTrainer(_ridge_cfg(schema, n_trials=4), schema).fit(df)
    coef = float(np.asarray(fit["artifacts"].coef).ravel()[0])
    assert abs(coef) < PLACEBO_COEF_ABS_MAX
