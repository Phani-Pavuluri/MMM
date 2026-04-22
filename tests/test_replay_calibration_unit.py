"""Synthetic replay: implied delta should track spend shift sign."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mmm.calibration.contracts import CalibrationUnit
from mmm.calibration.replay_lift import aggregate_replay_calibration_loss
from mmm.config.schema import DataConfig, Framework, MMMConfig, ModelForm
from mmm.data.schema import PanelSchema
from mmm.features.design_matrix import build_design_matrix
from mmm.models.ridge_bo.ridge import fit_ridge, predict_ridge


def test_replay_loss_positive_when_implied_mismatch():
    schema = PanelSchema("geo_id", "week", "revenue", ("c1",))
    rows = []
    for t in range(30):
        rows.append({"geo_id": "G0", "week": t, "revenue": 100.0 + t * 0.1, "c1": float(10 + t)})
    df = pd.DataFrame(rows)
    cfg = MMMConfig(
        framework=Framework.RIDGE_BO,
        model_form=ModelForm.SEMI_LOG,
        data=DataConfig(
            path=None,
            geo_column="geo_id",
            week_column="week",
            target_column="revenue",
            channel_columns=["c1"],
            control_columns=[],
        ),
    )
    bundle = build_design_matrix(df, schema, cfg, decay=0.5, hill_half=1.0, hill_slope=2.0)
    coef, intercept = fit_ridge(bundle.X, bundle.y_modeling, alpha=1.0)

    def predict_fn(dfp: pd.DataFrame) -> np.ndarray:
        b = build_design_matrix(dfp, schema, cfg, decay=0.5, hill_half=1.0, hill_slope=2.0)
        return np.exp(predict_ridge(b.X, coef, intercept))

    df_obs = df.copy()
    df_cf = df.copy()
    df_cf.loc[df_cf["week"] >= 20, "c1"] = df_cf.loc[df_cf["week"] >= 20, "c1"] * 0.5
    r = predict_fn(df_obs) - predict_fn(df_cf)
    implied = float(np.mean(r[df["week"].to_numpy() >= 20]))
    unit = CalibrationUnit(
        unit_id="u1",
        treated_channel_names=["c1"],
        observed_spend_frame=df_obs,
        counterfactual_spend_frame=df_cf,
        observed_lift=implied * 0.5,
        lift_se=1.0,
        target_kpi="revenue",
        geo_ids=["G0"],
        replay_estimand={
            "geo_scope": "listed",
            "geo_ids": ["G0"],
            "week_start": 20,
            "week_end": 29,
            "aggregation": "mean",
            "target_kpi_column": "revenue",
            "lift_scale": "mean_kpi_level_delta",
        },
    )
    loss, meta = aggregate_replay_calibration_loss(
        [unit], predict_fn, schema=schema, target_col=schema.target_column
    )
    assert loss > 0.0
    assert meta["n_units"] == 1
