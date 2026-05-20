"""LOG_LOG DGP: log(y) ~ log(media); level predictions via exp(link)."""

from __future__ import annotations

import numpy as np

from mmm.config.schema import CVConfig, Framework, MMMConfig, ModelForm
from mmm.features.design_matrix import build_design_matrix
from mmm.models.ridge_bo.trainer import RidgeBOMMMTrainer
from mmm.utils.synthetic import SyntheticGeoPanelSpec, generate_geo_panel


def test_log_log_level_recovery_within_tolerance() -> None:
    spec = SyntheticGeoPanelSpec(
        n_geos=4,
        n_weeks=60,
        channels=("search",),
        betas=(0.6,),
        decay=0.55,
        noise=0.02,
    )
    df, schema = generate_geo_panel(spec, seed=11)
    cfg = MMMConfig(
        framework=Framework.RIDGE_BO,
        model_form=ModelForm.LOG_LOG,
        data={
            "path": None,
            "geo_column": schema.geo_column,
            "week_column": schema.week_column,
            "target_column": schema.target_column,
            "channel_columns": list(schema.channel_columns),
        },
        cv=CVConfig(mode="rolling", n_splits=2, min_train_weeks=16, horizon_weeks=4),
        ridge_bo={"n_trials": 6},
    )
    tr = RidgeBOMMMTrainer(cfg, schema)
    tr.fit(df)
    yhat = tr.predict(df)
    y = df[schema.target_column].to_numpy(dtype=float)
    mape = float(np.mean(np.abs(y - yhat) / (np.abs(y) + 1e-9)))
    assert mape < 0.15, f"LOG_LOG level MAPE too high: {mape:.3f}"

    art = tr._artifacts
    bundle = build_design_matrix(
        df,
        schema,
        cfg,
        decay=art.best_params["decay"],
        hill_half=art.best_params["hill_half"],
        hill_slope=art.best_params["hill_slope"],
    )
    assert bundle.feature_lineage["target"]["form"] == "log_log"
