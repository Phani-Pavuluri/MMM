"""Ridge+BO wall-clock budget regression (Phase 4 — single-node path)."""

from __future__ import annotations

import time

from mmm.config.schema import CVConfig, Framework, MMMConfig, ModelForm
from mmm.models.ridge_bo.trainer import RidgeBOMMMTrainer
from mmm.utils.synthetic import SyntheticGeoPanelSpec, generate_geo_panel


def test_ridge_bo_small_panel_fit_finishes_under_wall_clock_budget() -> None:
    df, schema = generate_geo_panel(SyntheticGeoPanelSpec(n_geos=2, n_weeks=30, channels=("a", "b"), betas=(0.2, 0.2)))
    cfg = MMMConfig(
        framework=Framework.RIDGE_BO,
        model_form=ModelForm.SEMI_LOG,
        data={
            "path": None,
            "geo_column": schema.geo_column,
            "week_column": schema.week_column,
            "target_column": schema.target_column,
            "channel_columns": list(schema.channel_columns),
        },
        cv=CVConfig(mode="rolling", n_splits=2, min_train_weeks=10, horizon_weeks=2),
        ridge_bo={"n_trials": 4},
    )
    tr = RidgeBOMMMTrainer(cfg, schema)
    t0 = time.perf_counter()
    out = tr.fit(df)
    elapsed = time.perf_counter() - t0
    assert elapsed < 90.0, f"fit too slow: {elapsed:.1f}s"
    tel = out.get("ridge_bo_telemetry") or {}
    assert tel.get("n_hyperparameter_evaluations", 0) >= 1
    assert float(tel.get("total_eval_wall_time_ms", 0.0)) > 0.0
