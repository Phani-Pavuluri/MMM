"""Replay train/holdout split: reproducibility, BO objective, holdout diagnostics."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd

from mmm.calibration.contracts import CalibrationUnit
from mmm.calibration.replay_split import split_replay_units
from mmm.calibration.units_io import write_calibration_units_to_json
from mmm.config.schema import CVConfig, Framework, MMMConfig
from mmm.evaluation.replay_holdout_validation import build_replay_holdout_validation
from mmm.models.ridge_bo.trainer import RidgeBOMMMTrainer
from mmm.utils.synthetic import SyntheticGeoPanelSpec, generate_geo_panel


def _units(n: int) -> list[CalibrationUnit]:
    out: list[CalibrationUnit] = []
    for i in range(n):
        obs = pd.DataFrame({"geo_id": ["G0"], "week_start_date": [1], "revenue": [100.0], "search": [10.0]})
        cf = obs.copy()
        out.append(
            CalibrationUnit(
                unit_id=f"u{i}",
                treated_channel_names=["search"],
                observed_spend_frame=obs,
                counterfactual_spend_frame=cf,
                observed_lift=0.01,
                lift_se=0.1,
                target_kpi="revenue",
                replay_estimand={
                    "geo_scope": "listed",
                    "geo_ids": ["G0"],
                    "week_start": 1,
                    "week_end": 1,
                    "aggregation": "mean",
                    "target_kpi_column": "revenue",
                    "lift_scale": "mean_kpi_level_delta",
                },
            )
        )
    return out


def test_split_reproducibility() -> None:
    units = _units(6)
    a_tr, a_ho, _ = split_replay_units(units, holdout_fraction=0.33, min_train_units=2, min_holdout_units=1, seed=7)
    b_tr, b_ho, _ = split_replay_units(units, holdout_fraction=0.33, min_train_units=2, min_holdout_units=1, seed=7)
    assert [u.unit_id for u in a_tr] == [u.unit_id for u in b_tr]
    assert [u.unit_id for u in a_ho] == [u.unit_id for u in b_ho]


def test_too_few_units_graceful() -> None:
    units = _units(2)
    tr, ho, meta = split_replay_units(
        units, holdout_fraction=0.5, min_train_units=2, min_holdout_units=2, seed=0
    )
    assert tr == []
    assert ho == []
    assert meta.get("holdout_not_available_reason")


def test_bo_uses_train_replay_only_when_split_enabled(tmp_path: Path) -> None:
    units = _units(4)
    path = tmp_path / "units.json"
    write_calibration_units_to_json(units, path)
    spec = SyntheticGeoPanelSpec(n_geos=2, n_weeks=40, channels=("search",), betas=(0.5,))
    df, schema = generate_geo_panel(spec, seed=1)
    cfg = MMMConfig(
        framework=Framework.RIDGE_BO,
        data={
            "path": None,
            "geo_column": schema.geo_column,
            "week_column": schema.week_column,
            "target_column": schema.target_column,
            "channel_columns": list(schema.channel_columns),
        },
        cv=CVConfig(mode="rolling", n_splits=2, min_train_weeks=15, horizon_weeks=3),
        ridge_bo={"n_trials": 2},
        calibration={
            "use_replay_calibration": True,
            "replay_units_path": str(path),
            "use_replay_holdout_split": True,
            "replay_holdout_fraction": 0.5,
            "replay_holdout_min_train_units": 1,
            "replay_holdout_min_holdout_units": 1,
        },
        random_seed=99,
    )
    tr = RidgeBOMMMTrainer(cfg, schema)
    train_ids = {u.unit_id for u in tr._replay_units_train}
    hold_ids = {u.unit_id for u in tr._replay_units_holdout}
    assert train_ids and hold_ids and train_ids.isdisjoint(hold_ids)

    captured: list[list] = []

    def _capture(units, predict_level, **kwargs):
        captured.append(list(units))
        return 0.1, {"n_units": len(units)}

    with patch("mmm.models.ridge_bo.trainer.aggregate_replay_calibration_loss", side_effect=_capture):
        tr.fit(df)
    assert captured
    assert {u.unit_id for u in captured[0]} == train_ids


def test_holdout_validation_emitted(tmp_path: Path) -> None:
    units = _units(4)
    path = tmp_path / "units.json"
    write_calibration_units_to_json(units, path)
    spec = SyntheticGeoPanelSpec(n_geos=2, n_weeks=35, channels=("search",), betas=(0.4,))
    df, schema = generate_geo_panel(spec, seed=2)
    cfg = MMMConfig(
        framework=Framework.RIDGE_BO,
        data={
            "path": None,
            "geo_column": schema.geo_column,
            "week_column": schema.week_column,
            "target_column": schema.target_column,
            "channel_columns": list(schema.channel_columns),
        },
        cv=CVConfig(mode="rolling", n_splits=2, min_train_weeks=12, horizon_weeks=3),
        ridge_bo={"n_trials": 2},
        calibration={
            "use_replay_calibration": True,
            "replay_units_path": str(path),
            "use_replay_holdout_split": True,
        },
        random_seed=1,
    )
    fit = RidgeBOMMMTrainer(cfg, schema).fit(df)
    rhv = build_replay_holdout_validation(df, schema, cfg, fit)
    assert rhv.get("n_train_replay_units", 0) >= 1
    assert "train_replay_loss" in rhv
    if rhv.get("status") == "ok" and rhv.get("n_holdout_replay_units", 0) > 0:
        assert rhv.get("holdout_replay_loss") is not None
