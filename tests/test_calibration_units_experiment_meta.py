"""Experiment platform metadata on calibration units."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from mmm.calibration.contracts import CalibrationUnit
from mmm.calibration.units_io import load_calibration_units_from_json, write_calibration_units_to_json


def test_replay_units_roundtrip_experiment_fields(tmp_path: Path) -> None:
    obs = pd.DataFrame({"g": [1], "w": [1], "y": [1.0], "c1": [1.0]})
    u = CalibrationUnit(
        unit_id="u1",
        treated_channel_names=["c1"],
        observed_spend_frame=obs,
        counterfactual_spend_frame=obs.copy(),
        observed_lift=0.1,
        lift_se=0.05,
        target_kpi="y",
        estimand="ATT",
        lift_scale="mean_kpi_level_delta",
        replay_estimand={
            "geo_scope": "listed",
            "geo_ids": ["1"],
            "week_start": 0,
            "week_end": 5,
            "aggregation": "mean",
            "target_kpi_column": "y",
            "lift_scale": "mean_kpi_level_delta",
        },
        experiment_id="exp-platform-42",
        payload_version="2026-04-21",
        payload_sha256="abc123",
        calibration_readiness="approved",
    )
    path = tmp_path / "units.json"
    write_calibration_units_to_json([u], path)
    loaded = load_calibration_units_from_json(path)
    assert len(loaded) == 1
    r = loaded[0]
    assert r.experiment_id == "exp-platform-42"
    assert r.payload_version == "2026-04-21"
    assert r.payload_sha256 == "abc123"
    assert r.calibration_readiness == "approved"
