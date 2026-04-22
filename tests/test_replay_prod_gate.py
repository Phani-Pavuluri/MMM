"""Production replay unit requirements (estimand, lift scale, SE, frames)."""

from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from mmm.calibration.contracts import CalibrationUnit
from mmm.calibration.replay_prod_gate import assert_replay_production_ready
from mmm.config.schema import CalibrationConfig, DataConfig, Framework, MMMConfig, ModelForm, RunEnvironment
from mmm.data.schema import PanelSchema


_OBS = pd.DataFrame({"g": [1], "w": [1], "y": [1.0], "c1": [1.0]})
_BASE_UNIT = CalibrationUnit(
    unit_id="u1",
    treated_channel_names=["c1"],
    observed_spend_frame=_OBS,
    counterfactual_spend_frame=_OBS.copy(),
    observed_lift=0.05,
    lift_se=0.02,
    target_kpi="y",
    geo_ids=["g"],
    estimand="geo_time_ATT",
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
)


def _minimal_unit(**kwargs: object) -> CalibrationUnit:
    return replace(_BASE_UNIT, **kwargs)


def test_prod_replay_gate_ok_complete_unit() -> None:
    cfg = MMMConfig(
        framework=Framework.RIDGE_BO,
        model_form=ModelForm.SEMI_LOG,
        data=DataConfig(
            path=None,
            geo_column="g",
            week_column="w",
            channel_columns=["c1"],
            target_column="y",
        ),
        run_environment=RunEnvironment.PROD,
        calibration=CalibrationConfig(use_replay_calibration=True, replay_units_path="x.json"),
    )
    schema = PanelSchema("g", "w", "y", ("c1",))
    assert_replay_production_ready(cfg, [_minimal_unit()], schema=schema)


def test_prod_replay_gate_missing_estimand() -> None:
    cfg = MMMConfig(
        framework=Framework.RIDGE_BO,
        model_form=ModelForm.SEMI_LOG,
        data=DataConfig(
            path=None,
            geo_column="g",
            week_column="w",
            channel_columns=["c1"],
            target_column="y",
        ),
        run_environment=RunEnvironment.PROD,
        calibration=CalibrationConfig(use_replay_calibration=True, replay_units_path="x.json"),
    )
    schema = PanelSchema("g", "w", "y", ("c1",))
    with pytest.raises(ValueError, match="estimand"):
        assert_replay_production_ready(cfg, [_minimal_unit(estimand="")], schema=schema)


def test_prod_replay_gate_rejects_unknown_lift_scale() -> None:
    cfg = MMMConfig(
        framework=Framework.RIDGE_BO,
        model_form=ModelForm.SEMI_LOG,
        data=DataConfig(
            path=None,
            geo_column="g",
            week_column="w",
            channel_columns=["c1"],
            target_column="y",
        ),
        run_environment=RunEnvironment.PROD,
        calibration=CalibrationConfig(use_replay_calibration=True, replay_units_path="x.json"),
    )
    schema = PanelSchema("g", "w", "y", ("c1",))
    re = dict(_BASE_UNIT.replay_estimand or {})
    re["lift_scale"] = "legacy_relative_lift"
    bad = _minimal_unit(lift_scale="legacy_relative_lift", replay_estimand=re)
    with pytest.raises(ValueError, match="lift_scale"):
        assert_replay_production_ready(cfg, [bad], schema=schema)


def test_prod_replay_gate_non_prod_skips() -> None:
    cfg = MMMConfig(
        framework=Framework.RIDGE_BO,
        model_form=ModelForm.SEMI_LOG,
        data=DataConfig(path=None, channel_columns=["c1"], target_column="y"),
        run_environment=RunEnvironment.RESEARCH,
        calibration=CalibrationConfig(use_replay_calibration=True, replay_units_path="x.json"),
    )
    assert_replay_production_ready(cfg, [_minimal_unit(estimand="")])
