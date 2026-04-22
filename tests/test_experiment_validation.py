import pandas as pd

from mmm.calibration.contracts import CalibrationUnit
from mmm.calibration.experiment_validation import (
    validate_calibration_unit_against_panel,
    validate_spend_shift_against_panel,
)
from mmm.calibration.replay_etl import SpendShiftSpec
from mmm.data.schema import PanelSchema


def test_validate_spend_shift_rejects_bad_geo():
    schema = PanelSchema("g", "w", "y", ("c1",))
    panel = pd.DataFrame({"g": ["A"], "w": [1], "y": [1.0], "c1": [1.0]})
    sp = SpendShiftSpec(
        unit_id="u1",
        channel="c1",
        spend_multiplier=0.9,
        geo_ids=["ZZZ"],
        week_start=1,
        week_end=1,
    )
    r = validate_spend_shift_against_panel(sp, panel, schema, expected_target_kpi="y", unit_kpi="y")
    assert not r.accepted
    assert any(i.code == "geo_mismatch" for i in r.issues)


def test_validate_spend_shift_rejects_empty_geo():
    schema = PanelSchema("g", "w", "y", ("c1",))
    panel = pd.DataFrame({"g": ["A"], "w": [1], "y": [1.0], "c1": [1.0]})
    sp = SpendShiftSpec(
        unit_id="u1",
        channel="c1",
        spend_multiplier=0.9,
        geo_ids=[],
        week_start=1,
        week_end=1,
    )
    r = validate_spend_shift_against_panel(sp, panel, schema, expected_target_kpi="y", unit_kpi="y")
    assert not r.accepted


def test_validate_calibration_unit_kpi_mismatch():
    schema = PanelSchema("g", "w", "rev", ("c1",))
    obs = pd.DataFrame({"g": ["A"], "w": [1], "rev": [1.0], "c1": [1.0]})
    u = CalibrationUnit(
        unit_id="u1",
        treated_channel_names=["c1"],
        observed_spend_frame=obs,
        counterfactual_spend_frame=obs.copy(),
        target_kpi="wrong_kpi",
        geo_ids=["A"],
    )
    panel = obs.copy()
    r = validate_calibration_unit_against_panel(u, panel, schema, expected_target_kpi="rev")
    assert not r.accepted
