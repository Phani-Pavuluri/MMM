import pandas as pd

from mmm.calibration.matching import match_experiments
from mmm.calibration.schema import ExperimentObservation


def test_match_filters_channel():
    exps = [
        ExperimentObservation(
            experiment_id="e1", geo_id="G0", channel="search", lift=0.1, lift_se=0.02
        ),
        ExperimentObservation(
            experiment_id="e2", geo_id="G0", channel="unknown", lift=0.1, lift_se=None
        ),
    ]
    m = match_experiments(exps, available_geos={"G0"}, available_channels={"search"}, match_levels=["geo"])
    assert len(m) == 1


def test_time_window_requires_overlap_with_panel_bounds():
    exps = [
        ExperimentObservation(
            experiment_id="e1",
            geo_id="G0",
            channel="search",
            lift=0.1,
            lift_se=0.02,
            start_week="2024-01-01",
            end_week="2024-01-07",
        ),
        ExperimentObservation(
            experiment_id="e2",
            geo_id="G0",
            channel="search",
            lift=0.1,
            lift_se=0.02,
            start_week="2025-01-01",
            end_week="2025-01-07",
        ),
    ]
    p_min = pd.Timestamp("2024-01-01")
    p_max = pd.Timestamp("2024-06-01")
    m = match_experiments(
        exps,
        available_geos={"G0"},
        available_channels={"search"},
        match_levels=["geo", "time_window"],
        panel_week_min=p_min,
        panel_week_max=p_max,
    )
    assert len(m) == 1 and m[0].obs.experiment_id == "e1"


def test_device_requires_allowed_set_when_match_levels_include_device():
    exps = [
        ExperimentObservation(
            experiment_id="e1",
            geo_id="G0",
            channel="search",
            lift=0.1,
            lift_se=0.02,
            device="mobile",
        ),
    ]
    m_empty = match_experiments(
        exps,
        available_geos={"G0"},
        available_channels={"search"},
        match_levels=["geo", "device"],
        allowed_devices=None,
    )
    assert len(m_empty) == 0
    m_ok = match_experiments(
        exps,
        available_geos={"G0"},
        available_channels={"search"},
        match_levels=["geo", "device"],
        allowed_devices={"mobile", "desktop"},
    )
    assert len(m_ok) == 1
