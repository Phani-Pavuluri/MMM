import pandas as pd
import yaml

from mmm.calibration.replay_etl import (
    SpendShiftSpec,
    build_replay_units_from_panel_shifts,
    ingest_validate_and_build_replay_units,
    load_spend_shift_specs,
)
from mmm.calibration.units_io import load_calibration_units_from_json, write_calibration_units_to_json
from mmm.config.schema import DataConfig
from mmm.data.loader import DatasetBuilder
from mmm.data.schema import PanelSchema


def test_replay_etl_build_and_json_roundtrip(tmp_path):
    schema = PanelSchema("geo", "week", "rev", ("search",))
    rows = []
    for w in range(10, 25):
        rows.append({"geo": "G1", "week": w, "rev": 100.0, "search": float(10 + w)})
    panel = pd.DataFrame(rows)
    cfg = DataConfig(
        path=None,
        geo_column="geo",
        week_column="week",
        target_column="rev",
        channel_columns=["search"],
    )
    panel = DatasetBuilder(cfg, schema).build(panel)
    shifts_yaml = tmp_path / "sh.yaml"
    shifts_yaml.write_text(
        yaml.dump(
            {
                "shifts": [
                    {
                        "unit_id": "u1",
                        "geo_ids": ["G1"],
                        "week_start": 12,
                        "week_end": 18,
                        "channel": "search",
                        "spend_multiplier": 0.9,
                        "observed_lift": 0.01,
                        "lift_se": 0.05,
                        "estimand": "geo_time_ATT",
                        "lift_scale": "mean_kpi_level_delta",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    specs = load_spend_shift_specs(shifts_yaml)
    units = build_replay_units_from_panel_shifts(panel, schema, specs, target_kpi="rev")
    assert len(units) == 1
    assert units[0].observed_spend_frame is not None
    out_json = tmp_path / "u.json"
    write_calibration_units_to_json(units, out_json)
    back = load_calibration_units_from_json(out_json)
    assert len(back) == 1
    assert back[0].unit_id == "u1"
    assert back[0].estimand == "geo_time_ATT"
    assert back[0].lift_scale == "mean_kpi_level_delta"


def test_ingest_validate_drops_bad_geo_shift():
    schema = PanelSchema("geo", "week", "rev", ("search",))
    rows = [{"geo": "G1", "week": w, "rev": 100.0, "search": float(10 + w)} for w in range(10, 25)]
    panel = pd.DataFrame(rows)
    cfg = DataConfig(
        path=None,
        geo_column="geo",
        week_column="week",
        target_column="rev",
        channel_columns=["search"],
    )
    panel = DatasetBuilder(cfg, schema).build(panel)
    bad = SpendShiftSpec(
        unit_id="bad",
        channel="search",
        spend_multiplier=0.9,
        geo_ids=["NOPE"],
        week_start=12,
        week_end=18,
    )
    units, reps = ingest_validate_and_build_replay_units(
        panel, schema, [bad], target_kpi="rev", expected_target_kpi="rev"
    )
    assert units == []
    assert not reps[0]["accepted"]
