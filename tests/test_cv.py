import numpy as np
import pandas as pd
import pytest

from mmm.config.schema import CVConfig, CVMode, CVSplitAxis
from mmm.data.schema import PanelSchema
from mmm.validation.cv import (
    GeoBlockedHoldoutCV,
    RollingCalendarWindowCV,
    RollingWindowCV,
    auto_cv_mode,
)


def _panel(n=120):
    rows = []
    for g in range(2):
        for t in range(n // 2):
            rows.append({"geo_id": f"G{g}", "week": t, "revenue": 1.0, "c1": 1.0, "c2": 1.0})
    return pd.DataFrame(rows)


def test_rolling_splits_nonempty():
    df = _panel()
    schema = PanelSchema("geo_id", "week", "revenue", ("c1", "c2"))
    cv = RollingWindowCV(n_splits=3, min_train_weeks=10, horizon_weeks=2, gap_weeks=0)
    splits = cv.split(df, schema)
    assert len(splits) >= 1
    for tr, va in splits:
        assert tr.dtype == bool
        assert not np.any(tr & va)


def test_calendar_rolling_splits_nonempty():
    df = _panel()
    schema = PanelSchema("geo_id", "week", "revenue", ("c1", "c2"))
    cv = RollingCalendarWindowCV(n_splits=3, min_train_weeks=10, horizon_weeks=2, gap_weeks=0)
    splits = cv.split(df, schema)
    assert len(splits) >= 1
    for tr, va in splits:
        assert len(tr) == len(df)
        assert not np.any(tr & va)


def test_calendar_cv_rejects_non_numeric_week():
    rows = [
        {"geo_id": "G0", "week": 0, "revenue": 1.0, "c1": 1.0, "c2": 1.0},
        {"geo_id": "G0", "week": "bad", "revenue": 1.0, "c1": 1.0, "c2": 1.0},
    ]
    df = pd.DataFrame(rows)
    schema = PanelSchema("geo_id", "week", "revenue", ("c1", "c2"))
    cv = RollingCalendarWindowCV(n_splits=2, min_train_weeks=1, horizon_weeks=1, gap_weeks=0)
    with pytest.raises(ValueError, match="Calendar CV"):
        cv.split(df, schema)


def test_geo_blocked_splits_no_overlap():
    df = _panel(n=120)
    schema = PanelSchema("geo_id", "week", "revenue", ("c1", "c2"))
    cv = GeoBlockedHoldoutCV(n_splits=2, seed=0, min_train_geos=1)
    splits = cv.split(df, schema)
    assert splits
    for tr, va in splits:
        assert not (tr & va).any()
        assert tr.sum() + va.sum() == len(df)


def test_auto_cv_respects_split_axis():
    df = _panel()
    schema = PanelSchema("geo_id", "week", "revenue", ("c1", "c2"))
    cfg_cal = CVConfig(
        mode=CVMode.ROLLING,
        n_splits=3,
        min_train_weeks=10,
        horizon_weeks=2,
        split_axis=CVSplitAxis.CALENDAR_WEEK,
    )
    assert len(auto_cv_mode(df, schema, cfg_cal).split(df, schema)) >= 1
    cfg_geo = CVConfig(
        mode=CVMode.ROLLING,
        n_splits=3,
        min_train_weeks=10,
        horizon_weeks=2,
        split_axis=CVSplitAxis.GEO_RANK,
    )
    assert len(auto_cv_mode(df, schema, cfg_geo).split(df, schema)) >= 1
    cfg_blk = CVConfig(
        mode=CVMode.ROLLING,
        n_splits=2,
        min_train_weeks=10,
        horizon_weeks=2,
        split_axis=CVSplitAxis.GEO_BLOCKED,
        geo_blocked_seed=1,
    )
    assert len(auto_cv_mode(df, schema, cfg_blk).split(df, schema)) >= 1
