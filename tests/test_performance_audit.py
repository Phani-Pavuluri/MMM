"""Performance audit artifact."""

from __future__ import annotations

import pandas as pd

from mmm.config.schema import MMMConfig
from mmm.data.schema import PanelSchema
from mmm.evaluation.performance_audit import build_performance_report


def test_performance_report_emits_timings() -> None:
    df = pd.DataFrame({"g": ["a", "b"], "w": [1, 2], "y": [1.0, 2.0], "c1": [1.0, 3.0]})
    schema = PanelSchema("g", "w", "y", ("c1",))
    cfg = MMMConfig(data={"channel_columns": ["c1"]})
    rep = build_performance_report(panel=df, schema=schema, config=cfg, fit_out=None)
    assert rep["diagnostic_only"] is True
    assert len(rep["timings"]) >= 1
    assert rep["optimization_hints"]
