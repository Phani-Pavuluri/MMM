"""Control governance diagnostics — guidance only."""

from __future__ import annotations

import pandas as pd

from mmm.config.schema import MMMConfig
from mmm.data.schema import PanelSchema
from mmm.governance.control_diagnostics import build_control_governance_report


def test_control_governance_never_mutates_schema() -> None:
    df = pd.DataFrame({"g": ["a"], "w": [1], "y": [1.0], "c1": [1.0], "conversion_rate": [0.1]})
    schema = PanelSchema("g", "w", "y", ("c1",), control_columns=("conversion_rate",))
    cfg = MMMConfig(data={"channel_columns": ["c1"], "control_columns": ["conversion_rate"]})
    before = list(schema.control_columns)
    rep = build_control_governance_report(config=cfg, schema=schema, panel=df)
    assert list(schema.control_columns) == before
    assert rep["policy_note"]
    assert rep["potentially_post_treatment"]
    assert "never auto-inserted" in rep["policy_note"].lower()
