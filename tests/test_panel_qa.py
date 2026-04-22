"""Panel QA structural checks and prod training gate."""

from __future__ import annotations

import pandas as pd
import pytest

from mmm.config.extensions import PanelQAConfig
from mmm.config.schema import DataConfig, Framework, MMMConfig, ModelForm, RunEnvironment
from mmm.data.panel_qa import assert_panel_qa_allows_training, run_panel_qa
from mmm.data.schema import PanelSchema


def _schema() -> PanelSchema:
    return PanelSchema("g", "w", "y", ("c1",))


def test_panel_qa_duplicate_geo_week_is_block() -> None:
    df = pd.DataFrame(
        {
            "g": ["a", "a"],
            "w": [1, 1],
            "y": [1.0, 2.0],
            "c1": [0.5, 0.5],
        }
    )
    rep = run_panel_qa(df, _schema(), PanelQAConfig())
    assert rep["max_severity"] == "block"
    assert any(i["code"] == "duplicate_geo_week_rows" for i in rep["issues"])


def test_panel_qa_disabled_is_info() -> None:
    df = pd.DataFrame({"g": ["a"], "w": [1], "y": [1.0], "c1": [0.5]})
    rep = run_panel_qa(df, _schema(), PanelQAConfig(enabled=False))
    assert rep["enabled"] is False
    assert rep["max_severity"] == "info"


def test_assert_panel_qa_prod_block_raises() -> None:
    df = pd.DataFrame(
        {
            "g": ["a", "a"],
            "w": [1, 1],
            "y": [1.0, 2.0],
            "c1": [0.5, 0.5],
        }
    )
    cfg = MMMConfig(
        framework=Framework.RIDGE_BO,
        model_form=ModelForm.SEMI_LOG,
        data=DataConfig(path=None, geo_column="g", week_column="w", channel_columns=["c1"], target_column="y"),
        run_environment=RunEnvironment.PROD,
        extensions={"panel_qa": {"prod_block_severity": "block"}},
    )
    with pytest.raises(PermissionError, match="panel QA"):
        assert_panel_qa_allows_training(df, _schema(), cfg)
