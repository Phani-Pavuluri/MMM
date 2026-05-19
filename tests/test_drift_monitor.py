"""Drift monitor operational diagnostics."""

from __future__ import annotations

import pandas as pd

from mmm.config.schema import MMMConfig
from mmm.contracts.seed_resolution import resolve_seed_contract
from mmm.data.fingerprint import fingerprint_panel
from mmm.data.schema import PanelSchema
from mmm.evaluation.drift_monitor import build_drift_report


def _panel() -> tuple[pd.DataFrame, PanelSchema]:
    df = pd.DataFrame(
        {"g": ["a", "b"], "w": [1, 2], "y": [1.0, 2.0], "c1": [1.0, 2.0]},
    )
    return df, PanelSchema("g", "w", "y", ("c1",))


def test_no_drift_when_reference_matches() -> None:
    df, schema = _panel()
    cfg = MMMConfig(data={"channel_columns": ["c1"]})
    seeds = resolve_seed_contract(cfg)
    ref_fp = fingerprint_panel(df, schema, config=cfg, seed_resolution=seeds)
    rep = build_drift_report(
        panel=df,
        schema=schema,
        config=cfg,
        reference_fingerprint=ref_fp,
        reference_panel=df,
        seed_resolution=seeds,
    )
    assert rep["severity"] == "none"
    assert rep["recommended_action"] == "monitor"


def test_channel_drift_detected() -> None:
    df, schema = _panel()
    df2 = df.copy()
    df2["c1"] = df2["c1"] * 5.0
    cfg = MMMConfig(data={"channel_columns": ["c1"]})
    seeds = resolve_seed_contract(cfg)
    rep = build_drift_report(
        panel=df2,
        schema=schema,
        config=cfg,
        reference_panel=df,
        seed_resolution=seeds,
    )
    assert any(d["kind"] == "channel_spend_distribution" for d in rep["detected_drifts"])
