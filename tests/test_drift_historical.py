"""Historical drift comparison across prior runs."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from mmm.config.schema import MMMConfig
from mmm.contracts.seed_resolution import resolve_seed_contract
from mmm.data.fingerprint import fingerprint_panel
from mmm.data.schema import PanelSchema
from mmm.evaluation.drift_history import load_historical_reference, panel_distribution_snapshot
from mmm.evaluation.drift_monitor import build_drift_report


def _panel() -> tuple[pd.DataFrame, PanelSchema]:
    df = pd.DataFrame(
        {"g": ["a", "b"], "w": [1, 2], "y": [1.0, 2.0], "c1": [1.0, 2.0]},
    )
    return df, PanelSchema("g", "w", "y", ("c1",))


def test_load_historical_reference_from_prior_run(tmp_path: Path) -> None:
    df, schema = _panel()
    cfg = MMMConfig(data={"channel_columns": ["c1"]})
    seeds = resolve_seed_contract(cfg)
    fp = fingerprint_panel(df, schema, config=cfg, seed_resolution=seeds)
    dist = panel_distribution_snapshot(df, schema)
    report = {
        "data_fingerprint": fp,
        "drift_report": {"current_panel_distribution_snapshot": dist},
        "post_fit_validation": {"in_sample_rmse": 0.5},
    }
    (tmp_path / "extension_report.json").write_text(json.dumps(report), encoding="utf-8")
    hist = load_historical_reference(tmp_path)
    assert hist is not None
    assert hist["panel_fingerprint"]["sha256_combined"] or hist["panel_fingerprint"]


def test_drift_report_includes_historical_context(tmp_path: Path) -> None:
    df, schema = _panel()
    df2 = df.copy()
    df2["c1"] = df2["c1"] * 4.0
    cfg = MMMConfig(data={"channel_columns": ["c1"]})
    seeds = resolve_seed_contract(cfg)
    hist = {
        "source_run_dir": str(tmp_path),
        "panel_fingerprint": fingerprint_panel(df, schema, config=cfg, seed_resolution=seeds),
        "panel_distribution_snapshot": panel_distribution_snapshot(df, schema),
        "model_outputs": {"in_sample_rmse": 0.5},
    }
    rep = build_drift_report(
        panel=df2,
        schema=schema,
        config=cfg,
        historical_reference=hist,
        current_model_outputs={"in_sample_rmse": 1.2},
        seed_resolution=seeds,
    )
    assert rep["historical_reference"]["loaded"] is True
    assert rep["drift_severity"] in ("informational", "warning", "critical")
