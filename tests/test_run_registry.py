"""Accepted-run registry for historical drift."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from mmm.config.schema import MMMConfig
from mmm.contracts.seed_resolution import resolve_seed_contract
from mmm.data.fingerprint import fingerprint_panel
from mmm.data.schema import PanelSchema
from mmm.evaluation.drift_monitor import build_drift_report
from mmm.evaluation.run_registry import AcceptedRunRegistry


def _write_prior_run(tmp_path: Path) -> Path:
    df = pd.DataFrame({"g": ["a"], "w": [1], "y": [1.0], "c1": [1.0]})
    schema = PanelSchema("g", "w", "y", ("c1",))
    cfg = MMMConfig(data={"channel_columns": ["c1"]})
    seeds = resolve_seed_contract(cfg)
    fp = fingerprint_panel(df, schema, config=cfg, seed_resolution=seeds)
    run_dir = tmp_path / "prior_run"
    run_dir.mkdir()
    report = {
        "data_fingerprint": fp,
        "drift_report": {"current_panel_distribution_snapshot": {"media": {"c1": {"mean": 1.0}}}},
        "post_fit_validation": {"in_sample_rmse": 0.4},
    }
    (run_dir / "extension_report.json").write_text(json.dumps(report), encoding="utf-8")
    return run_dir


def test_registry_excludes_same_run(tmp_path: Path) -> None:
    prior = _write_prior_run(tmp_path)
    reg = AcceptedRunRegistry(tmp_path / "registry")
    reg.register_accepted_run(prior, run_id="prior_run")

    df2 = pd.DataFrame({"g": ["a"], "w": [1], "y": [1.0], "c1": [5.0]})
    schema = PanelSchema("g", "w", "y", ("c1",))
    cfg = MMMConfig(data={"channel_columns": ["c1"]}, run_id="current_run")
    seeds = resolve_seed_contract(cfg)
    hist = reg.historical_reference_for_drift(exclude_run_id="current_run")
    assert hist is not None
    rep = build_drift_report(
        panel=df2,
        schema=schema,
        config=cfg,
        historical_reference=hist,
        current_model_outputs={"in_sample_rmse": 0.9},
        seed_resolution=seeds,
        current_run_id="current_run",
    )
    assert rep.get("drift_trend") in ("gradual_drift", "sudden_drift", "stable")
    assert rep["historical_reference"]["loaded"] is True
