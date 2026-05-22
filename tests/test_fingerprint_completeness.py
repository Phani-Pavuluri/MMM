"""Fingerprint v2 includes schema, config, seeds; ignores volatile metadata."""

from __future__ import annotations

import pandas as pd

from mmm.config.schema import MMMConfig, ModelForm
from mmm.contracts.seed_resolution import resolve_seed_contract
from mmm.data.fingerprint import fingerprint_panel
from mmm.data.schema import PanelSchema


def _panel() -> tuple[pd.DataFrame, PanelSchema]:
    df = pd.DataFrame(
        {
            "g": ["a", "a", "b", "b"],
            "w": [1, 2, 1, 2],
            "y": [10.0, 11.0, 9.0, 10.0],
            "c1": [1.0, 2.0, 1.5, 2.5],
            "ctrl": [0.1, 0.2, 0.1, 0.2],
        }
    )
    schema = PanelSchema("g", "w", "y", ("c1",), ("ctrl",))
    return df, schema


def test_control_change_alters_fingerprint() -> None:
    df, schema = _panel()
    cfg = MMMConfig(data={"channel_columns": ["c1"], "control_columns": ["ctrl"]})
    seeds = resolve_seed_contract(cfg)
    fp1 = fingerprint_panel(df, schema, config=cfg, seed_resolution=seeds)
    df2 = df.copy()
    df2["ctrl"] = df2["ctrl"] + 1.0
    fp2 = fingerprint_panel(df2, schema, config=cfg, seed_resolution=seeds)
    assert fp1["sha256_combined"] != fp2["sha256_combined"]


def test_seed_change_alters_fingerprint() -> None:
    df, schema = _panel()
    cfg1 = MMMConfig(random_seed=1, data={"channel_columns": ["c1"], "control_columns": ["ctrl"]})
    s1 = resolve_seed_contract(cfg1)
    cfg2 = MMMConfig(random_seed=2, data={"channel_columns": ["c1"], "control_columns": ["ctrl"]})
    s2 = resolve_seed_contract(cfg2)
    fp1 = fingerprint_panel(df, schema, config=cfg1, seed_resolution=s1)
    fp2 = fingerprint_panel(df, schema, config=cfg2, seed_resolution=s2)
    assert fp1["sha256_combined"] != fp2["sha256_combined"]


def test_channel_change_alters_fingerprint() -> None:
    df, schema = _panel()
    cfg = MMMConfig(data={"channel_columns": ["c1"], "control_columns": ["ctrl"]})
    seeds = resolve_seed_contract(cfg)
    fp1 = fingerprint_panel(df, schema, config=cfg, seed_resolution=seeds)
    df2 = df.copy()
    df2["c1"] = df2["c1"] * 2.0
    fp2 = fingerprint_panel(df2, schema, config=cfg, seed_resolution=seeds)
    assert fp1["sha256_combined"] != fp2["sha256_combined"]


def test_run_id_does_not_alter_fingerprint() -> None:
    df, schema = _panel()
    cfg1 = MMMConfig(run_id="run-a", data={"channel_columns": ["c1"], "control_columns": ["ctrl"]})
    cfg2 = MMMConfig(run_id="run-b", data={"channel_columns": ["c1"], "control_columns": ["ctrl"]})
    s = resolve_seed_contract(cfg1)
    resolve_seed_contract(cfg2)
    fp1 = fingerprint_panel(df, schema, config=cfg1, seed_resolution=s)
    fp2 = fingerprint_panel(df, schema, config=cfg2, seed_resolution=s)
    assert fp1["sha256_combined"] == fp2["sha256_combined"]


def test_fingerprint_details_artifact() -> None:
    df, schema = _panel()
    cfg = MMMConfig(
        model_form=ModelForm.SEMI_LOG,
        data={"channel_columns": ["c1"], "control_columns": ["ctrl"], "data_version_id": "snap-1"},
    )
    seeds = resolve_seed_contract(cfg)
    fp = fingerprint_panel(df, schema, config=cfg, seed_resolution=seeds)
    details = fp["fingerprint_details"]
    assert details["fingerprint_version"] == "fingerprint_v2"
    assert "control_columns" in details["included_fields"]
    assert "run_id" in details["omitted_fields"]
