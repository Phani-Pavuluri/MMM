"""End-to-end artifact lifecycle roundtrip (local supported backend)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from mmm.artifacts.compatibility import verify_run_roundtrip
from mmm.artifacts.lifecycle import persist_training_artifacts
from mmm.artifacts.run_loader import load_run_json, load_run_lineage
from mmm.artifacts.stores.local import LocalArtifactStore
from mmm.config.schema import MMMConfig
from mmm.contracts.seed_resolution import resolve_seed_contract
from mmm.data.fingerprint import fingerprint_panel
from mmm.data.schema import PanelSchema


def _minimal_config() -> MMMConfig:
    return MMMConfig(data={"channel_columns": ["c1"], "control_columns": []})


def test_local_lifecycle_roundtrip(tmp_path: Path) -> None:
    df = pd.DataFrame({"g": ["a"], "w": [1], "y": [1.0], "c1": [2.0]})
    schema = PanelSchema("g", "w", "y", ("c1",))
    cfg = _minimal_config()
    seeds = resolve_seed_contract(cfg)
    fp = fingerprint_panel(df, schema, config=cfg, seed_resolution=seeds)

    store = LocalArtifactStore(tmp_path)
    store.start_run("e2e_run", metadata={"test": True})
    ext = {
        "data_fingerprint": fp,
        "decision_bundle": {
            "artifact_tier": "research",
            "framework": "ridge_bo",
            "decision_uncertainty": {"uncertainty_available": False},
        },
        "seed_resolution": seeds,
    }
    persist_training_artifacts(
        store,
        extension_report=ext,
        data_fingerprint=fp,
        seed_resolution=seeds,
        config_fingerprint={"sha256": "abc", "version": "test"},
        model_card_md="# Model card\n\n## Intended use\n\ntest\n",
    )
    store.end_run()

    rt = verify_run_roundtrip(store.run_path)
    assert rt["ok"], rt
    lineage = load_run_lineage(store.run_path)
    assert lineage["roundtrip"]["ok"]
    bundle = load_run_json(store.run_path, "decision_bundle")
    assert bundle["artifact_tier"] == "research"
    assert (store.run_path / "model_card.md").is_file()


def test_compatibility_check_store(tmp_path: Path) -> None:
    from mmm.artifacts.compatibility import check_store_capabilities

    store = LocalArtifactStore(tmp_path)
    caps = check_store_capabilities(store)
    assert all(caps.values())
