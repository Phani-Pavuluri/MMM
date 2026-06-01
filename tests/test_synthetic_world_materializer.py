"""Synthetic world bundle materialization and validation (Phase 2A)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from mmm.validation.synthetic.materializer import materialize_world
from mmm.validation.synthetic.validator import validate_bundle, verify_checksums

REPO_ROOT = Path(__file__).resolve().parents[1]
WORLD_001 = REPO_ROOT / "validation" / "worlds" / "WORLD-001-baseline"


@pytest.fixture(scope="module")
def world_001_bundle() -> Path:
    assert (WORLD_001 / "world_truth.json").is_file()
    materialize_world(WORLD_001, overwrite=True)
    return WORLD_001


def test_materialize_world_001_complete_bundle(world_001_bundle: Path) -> None:
    for name in ("panel.parquet", "metadata.json", "checksums.json", "decision_truth.json"):
        assert (world_001_bundle / name).is_file(), name
    assert not (world_001_bundle / "replay_units.json").exists()


def test_materialize_produces_identical_checksums(world_001_bundle: Path) -> None:
    first = json.loads((world_001_bundle / "checksums.json").read_text(encoding="utf-8"))
    materialize_world(world_001_bundle, overwrite=True)
    second = json.loads((world_001_bundle / "checksums.json").read_text(encoding="utf-8"))
    assert first == second
    assert first["panel_sha256"] == second["panel_sha256"]


def test_validator_passes_world_001(world_001_bundle: Path) -> None:
    result = validate_bundle(world_001_bundle, max_level=3)
    assert result.passed, result.hard_failures


def test_validator_catches_missing_metadata_field(tmp_path: Path) -> None:
    bundle = tmp_path / "WORLD-001-baseline"
    shutil.copytree(WORLD_001, bundle)
    materialize_world(bundle, overwrite=True)
    meta = json.loads((bundle / "metadata.json").read_text(encoding="utf-8"))
    del meta["world_id"]
    (bundle / "metadata.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result = validate_bundle(bundle, max_level=1)
    assert not result.passed
    assert any("L1-006" in f for f in result.hard_failures)


def test_validator_catches_checksum_mismatch(tmp_path: Path) -> None:
    bundle = tmp_path / "WORLD-001-baseline"
    shutil.copytree(WORLD_001, bundle)
    materialize_world(bundle, overwrite=True)
    panel_path = bundle / "panel.parquet"
    df = pd.read_parquet(panel_path)
    df.iloc[0, df.columns.get_loc("revenue")] = 999.0
    df.to_parquet(panel_path, index=False)
    failures = verify_checksums(bundle)
    assert any("panel_sha256" in f for f in failures)


def test_panel_columns(world_001_bundle: Path) -> None:
    df = pd.read_parquet(world_001_bundle / "panel.parquet")
    assert set(df.columns) == {"geo_id", "week_start_date", "revenue", "search", "social"}
    assert len(df) == 2 * 8


def test_metadata_references_world(world_001_bundle: Path) -> None:
    meta = json.loads((world_001_bundle / "metadata.json").read_text(encoding="utf-8"))
    assert meta["world_id"] == "WORLD-001-baseline"
    assert meta["materialization_version"] == "materialize_v1.0.0"
    assert meta["checksum_version"] == "checksums_v1"


def test_decision_truth_index_no_coefficients(world_001_bundle: Path) -> None:
    text = (world_001_bundle / "decision_truth.json").read_text(encoding="utf-8").lower()
    assert "true_beta" not in text
    assert "bau_vs_bump_search" in text


def test_materialize_tmp_copy_reproducible(tmp_path: Path) -> None:
    bundle = tmp_path / "WORLD-001-baseline"
    shutil.copytree(WORLD_001, bundle, dirs_exist_ok=True)
    for f in ("panel.parquet", "metadata.json", "checksums.json", "decision_truth.json"):
        p = bundle / f
        if p.exists():
            p.unlink()
    materialize_world(bundle, overwrite=True)
    a = json.loads((bundle / "checksums.json").read_text(encoding="utf-8"))
    materialize_world(bundle, overwrite=True)
    b = json.loads((bundle / "checksums.json").read_text(encoding="utf-8"))
    assert a == b
