"""WORLD-002 replay bundle materialization and loader compatibility (Phase 2B)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from mmm.calibration.replay_estimand import ReplayEstimandSpec
from mmm.calibration.units_io import load_calibration_units_from_json
from mmm.validation.synthetic.materializer import materialize_world
from mmm.validation.synthetic.validator import validate_bundle, verify_checksums

REPO_ROOT = Path(__file__).resolve().parents[1]
WORLD_002 = REPO_ROOT / "validation" / "worlds" / "WORLD-002-replay"


@pytest.fixture(scope="module")
def world_002_bundle() -> Path:
    assert (WORLD_002 / "world_truth.json").is_file()
    materialize_world(WORLD_002, overwrite=True)
    return WORLD_002


def test_materialize_world_002_writes_replay_units(world_002_bundle: Path) -> None:
    assert (world_002_bundle / "replay_units.json").is_file()
    assert (world_002_bundle / "panel.parquet").is_file()
    assert not (world_002_bundle / "decision_truth.json").exists()


def test_replay_checksum_stable_on_rematerialize(world_002_bundle: Path) -> None:
    first = json.loads((world_002_bundle / "checksums.json").read_text(encoding="utf-8"))["replay_sha256"]
    materialize_world(world_002_bundle, overwrite=True)
    second = json.loads((world_002_bundle / "checksums.json").read_text(encoding="utf-8"))["replay_sha256"]
    assert first == second


def test_load_calibration_units_from_json(world_002_bundle: Path) -> None:
    units = load_calibration_units_from_json(world_002_bundle / "replay_units.json")
    assert len(units) == 1
    u = units[0]
    assert u.unit_id == "replay_u1"
    assert u.experiment_id == "exp-search-001"
    assert u.treated_channel_names == ["search"]
    assert u.observed_lift == pytest.approx(0.02)
    assert u.lift_se == pytest.approx(0.08)
    assert u.lift_scale == "mean_kpi_level_delta"
    assert u.estimand == "geo_time_ATT"
    assert u.replay_estimand is not None
    assert u.replay_estimand.get("replay_transform_mode") == "full_panel_transform_estimand_mask"
    ReplayEstimandSpec.from_dict(u.replay_estimand)


def test_replay_units_trace_fields(world_002_bundle: Path) -> None:
    raw = json.loads((world_002_bundle / "replay_units.json").read_text(encoding="utf-8"))
    row = raw[0]
    assert row["world_id"] == "WORLD-002-replay"
    assert row["channel"] == "search"
    assert row["geo_scope"] == "listed"
    assert row["time_window"]["week_start"] == "2020-01-20"
    assert row["lift"] == pytest.approx(0.02)
    assert row["standard_error"] == pytest.approx(0.08)


def test_validator_passes_world_002(world_002_bundle: Path) -> None:
    result = validate_bundle(world_002_bundle, max_level=3)
    assert result.passed, result.hard_failures


def test_validator_unknown_experiment_reference(tmp_path: Path) -> None:
    bundle = tmp_path / "WORLD-002-replay"
    shutil.copytree(WORLD_002, bundle)
    materialize_world(bundle, overwrite=True)
    rows = json.loads((bundle / "replay_units.json").read_text(encoding="utf-8"))
    rows.append({**rows[0], "unit_id": "ghost_unit", "experiment_id": "ghost"})
    (bundle / "replay_units.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result = validate_bundle(bundle, max_level=3)
    assert not result.passed
    assert any("L3-replay-unknown-unit" in f for f in result.hard_failures)


def test_validator_replay_window_outside_time_truth(tmp_path: Path) -> None:
    bundle = tmp_path / "WORLD-002-replay"
    shutil.copytree(WORLD_002, bundle)
    materialize_world(bundle, overwrite=True)
    rows = json.loads((bundle / "replay_units.json").read_text(encoding="utf-8"))
    rows[0]["time_window"] = {"week_start": "2019-01-01", "week_end": "2019-02-01"}
    rows[0]["replay_estimand"]["week_start"] = "2019-01-01"
    rows[0]["replay_estimand"]["week_end"] = "2019-02-01"
    (bundle / "replay_units.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result = validate_bundle(bundle, max_level=3)
    assert not result.passed
    assert any("L3-replay-window" in f for f in result.hard_failures)


def test_validator_replay_channel_missing_from_media_truth(tmp_path: Path) -> None:
    bundle = tmp_path / "WORLD-002-replay"
    shutil.copytree(WORLD_002, bundle)
    materialize_world(bundle, overwrite=True)
    rows = json.loads((bundle / "replay_units.json").read_text(encoding="utf-8"))
    rows[0]["channel"] = "unknown_channel"
    rows[0]["treated_channel_names"] = ["unknown_channel"]
    (bundle / "replay_units.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result = validate_bundle(bundle, max_level=3)
    assert not result.passed
    assert any("L3-replay-channel" in f for f in result.hard_failures)


def test_validator_replay_checksum_mismatch_after_edit(tmp_path: Path) -> None:
    bundle = tmp_path / "WORLD-002-replay"
    shutil.copytree(WORLD_002, bundle)
    materialize_world(bundle, overwrite=True)
    replay_path = bundle / "replay_units.json"
    text = replay_path.read_text(encoding="utf-8")
    replay_path.write_text(text.replace("0.02", "0.99", 1), encoding="utf-8")
    failures = verify_checksums(bundle)
    assert any("replay_sha256" in f for f in failures)
