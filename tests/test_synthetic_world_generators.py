"""Phase 3A — deterministic GroundTruthWorld generators (truth only)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mmm.calibration.replay_estimand import ReplayEstimandSpec
from mmm.calibration.units_io import load_calibration_units_from_json
from mmm.validation.synthetic.generators import (
    GENERATOR_VERSION,
    generate_baseline_world_truth,
    generate_replay_world_truth,
    write_world_truth,
)
from mmm.validation.synthetic.materializer import materialize_world
from mmm.validation.synthetic.validator import validate_bundle

REPO_ROOT = Path(__file__).resolve().parents[1]
WORLD_003 = REPO_ROOT / "validation" / "worlds" / "WORLD-003-generated-baseline"
WORLD_004 = REPO_ROOT / "validation" / "worlds" / "WORLD-004-generated-replay"


def _canonical_truth_bytes(truth: dict) -> bytes:
    return json.dumps(truth, sort_keys=True, separators=(",", ":")).encode("utf-8")


def test_same_seed_produces_identical_world_truth() -> None:
    a = generate_baseline_world_truth(42, "WORLD-test-baseline")
    b = generate_baseline_world_truth(42, "WORLD-test-baseline")
    assert _canonical_truth_bytes(a) == _canonical_truth_bytes(b)
    assert a["metadata"]["world_generator_version"] == GENERATOR_VERSION

    c = generate_replay_world_truth(99, "WORLD-test-replay")
    d = generate_replay_world_truth(99, "WORLD-test-replay")
    assert _canonical_truth_bytes(c) == _canonical_truth_bytes(d)


def test_same_seed_different_world_id_changes_metadata_only() -> None:
    a = generate_baseline_world_truth(42, "WORLD-a")
    b = generate_baseline_world_truth(42, "WORLD-b")
    assert a["metadata"]["world_id"] != b["metadata"]["world_id"]
    assert a["time_truth"] == b["time_truth"]
    assert a["coefficient_truth"] == b["coefficient_truth"]


def test_different_seed_produces_controlled_differences() -> None:
    t1 = generate_baseline_world_truth(10, "WORLD-a")
    t2 = generate_baseline_world_truth(11, "WORLD-b")
    assert t1["time_truth"]["n_periods"] != t2["time_truth"]["n_periods"] or t1["geo_truth"]["n_geos"] != t2[
        "geo_truth"
    ]["n_geos"]
    assert t1["metadata"]["generation_seed"] != t2["metadata"]["generation_seed"]

    r1 = generate_replay_world_truth(20, "WORLD-r1")
    r2 = generate_replay_world_truth(21, "WORLD-r2")
    u1 = r1["experiment_truth"]["units"][0]
    u2 = r2["experiment_truth"]["units"][0]
    assert (
        u1["lift_definition"]["value"] != u2["lift_definition"]["value"]
        or u1["week_start"] != u2["week_start"]
        or r1["time_truth"]["n_periods"] != r2["time_truth"]["n_periods"]
    )


def test_write_world_truth_does_not_materialize_derived_artifacts(tmp_path: Path) -> None:
    bundle = tmp_path / "WORLD-gen-only"
    truth = generate_baseline_world_truth(7, "WORLD-gen-only")
    write_world_truth(bundle, truth)
    assert (bundle / "world_truth.json").is_file()
    assert not (bundle / "panel.parquet").exists()
    assert not (bundle / "replay_units.json").exists()
    assert not (bundle / "checksums.json").exists()


@pytest.fixture(scope="module")
def world_003_bundle() -> Path:
    assert (WORLD_003 / "world_truth.json").is_file()
    materialize_world(WORLD_003, overwrite=True)
    return WORLD_003


@pytest.fixture(scope="module")
def world_004_bundle() -> Path:
    assert (WORLD_004 / "world_truth.json").is_file()
    materialize_world(WORLD_004, overwrite=True)
    return WORLD_004


def test_world_003_generated_baseline_materializes(world_003_bundle: Path) -> None:
    meta = json.loads((world_003_bundle / "metadata.json").read_text(encoding="utf-8"))
    assert meta["world_id"] == "WORLD-003-generated-baseline"
    assert (world_003_bundle / "panel.parquet").is_file()
    assert (world_003_bundle / "decision_truth.json").is_file()
    assert not (world_003_bundle / "replay_units.json").exists()
    truth = json.loads((world_003_bundle / "world_truth.json").read_text(encoding="utf-8"))
    assert truth["metadata"]["world_generator_version"] == GENERATOR_VERSION
    assert truth["metadata"]["archetype_id"] == "baseline_world"


def test_world_004_generated_replay_materializes_and_loads(world_004_bundle: Path) -> None:
    assert (world_004_bundle / "replay_units.json").is_file()
    units = load_calibration_units_from_json(world_004_bundle / "replay_units.json")
    assert len(units) == 1
    ReplayEstimandSpec.from_dict(units[0].replay_estimand)
    truth = json.loads((world_004_bundle / "world_truth.json").read_text(encoding="utf-8"))
    assert truth["metadata"]["archetype_id"] == "experiment_world"
    assert len(truth["experiment_truth"]["units"]) == 1


def test_generated_bundles_pass_validator(world_003_bundle: Path, world_004_bundle: Path) -> None:
    for bundle in (world_003_bundle, world_004_bundle):
        result = validate_bundle(bundle, max_level=3)
        assert result.passed, (bundle.name, result.hard_failures)


def test_generated_replay_bundle_has_no_replay_failures(world_004_bundle: Path) -> None:
    result = validate_bundle(world_004_bundle, max_level=3)
    replay_fails = [f for f in result.hard_failures if "replay" in f.lower()]
    assert not replay_fails, replay_fails
