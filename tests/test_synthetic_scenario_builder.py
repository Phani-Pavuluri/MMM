"""Phase 3B — ScenarioBuilder MVP (deterministic scenario → world_truth)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mmm.calibration.units_io import load_calibration_units_from_json
from mmm.validation.synthetic.materializer import materialize_world
from mmm.validation.synthetic.scenario_builder import (
    SCENARIO_BUILDER_VERSION,
    WORLD_005_LOW_NOISE,
    WORLD_006_HIGH_COLLINEARITY,
    WORLD_007_REPLAY_DRIFT,
    ScenarioSpec,
    build_world_truth,
    write_scenario_world,
)
from mmm.validation.synthetic.validator import validate_bundle

REPO_ROOT = Path(__file__).resolve().parents[1]
WORLD_005 = REPO_ROOT / "validation" / "worlds" / "WORLD-005-scenario-low-noise"
WORLD_006 = REPO_ROOT / "validation" / "worlds" / "WORLD-006-scenario-high-collinearity"
WORLD_007 = REPO_ROOT / "validation" / "worlds" / "WORLD-007-scenario-replay-drift"


def _canonical_bytes(truth: dict) -> bytes:
    return json.dumps(truth, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _base_spec(**overrides: object) -> ScenarioSpec:
    d = WORLD_005_LOW_NOISE.to_dict()
    d.update(overrides)
    if "channels" in overrides:
        d["channels"] = list(overrides["channels"])  # type: ignore[arg-type]
    return ScenarioSpec.from_dict(d)


def test_same_spec_produces_identical_world_truth() -> None:
    spec = WORLD_005_LOW_NOISE
    a = build_world_truth(spec)
    b = build_world_truth(spec)
    assert _canonical_bytes(a) == _canonical_bytes(b)
    assert SCENARIO_BUILDER_VERSION in " ".join(a["metadata"]["scenario_tags"])


def test_different_seed_produces_controlled_difference() -> None:
    s1 = _base_spec(seed=100)
    s2 = _base_spec(seed=101)
    t1 = build_world_truth(s1)
    t2 = build_world_truth(s2)
    assert t1["metadata"]["generation_seed"] != t2["metadata"]["generation_seed"]
    assert (
        t1["coefficient_truth"]["true_beta_by_channel"] != t2["coefficient_truth"]["true_beta_by_channel"]
        or t1["geo_truth"]["n_geos"] != t2["geo_truth"]["n_geos"]
    )


def test_changing_n_geos_changes_geo_truth_only() -> None:
    s2 = _base_spec(n_geos=4, world_id="WORLD-geo-test")
    s3 = _base_spec(n_geos=5, world_id="WORLD-geo-test")
    t2 = build_world_truth(s2)
    t3 = build_world_truth(s3)
    assert t2["geo_truth"] != t3["geo_truth"]
    assert t2["time_truth"] == t3["time_truth"]
    assert t2["media_truth"]["channels"] == t3["media_truth"]["channels"]
    assert t2["coefficient_truth"] == t3["coefficient_truth"]


def test_replay_family_with_experiment_quality_creates_units() -> None:
    spec = ScenarioSpec.from_dict(
        {
            **WORLD_007_REPLAY_DRIFT.to_dict(),
            "experiment_quality": "high",
            "drift": False,
            "world_id": "WORLD-replay-units-test",
        }
    )
    truth = build_world_truth(spec)
    assert len(truth["experiment_truth"]["units"]) == 1
    assert truth["governance_truth"]["replay_calibration_active"] is True


def test_replay_none_experiment_quality_has_no_units() -> None:
    spec = ScenarioSpec.from_dict(
        {
            **WORLD_007_REPLAY_DRIFT.to_dict(),
            "experiment_quality": "none",
            "drift": False,
            "world_id": "WORLD-replay-empty-test",
        }
    )
    truth = build_world_truth(spec)
    assert truth["experiment_truth"]["units"] == []
    assert truth["governance_truth"]["replay_calibration_active"] is False


def test_high_collinearity_sets_expected_warning() -> None:
    truth = build_world_truth(WORLD_006_HIGH_COLLINEARITY)
    warnings = truth["artifact_truth"]["expected_warnings"]
    assert any(w.get("warning_id") == "identifiability_collinearity" for w in warnings)
    assert truth["media_truth"]["spend_process_spec"]["correlation_level"] == "severe"


def test_drift_scenario_sets_drift_truth() -> None:
    truth = build_world_truth(WORLD_007_REPLAY_DRIFT)
    drift = truth["drift_truth"]
    assert drift["changepoints"]
    assert drift["coefficient_drift"]
    assert drift["changepoints"][0]["period_index"] < truth["time_truth"]["n_periods"]


def test_write_scenario_world_writes_truth_only(tmp_path: Path) -> None:
    spec = _base_spec(world_id="WORLD-write-only", seed=77)
    bundle = tmp_path / spec.world_id
    write_scenario_world(bundle, spec)
    assert (bundle / "world_truth.json").is_file()
    assert not (bundle / "panel.parquet").exists()
    assert not (bundle / "replay_units.json").exists()
    assert not (bundle / "checksums.json").exists()


@pytest.fixture(scope="module")
def scenario_bundles_materialized() -> list[Path]:
    bundles = []
    for spec, path in (
        (WORLD_005_LOW_NOISE, WORLD_005),
        (WORLD_006_HIGH_COLLINEARITY, WORLD_006),
        (WORLD_007_REPLAY_DRIFT, WORLD_007),
    ):
        assert (path / "world_truth.json").is_file(), path
        write_scenario_world(path, spec)
        materialize_world(path, overwrite=True)
        bundles.append(path)
    return bundles


def test_scenario_bundles_pass_validator_l3(scenario_bundles_materialized: list[Path]) -> None:
    for bundle in scenario_bundles_materialized:
        result = validate_bundle(bundle, max_level=3)
        assert result.passed, (bundle.name, result.hard_failures)


def test_world_007_replay_loads(scenario_bundles_materialized: list[Path]) -> None:
    units = load_calibration_units_from_json(WORLD_007 / "replay_units.json")
    assert len(units) == 1
