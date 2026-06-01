"""Phase 4B-4 — replay calibration recovery world WORLD-010."""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import pytest

from mmm.calibration.units_io import load_calibration_units_from_json
from mmm.data.schema import PanelSchema
from mmm.validation.synthetic.certification_runner import run_world_certification
from mmm.validation.synthetic.dgp_materializer import compute_dgp_series, materialize_dgp_world
from mmm.validation.synthetic.recovery_certification import (
    REPLAY_RECOVERY_WORLD_IDS,
    build_recovery_mmm_config,
    is_replay_recovery_eligible,
    run_recovery_certification,
)
from mmm.validation.synthetic.replay_truth import (
    build_world_010_truth,
    compute_true_replay_lift,
    detect_window_slice_adstock_reset,
    enrich_world_010_experiment_truth,
    validate_replay_lift_surface,
)
from mmm.validation.synthetic.validator import validate_bundle

REPO_ROOT = Path(__file__).resolve().parents[1]
WORLD_010 = REPO_ROOT / "validation" / "worlds" / "WORLD-010-replay-recovery"


@pytest.fixture(scope="module")
def world_010_bundle(tmp_path_factory: pytest.TempPathFactory) -> Path:
    bundle = tmp_path_factory.mktemp("w010") / "WORLD-010-replay-recovery"
    shutil.copytree(WORLD_010, bundle, dirs_exist_ok=True)
    materialize_dgp_world(bundle, overwrite=True)
    return bundle


def test_world_010_present_and_eligible() -> None:
    assert (WORLD_010 / "world_truth.json").is_file()
    truth = json.loads((WORLD_010 / "world_truth.json").read_text(encoding="utf-8"))
    assert truth["metadata"]["world_id"] in REPLAY_RECOVERY_WORLD_IDS
    assert is_replay_recovery_eligible(truth)
    unit = truth["experiment_truth"]["units"][0]
    assert unit["replay_transform_mode"] == "full_panel_transform_estimand_mask"
    assert float(unit["lift_definition"]["value"]) != 0.0


def test_world_010_materializes(world_010_bundle: Path) -> None:
    assert (world_010_bundle / "panel.parquet").is_file()
    assert (world_010_bundle / "replay_units.json").is_file()
    assert (world_010_bundle / "dgp_diagnostics.parquet").is_file()


def test_replay_units_loads(world_010_bundle: Path) -> None:
    units = load_calibration_units_from_json(world_010_bundle / "replay_units.json")
    assert len(units) == 1
    assert units[0].replay_estimand is not None


def test_true_lift_full_panel_estimand_mask(tmp_path: Path) -> None:
    truth = build_world_010_truth()
    panel, _ = compute_dgp_series(truth)
    panel_path = tmp_path / "panel.parquet"
    panel.to_parquet(panel_path, index=False)
    config = build_recovery_mmm_config(truth, panel_path=panel_path)
    schema = PanelSchema(
        "geo_id",
        "week_start_date",
        "revenue",
        ("search",),
        (),
    )
    unit = truth["experiment_truth"]["units"][0]
    result = compute_true_replay_lift(truth, panel, schema, config, unit)
    validate_replay_lift_surface(result["true_experiment_lift"])
    assert result["replay_transform_mode"] == "full_panel_transform_estimand_mask"
    assert result["pre_window_adstock_preserved"] is True


def test_pre_window_adstock_affects_lift(tmp_path: Path) -> None:
    truth = build_world_010_truth()
    panel, _ = compute_dgp_series(truth)
    panel_path = tmp_path / "panel.parquet"
    panel.to_parquet(panel_path, index=False)
    config = build_recovery_mmm_config(truth, panel_path=panel_path)
    schema = PanelSchema("geo_id", "week_start_date", "revenue", ("search",), ())
    unit = truth["experiment_truth"]["units"][0]
    with_carry = compute_true_replay_lift(truth, panel, schema, config, unit)["true_experiment_lift"]
    flat = truth.copy()
    flat["media_truth"] = dict(truth["media_truth"])
    flat["media_truth"]["spend_process_spec"] = {
        "kind": "constant",
        "level": 10.0,
        "correlation_level": "low",
    }
    panel_flat, _ = compute_dgp_series(flat)
    without_carry = compute_true_replay_lift(flat, panel_flat, schema, config, unit)[
        "true_experiment_lift"
    ]
    assert abs(with_carry - without_carry) > 1e-5


def test_window_slice_reset_detected(tmp_path: Path) -> None:
    truth = build_world_010_truth()
    panel, _ = compute_dgp_series(truth)
    panel_path = tmp_path / "panel.parquet"
    panel.to_parquet(panel_path, index=False)
    config = build_recovery_mmm_config(truth, panel_path=panel_path)
    schema = PanelSchema("geo_id", "week_start_date", "revenue", ("search",), ())
    unit = truth["experiment_truth"]["units"][0]
    assert detect_window_slice_adstock_reset(truth, panel, schema, config, unit) is False


def test_train_and_replay_path_complete(world_010_bundle: Path) -> None:
    result = run_recovery_certification(world_010_bundle)
    train = next(c for c in result.checks if c.check_id == "REC-4B4-TRAIN")
    replay = next(c for c in result.checks if c.check_id == "REC-4B4-REPLAY")
    assert train.status == "pass"
    assert replay.status == "pass"


def test_val_006_executes_not_skipped(world_010_bundle: Path) -> None:
    result = run_recovery_certification(world_010_bundle)
    replay = next(c for c in result.checks if c.check_id == "REC-4B4-REPLAY")
    assert replay.skip_reason is None
    assert replay.details.get("registry_validation_id") == "VAL-006"
    cert = run_world_certification(world_010_bundle, write_report=False)
    skipped = {r["check_id"] for r in cert.report["skipped_validations"]}
    assert "VAL-006" not in skipped


def test_fitted_lift_matches_truth(world_010_bundle: Path) -> None:
    result = run_recovery_certification(world_010_bundle)
    replay = next(c for c in result.checks if c.check_id == "REC-4B4-REPLAY")
    details = replay.details
    assert details["replay_recovery_status"] == "pass"
    assert float(details["replay_lift_error"]) <= 0.02 + 0.25 * abs(details["true_experiment_lift"])
    assert details["pre_window_adstock_preserved"] is True


def test_corrupted_true_lift_fails(world_010_bundle: Path) -> None:
    truth = json.loads((world_010_bundle / "world_truth.json").read_text(encoding="utf-8"))
    bad = copy.deepcopy(truth)
    bad["experiment_truth"]["units"][0]["lift_definition"]["value"] = -999.0
    result = run_recovery_certification(world_010_bundle, truth_override=bad)
    replay = next(c for c in result.checks if c.check_id == "REC-4B4-REPLAY")
    assert replay.status == "fail"


def test_validator_passes(world_010_bundle: Path) -> None:
    outcome = validate_bundle(world_010_bundle, max_level=3)
    assert outcome.passed


def test_enrich_world_010_sets_lift(tmp_path: Path) -> None:
    truth = build_world_010_truth()
    panel, _ = compute_dgp_series(truth)
    panel_path = tmp_path / "panel.parquet"
    panel.to_parquet(panel_path, index=False)
    config = build_recovery_mmm_config(truth, panel_path=panel_path)
    schema = PanelSchema("geo_id", "week_start_date", "revenue", ("search",), ())
    enriched = enrich_world_010_experiment_truth(truth, panel, schema, config)
    lift = enriched["experiment_truth"]["units"][0]["lift_definition"]["value"]
    validate_replay_lift_surface(float(lift))
