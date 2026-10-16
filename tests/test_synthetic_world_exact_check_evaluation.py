"""World-parameterized exact transform-check evaluation regressions."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from mmm.governance.synthetic_certification import (
    EXACT_CHECK_NAMES,
    _resolve_certification_level,
    run_exact_check,
)
from mmm.validation.synthetic.world_exact_check_evaluation import (
    WORLD_CHECK_ORDER,
    evaluate_world_exact_checks,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
WORLDS_ROOT = REPO_ROOT / "validation" / "worlds"

EXPECTED_WORLDS = [
    "WORLD-001-baseline",
    "WORLD-002-replay",
    "WORLD-003-generated-baseline",
    "WORLD-004-generated-replay",
    "WORLD-005-scenario-low-noise",
    "WORLD-006-scenario-high-collinearity",
    "WORLD-007-scenario-replay-drift",
    "WORLD-008-exact-recovery",
    "WORLD-009-optimizer-recovery",
    "WORLD-010-replay-recovery",
]


def _truth(world: str) -> dict:
    return json.loads((WORLDS_ROOT / world / "world_truth.json").read_text(encoding="utf-8"))


def test_covered_checks_use_recorded_world_values_deterministically() -> None:
    assert set(WORLD_CHECK_ORDER) == {
        "geometric_adstock_carryover",
        "hill_saturation_analytic",
        "geometric_adstock_design_matrix",
        "hill_saturation_monotone_design_matrix",
    }
    for world in EXPECTED_WORLDS:
        first = evaluate_world_exact_checks(world, _truth(world))
        second = evaluate_world_exact_checks(world, _truth(world))
        assert [row["check_id"] for row in first] == list(WORLD_CHECK_ORDER)
        assert first == second
        for row in first:
            assert row["world_id"] == world
            assert row["status"] == "pass"
            assert row["skip_reason"] is None
            assert row["channels"]
    rows = {row["check_id"]: row for row in evaluate_world_exact_checks(
        "WORLD-008-exact-recovery", _truth("WORLD-008-exact-recovery"))}
    assert rows["geometric_adstock_carryover"]["evidence_refs"]["world_values"]["display"][
        "adstock_decay_by_channel"] == 0.55


def test_malformed_truth_fails_closed_without_silent_skips() -> None:
    truth = _truth("WORLD-001-baseline")
    for bad in [None, [], {}, {"transform_truth": None}, {"transform_truth": {}}]:
        for row in evaluate_world_exact_checks("WORLD-001-baseline", bad):
            assert row["status"] == "fail"
            assert row["skip_reason"] is None
    broken = json.loads(json.dumps(truth))
    broken["transform_truth"]["adstock_decay_by_channel"] = {"search": float("inf")}
    for row in evaluate_world_exact_checks("WORLD-001-baseline", broken):
        if row["check_id"] in ("geometric_adstock_carryover", "geometric_adstock_design_matrix"):
            assert row["status"] == "fail"
    mismatch = json.loads(json.dumps(truth))
    mismatch["transform_truth"]["hill_half_max_by_channel"] = {"other": 10.0}
    for row in evaluate_world_exact_checks("WORLD-001-baseline", mismatch):
        if row["check_id"] in ("hill_saturation_analytic", "hill_saturation_monotone_design_matrix"):
            assert row["status"] == "fail"
            assert "mismatch" in row["message"]


def test_one_bad_channel_fails_row_with_every_channel_visible() -> None:
    truth = json.loads(json.dumps(_truth("WORLD-001-baseline")))
    truth["transform_truth"]["adstock_decay_by_channel"] = {"search": 0.5, "social": 1.5}
    rows = {row["check_id"]: row
            for row in evaluate_world_exact_checks("WORLD-001-baseline", truth)}
    row = rows["geometric_adstock_carryover"]
    assert row["status"] == "fail"
    assert [c["channel"] for c in row["channels"]] == ["search", "social"]
    assert [c["status"] for c in row["channels"]] == ["pass", "fail"]
    assert "social" in row["message"]


def test_additive_key_leaves_level_and_readiness_inputs_identical() -> None:
    checks = [{"name": name, "status": "pass"} for name in EXACT_CHECK_NAMES]
    before = _resolve_certification_level(checks=checks, check_names=tuple(EXACT_CHECK_NAMES))
    rows = evaluate_world_exact_checks("WORLD-001-baseline", _truth("WORLD-001-baseline"))
    report = {"checks": checks, "world_exact_check_evaluations": rows}
    after = _resolve_certification_level(checks=report["checks"], check_names=tuple(EXACT_CHECK_NAMES))
    assert before == after == "exact"
    assert "world_exact_check_evaluations" not in EXACT_CHECK_NAMES
    for check_name in EXACT_CHECK_NAMES:
        run_exact_check(check_name)


def test_certification_run_embeds_diagnostic_rows_without_changing_existing_keys(tmp_path) -> None:
    from mmm.validation.synthetic.certification_runner import run_world_certification

    source = WORLDS_ROOT / "WORLD-001-baseline"
    bundle = tmp_path / "WORLD-001-baseline"
    shutil.copytree(source, bundle)
    result = run_world_certification(bundle, write_report=True)
    assert result.passed
    report = json.loads((bundle / "synthetic_world_certification_report.json").read_text(encoding="utf-8"))
    rows = report["world_exact_check_evaluations"]
    assert [row["check_id"] for row in rows] == list(WORLD_CHECK_ORDER)
    assert all(row["status"] == "pass" for row in rows)
    assert all(row["world_id"] == report["world_id"] for row in rows)
