"""World-parameterized exact transform-check evaluation regressions."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from mmm.governance.synthetic_certification import (
    EXACT_CHECK_NAMES,
    _check_geometric_adstock_carryover,
    _check_geometric_adstock_design_matrix,
    _check_hill_monotone_design_matrix,
    _check_hill_saturation_analytic,
    _resolve_certification_level,
    run_exact_check,
    run_synthetic_certification_suite,
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


def _assert_schema(row: dict) -> None:
    assert isinstance(row["check_id"], str) and isinstance(row["world_id"], str)
    assert row["status"] in ("pass", "fail")
    assert row["skip_reason"] is None
    for key in ("expected", "actual"):
        assert isinstance(row[key], (int, float, str)) and not isinstance(row[key], bool)
    assert row["tolerance"] is None or isinstance(row["tolerance"], (int, float))
    assert isinstance(row["message"], str) and isinstance(row["evidence_refs"], dict)
    for entry in row["channels"]:
        assert isinstance(entry["channel"], str) and entry["status"] in ("pass", "fail")


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
            _assert_schema(row)
            assert row["channels"]
    rows = {row["check_id"]: row for row in evaluate_world_exact_checks(
        "WORLD-008-exact-recovery", _truth("WORLD-008-exact-recovery"))}
    assert rows["geometric_adstock_carryover"]["evidence_refs"]["world_values"][
        "adstock_decay_by_channel"]["display"] == 0.55
    assert rows["geometric_adstock_carryover"]["actual"] == 0.55 * 100.0
    assert rows["geometric_adstock_carryover"]["tolerance"] == 1e-12


def test_malformed_truth_fails_closed_without_silent_skips() -> None:
    truth = _truth("WORLD-001-baseline")
    for bad in [None, [], {}, {"transform_truth": None}, {"transform_truth": {}}]:
        for row in evaluate_world_exact_checks("WORLD-001-baseline", bad):
            assert row["status"] == "fail"
            _assert_schema(row)
    broken = json.loads(json.dumps(truth))
    broken["transform_truth"]["adstock_decay_by_channel"] = {"search": float("inf")}
    for row in evaluate_world_exact_checks("WORLD-001-baseline", broken):
        if row["check_id"] in ("geometric_adstock_carryover", "geometric_adstock_design_matrix"):
            assert row["status"] == "fail"
    oversized = json.loads(json.dumps(truth))
    oversized["transform_truth"]["adstock_decay_by_channel"] = {"search": 10**400, "social": 0.5}
    for row in evaluate_world_exact_checks("WORLD-001-baseline", oversized):
        if row["check_id"] in ("geometric_adstock_carryover", "geometric_adstock_design_matrix"):
            assert row["status"] == "fail"
            assert "OverflowError" in row["channels"][0]["actual"] or "non-finite" in row["channels"][0]["actual"]
            assert [c["channel"] for c in row["channels"]] == ["search", "social"]
    mismatch = json.loads(json.dumps(truth))
    mismatch["transform_truth"]["hill_half_max_by_channel"] = {"other": 10.0}
    for row in evaluate_world_exact_checks("WORLD-001-baseline", mismatch):
        if row["check_id"] in ("hill_saturation_analytic", "hill_saturation_monotone_design_matrix"):
            assert row["status"] == "fail"
            by_channel = {c["channel"]: c for c in row["channels"]}
            assert set(by_channel) == {"other", "search", "social"}
            assert "unknown extra" in by_channel["other"]["actual"]
            assert "mismatch" in by_channel["search"]["actual"]
            assert "binding channel other" in row["message"]


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
    assert isinstance(row["actual"], str)


def test_zero_arg_predicates_return_none_with_shared_measurements() -> None:
    assert _check_geometric_adstock_carryover() is None
    assert _check_hill_saturation_analytic() is None
    assert _check_geometric_adstock_design_matrix() is None
    assert _check_hill_monotone_design_matrix() is None
    from mmm.validation.synthetic.world_exact_check_evaluation import (
        _measure_carryover,
        _measure_design_carryover,
        _measure_hill_analytic,
        _measure_hill_monotone,
    )
    assert _measure_carryover(decay=0.5)["week1_carryover"] == 50.0
    assert _measure_hill_analytic(half_max=10.0, slope=2.0)["expected"] == 5.0**2.0 / (10.0**2.0 + 5.0**2.0 + 1e-12)
    assert _measure_design_carryover(decay=0.5)["impulse_response"] > 0.0
    assert _measure_hill_monotone(hill_half=10.0, hill_slope=2.0)["min_diff"] >= -1e-9


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


def test_certification_suite_outputs_unchanged() -> None:
    rep = run_synthetic_certification_suite(mode="exact")
    assert rep["certification_level"] == "exact"
    assert rep["certification_status"] == "pass"
    assert rep["n_pass"] == len(EXACT_CHECK_NAMES)
    assert {c["name"] for c in rep["checks"]} == set(EXACT_CHECK_NAMES)


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
    for row in rows:
        _assert_schema(row)
    for key in ("report_kind", "world_id", "overall_status", "executed_validations",
                "failed_validations", "validation_results", "contract_compatibility"):
        assert key in report


def _rows_for(world: str, mutate) -> dict:
    truth = json.loads(json.dumps(_truth(world)))
    mutate(truth)
    return {row["check_id"]: row for row in evaluate_world_exact_checks(world, truth)}


def test_blank_and_nonstring_identifiers_fail_explicitly() -> None:
    rows = _rows_for("WORLD-001-baseline", lambda t: t["transform_truth"].update(
        {"adstock_decay_by_channel": {"search": 0.5, "": 0.6}}))
    row = rows["geometric_adstock_carryover"]
    assert row["status"] == "fail"
    by_channel = {c["channel"]: c for c in row["channels"]}
    assert by_channel["search"]["status"] == "pass"
    assert by_channel[""]["status"] == "fail"
    assert "malformed channel identifier" in by_channel[""]["actual"]
    rows = _rows_for("WORLD-001-baseline", lambda t: t["transform_truth"].update(
        {"adstock_decay_by_channel": {"search": 0.5, 7: 0.6}}))
    row = rows["geometric_adstock_carryover"]
    assert row["status"] == "fail"
    assert any(c["status"] == "fail" and "malformed" in c["actual"] for c in row["channels"])


def test_missing_slope_keeps_valid_search_visible() -> None:
    rows = _rows_for("WORLD-001-baseline", lambda t: t["transform_truth"].update(
        {"hill_half_max_by_channel": {"search": 10.0, "social": 10.0},
         "hill_slope_by_channel": {"search": 2.0}}))
    row = rows["hill_saturation_analytic"]
    assert row["status"] == "fail"
    by_channel = {c["channel"]: c for c in row["channels"]}
    assert by_channel["search"]["status"] == "pass"
    assert by_channel["social"]["status"] == "fail"
    assert "mismatch" in by_channel["social"]["actual"]


def test_unknown_extra_channel_and_missing_declared_value_fail() -> None:
    rows = _rows_for("WORLD-001-baseline", lambda t: t["transform_truth"].update(
        {"adstock_decay_by_channel": {"search": 0.5, "social": 0.5, "xm": 0.4}}))
    row = rows["geometric_adstock_carryover"]
    assert row["status"] == "fail"
    by_channel = {c["channel"]: c for c in row["channels"]}
    assert by_channel["search"]["status"] == "pass"
    assert by_channel["social"]["status"] == "pass"
    assert by_channel["xm"]["status"] == "fail"
    assert "unknown extra" in by_channel["xm"]["actual"]
    rows = _rows_for("WORLD-001-baseline", lambda t: t["transform_truth"].update(
        {"adstock_decay_by_channel": {"search": 0.5}}))
    row = rows["geometric_adstock_carryover"]
    assert row["status"] == "fail"
    assert row["channels"][1] == {
        "channel": "social", "expected": "required recorded parameter present",
        "actual": "channel coverage mismatch: missing adstock_decay_by_channel for channel 'social'",
        "tolerance": None, "status": "fail",
    } or "mismatch" in row["channels"][1]["actual"]


def test_invalid_inventory_and_maps_fail_closed() -> None:
    cases = [
        (lambda t: t["media_truth"].update({"channels": []}), None),
        (lambda t: t["media_truth"].update({"channels": ["search", "search"]}), None),
        (lambda t: t["media_truth"].update({"channels": "search"}), None),
        (lambda t: t["media_truth"].update({"channels": [""]}), None),
        (lambda t: t.pop("transform_truth"), None),
        (lambda t: t.update({"transform_truth": []}), None),
        (lambda t: t["transform_truth"].pop("adstock_decay_by_channel"),
         ("geometric_adstock_carryover", "geometric_adstock_design_matrix")),
    ]
    for mutate, affected in cases:
        for row in evaluate_world_exact_checks("WORLD-001-baseline", _mutated(mutate)):
            if affected is None or row["check_id"] in affected:
                assert row["status"] == "fail", row
            _assert_schema(row)


def _mutated(mutate):
    truth = json.loads(json.dumps(_truth("WORLD-001-baseline")))
    mutate(truth)
    return truth


def test_nonfinite_and_domain_invalid_values_fail_with_siblings_visible() -> None:
    base = {"search": 0.5, "social": 0.5}
    cases = {
        "nan": float("nan"),
        "inf": float("inf"),
        "zero_domain": 0.0,
        "above_domain": 1.5,
        "oversized": 10**400,
        "string": "fast",
        "boolean": True,
    }
    for label, bad in cases.items():
        decay = dict(base)
        decay["social"] = bad
        truth = json.loads(json.dumps(_truth("WORLD-001-baseline"), default=str))
        truth["transform_truth"]["adstock_decay_by_channel"] = decay
        rows = {row["check_id"]: row for row in evaluate_world_exact_checks("WORLD-001-baseline", truth)}
        row = rows["geometric_adstock_carryover"]
        assert row["status"] == "fail", label
        by_channel = {c["channel"]: c for c in row["channels"]}
        assert by_channel["search"]["status"] == "pass", label
        assert by_channel["social"]["status"] == "fail", label


def test_measured_evidence_refs_match_produced_measurements() -> None:
    rows = {row["check_id"]: row for row in evaluate_world_exact_checks(
        "WORLD-001-baseline", _truth("WORLD-001-baseline"))}
    for row in rows.values():
        measured = row["evidence_refs"]["measured"]
        assert set(measured) == {c["channel"] for c in row["channels"] if c["status"] == "pass"}
        for detail in measured.values():
            assert isinstance(detail, dict) and detail
    assert rows["geometric_adstock_carryover"]["evidence_refs"]["measured"]["search"]["week1_carryover"] == 0.5 * 100.0


def test_whitespace_identifier_fails_closed() -> None:
    rows = _rows_for("WORLD-001-baseline", lambda t: t["transform_truth"].update(
        {"adstock_decay_by_channel": {"search": 0.5, " ": 0.5}}))
    row = rows["geometric_adstock_carryover"]
    assert row["status"] == "fail"
    by_channel = {c["channel"]: c for c in row["channels"]}
    assert by_channel["search"]["status"] == "pass"
    assert by_channel[" "]["status"] == "fail"
    assert "malformed" in by_channel[" "]["actual"]


def test_binding_follows_lexical_order_not_declared_order() -> None:
    truth = json.loads(json.dumps(_truth("WORLD-001-baseline")))
    truth["media_truth"]["channels"] = ["social", "search"]
    truth["transform_truth"]["adstock_decay_by_channel"] = {"search": 1.5, "social": 1.5}
    rows = {row["check_id"]: row
            for row in evaluate_world_exact_checks("WORLD-001-baseline", truth)}
    row = rows["geometric_adstock_carryover"]
    assert row["status"] == "fail"
    assert [c["channel"] for c in row["channels"]] == ["search", "social"]
    assert "binding channel search" in row["message"]


def test_malformed_inventory_entry_preserves_valid_siblings() -> None:
    rows = _rows_for("WORLD-001-baseline", lambda t: t["media_truth"].update(
        {"channels": ["search", "social", ""]}))
    row = rows["geometric_adstock_carryover"]
    assert row["status"] == "fail"
    by_channel = {c["channel"]: c for c in row["channels"]}
    assert by_channel["search"]["status"] == "pass"
    assert by_channel["social"]["status"] == "pass"
    assert by_channel[""]["status"] == "fail"
