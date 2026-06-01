"""Phase 5A — lattice sweep MVP (deterministic ScenarioBuilder grid)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mmm.validation.synthetic.certification_registry import REPORT_ARTIFACT_NAME
from mmm.validation.synthetic.lattice_sweep import (
    LATTICE_WORLD_PREFIX,
    encode_world_id,
    mvp_lattice_specs,
    run_lattice_sweep,
    run_single_world,
    scenario_spec_from_axes,
)
from mmm.validation.synthetic.scenario_builder import build_world_truth

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_encode_world_id_is_deterministic() -> None:
    a = encode_world_id(
        family="baseline",
        noise_level="low",
        correlation_level="severe",
        drift=True,
        experiment_quality="none",
    )
    b = encode_world_id(
        family="baseline",
        noise_level="low",
        correlation_level="severe",
        drift=True,
        experiment_quality="none",
    )
    assert a == b
    assert a.startswith(LATTICE_WORLD_PREFIX)
    assert "noise-low" in a
    assert "corr-severe" in a
    assert "drift-on" in a


def test_mvp_lattice_world_count() -> None:
    specs = mvp_lattice_specs()
    assert len(specs) == 12
    ids = [s.world_id for s in specs]
    assert len(ids) == len(set(ids))


def test_lattice_same_spec_identical_truth() -> None:
    spec = scenario_spec_from_axes(
        family="baseline",
        noise_level="high",
        correlation_level="low",
        drift=False,
        experiment_quality="none",
    )
    t1 = build_world_truth(spec)
    t2 = build_world_truth(spec)
    assert json.dumps(t1, sort_keys=True) == json.dumps(t2, sort_keys=True)


@pytest.fixture
def lattice_tmp(tmp_path: Path) -> Path:
    return tmp_path


def test_single_world_materializes_and_certifies(lattice_tmp: Path) -> None:
    spec = scenario_spec_from_axes(
        family="baseline",
        noise_level="low",
        correlation_level="low",
        drift=False,
        experiment_quality="none",
    )
    bundle = lattice_tmp / spec.world_id
    outcome = run_single_world(bundle, spec)
    assert outcome.truth_written
    assert outcome.materialized
    assert outcome.certified
    assert (bundle / "world_truth.json").is_file()
    assert (bundle / REPORT_ARTIFACT_NAME).is_file()
    assert outcome.overall_status in ("pass", "fail", "partial")


def test_lattice_sweep_includes_all_worlds(lattice_tmp: Path) -> None:
    specs = mvp_lattice_specs()[:4]
    report = run_lattice_sweep(lattice_tmp, specs=specs, lattice_subdir="lattice-test")
    assert report["world_count"] == 4
    assert set(report["world_ids"]) == {s.world_id for s in specs}
    assert len(report["per_world_status"]) == 4
    assert report["scorecard_summary"]["reliability_score"] is not None


def test_severe_collinearity_axis_has_identifiability_warning(lattice_tmp: Path) -> None:
    spec = scenario_spec_from_axes(
        family="baseline",
        noise_level="medium",
        correlation_level="severe",
        drift=False,
        experiment_quality="none",
    )
    truth = build_world_truth(spec)
    warnings = (truth.get("artifact_truth") or {}).get("expected_warnings") or []
    assert any(w.get("warning_id") == "identifiability_collinearity" for w in warnings)

    bundle = lattice_tmp / spec.world_id
    outcome = run_single_world(bundle, spec)
    assert outcome.report is not None
    assert any("identifiability_collinearity" in w for w in (outcome.report.get("warnings") or []))


def test_drift_axis_truth_has_changepoints(lattice_tmp: Path) -> None:
    spec = scenario_spec_from_axes(
        family="baseline",
        noise_level="low",
        correlation_level="low",
        drift=True,
        experiment_quality="none",
    )
    truth = build_world_truth(spec)
    drift = truth.get("drift_truth") or {}
    assert drift.get("changepoints") or drift.get("coefficient_drift")

    bundle = lattice_tmp / spec.world_id
    outcome = run_single_world(bundle, spec)
    assert outcome.report is not None
    skipped = {s["check_id"] for s in outcome.report.get("skipped_validations") or []}
    assert "VAL-012" in skipped or any(
        r.get("check_id") == "VAL-012" and r.get("status") == "skipped"
        for r in outcome.report.get("validation_results") or []
    )


def test_replay_family_runs_replay_compatibility_check(lattice_tmp: Path) -> None:
    spec = scenario_spec_from_axes(
        family="replay",
        noise_level="low",
        correlation_level="low",
        drift=False,
        experiment_quality="medium",
    )
    bundle = lattice_tmp / spec.world_id
    outcome = run_single_world(bundle, spec)
    assert outcome.report is not None
    checks = {r["check_id"]: r["status"] for r in outcome.report.get("validation_results") or []}
    assert checks.get("CERT-4A-003") in ("pass", "fail", "skipped")


def test_skipped_validations_summarized_not_counted_as_unexpected_failures(lattice_tmp: Path) -> None:
    report = run_lattice_sweep(lattice_tmp, specs=mvp_lattice_specs()[:2], lattice_subdir="skip-test")
    skip_sum = report["skipped_validation_summary"]
    assert skip_sum["total_skipped_rows"] >= 0
    assert "by_skip_reason" in skip_sum
    for _wid, status in report["per_world_status"].items():
        if status.get("error"):
            continue
        assert status["certified"] or status.get("error")


def test_failure_preserved_on_materialize_error(lattice_tmp: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from mmm.validation.synthetic import lattice_sweep as ls

    spec = mvp_lattice_specs()[0]
    bundle = lattice_tmp / spec.world_id

    def boom(_path: Path, **_kw: object) -> None:
        raise RuntimeError("materialize failed on purpose")

    monkeypatch.setattr(ls, "materialize_world", boom)
    outcome = run_single_world(bundle, spec)
    assert outcome.overall_status == "error"
    assert outcome.error is not None
    assert not outcome.materialized
