"""Phase 5B — behavioral lattice sweep (rich DGP + recovery)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mmm.validation.synthetic.behavioral_lattice_sweep import (
    BEHAVIORAL_WORLD_PREFIX,
    BehavioralWorldSpec,
    behavioral_spec_from_cell,
    build_behavioral_world_truth,
    encode_behavioral_world_id,
    extract_recovery_metrics,
    mvp_behavioral_lattice_specs,
    resolve_behavioral_mode,
    run_behavioral_lattice_sweep,
    run_single_behavioral_world,
)
from mmm.validation.synthetic.certification_registry import REPORT_ARTIFACT_NAME
from mmm.validation.synthetic.dgp_materializer import materialize_dgp_world

REPO_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_REPORT_FIELDS = frozenset(
    {
        "sweep_id",
        "sweep_version",
        "generated_at",
        "world_count",
        "world_ids",
        "axes",
        "per_world_certification_status",
        "per_axis_recovery_summary",
        "capability_recovery_summary",
        "scorecard_summary",
        "failures",
        "partials",
        "skips",
        "limitations",
        "recommended_followups",
    }
)


def test_encode_world_id_deterministic() -> None:
    a = encode_behavioral_world_id(
        world_type="exact_recovery",
        noise_level="zero",
        correlation_level="low",
        drift=False,
        replay=False,
    )
    b = encode_behavioral_world_id(
        world_type="exact_recovery",
        noise_level="zero",
        correlation_level="low",
        drift=False,
        replay=False,
    )
    assert a == b
    assert a.startswith(BEHAVIORAL_WORLD_PREFIX)


def test_mvp_lattice_bounded_and_deterministic() -> None:
    specs = mvp_behavioral_lattice_specs()
    assert 8 <= len(specs) <= 12
    ids = [s.world_id for s in specs]
    assert len(ids) == len(set(ids))
    again = [s.world_id for s in mvp_behavioral_lattice_specs()]
    assert ids == again


def test_same_cell_same_truth() -> None:
    spec = behavioral_spec_from_cell("exact_recovery", "zero", "low", False, False)
    t1 = build_behavioral_world_truth(spec)
    t2 = build_behavioral_world_truth(spec)
    assert json.dumps(t1, sort_keys=True) == json.dumps(t2, sort_keys=True)
    assert "dgp:exact_recovery" in t1["metadata"]["scenario_tags"]


def test_unsupported_replay_axis() -> None:
    mode = resolve_behavioral_mode(
        world_type="replay",
        drift=False,
        replay=False,
        correlation_level="low",
    )
    assert mode == "unsupported"


@pytest.fixture
def exact_spec() -> BehavioralWorldSpec:
    return behavioral_spec_from_cell("exact_recovery", "zero", "low", False, False)


def test_rich_dgp_materializes(tmp_path: Path, exact_spec: BehavioralWorldSpec) -> None:
    bundle = tmp_path / exact_spec.world_id
    truth = build_behavioral_world_truth(exact_spec)
    from mmm.validation.synthetic.generators import write_world_truth

    write_world_truth(bundle, truth)
    materialize_dgp_world(bundle, overwrite=True)
    assert (bundle / "panel.parquet").is_file()
    assert (bundle / "dgp_diagnostics.parquet").is_file()


def test_recovery_certification_runs(tmp_path: Path, exact_spec: BehavioralWorldSpec) -> None:
    bundle = tmp_path / exact_spec.world_id
    outcome = run_single_behavioral_world(bundle, exact_spec)
    assert outcome.materialized
    assert outcome.certified
    assert outcome.report is not None
    assert (bundle / REPORT_ARTIFACT_NAME).is_file()
    metrics = extract_recovery_metrics(outcome.report)
    assert metrics["coefficient_recovery_status"] in ("pass", "fail", "partial", "skipped")
    assert metrics["delta_mu_recovery_status"] in ("pass", "fail", "partial", "skipped")
    assert metrics["contract_compatibility_status"] == "pass"


def test_optimizer_world_has_optimizer_metric(tmp_path: Path) -> None:
    spec = behavioral_spec_from_cell("optimizer", "zero", "low", False, False)
    outcome = run_single_behavioral_world(tmp_path / spec.world_id, spec)
    assert outcome.certified and outcome.report
    metrics = extract_recovery_metrics(outcome.report)
    assert metrics["optimizer_recovery_status"] in ("pass", "fail", "skipped")


def test_replay_world_has_replay_metric(tmp_path: Path) -> None:
    spec = behavioral_spec_from_cell("replay", "zero", "low", False, True)
    outcome = run_single_behavioral_world(tmp_path / spec.world_id, spec)
    assert outcome.certified and outcome.report
    metrics = extract_recovery_metrics(outcome.report)
    assert metrics["replay_recovery_status"] in ("pass", "fail", "skipped")


def test_drift_partial_or_pass_without_full_runner(tmp_path: Path) -> None:
    spec = behavioral_spec_from_cell("drift", "zero", "low", True, False)
    outcome = run_single_behavioral_world(tmp_path / spec.world_id, spec)
    assert outcome.certified and outcome.report
    metrics = extract_recovery_metrics(outcome.report)
    assert metrics["drift_behavior_status"] in ("pass", "partial")
    assert metrics["coefficient_recovery_status"] in (None, "skipped")


def test_identifiability_coef_not_failed(tmp_path: Path) -> None:
    spec = behavioral_spec_from_cell("identifiability", "low", "severe", False, False)
    outcome = run_single_behavioral_world(tmp_path / spec.world_id, spec)
    assert outcome.certified and outcome.report
    metrics = extract_recovery_metrics(outcome.report)
    assert metrics["coefficient_recovery_status"] in (None, "skipped")
    assert metrics["identifiability_behavior_status"] in ("pass", "fail", "partial", "skipped")


def test_lattice_sweep_report_schema(tmp_path: Path) -> None:
    specs = mvp_behavioral_lattice_specs()[:3]
    report = run_behavioral_lattice_sweep(tmp_path, specs=specs, lattice_subdir="b5-test")
    assert set(report.keys()) >= REQUIRED_REPORT_FIELDS
    assert report["world_count"] == 3
    assert "behavioral_score" in report["scorecard_summary"]
    assert "structural_score" in report["scorecard_summary"]
    assert "coverage_ratio" in report["scorecard_summary"]
    assert report["per_axis_recovery_summary"]["world_type"]
    assert isinstance(report["skips"], list)


def test_failures_not_swallowed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from mmm.validation.synthetic import behavioral_lattice_sweep as bls

    spec = mvp_behavioral_lattice_specs()[0]

    def boom(*_a: object, **_k: object) -> None:
        raise RuntimeError("dgp failed")

    monkeypatch.setattr(bls, "materialize_dgp_world", boom)
    outcome = run_single_behavioral_world(tmp_path / spec.world_id, spec)
    assert outcome.overall_status == "error"
    assert outcome.error is not None
