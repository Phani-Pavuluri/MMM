"""Deterministic public-simulation producer fixture verification."""

import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import mmm.contracts.public_simulation as public_simulation
from mmm.contracts.mip_failure import MMMFailureCode
from mmm.contracts.public_simulation import (
    MMM_PUBLIC_SIMULATION_SCHEMA_VERSION,
    MMMPublicSimulationExport,
    MMMSimulationPlan,
    MMMSimulationPlanItem,
    MMMSimulationPlanRole,
    MMMSimulationScope,
    MMMSimulationStatus,
    build_mmm_public_simulation_export,
    build_mmm_public_simulation_export_from_payloads,
)
from mmm.contracts.supported_range import (
    MMMRangeAvailabilityStatus,
    MMMRangeBound,
    MMMRangeEvidenceBasis,
    MMMRangeScope,
    MMMSupportedRangeEvidence,
    MMMSupportedRangeRecord,
    MMMSupportedRangeSimulationEligibility,
)

ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "mip_export" / "simulation_v1"
NOW = datetime(2026, 7, 15, tzinfo=timezone.utc)
FORBIDDEN = (
    "/Users/",
    "Traceback",
    "stack trace",
    "DataFrame",
    "Exception(",
    "secret",
    "password",
    "file://",
    "http://",
    "https://",
)


def _scope() -> MMMSimulationScope:
    return MMMSimulationScope(
        metric_id="revenue",
        model_id="ridge-simulation-model-001",
        model_family="ridge",
        model_version="v1",
        configuration_hash="config-simulation-v1",
        panel_id="panel-simulation-v1",
        geography="national",
        segment="all",
        evaluation_start=NOW,
        evaluation_end=NOW.replace(day=16),
        panel_grain="weekly",
        spend_unit="USD",
        spend_scale="RAW",
    )


def _plan(
    role: MMMSimulationPlanRole, spend: float, *, plan_id: str, scope: MMMSimulationScope | None = None
) -> MMMSimulationPlan:
    return MMMSimulationPlan(
        plan_id=plan_id,
        role=role,
        spend_unit="USD",
        evaluation_time_window="2026-W29",
        items=[MMMSimulationPlanItem(channel_id="search", spend=spend, spend_unit="USD")],
        total_spend=spend,
        scope=scope,
    )


def _evidence(
    run_id: str, *, eligibility: MMMSupportedRangeSimulationEligibility, upper: float = 20.0
) -> MMMSupportedRangeEvidence:
    scope = _scope()
    record = MMMSupportedRangeRecord(
        range_record_id=f"range-record:{run_id}",
        run_id=run_id,
        model_id=scope.model_id,
        model_family=scope.model_family,
        model_version=scope.model_version,
        configuration_hash=scope.configuration_hash,
        scope=MMMRangeScope(
            channel="search",
            kpi=scope.metric_id,
            geography=scope.geography,
            segment=scope.segment,
            data_grain=scope.panel_grain,
        ),
        supported_lower=MMMRangeBound(value=0, unit="USD"),
        supported_upper=MMMRangeBound(value=upper, unit="USD"),
        evidence_basis=[MMMRangeEvidenceBasis.OBSERVED_DATA],
        availability_status=MMMRangeAvailabilityStatus.AVAILABLE,
        simulation_eligibility=eligibility,
    )
    return MMMSupportedRangeEvidence(
        evidence_id=f"range-evidence:{run_id}",
        run_id=run_id,
        created_at=NOW,
        producer_package_version="0.1.0",
        records=[record],
    )


def _context() -> SimpleNamespace:
    return SimpleNamespace(schema=SimpleNamespace(channel_columns=("search",), target_column="revenue"))


def _success_result(baseline: float, candidate: float) -> SimpleNamespace:
    return SimpleNamespace(
        aggregation_semantics="full_panel", baseline_mu=baseline, plan_mu=candidate, delta_mu=candidate - baseline
    )


def fixture_payloads() -> dict[str, str]:
    scope = _scope()
    with patch.object(
        public_simulation, "simulate", side_effect=[_success_result(100.0, 120.0), _success_result(100.0, 100.0)]
    ):
        in_range = build_mmm_public_simulation_export(
            export_id="simulation-success-in-range",
            run_id="simulation-run-success-in-range",
            created_at=NOW,
            ctx=_context(),
            baseline_plan=_plan(MMMSimulationPlanRole.BASELINE, 10.0, plan_id="baseline-success", scope=scope),
            candidate_plan=_plan(MMMSimulationPlanRole.CANDIDATE, 20.0, plan_id="candidate-success", scope=scope),
            supported_range_evidence=_evidence(
                "simulation-run-success-in-range",
                eligibility=MMMSupportedRangeSimulationEligibility.ELIGIBLE_FOR_TECHNICAL_SIMULATION,
            ),
        )
        equal = build_mmm_public_simulation_export(
            export_id="simulation-success-equal",
            run_id="simulation-run-success-equal",
            created_at=NOW,
            ctx=_context(),
            baseline_plan=_plan(MMMSimulationPlanRole.BASELINE, 10.0, plan_id="baseline-equal", scope=scope),
            candidate_plan=_plan(MMMSimulationPlanRole.CANDIDATE, 10.0, plan_id="candidate-equal", scope=scope),
            supported_range_evidence=_evidence(
                "simulation-run-success-equal",
                eligibility=MMMSupportedRangeSimulationEligibility.ELIGIBLE_FOR_TECHNICAL_SIMULATION,
            ),
        )
    range_blocked = build_mmm_public_simulation_export(
        export_id="simulation-blocked-unusable-range",
        run_id="simulation-run-blocked-unusable-range",
        created_at=NOW,
        ctx=_context(),
        baseline_plan=_plan(MMMSimulationPlanRole.BASELINE, 10.0, plan_id="baseline-unusable", scope=scope),
        candidate_plan=_plan(MMMSimulationPlanRole.CANDIDATE, 20.0, plan_id="candidate-unusable", scope=scope),
        supported_range_evidence=_evidence(
            "simulation-run-blocked-unusable-range", eligibility=MMMSupportedRangeSimulationEligibility.NOT_ASSESSED
        ),
    )
    extrapolated = build_mmm_public_simulation_export(
        export_id="simulation-blocked-extrapolation",
        run_id="simulation-run-blocked-extrapolation",
        created_at=NOW,
        ctx=_context(),
        baseline_plan=_plan(MMMSimulationPlanRole.BASELINE, 10.0, plan_id="baseline-extrapolation", scope=scope),
        candidate_plan=_plan(MMMSimulationPlanRole.CANDIDATE, 21.0, plan_id="candidate-extrapolation", scope=scope),
        supported_range_evidence=_evidence(
            "simulation-run-blocked-extrapolation",
            eligibility=MMMSupportedRangeSimulationEligibility.ELIGIBLE_FOR_TECHNICAL_SIMULATION,
        ),
    )
    failed_malformed = build_mmm_public_simulation_export_from_payloads(
        export_id="simulation-failed-malformed",
        run_id="simulation-run-failed-malformed",
        created_at=NOW,
        ctx=_context(),
        baseline_payload={},
        candidate_payload={},
        supported_range_evidence=None,
    )
    failed_scope = build_mmm_public_simulation_export(
        export_id="simulation-failed-scope-mismatch",
        run_id="simulation-run-failed-scope-mismatch",
        created_at=NOW,
        ctx=_context(),
        baseline_plan=_plan(MMMSimulationPlanRole.BASELINE, 10.0, plan_id="baseline-scope-mismatch", scope=scope),
        candidate_plan=_plan(
            MMMSimulationPlanRole.CANDIDATE,
            10.0,
            plan_id="candidate-scope-mismatch",
            scope=scope.model_copy(update={"metric_id": "orders"}),
        ),
        supported_range_evidence=None,
    )
    exports = {
        "ridge_in_range_success.json": in_range,
        "ridge_equal_plan_success.json": equal,
        "blocked_unusable_range_evidence.json": range_blocked,
        "blocked_unsupported_extrapolation.json": extrapolated,
        "failed_malformed_caller_plan.json": failed_malformed,
        "failed_scope_mismatch.json": failed_scope,
    }
    return {name: export.to_json() + "\n" for name, export in exports.items()}


def fixture_index() -> str:
    scenarios = [
        (
            "blocked_unsupported_extrapolation",
            "blocked_unsupported_extrapolation.json",
            "simulation-run-blocked-extrapolation",
            "BLOCKED",
            "UNSUPPORTED_EXTRAPOLATION",
            False,
        ),
        (
            "blocked_unusable_range_evidence",
            "blocked_unusable_range_evidence.json",
            "simulation-run-blocked-unusable-range",
            "BLOCKED",
            "SUPPORTED_RANGE_EVIDENCE_UNUSABLE",
            False,
        ),
        (
            "failed_malformed_caller_plan",
            "failed_malformed_caller_plan.json",
            "simulation-run-failed-malformed",
            "FAILED",
            "INVALID_PLAN_INPUT",
            False,
        ),
        (
            "failed_scope_mismatch",
            "failed_scope_mismatch.json",
            "simulation-run-failed-scope-mismatch",
            "FAILED",
            "INVALID_PLAN_INPUT",
            False,
        ),
        (
            "ridge_equal_plan_success",
            "ridge_equal_plan_success.json",
            "simulation-run-success-equal",
            "SUCCEEDED",
            None,
            True,
        ),
        (
            "ridge_in_range_success",
            "ridge_in_range_success.json",
            "simulation-run-success-in-range",
            "SUCCEEDED",
            None,
            True,
        ),
    ]
    payload = {
        "fixture_set_id": "mmm_public_simulation_v1",
        "schema_version": "mmm_public_simulation_fixture_set_v1",
        "producer_package": "mmm",
        "artifact_kind": "MMMPublicSimulationExport",
        "artifact_schema_version": "mmm_public_simulation_export_v1",
        "scenarios": [
            {
                "scenario_id": scenario_id,
                "fixture": fixture,
                "expected_status": status,
                "expected_failure_code": code,
                "expected_artifact_kind": "MMMPublicSimulationExport",
                "expected_artifact_schema_version": MMM_PUBLIC_SIMULATION_SCHEMA_VERSION,
                "expected_artifact_id": f"mmm_public_simulation:{run_id}",
                "comparison_present": comparison,
                "successful_output_present": comparison,
                "run_manifest_present": True,
                "analytical_outcome_present": True,
                "numeric_tolerance": 1e-9,
                "source_builder": "mmm.contracts.public_simulation",
            }
            for scenario_id, fixture, run_id, status, code, comparison in scenarios
        ],
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"


def write_fixtures(root: Path = ROOT) -> None:
    """Regenerate deterministic fixture bytes from the governed public wrapper."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "index.json").write_text(fixture_index())
    for name, payload in fixture_payloads().items():
        (root / name).write_text(payload)


def test_simulation_fixtures_are_generated_by_the_public_wrapper_and_are_stable() -> None:
    generated = fixture_payloads()
    assert fixture_payloads() == generated
    assert fixture_index() == fixture_index()
    for name, payload in generated.items():
        assert (ROOT / name).read_text() == payload
        export = MMMPublicSimulationExport.from_json(payload)
        assert export.to_json() + "\n" == payload


def test_simulation_fixture_generator_writes_identical_bytes_twice(tmp_path: Path) -> None:
    write_fixtures(tmp_path)
    first = {path.name: path.read_bytes() for path in sorted(tmp_path.glob("*.json"))}
    write_fixtures(tmp_path)
    assert {path.name: path.read_bytes() for path in sorted(tmp_path.glob("*.json"))} == first


def test_simulation_fixture_index_and_terminal_linkage_are_complete_and_safe() -> None:
    index = json.loads((ROOT / "index.json").read_text())
    assert index["artifact_kind"] == "MMMPublicSimulationExport"
    assert index["artifact_schema_version"] == "mmm_public_simulation_export_v1"
    assert [item["scenario_id"] for item in index["scenarios"]] == sorted(
        item["scenario_id"] for item in index["scenarios"]
    )
    for scenario in index["scenarios"]:
        payload = (ROOT / scenario["fixture"]).read_text()
        export = MMMPublicSimulationExport.from_json(payload)
        assert export.status.value == scenario["expected_status"]
        assert export.artifact_kind == scenario["expected_artifact_kind"]
        assert export.artifact_schema_version == scenario["expected_artifact_schema_version"]
        assert export.artifact_id == scenario["expected_artifact_id"]
        assert scenario["numeric_tolerance"] == 1e-9
        assert (
            export.run_manifest
            and export.export_manifest_outcome
            and export.export_manifest_outcome.outcome_kind == "analytical_artifact"
        )
        analytical = export.export_manifest_outcome.analytical_outcome
        assert analytical and analytical.run_id == export.run_id == export.run_manifest.run_id
        assert (
            analytical.status.value == export.status.value
            and analytical.producer_package_version == export.producer_package_version
        )
        assert export.run_manifest.producer_package_name == export.producer_package_name
        assert not any(marker.lower() in payload.lower() for marker in FORBIDDEN)
        if export.status in {MMMSimulationStatus.SUCCEEDED, MMMSimulationStatus.SUCCEEDED_WITH_LIMITATIONS}:
            assert export.comparison is not None and analytical.output_artifact == export.run_manifest.successful_export
            assert scenario["successful_output_present"] is True
            assert export.comparison.metrics[0].uncertainty.status.value == "UNAVAILABLE"
        elif export.status == MMMSimulationStatus.BLOCKED:
            assert export.comparison is None and export.blocking_references and export.failure_packet
            assert analytical.output_artifact is None and analytical.failure_packet == export.failure_packet
            assert scenario["successful_output_present"] is False
        else:
            assert export.comparison is None and not export.blocking_references and export.failure_packet
            assert analytical.output_artifact is None and analytical.failure_packet == export.failure_packet
            assert scenario["successful_output_present"] is False
        assert (export.failure_packet.code.value if export.failure_packet else None) == scenario[
            "expected_failure_code"
        ]
    equal = MMMPublicSimulationExport.from_json((ROOT / "ridge_equal_plan_success.json").read_text())
    assert equal.comparison and equal.comparison.metrics[0].delta_mu == 0
    extrapolated = MMMPublicSimulationExport.from_json((ROOT / "blocked_unsupported_extrapolation.json").read_text())
    unusable = MMMPublicSimulationExport.from_json((ROOT / "blocked_unusable_range_evidence.json").read_text())
    assert extrapolated.failure_packet and extrapolated.failure_packet.code == MMMFailureCode.UNSUPPORTED_EXTRAPOLATION
    assert unusable.failure_packet and unusable.failure_packet.code == MMMFailureCode.SUPPORTED_RANGE_EVIDENCE_UNUSABLE


if __name__ == "__main__":
    write_fixtures()
