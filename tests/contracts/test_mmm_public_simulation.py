from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

import mmm.contracts.public_simulation as public_simulation
from mmm.contracts.mip_failure import MMMFailureCode, MMMFailureStage, build_mmm_failure_packet
from mmm.contracts.public_simulation import (
    MMM_PACKAGE_VERSION,
    MMM_PUBLIC_SIMULATION_SCHEMA_VERSION,
    MMMPublicSimulationExport,
    MMMSimulationComparison,
    MMMSimulationMetricResult,
    MMMSimulationPlan,
    MMMSimulationPlanItem,
    MMMSimulationPlanRole,
    MMMSimulationScope,
    MMMSimulationStatus,
    MMMSimulationUncertaintyStatus,
    _resolve_range_record,
    build_mmm_public_simulation_export,
    build_mmm_public_simulation_export_from_payloads,
)
from mmm.contracts.run_manifest import (
    MMMAnalyticalArtifactOutcome,
    MMMArtifactReference,
    MMMExportManifestOutcome,
    MMMRunManifest,
    MMMRunStatus,
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

NOW = datetime(2026, 7, 15, tzinfo=timezone.utc)


def plan(role, spend=10.0):
    return MMMSimulationPlan(
        plan_id=role.value,
        role=role,
        spend_unit="USD",
        evaluation_time_window="2026",
        items=[MMMSimulationPlanItem(channel_id="search", spend=spend, spend_unit="USD")],
        total_spend=spend,
    )


def comparison():
    return MMMSimulationComparison(
        comparison_id="c",
        run_id="run",
        baseline_plan_id="BASELINE",
        candidate_plan_id="CANDIDATE",
        status=MMMSimulationStatus.SUCCEEDED,
        technical_summary="technical",
        metrics=[
            MMMSimulationMetricResult(
                metric_id="revenue",
                estimand="full_panel_delta_mu",
                aggregation_scope="full",
                baseline_mu=1,
                candidate_mu=2,
                delta_mu=1,
                unit="model",
            )
        ],
    )


def packet():
    return build_mmm_failure_packet(
        failure_id="f",
        created_at=NOW,
        run_id="run",
        code=MMMFailureCode.UNSUPPORTED_EXTRAPOLATION,
        stage=MMMFailureStage.SIMULATION,
        source_component="mmm.public_simulation",
        technical_summary="Blocked range",
        affected_resource="candidate",
    )


def failed_packet():
    return build_mmm_failure_packet(
        failure_id="failed",
        created_at=NOW,
        run_id="run",
        code=MMMFailureCode.INVALID_PLAN_INPUT,
        stage=MMMFailureStage.SIMULATION,
        source_component="mmm.public_simulation",
        technical_summary="Invalid plan",
        affected_resource="candidate",
        failure_status="failed",
    )


def manifest(status=MMMRunStatus.SUCCEEDED, limitation_ids=None):
    return MMMRunManifest(
        manifest_id="manifest:e",
        run_id="run",
        created_at=NOW,
        started_at=NOW,
        completed_at=NOW,
        producer_package_version=MMM_PACKAGE_VERSION,
        status=status,
        model_family="ridge",
        successful_export=MMMArtifactReference(
            artifact_type="MMMPublicSimulationExport",
            artifact_id="mmm_public_simulation:run",
            contract_version=MMM_PUBLIC_SIMULATION_SCHEMA_VERSION,
            logical_name="mmm_public_simulation",
        ),
        limitation_ids=limitation_ids or [],
    )


def blocked_manifest(failure):
    return MMMRunManifest(
        manifest_id="manifest:e",
        run_id="run",
        created_at=NOW,
        started_at=NOW,
        completed_at=NOW,
        producer_package_version=MMM_PACKAGE_VERSION,
        status=MMMRunStatus.BLOCKED,
        model_family="ridge",
        failure_packet=failure,
    )


def failed_manifest(failure):
    return MMMRunManifest(
        manifest_id="manifest:e",
        run_id="run",
        created_at=NOW,
        started_at=NOW,
        completed_at=NOW,
        producer_package_version=MMM_PACKAGE_VERSION,
        status=MMMRunStatus.FAILED,
        failure_packet=failure,
    )


def analytical_outcome(run_manifest, status=MMMRunStatus.SUCCEEDED, limitation_ids=None):
    artifact = run_manifest.successful_export
    analytical = MMMAnalyticalArtifactOutcome(
        status=status,
        run_id="run",
        producer_package_version=MMM_PACKAGE_VERSION,
        output_artifact=artifact,
        limitation_ids=limitation_ids or [],
    )
    return MMMExportManifestOutcome(
        outcome_kind="analytical_artifact",
        analytical_outcome=analytical,
        run_manifest=run_manifest,
        supported_range_evidence_id=run_manifest.supported_range_evidence_id,
    )


def blocked_analytical_outcome(run_manifest, failure, blockers):
    analytical = MMMAnalyticalArtifactOutcome(
        status=MMMRunStatus.BLOCKED,
        run_id="run",
        producer_package_version=MMM_PACKAGE_VERSION,
        failure_packet=failure,
        blocker_references=blockers,
    )
    return MMMExportManifestOutcome(
        outcome_kind="analytical_artifact",
        analytical_outcome=analytical,
        run_manifest=run_manifest,
        supported_range_evidence_id="range" if run_manifest.supported_range_evidence_id else None,
    )


def failed_analytical_outcome(run_manifest, failure):
    analytical = MMMAnalyticalArtifactOutcome(
        status=MMMRunStatus.FAILED, run_id="run", producer_package_version=MMM_PACKAGE_VERSION, failure_packet=failure
    )
    return MMMExportManifestOutcome(
        outcome_kind="analytical_artifact", analytical_outcome=analytical, run_manifest=run_manifest
    )


def export(**kw):
    d = dict(
        export_id="e",
        run_id="run",
        created_at=NOW,
        baseline_plan=plan(MMMSimulationPlanRole.BASELINE),
        candidate_plan=plan(MMMSimulationPlanRole.CANDIDATE),
        supported_range_evidence_id="range",
        comparison=comparison(),
    )
    d.update(kw)
    if d.get("status", MMMSimulationStatus.SUCCEEDED) == MMMSimulationStatus.SUCCEEDED:
        d.setdefault("run_manifest", manifest())
    if d.get("status") == MMMSimulationStatus.SUCCEEDED_WITH_LIMITATIONS:
        d.setdefault("run_manifest", manifest(MMMRunStatus.SUCCEEDED_WITH_LIMITATIONS, d.get("limitation_references")))
    if (
        d.get("status", MMMSimulationStatus.SUCCEEDED) == MMMSimulationStatus.SUCCEEDED
        and d.get("run_manifest") is not None
    ):
        d.setdefault("export_manifest_outcome", analytical_outcome(d["run_manifest"]))
    if d.get("status") == MMMSimulationStatus.SUCCEEDED_WITH_LIMITATIONS and d.get("run_manifest") is not None:
        d.setdefault(
            "export_manifest_outcome",
            analytical_outcome(
                d["run_manifest"], MMMRunStatus.SUCCEEDED_WITH_LIMITATIONS, d.get("limitation_references")
            ),
        )
    if d.get("status") == MMMSimulationStatus.BLOCKED and d.get("failure_packet") is not None:
        d.setdefault("run_manifest", blocked_manifest(d["failure_packet"]))
    if (
        d.get("status") == MMMSimulationStatus.BLOCKED
        and d.get("run_manifest") is not None
        and d.get("failure_packet") is not None
    ):
        d.setdefault(
            "export_manifest_outcome",
            blocked_analytical_outcome(d["run_manifest"], d["failure_packet"], d.get("blocking_references", [])),
        )
    if d.get("status") == MMMSimulationStatus.FAILED and d.get("failure_packet") is not None:
        d.setdefault("run_manifest", failed_manifest(d["failure_packet"]))
    if (
        d.get("status") == MMMSimulationStatus.FAILED
        and d.get("run_manifest") is not None
        and d.get("failure_packet") is not None
    ):
        d.setdefault("export_manifest_outcome", failed_analytical_outcome(d["run_manifest"], d["failure_packet"]))
    return MMMPublicSimulationExport(**d)


def test_terminal_states_and_serialization():
    assert MMMPublicSimulationExport.from_json(export().to_json()) == export()
    assert export().producer_package_name == "mmm" and export().producer_package_version
    assert export().artifact_kind == "MMMPublicSimulationExport" and export().artifact_id == "mmm_public_simulation:run"
    assert (
        export(status=MMMSimulationStatus.SUCCEEDED_WITH_LIMITATIONS, limitation_references=["lim"]).status
        == MMMSimulationStatus.SUCCEEDED_WITH_LIMITATIONS
    )
    assert (
        export(
            status=MMMSimulationStatus.BLOCKED, comparison=None, failure_packet=packet(), blocking_references=["range"]
        ).status
        == MMMSimulationStatus.BLOCKED
    )
    assert (
        export(status=MMMSimulationStatus.FAILED, comparison=None, failure_packet=failed_packet()).status
        == MMMSimulationStatus.FAILED
    )
    with pytest.raises(ValidationError):
        export(status=MMMSimulationStatus.BLOCKED, comparison=None, failure_packet=packet())
    with pytest.raises(ValidationError):
        export(status=MMMSimulationStatus.FAILED)


def test_success_manifest_linkage_is_strict_and_round_trips():
    out = export()
    assert out.run_manifest and out.run_manifest.status == MMMRunStatus.SUCCEEDED
    assert out.run_manifest.run_id == out.run_id
    assert out.run_manifest.producer_package_name == out.producer_package_name
    assert out.run_manifest.producer_package_version == out.producer_package_version
    assert out.run_manifest.successful_export and out.run_manifest.successful_export.artifact_id == out.artifact_id
    assert (
        out.export_manifest_outcome
        and out.export_manifest_outcome.outcome_kind == "analytical_artifact"
        and out.export_manifest_outcome.analytical_outcome
    )
    assert out.run_manifest.failure_packet is None and not out.blocking_references
    assert MMMPublicSimulationExport.from_json(out.to_json()) == out
    with pytest.raises(ValidationError):
        export(run_manifest=None)
    with pytest.raises(ValidationError):
        export(export_manifest_outcome=None)
    limited = export(status=MMMSimulationStatus.SUCCEEDED_WITH_LIMITATIONS, limitation_references=["limitation:range"])
    assert limited.run_manifest and limited.run_manifest.status == MMMRunStatus.SUCCEEDED_WITH_LIMITATIONS
    assert limited.run_manifest.limitation_ids == ["limitation:range"]
    assert limited.export_manifest_outcome and limited.export_manifest_outcome.analytical_outcome.limitation_ids == [
        "limitation:range"
    ]
    with pytest.raises(ValidationError):
        export(
            status=MMMSimulationStatus.SUCCEEDED_WITH_LIMITATIONS,
            limitation_references=["limitation:range"],
            run_manifest=None,
        )
    with pytest.raises(ValidationError):
        export(
            run_manifest=manifest(status=MMMRunStatus.SUCCEEDED_WITH_LIMITATIONS, limitation_ids=["limitation:range"])
        )
    with pytest.raises(ValidationError):
        export(run_manifest=manifest().model_copy(update={"model_family": "bayesian"}))
    with pytest.raises(ValidationError):
        export(run_manifest=manifest().model_copy(update={"producer_package_name": "other"}))
    with pytest.raises(ValidationError):
        export(
            run_manifest=manifest().model_copy(
                update={
                    "successful_export": MMMArtifactReference(
                        artifact_type="MMMPublicSimulationExport",
                        artifact_id="wrong",
                        contract_version=MMM_PUBLIC_SIMULATION_SCHEMA_VERSION,
                    )
                }
            )
        )
    with pytest.raises(ValidationError):
        export(
            export_manifest_outcome=analytical_outcome(manifest()).model_copy(update={"outcome_kind": "export_bundle"})
        )
    mismatched = analytical_outcome(manifest()).model_copy(
        update={
            "analytical_outcome": analytical_outcome(manifest()).analytical_outcome.model_copy(
                update={"run_id": "other"}
            )
        }
    )
    with pytest.raises(ValidationError):
        export(export_manifest_outcome=mismatched)
    mismatched_artifact = analytical_outcome(manifest()).model_copy(
        update={
            "analytical_outcome": analytical_outcome(manifest()).analytical_outcome.model_copy(
                update={
                    "output_artifact": MMMArtifactReference(
                        artifact_type="MMMPublicSimulationExport",
                        artifact_id="wrong",
                        contract_version=MMM_PUBLIC_SIMULATION_SCHEMA_VERSION,
                    )
                }
            )
        }
    )
    with pytest.raises(ValidationError):
        export(export_manifest_outcome=mismatched_artifact)
    mismatched_status = analytical_outcome(manifest()).model_copy(
        update={
            "analytical_outcome": analytical_outcome(manifest()).analytical_outcome.model_copy(
                update={"status": MMMRunStatus.SUCCEEDED_WITH_LIMITATIONS}
            )
        }
    )
    with pytest.raises(ValidationError):
        export(export_manifest_outcome=mismatched_status)
    mismatched_producer = analytical_outcome(manifest()).model_copy(
        update={
            "analytical_outcome": analytical_outcome(manifest()).analytical_outcome.model_copy(
                update={"producer_package_version": "0.2.0"}
            )
        }
    )
    with pytest.raises(ValidationError):
        export(export_manifest_outcome=mismatched_producer)
    bad_limited = analytical_outcome(
        manifest(MMMRunStatus.SUCCEEDED_WITH_LIMITATIONS, ["limitation:range"]),
        MMMRunStatus.SUCCEEDED_WITH_LIMITATIONS,
        ["limitation:range"],
    ).model_copy(
        update={
            "analytical_outcome": analytical_outcome(
                manifest(MMMRunStatus.SUCCEEDED_WITH_LIMITATIONS, ["limitation:range"]),
                MMMRunStatus.SUCCEEDED_WITH_LIMITATIONS,
                ["limitation:range"],
            ).analytical_outcome.model_copy(update={"limitation_ids": ["limitation:other"]})
        }
    )
    with pytest.raises(ValidationError):
        export(
            status=MMMSimulationStatus.SUCCEEDED_WITH_LIMITATIONS,
            limitation_references=["limitation:range"],
            export_manifest_outcome=bad_limited,
        )
    assert (
        export(
            status=MMMSimulationStatus.BLOCKED, comparison=None, failure_packet=packet(), blocking_references=["range"]
        ).run_manifest
        is not None
    )
    assert (
        export(status=MMMSimulationStatus.FAILED, comparison=None, failure_packet=failed_packet()).run_manifest
        is not None
    )


def test_blocked_manifest_linkage_is_strict_and_round_trips():
    out = export(
        status=MMMSimulationStatus.BLOCKED,
        comparison=None,
        failure_packet=packet(),
        blocking_references=["missing_or_scope_incompatible"],
    )
    assert out.run_manifest and out.run_manifest.status == MMMRunStatus.BLOCKED
    assert out.run_manifest.run_id == out.run_id and out.run_manifest.failure_packet == out.failure_packet
    assert out.run_manifest.successful_export is None and out.run_manifest.steps == []
    assert (
        out.export_manifest_outcome
        and out.export_manifest_outcome.outcome_kind == "analytical_artifact"
        and out.export_manifest_outcome.analytical_outcome
    )
    assert (
        out.export_manifest_outcome.analytical_outcome.failure_packet == out.failure_packet
        and out.export_manifest_outcome.analytical_outcome.blocker_references == out.blocking_references
    )
    assert MMMPublicSimulationExport.from_json(out.to_json()) == out
    with pytest.raises(ValidationError):
        export(
            status=MMMSimulationStatus.BLOCKED,
            comparison=None,
            failure_packet=packet(),
            blocking_references=["range"],
            export_manifest_outcome=None,
        )
    with pytest.raises(ValidationError):
        export(
            status=MMMSimulationStatus.BLOCKED,
            comparison=None,
            failure_packet=packet(),
            blocking_references=["range"],
            run_manifest=None,
        )
    with pytest.raises(ValidationError):
        export(
            status=MMMSimulationStatus.BLOCKED,
            comparison=None,
            failure_packet=packet(),
            blocking_references=["range"],
            run_manifest=blocked_manifest(packet()).model_copy(update={"run_id": "other"}),
        )
    with pytest.raises(ValidationError):
        export(
            status=MMMSimulationStatus.BLOCKED,
            comparison=None,
            failure_packet=packet(),
            blocking_references=["range"],
            run_manifest=blocked_manifest(packet()).model_copy(update={"producer_package_name": "other"}),
        )
    with pytest.raises(ValidationError):
        export(
            status=MMMSimulationStatus.BLOCKED,
            comparison=None,
            failure_packet=packet(),
            blocking_references=["range"],
            run_manifest=blocked_manifest(packet()).model_copy(update={"model_family": "bayesian"}),
        )
    other = build_mmm_failure_packet(
        failure_id="other",
        created_at=NOW,
        run_id="run",
        code=MMMFailureCode.SUPPORTED_RANGE_EVIDENCE_UNUSABLE,
        stage=MMMFailureStage.SIMULATION,
        source_component="mmm.public_simulation",
        technical_summary="Blocked range",
        affected_resource="candidate",
    )
    with pytest.raises(ValidationError):
        export(
            status=MMMSimulationStatus.BLOCKED,
            comparison=None,
            failure_packet=packet(),
            blocking_references=["range"],
            run_manifest=blocked_manifest(other),
        )
    current = blocked_analytical_outcome(blocked_manifest(packet()), packet(), ["range"])
    with pytest.raises(ValidationError):
        export(
            status=MMMSimulationStatus.BLOCKED,
            comparison=None,
            failure_packet=packet(),
            blocking_references=["range"],
            export_manifest_outcome=current.model_copy(update={"outcome_kind": "export_bundle"}),
        )
    with pytest.raises(ValidationError):
        export(
            status=MMMSimulationStatus.BLOCKED,
            comparison=None,
            failure_packet=packet(),
            blocking_references=["range"],
            export_manifest_outcome=current.model_copy(
                update={"analytical_outcome": current.analytical_outcome.model_copy(update={"run_id": "other"})}
            ),
        )
    with pytest.raises(ValidationError):
        export(
            status=MMMSimulationStatus.BLOCKED,
            comparison=None,
            failure_packet=packet(),
            blocking_references=["range"],
            export_manifest_outcome=current.model_copy(
                update={
                    "analytical_outcome": current.analytical_outcome.model_copy(
                        update={"producer_package_version": "0.2.0"}
                    )
                }
            ),
        )
    with pytest.raises(ValidationError):
        export(
            status=MMMSimulationStatus.BLOCKED,
            comparison=None,
            failure_packet=packet(),
            blocking_references=["range"],
            export_manifest_outcome=current.model_copy(
                update={
                    "analytical_outcome": current.analytical_outcome.model_copy(update={"status": MMMRunStatus.FAILED})
                }
            ),
        )
    with pytest.raises(ValidationError):
        export(
            status=MMMSimulationStatus.BLOCKED,
            comparison=None,
            failure_packet=packet(),
            blocking_references=["range"],
            export_manifest_outcome=current.model_copy(
                update={"analytical_outcome": current.analytical_outcome.model_copy(update={"failure_packet": other})}
            ),
        )
    with pytest.raises(ValidationError):
        export(
            status=MMMSimulationStatus.BLOCKED,
            comparison=None,
            failure_packet=packet(),
            blocking_references=["range"],
            export_manifest_outcome=current.model_copy(
                update={
                    "analytical_outcome": current.analytical_outcome.model_copy(
                        update={"blocker_references": ["other"]}
                    )
                }
            ),
        )


def test_failed_manifest_linkage_is_strict_and_round_trips():
    out = export(status=MMMSimulationStatus.FAILED, comparison=None, failure_packet=failed_packet())
    assert out.run_manifest and out.run_manifest.status == MMMRunStatus.FAILED
    assert out.run_manifest.run_id == out.run_id and out.run_manifest.failure_packet == out.failure_packet
    assert out.run_manifest.successful_export is None and out.run_manifest.model_family is None
    assert (
        out.export_manifest_outcome
        and out.export_manifest_outcome.outcome_kind == "analytical_artifact"
        and out.export_manifest_outcome.analytical_outcome
    )
    assert (
        out.export_manifest_outcome.analytical_outcome.status == MMMRunStatus.FAILED
        and out.export_manifest_outcome.analytical_outcome.failure_packet == out.failure_packet
    )
    assert MMMPublicSimulationExport.from_json(out.to_json()) == out
    with pytest.raises(ValidationError):
        export(
            status=MMMSimulationStatus.FAILED,
            comparison=None,
            failure_packet=failed_packet(),
            export_manifest_outcome=None,
        )
    with pytest.raises(ValidationError):
        export(status=MMMSimulationStatus.FAILED, comparison=None, failure_packet=failed_packet(), run_manifest=None)
    with pytest.raises(ValidationError):
        export(
            status=MMMSimulationStatus.FAILED,
            comparison=None,
            failure_packet=failed_packet(),
            run_manifest=failed_manifest(failed_packet()).model_copy(update={"run_id": "other"}),
        )
    with pytest.raises(ValidationError):
        export(
            status=MMMSimulationStatus.FAILED,
            comparison=None,
            failure_packet=failed_packet(),
            run_manifest=failed_manifest(failed_packet()).model_copy(update={"producer_package_name": "other"}),
        )
    with pytest.raises(ValidationError):
        export(
            status=MMMSimulationStatus.FAILED,
            comparison=None,
            failure_packet=failed_packet(),
            run_manifest=failed_manifest(failed_packet()).model_copy(update={"model_family": "bayesian"}),
        )
    other = build_mmm_failure_packet(
        failure_id="other-failed",
        created_at=NOW,
        run_id="run",
        code=MMMFailureCode.INVALID_PLAN_INPUT,
        stage=MMMFailureStage.SIMULATION,
        source_component="mmm.public_simulation",
        technical_summary="Invalid plan",
        affected_resource="candidate",
        failure_status="failed",
    )
    with pytest.raises(ValidationError):
        export(
            status=MMMSimulationStatus.FAILED,
            comparison=None,
            failure_packet=failed_packet(),
            run_manifest=failed_manifest(other),
        )
    current = failed_analytical_outcome(failed_manifest(failed_packet()), failed_packet())
    with pytest.raises(ValidationError):
        export(
            status=MMMSimulationStatus.FAILED,
            comparison=None,
            failure_packet=failed_packet(),
            export_manifest_outcome=current.model_copy(update={"outcome_kind": "export_bundle"}),
        )
    with pytest.raises(ValidationError):
        export(
            status=MMMSimulationStatus.FAILED,
            comparison=None,
            failure_packet=failed_packet(),
            export_manifest_outcome=current.model_copy(
                update={"analytical_outcome": current.analytical_outcome.model_copy(update={"run_id": "other"})}
            ),
        )
    with pytest.raises(ValidationError):
        export(
            status=MMMSimulationStatus.FAILED,
            comparison=None,
            failure_packet=failed_packet(),
            export_manifest_outcome=current.model_copy(
                update={
                    "analytical_outcome": current.analytical_outcome.model_copy(
                        update={"producer_package_version": "0.2.0"}
                    )
                }
            ),
        )
    with pytest.raises(ValidationError):
        export(
            status=MMMSimulationStatus.FAILED,
            comparison=None,
            failure_packet=failed_packet(),
            export_manifest_outcome=current.model_copy(
                update={
                    "analytical_outcome": current.analytical_outcome.model_copy(update={"status": MMMRunStatus.BLOCKED})
                }
            ),
        )
    with pytest.raises(ValidationError):
        export(
            status=MMMSimulationStatus.FAILED,
            comparison=None,
            failure_packet=failed_packet(),
            export_manifest_outcome=current.model_copy(
                update={"analytical_outcome": current.analytical_outcome.model_copy(update={"failure_packet": other})}
            ),
        )
    with pytest.raises(ValidationError):
        export(
            status=MMMSimulationStatus.FAILED,
            comparison=None,
            failure_packet=failed_packet(),
            export_manifest_outcome=current.model_copy(
                update={
                    "analytical_outcome": current.analytical_outcome.model_copy(
                        update={"blocker_references": ["blocker"]}
                    )
                }
            ),
        )
    with pytest.raises(ValidationError):
        export(
            status=MMMSimulationStatus.FAILED,
            comparison=None,
            failure_packet=failed_packet(),
            export_manifest_outcome=current.model_copy(
                update={
                    "analytical_outcome": current.analytical_outcome.model_copy(
                        update={"limitation_ids": ["limitation:range"]}
                    )
                }
            ),
        )
    with pytest.raises(ValidationError):
        export(
            status=MMMSimulationStatus.FAILED,
            comparison=None,
            failure_packet=failed_packet(),
            export_manifest_outcome=current.model_copy(
                update={
                    "analytical_outcome": current.analytical_outcome.model_copy(
                        update={
                            "output_artifact": MMMArtifactReference(
                                artifact_type="MMMPublicSimulationExport",
                                artifact_id="wrong",
                                contract_version=MMM_PUBLIC_SIMULATION_SCHEMA_VERSION,
                            )
                        }
                    )
                }
            ),
        )


def test_plan_and_delta_validation():
    with pytest.raises(ValidationError):
        plan(MMMSimulationPlanRole.BASELINE, -1)
    with pytest.raises(ValidationError):
        MMMSimulationMetricResult(
            metric_id="m", estimand="e", aggregation_scope="full", baseline_mu=1, candidate_mu=2, delta_mu=0, unit="u"
        )


def test_typed_scope_is_strict_and_round_trips():
    scope = MMMSimulationScope(
        metric_id="revenue",
        model_id="model",
        model_family="ridge",
        panel_id="sha256:panel",
        evaluation_start=NOW,
        evaluation_end=NOW.replace(day=16),
        panel_grain="weekly",
        spend_unit="USD",
        spend_scale="RAW",
    )
    assert MMMSimulationScope.model_validate_json(scope.model_dump_json()) == scope
    with pytest.raises(ValidationError):
        MMMSimulationScope(
            metric_id="revenue",
            model_id="model",
            model_family="",
            panel_id="panel",
            evaluation_start=NOW,
            evaluation_end=NOW.replace(day=16),
            panel_grain="weekly",
            spend_unit="USD",
            spend_scale="RAW",
        )
    with pytest.raises(ValidationError):
        MMMSimulationScope(
            metric_id="revenue",
            model_id="model",
            model_family="ridge",
            panel_id="panel",
            evaluation_start=NOW,
            evaluation_end=NOW.replace(day=16),
            panel_grain="weekly",
            spend_unit="USD",
            spend_scale="TRANSFORMED",
        )


def test_scope_mismatch_is_failed_before_runtime_invocation():
    scope = MMMSimulationScope(
        metric_id="revenue",
        model_id="model",
        model_family="ridge",
        panel_id="panel",
        evaluation_start=NOW,
        evaluation_end=NOW.replace(day=16),
        panel_grain="weekly",
        spend_unit="USD",
        spend_scale="RAW",
    )
    other = scope.model_copy(update={"metric_id": "orders"})
    baseline = plan(MMMSimulationPlanRole.BASELINE).model_copy(update={"scope": scope})
    candidate = plan(MMMSimulationPlanRole.CANDIDATE).model_copy(update={"scope": other})
    out = build_mmm_public_simulation_export(
        export_id="scope",
        run_id="run",
        created_at=NOW,
        ctx=None,
        baseline_plan=baseline,
        candidate_plan=candidate,
        supported_range_evidence=None,
    )
    assert out.status == MMMSimulationStatus.FAILED and out.comparison is None and not out.blocking_references
    assert (
        out.failure_packet
        and out.failure_packet.code == MMMFailureCode.INVALID_PLAN_INPUT
        and out.failure_packet.run_id == out.run_id
    )
    assert (
        out.run_manifest
        and out.run_manifest.status == MMMRunStatus.FAILED
        and [ref.artifact_id for ref in out.run_manifest.steps[0].input_artifacts]
        == [baseline.plan_id, candidate.plan_id]
    )
    assert out.run_manifest.model_id == scope.model_id and out.run_manifest.model_family == "ridge"
    assert (
        out.export_manifest_outcome
        and out.export_manifest_outcome.analytical_outcome
        and out.export_manifest_outcome.analytical_outcome.failure_packet == out.failure_packet
    )
    assert (
        out.export_manifest_outcome
        and out.export_manifest_outcome.analytical_outcome
        and out.export_manifest_outcome.analytical_outcome.output_artifact is None
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("metric_id", "orders"),
        ("outcome_id", "sales"),
        ("estimand_id", "delta"),
        ("model_id", "other"),
        ("model_family", "bayesian"),
        ("model_version", "v2"),
        ("configuration_hash", "sha256:other"),
        ("panel_id", "other-panel"),
        ("geography", "west"),
        ("segment", "new"),
        ("evaluation_start", NOW.replace(hour=1)),
        ("evaluation_end", NOW.replace(day=17)),
        ("panel_grain", "daily"),
        ("spend_unit", "EUR"),
        ("spend_scale", "TRANSFORMED"),
        ("transformation_id", "log1p"),
    ],
)
def test_each_scope_dimension_mismatch_is_typed_failed(field, value):
    common = dict(
        metric_id="revenue",
        model_id="model",
        model_family="ridge",
        panel_id="panel",
        evaluation_start=NOW,
        evaluation_end=NOW.replace(day=16),
        panel_grain="weekly",
        spend_unit="USD",
        spend_scale="RAW",
        outcome_id="outcome",
        estimand_id="estimand",
        model_version="v1",
        configuration_hash="sha256:cfg",
        geography="national",
        segment="all",
    )
    baseline_scope = MMMSimulationScope(**common)
    if field == "spend_scale":
        common["transformation_id"] = "log1p"
    candidate_values = {**common, field: value}
    if field == "transformation_id":
        candidate_values["spend_scale"] = "TRANSFORMED"
    candidate_scope = MMMSimulationScope(**candidate_values)
    out = build_mmm_public_simulation_export(
        export_id="scope",
        run_id="run",
        created_at=NOW,
        ctx=None,
        baseline_plan=plan(MMMSimulationPlanRole.BASELINE).model_copy(update={"scope": baseline_scope}),
        candidate_plan=plan(MMMSimulationPlanRole.CANDIDATE).model_copy(update={"scope": candidate_scope}),
        supported_range_evidence=None,
    )
    assert out.status == MMMSimulationStatus.FAILED and out.comparison is None and not out.blocking_references
    assert (
        out.failure_packet
        and out.failure_packet.code == MMMFailureCode.INVALID_PLAN_INPUT
        and out.failure_packet.run_id == "run"
    )
    assert "traceback" not in out.to_json().lower() and "/users/" not in out.to_json().lower()
    assert (
        out.run_manifest
        and out.run_manifest.failure_packet == out.failure_packet
        and [ref.artifact_id for ref in out.run_manifest.steps[0].input_artifacts] == ["BASELINE", "CANDIDATE"]
    )
    assert (
        out.export_manifest_outcome
        and out.export_manifest_outcome.analytical_outcome
        and out.export_manifest_outcome.analytical_outcome.failure_packet == out.failure_packet
    )


def test_optional_scope_asymmetry_and_matching_scope_preserve_range_block():
    scope = MMMSimulationScope(
        metric_id="revenue",
        model_id="model",
        model_family="ridge",
        panel_id="panel",
        evaluation_start=NOW,
        evaluation_end=NOW.replace(day=16),
        panel_grain="weekly",
        spend_unit="USD",
        spend_scale="RAW",
    )
    b = plan(MMMSimulationPlanRole.BASELINE).model_copy(update={"scope": scope})
    c = plan(MMMSimulationPlanRole.CANDIDATE).model_copy(
        update={"scope": scope.model_copy(update={"geography": "national"})}
    )
    failed = build_mmm_public_simulation_export(
        export_id="scope",
        run_id="run",
        created_at=NOW,
        ctx=None,
        baseline_plan=b,
        candidate_plan=c,
        supported_range_evidence=None,
    )
    assert failed.status == MMMSimulationStatus.FAILED
    blocked = build_mmm_public_simulation_export(
        export_id="range",
        run_id="run",
        created_at=NOW,
        ctx=None,
        baseline_plan=b,
        candidate_plan=b.model_copy(update={"role": MMMSimulationPlanRole.CANDIDATE}),
        supported_range_evidence=None,
    )
    assert (
        blocked.status == MMMSimulationStatus.BLOCKED
        and blocked.comparison is None
        and blocked.run_manifest
        and blocked.run_manifest.status == MMMRunStatus.BLOCKED
    )


def test_exact_range_record_resolution_is_unambiguous_and_dimension_strict():
    scope = MMMSimulationScope(
        metric_id="revenue",
        model_id="model",
        model_family="ridge",
        panel_id="panel",
        evaluation_start=NOW,
        evaluation_end=NOW.replace(day=16),
        panel_grain="weekly",
        spend_unit="USD",
        spend_scale="RAW",
    )
    item = MMMSimulationPlanItem(channel_id="search", spend=10, spend_unit="USD")
    record = MMMSupportedRangeRecord(
        range_record_id="range",
        run_id="run",
        model_id="model",
        model_family="ridge",
        scope=MMMRangeScope(channel="search", kpi="revenue", data_grain="weekly"),
        supported_lower=MMMRangeBound(value=0, unit="USD"),
        supported_upper=MMMRangeBound(value=10, unit="USD"),
        evidence_basis=[MMMRangeEvidenceBasis.OBSERVED_DATA],
        availability_status=MMMRangeAvailabilityStatus.AVAILABLE,
    )
    evidence = MMMSupportedRangeEvidence(
        evidence_id="evidence", run_id="run", created_at=NOW, producer_package_version="0", records=[record]
    )
    assert _resolve_range_record(evidence, item, scope) == [record]
    for update in ({"channel_id": "social"}, {"spend_unit": "EUR"}):
        assert _resolve_range_record(evidence, item.model_copy(update=update), scope) == []
    duplicate = record.model_copy(update={"range_record_id": "range-2"})
    ambiguous = MMMSupportedRangeEvidence(
        evidence_id="evidence", run_id="run", created_at=NOW, producer_package_version="0", records=[record, duplicate]
    )
    assert len(_resolve_range_record(ambiguous, item, scope)) == 2


@pytest.mark.parametrize(
    "record_update,scope_update",
    [
        ({"scope": MMMRangeScope(channel="search", kpi="orders", data_grain="weekly")}, {}),
        ({"model_id": "other"}, {}),
        ({"model_family": "bayesian"}, {}),
        ({"model_version": "v2"}, {"model_version": "v1"}),
        ({"configuration_hash": "other"}, {"configuration_hash": "cfg"}),
        (
            {"scope": MMMRangeScope(channel="search", kpi="revenue", geography="west", data_grain="weekly")},
            {"geography": "national"},
        ),
        (
            {"scope": MMMRangeScope(channel="search", kpi="revenue", segment="new", data_grain="weekly")},
            {"segment": "all"},
        ),
        ({"scope": MMMRangeScope(channel="search", kpi="revenue", data_grain="daily")}, {}),
        (
            {"scope": MMMRangeScope(channel="search", kpi="revenue", data_grain="weekly", transformation_id="log1p")},
            {"spend_scale": "TRANSFORMED", "transformation_id": "sqrt"},
        ),
    ],
)
def test_remaining_range_scope_dimensions_do_not_resolve(record_update, scope_update):
    scope_values = {
        "metric_id": "revenue",
        "model_id": "model",
        "model_family": "ridge",
        "panel_id": "panel",
        "evaluation_start": NOW,
        "evaluation_end": NOW.replace(day=16),
        "panel_grain": "weekly",
        "spend_unit": "USD",
        "spend_scale": "RAW",
    }
    scope = MMMSimulationScope(**{**scope_values, **scope_update})
    item = MMMSimulationPlanItem(channel_id="search", spend=10, spend_unit="USD")
    data = dict(
        range_record_id="range",
        run_id="run",
        model_id="model",
        model_family="ridge",
        scope=MMMRangeScope(channel="search", kpi="revenue", data_grain="weekly"),
        supported_lower=MMMRangeBound(value=0, unit="USD"),
        supported_upper=MMMRangeBound(value=10, unit="USD"),
        evidence_basis=[MMMRangeEvidenceBasis.OBSERVED_DATA],
        availability_status=MMMRangeAvailabilityStatus.AVAILABLE,
    )
    data.update(record_update)
    record = MMMSupportedRangeRecord(**data)
    evidence = MMMSupportedRangeEvidence(
        evidence_id="evidence", run_id="run", created_at=NOW, producer_package_version="0", records=[record]
    )
    assert _resolve_range_record(evidence, item, scope) == []


def test_incompatible_range_evidence_short_circuits_runtime_and_keeps_candidate(monkeypatch):
    ctx = SimpleNamespace(schema=SimpleNamespace(channel_columns=("search",)))
    b = plan(MMMSimulationPlanRole.BASELINE)
    c = plan(MMMSimulationPlanRole.CANDIDATE, 20)
    record = MMMSupportedRangeRecord(
        range_record_id="range",
        run_id="run",
        model_id=None,
        model_family=None,
        scope=MMMRangeScope(channel="social"),
        supported_lower=MMMRangeBound(value=0, unit="USD"),
        supported_upper=MMMRangeBound(value=30, unit="USD"),
        evidence_basis=[MMMRangeEvidenceBasis.OBSERVED_DATA],
        availability_status=MMMRangeAvailabilityStatus.AVAILABLE,
    )
    evidence = MMMSupportedRangeEvidence(
        evidence_id="evidence", run_id="run", created_at=NOW, producer_package_version="0", records=[record]
    )
    called = Mock()
    monkeypatch.setattr(public_simulation, "simulate", called)
    out = build_mmm_public_simulation_export(
        export_id="blocked",
        run_id="run",
        created_at=NOW,
        ctx=ctx,
        baseline_plan=b,
        candidate_plan=c,
        supported_range_evidence=evidence,
    )
    assert (
        out.status == MMMSimulationStatus.BLOCKED
        and out.failure_packet
        and out.failure_packet.code == MMMFailureCode.SUPPORTED_RANGE_EVIDENCE_UNUSABLE
    )
    assert out.failure_packet.technical_context["reason"] == "missing_or_scope_incompatible" and not called.called
    assert (
        out.run_manifest
        and out.run_manifest.failure_packet == out.failure_packet
        and out.run_manifest.steps[0].status.name == "BLOCKED"
    )
    assert (
        out.export_manifest_outcome
        and out.export_manifest_outcome.analytical_outcome
        and out.export_manifest_outcome.analytical_outcome.blocker_references == out.blocking_references
    )
    assert c.items[0].spend == 20 and out.candidate_plan.items[0].spend == 20 and out.comparison is None


def test_missing_ambiguous_and_extrapolated_range_blocks_have_matching_manifests(monkeypatch):
    ctx = SimpleNamespace(schema=SimpleNamespace(channel_columns=("search",), target_column="revenue"))
    b = plan(MMMSimulationPlanRole.BASELINE)
    c = plan(MMMSimulationPlanRole.CANDIDATE, 20)
    called = Mock()
    monkeypatch.setattr(public_simulation, "simulate", called)
    missing = build_mmm_public_simulation_export(
        export_id="missing",
        run_id="run",
        created_at=NOW,
        ctx=ctx,
        baseline_plan=b,
        candidate_plan=c,
        supported_range_evidence=None,
    )
    record = MMMSupportedRangeRecord(
        range_record_id="range",
        run_id="run",
        model_id=None,
        model_family=None,
        scope=MMMRangeScope(channel="search"),
        supported_lower=MMMRangeBound(value=0, unit="USD"),
        supported_upper=MMMRangeBound(value=10, unit="USD"),
        evidence_basis=[MMMRangeEvidenceBasis.OBSERVED_DATA],
        availability_status=MMMRangeAvailabilityStatus.AVAILABLE,
        simulation_eligibility=MMMSupportedRangeSimulationEligibility.ELIGIBLE_FOR_TECHNICAL_SIMULATION,
    )
    ambiguous_evidence = MMMSupportedRangeEvidence(
        evidence_id="ambiguous-evidence",
        run_id="run",
        created_at=NOW,
        producer_package_version="0",
        records=[record, record.model_copy(update={"range_record_id": "range-2"})],
    )
    ambiguous = build_mmm_public_simulation_export(
        export_id="ambiguous",
        run_id="run",
        created_at=NOW,
        ctx=ctx,
        baseline_plan=b,
        candidate_plan=c,
        supported_range_evidence=ambiguous_evidence,
    )
    extrapolation_evidence = MMMSupportedRangeEvidence(
        evidence_id="range-evidence", run_id="run", created_at=NOW, producer_package_version="0", records=[record]
    )
    extrapolated = build_mmm_public_simulation_export(
        export_id="extrapolated",
        run_id="run",
        created_at=NOW,
        ctx=ctx,
        baseline_plan=b,
        candidate_plan=c,
        supported_range_evidence=extrapolation_evidence,
    )
    for out, code, reason in (
        (missing, MMMFailureCode.UNSUPPORTED_EXTRAPOLATION, "Required supported-range evidence is unavailable."),
        (ambiguous, MMMFailureCode.SUPPORTED_RANGE_EVIDENCE_UNUSABLE, "ambiguous"),
        (
            extrapolated,
            MMMFailureCode.UNSUPPORTED_EXTRAPOLATION,
            "Candidate spend exceeds the supported range and was not changed.",
        ),
    ):
        assert out.status == MMMSimulationStatus.BLOCKED and out.failure_packet and out.failure_packet.code == code
        assert (
            out.run_manifest
            and out.run_manifest.status == MMMRunStatus.BLOCKED
            and out.run_manifest.failure_packet == out.failure_packet
        )
        assert (
            out.run_manifest.successful_export is None
            and out.run_manifest.steps[0].status.name == "BLOCKED"
            and out.comparison is None
        )
        assert (
            out.export_manifest_outcome
            and out.export_manifest_outcome.outcome_kind == "analytical_artifact"
            and out.export_manifest_outcome.analytical_outcome
        )
        assert (
            out.export_manifest_outcome.analytical_outcome.status == MMMRunStatus.BLOCKED
            and out.export_manifest_outcome.analytical_outcome.failure_packet == out.failure_packet
            and out.export_manifest_outcome.analytical_outcome.blocker_references == out.blocking_references
        )
        assert out.blocking_references == [reason]
    assert not called.called and c.items[0].spend == 20


@pytest.mark.parametrize(
    "status,eligibility,reason",
    [
        (
            MMMRangeAvailabilityStatus.AVAILABLE,
            MMMSupportedRangeSimulationEligibility.NOT_ASSESSED,
            "simulation_eligibility_not_assessed",
        ),
        (
            MMMRangeAvailabilityStatus.AVAILABLE,
            MMMSupportedRangeSimulationEligibility.NOT_ELIGIBLE,
            "simulation_not_eligible",
        ),
        (
            MMMRangeAvailabilityStatus.PARTIALLY_AVAILABLE,
            MMMSupportedRangeSimulationEligibility.NOT_ASSESSED,
            "partially_available",
        ),
        (MMMRangeAvailabilityStatus.UNAVAILABLE, MMMSupportedRangeSimulationEligibility.NOT_ASSESSED, "unavailable"),
        (MMMRangeAvailabilityStatus.BLOCKED, MMMSupportedRangeSimulationEligibility.NOT_ASSESSED, "blocked"),
        (
            MMMRangeAvailabilityStatus.RESEARCH_ONLY,
            MMMSupportedRangeSimulationEligibility.NOT_ASSESSED,
            "research_only",
        ),
    ],
)
def test_ineligible_range_status_short_circuits_before_bounds(monkeypatch, status, eligibility, reason):
    ctx = SimpleNamespace(schema=SimpleNamespace(channel_columns=("search",)))
    b = plan(MMMSimulationPlanRole.BASELINE)
    c = plan(MMMSimulationPlanRole.CANDIDATE, 99)
    partial_data_evidence = ["dataset:partial"] if status == MMMRangeAvailabilityStatus.PARTIALLY_AVAILABLE else []
    bounds = (
        {}
        if status == MMMRangeAvailabilityStatus.UNAVAILABLE
        else {
            "supported_lower": MMMRangeBound(value=0, unit="USD"),
            "supported_upper": MMMRangeBound(value=10, unit="USD"),
        }
    )
    bounds.update(
        {"observed_lower": MMMRangeBound(value=0, unit="USD"), "observed_upper": MMMRangeBound(value=10, unit="USD")}
        if status == MMMRangeAvailabilityStatus.UNAVAILABLE
        else {}
    )
    record = MMMSupportedRangeRecord(
        range_record_id="range",
        run_id="run",
        model_id=None,
        model_family=None,
        scope=MMMRangeScope(channel="search"),
        evidence_basis=[MMMRangeEvidenceBasis.OBSERVED_DATA]
        if status != MMMRangeAvailabilityStatus.RESEARCH_ONLY
        else [MMMRangeEvidenceBasis.RESEARCH_ONLY],
        availability_status=status,
        simulation_eligibility=eligibility,
        data_evidence_references=partial_data_evidence,
        limitation_references=["lim"] if status == MMMRangeAvailabilityStatus.BLOCKED else [],
        **bounds,
    )
    evidence = MMMSupportedRangeEvidence(
        evidence_id="evidence", run_id="run", created_at=NOW, producer_package_version="0", records=[record]
    )
    called = Mock()
    monkeypatch.setattr(public_simulation, "simulate", called)
    out = build_mmm_public_simulation_export(
        export_id="blocked",
        run_id="run",
        created_at=NOW,
        ctx=ctx,
        baseline_plan=b,
        candidate_plan=c,
        supported_range_evidence=evidence,
    )
    assert (
        out.status == MMMSimulationStatus.BLOCKED
        and out.failure_packet
        and out.failure_packet.code == MMMFailureCode.SUPPORTED_RANGE_EVIDENCE_UNUSABLE
    )
    assert (
        out.failure_packet.technical_context["reason"] == reason
        and not called.called
        and out.comparison is None
        and c.items[0].spend == 99
    )
    assert (
        out.run_manifest
        and out.run_manifest.failure_packet == out.failure_packet
        and out.run_manifest.successful_export is None
    )
    assert (
        out.export_manifest_outcome
        and out.export_manifest_outcome.analytical_outcome
        and out.export_manifest_outcome.analytical_outcome.failure_packet == out.failure_packet
    )


def test_eligible_in_range_uses_governed_simulator_once_without_mutation(monkeypatch):
    ctx = SimpleNamespace(schema=SimpleNamespace(channel_columns=("search",), target_column="revenue"))
    b = plan(MMMSimulationPlanRole.BASELINE, 10)
    c = plan(MMMSimulationPlanRole.CANDIDATE, 20)
    record = MMMSupportedRangeRecord(
        range_record_id="range",
        run_id="run",
        model_id=None,
        model_family=None,
        scope=MMMRangeScope(channel="search"),
        supported_lower=MMMRangeBound(value=0, unit="USD"),
        supported_upper=MMMRangeBound(value=20, unit="USD"),
        evidence_basis=[MMMRangeEvidenceBasis.OBSERVED_DATA],
        availability_status=MMMRangeAvailabilityStatus.AVAILABLE,
        simulation_eligibility=MMMSupportedRangeSimulationEligibility.ELIGIBLE_FOR_TECHNICAL_SIMULATION,
    )
    evidence = MMMSupportedRangeEvidence(
        evidence_id="evidence", run_id="run", created_at=NOW, producer_package_version="0", records=[record]
    )
    result = SimpleNamespace(aggregation_semantics="full_panel", baseline_mu=1.0, plan_mu=3.0, delta_mu=2.0)
    called = Mock(return_value=result)
    monkeypatch.setattr(public_simulation, "simulate", called)
    out = build_mmm_public_simulation_export(
        export_id="success",
        run_id="run",
        created_at=NOW,
        ctx=ctx,
        baseline_plan=b,
        candidate_plan=c,
        supported_range_evidence=evidence,
    )
    assert (
        called.call_count == 1
        and called.call_args.args[0] == {"search": 20}
        and called.call_args.kwargs["baseline_plan"].spend_by_channel == {"search": 10}
    )
    assert (
        b.items[0].spend == 10
        and c.items[0].spend == 20
        and out.comparison.metrics[0].delta_mu == 2.0
        and out.comparison.metrics[0].uncertainty.status == MMMSimulationUncertaintyStatus.UNAVAILABLE
    )
    assert (
        out.run_manifest
        and out.run_manifest.status == MMMRunStatus.SUCCEEDED
        and out.run_manifest.steps[0].input_artifacts[0].artifact_id == b.plan_id
    )
    assert (
        out.export_manifest_outcome
        and out.export_manifest_outcome.analytical_outcome
        and out.export_manifest_outcome.analytical_outcome.output_artifact == out.run_manifest.successful_export
    )


def test_unexpected_runtime_exception_propagates_without_a_failed_manifest(monkeypatch):
    ctx = SimpleNamespace(schema=SimpleNamespace(channel_columns=("search",), target_column="revenue"))
    b = plan(MMMSimulationPlanRole.BASELINE, 10)
    c = plan(MMMSimulationPlanRole.CANDIDATE, 20)
    record = MMMSupportedRangeRecord(
        range_record_id="range",
        run_id="run",
        model_id=None,
        model_family=None,
        scope=MMMRangeScope(channel="search"),
        supported_lower=MMMRangeBound(value=0, unit="USD"),
        supported_upper=MMMRangeBound(value=20, unit="USD"),
        evidence_basis=[MMMRangeEvidenceBasis.OBSERVED_DATA],
        availability_status=MMMRangeAvailabilityStatus.AVAILABLE,
        simulation_eligibility=MMMSupportedRangeSimulationEligibility.ELIGIBLE_FOR_TECHNICAL_SIMULATION,
    )
    evidence = MMMSupportedRangeEvidence(
        evidence_id="evidence", run_id="run", created_at=NOW, producer_package_version="0", records=[record]
    )
    monkeypatch.setattr(public_simulation, "simulate", Mock(side_effect=RuntimeError("internal defect")))
    with pytest.raises(RuntimeError, match="internal defect"):
        build_mmm_public_simulation_export(
            export_id="runtime-error",
            run_id="run",
            created_at=NOW,
            ctx=ctx,
            baseline_plan=b,
            candidate_plan=c,
            supported_range_evidence=evidence,
        )


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {
            "plan_id": "x",
            "role": "BASELINE",
            "spend_unit": "USD",
            "evaluation_time_window": "2026",
            "items": [{"channel_id": "search", "spend": -1, "spend_unit": "USD"}],
            "total_spend": -1,
        },
        {
            "plan_id": "x",
            "role": "BASELINE",
            "spend_unit": "USD",
            "evaluation_time_window": "2026",
            "items": [
                {"channel_id": "search", "spend": 1, "spend_unit": "USD"},
                {"channel_id": "search", "spend": 1, "spend_unit": "USD"},
            ],
            "total_spend": 2,
        },
    ],
)
def test_invalid_payload_is_typed_failed(payload):
    out = build_mmm_public_simulation_export_from_payloads(
        export_id="invalid",
        run_id="run",
        created_at=NOW,
        ctx=None,
        baseline_payload=payload,
        candidate_payload=payload,
        supported_range_evidence=None,
    )
    assert out.status == MMMSimulationStatus.FAILED and out.comparison is None and not out.blocking_references
    assert (
        out.failure_packet
        and out.failure_packet.code == MMMFailureCode.INVALID_PLAN_INPUT
        and out.failure_packet.run_id == out.run_id
    )


def _payload(role="BASELINE", **changes):
    data = {
        "plan_id": "p",
        "role": role,
        "spend_unit": "USD",
        "evaluation_time_window": "2026",
        "items": [{"channel_id": "search", "spend": 10.0, "spend_unit": "USD"}],
        "total_spend": 10.0,
    }
    data.update(changes)
    return data


@pytest.mark.parametrize(
    "baseline,candidate",
    [
        ({}, _payload("CANDIDATE")),
        (_payload("BASELINE"), {}),
        (_payload("CANDIDATE"), _payload("CANDIDATE")),
        (_payload("BASELINE"), _payload("BASELINE")),
        (
            _payload(
                "BASELINE",
                items=[
                    {"channel_id": "search", "spend": 1, "spend_unit": "USD"},
                    {"channel_id": "search", "spend": 1, "spend_unit": "USD"},
                ],
                total_spend=2,
            ),
            _payload("CANDIDATE"),
        ),
        (
            _payload("BASELINE", items=[{"channel_id": "search", "spend": float("nan"), "spend_unit": "USD"}]),
            _payload("CANDIDATE"),
        ),
        (_payload("BASELINE", total_spend=9), _payload("CANDIDATE")),
        (
            _payload("BASELINE", items=[{"channel_id": "search", "spend": 10, "spend_unit": "EUR"}]),
            _payload("CANDIDATE"),
        ),
        (_payload("BASELINE", items=[{"spend": 10, "spend_unit": "USD"}]), _payload("CANDIDATE")),
    ],
)
def test_all_malformed_plan_payloads_are_sanitized_failed(baseline, candidate):
    out = build_mmm_public_simulation_export_from_payloads(
        export_id="invalid",
        run_id="run",
        created_at=NOW,
        ctx=None,
        baseline_payload=baseline,
        candidate_payload=candidate,
        supported_range_evidence=None,
    )
    assert out.status == MMMSimulationStatus.FAILED and out.comparison is None and not out.blocking_references
    assert (
        out.failure_packet
        and out.failure_packet.code == MMMFailureCode.INVALID_PLAN_INPUT
        and out.failure_packet.run_id == "run"
    )
    assert "traceback" not in out.to_json().lower() and "/users/" not in out.to_json().lower()
