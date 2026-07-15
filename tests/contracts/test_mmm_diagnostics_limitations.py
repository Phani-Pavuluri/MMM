from datetime import datetime, timezone
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from mmm.contracts.diagnostics_limitations import (
    MMMAffectedScope, MMMClaimEffect, MMMDiagnosticCategory, MMMDiagnosticRecord,
    MMMDiagnosticSeverity, MMMDiagnosticStatus, MMMDiagnosticsLimitations,
    MMMLimitationRecord, MMMTechnicalClaim, MMMTechnicalClaimDisposition,
)
from mmm.contracts.mip_export_adapter import MMMExportRuntimeContext, adapt_runtime_artifacts_to_export_manifest_outcome

NOW = datetime(2026, 7, 14, 12, 0, tzinfo=timezone.utc)


def _effect(disposition=MMMTechnicalClaimDisposition.SUPPORTED):
    return MMMClaimEffect(claim=MMMTechnicalClaim.MODEL_FIT, disposition=disposition, evidence_references=["diag-001"] if disposition == MMMTechnicalClaimDisposition.SUPPORTED else [], blocking_references=["lim-001"] if disposition == MMMTechnicalClaimDisposition.BLOCKED else [], restriction="validated range" if disposition == MMMTechnicalClaimDisposition.RESTRICTED else None)


def _diagnostic(**changes):
    data = dict(sequence=0, diagnostic_id="diag-001", diagnostic_code="ridge_fit_stability", category=MMMDiagnosticCategory.MODEL_STABILITY, contract_version="v1", producer_component="mmm.diagnostics", created_at=NOW, status=MMMDiagnosticStatus.PASSED, severity=MMMDiagnosticSeverity.LOW, technical_summary="Stable diagnostic evidence.", claim_effects=[_effect()])
    data.update(changes)
    return MMMDiagnosticRecord(**data)


def _aggregate(**changes):
    data = dict(aggregate_id="diagnostics-001", run_id="run-001", created_at=NOW, producer_package_version="0.1.0", diagnostics=[_diagnostic()])
    data.update(changes)
    return MMMDiagnosticsLimitations(**data)


def test_typed_diagnostics_and_limitations_round_trip_with_explicit_claim_effects():
    aggregate = _aggregate(limitations=[MMMLimitationRecord(sequence=0, limitation_id="lim-001", limitation_code="unsupported_extrapolation", category=MMMDiagnosticCategory.EXTRAPOLATION, severity=MMMDiagnosticSeverity.HIGH, technical_summary="Out-of-range plans remain blocked.", claim_effects=[MMMClaimEffect(claim=MMMTechnicalClaim.EXTRAPOLATIVE_SIMULATION, disposition=MMMTechnicalClaimDisposition.BLOCKED, blocking_references=["lim-001"])])])
    assert MMMDiagnosticsLimitations.from_json(aggregate.to_json()) == aggregate
    assert aggregate.claim_dispositions["EXTRAPOLATIVE_SIMULATION"] == "BLOCKED"


def test_unavailable_passed_and_claim_disposition_consistency_fails_closed():
    with pytest.raises(ValidationError, match="unavailable"):
        _diagnostic(status=MMMDiagnosticStatus.UNAVAILABLE)
    with pytest.raises(ValidationError, match="affirmative evidence"):
        MMMClaimEffect(claim=MMMTechnicalClaim.MODEL_FIT, disposition=MMMTechnicalClaimDisposition.SUPPORTED)


def test_scope_research_only_and_manifest_linkage_are_additive():
    limitation = MMMLimitationRecord(sequence=0, limitation_id="lim-bayes", limitation_code="research_only", category=MMMDiagnosticCategory.PROMOTION, severity=MMMDiagnosticSeverity.HIGH, technical_summary="Bayesian output remains research only.", research_only=True, affected_scope=MMMAffectedScope(model_family="bayesian"), claim_effects=[MMMClaimEffect(claim=MMMTechnicalClaim.PRODUCTION_USE, disposition=MMMTechnicalClaimDisposition.BLOCKED, blocking_references=["lim-bayes"])])
    aggregate = _aggregate(limitations=[limitation])
    context = MMMExportRuntimeContext(model_run_id="run-001", training_data_fingerprint="sha256:panel", model_artifact_fingerprint="sha256:model", generated_at="2026-07-14T12:00:00Z", package_version="0.1.0", git_commit="abc", model_form="ridge", estimand="incremental_sales", time_window="2026", geo_scope="national", channel_scope=("search",), outcome_metric="revenue", spend_metric="spend", currency="USD")
    outcome = adapt_runtime_artifacts_to_export_manifest_outcome(context=context, manifest_id="manifest-001", created_at=NOW, diagnostics_limitations=aggregate, extension_report={})
    assert outcome.run_manifest.diagnostics_limitations_id == "diagnostics-001"


@pytest.mark.parametrize("name", [
    "diagnostics_limitations_success.json", "diagnostics_limitations_warning.json",
    "diagnostics_limitations_insufficient_history.json", "diagnostics_limitations_extrapolation.json",
    "diagnostics_limitations_research_only.json",
])
def test_deterministic_producer_fixtures_round_trip(name):
    payload = json.loads((Path(__file__).resolve().parents[1] / "fixtures" / "mip_export" / name).read_text())
    aggregate = MMMDiagnosticsLimitations.model_validate(payload)
    assert MMMDiagnosticsLimitations.from_json(aggregate.to_json()) == aggregate
