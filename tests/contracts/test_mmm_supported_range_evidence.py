"""Regression coverage for governed MMM supported-range evidence."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from mmm.contracts.diagnostics_limitations import MMMClaimEffect, MMMTechnicalClaim, MMMTechnicalClaimDisposition
from mmm.contracts.mip_export_adapter import MMMExportRuntimeContext, adapt_runtime_artifacts_to_export_manifest_outcome
from mmm.contracts.supported_range import (
    MMM_SUPPORTED_RANGE_SCHEMA_VERSION, MMMExtrapolationClassification,
    MMMRangeAvailabilityStatus, MMMRangeBound, MMMRangeEvidenceBasis,
    MMMRangeRelation, MMMRangeRestriction, MMMRangeScale, MMMRangeScope,
    MMMSupportedRangeEvidence, MMMSupportedRangeRecord,
)

NOW = datetime(2026, 7, 15, 12, 0, tzinfo=timezone.utc)
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "mip_export" / "supported_range_v1"


def _bound(value: float, **overrides: object) -> MMMRangeBound:
    data: dict[str, object] = {"value": value, "unit": "USD", "scale": MMMRangeScale.RAW}
    data.update(overrides)
    return MMMRangeBound(**data)


def _scope(**overrides: object) -> MMMRangeScope:
    data: dict[str, object] = {"channel": "search", "kpi": "revenue", "geography": "national", "time_window": "2026-01-01/2026-06-30", "data_grain": "weekly"}
    data.update(overrides)
    return MMMRangeScope(**data)


def _restriction(code: str = "unsupported_range") -> MMMRangeRestriction:
    return MMMRangeRestriction(
        restriction_code=code, technical_summary="Claims outside governed range remain blocked.",
        affected_scope=_scope(), evidence_references=["diagnostic:range-001"],
        affected_technical_claims=[MMMTechnicalClaim.EXTRAPOLATIVE_SIMULATION, MMMTechnicalClaim.BUDGET_OPTIMIZATION_INPUT],
        claim_effects=[MMMClaimEffect(claim=MMMTechnicalClaim.EXTRAPOLATIVE_SIMULATION, disposition=MMMTechnicalClaimDisposition.BLOCKED, blocking_references=["diagnostic:range-001"])],
    )


def _record(**overrides: object) -> MMMSupportedRangeRecord:
    data: dict[str, object] = {
        "range_record_id": "range-search-001", "run_id": "run-range-001", "model_id": "model-ridge-001", "model_family": "ridge", "configuration_hash": "sha256:config", "scope": _scope(),
        "observed_lower": _bound(100.0), "observed_upper": _bound(1000.0),
        "supported_lower": _bound(100.0), "supported_upper": _bound(1000.0),
        "evidence_basis": [MMMRangeEvidenceBasis.OBSERVED_DATA, MMMRangeEvidenceBasis.TRAINING_DOMAIN],
        "availability_status": MMMRangeAvailabilityStatus.AVAILABLE,
        "range_relation": MMMRangeRelation.WITHIN_SUPPORTED_RANGE,
        "extrapolation_classification": MMMExtrapolationClassification.INTERPOLATION,
        "data_evidence_references": ["dataset:training-panel-001"],
    }
    data.update(overrides)
    return MMMSupportedRangeRecord(**data)


def _evidence(**overrides: object) -> MMMSupportedRangeEvidence:
    data: dict[str, object] = {"evidence_id": "range-evidence-001", "run_id": "run-range-001", "created_at": NOW, "producer_package_version": "0.1.0", "records": [_record()]}
    data.update(overrides)
    return MMMSupportedRangeEvidence(**data)


def test_public_contract_is_versioned_deterministic_and_preserves_observed_supported_validated_distinction() -> None:
    evidence = _evidence(records=[_record(validated_lower=_bound(125), validated_upper=_bound(950), validation_evidence_references=["validation:holdout-001"])])
    assert evidence.schema_version == MMM_SUPPORTED_RANGE_SCHEMA_VERSION
    assert evidence.summary_counts == {"available": 1, "partially_available": 0, "unavailable": 0, "blocked": 0, "research_only": 0, "supported": 1, "restricted": 0, "unsupported_extrapolation": 0}
    assert evidence.records[0].observed_lower != evidence.records[0].supported_lower
    assert MMMSupportedRangeEvidence.from_json(evidence.to_json()) == evidence
    assert evidence.to_json() == evidence.to_json()


def test_bounds_units_transforms_and_status_rules_fail_closed() -> None:
    with pytest.raises(ValidationError, match="cannot exceed"):
        _record(supported_lower=_bound(1001), supported_upper=_bound(1000))
    with pytest.raises(ValidationError, match="finite"):
        MMMRangeBound(value=float("nan"), unit="USD")
    with pytest.raises(ValidationError, match="compatible"):
        _record(supported_upper=MMMRangeBound(value=1000, unit="EUR"))
    with pytest.raises(ValidationError, match="transformation_id"):
        MMMRangeBound(value=1, unit="index", scale=MMMRangeScale.TRANSFORMED)
    with pytest.raises(ValidationError, match="cannot claim supported"):
        _record(availability_status=MMMRangeAvailabilityStatus.UNAVAILABLE)
    with pytest.raises(ValidationError, match="validation evidence"):
        _record(validated_lower=_bound(150), validated_upper=_bound(900))


def test_unavailable_blocked_research_only_and_pre_model_evidence_are_truthful() -> None:
    unavailable = _record(range_record_id="range-observed-only", model_id=None, observed_lower=_bound(100), observed_upper=_bound(1000), supported_lower=None, supported_upper=None, availability_status=MMMRangeAvailabilityStatus.UNAVAILABLE, range_relation=MMMRangeRelation.UNKNOWN, extrapolation_classification=MMMExtrapolationClassification.UNKNOWN)
    blocked = _record(range_record_id="range-blocked", availability_status=MMMRangeAvailabilityStatus.BLOCKED, extrapolation_classification=MMMExtrapolationClassification.UNSUPPORTED_EXTRAPOLATION, range_relation=MMMRangeRelation.OUTSIDE_SUPPORTED_RANGE, restrictions=[_restriction()], limitation_references=["limitation:range-001"])
    research = _record(range_record_id="range-bayes", model_family="bayesian", availability_status=MMMRangeAvailabilityStatus.RESEARCH_ONLY, evidence_basis=[MMMRangeEvidenceBasis.RESEARCH_ONLY], restrictions=[_restriction("research_only")], extrapolation_classification=MMMExtrapolationClassification.UNKNOWN, range_relation=MMMRangeRelation.UNKNOWN)
    aggregate = _evidence(records=sorted([unavailable, blocked, research], key=lambda item: item.range_record_id))
    assert unavailable.model_id is None
    assert aggregate.summary_counts["unavailable"] == 1
    assert aggregate.summary_counts["blocked"] == 1
    assert aggregate.summary_counts["research_only"] == 1


def test_extrapolation_and_uncertainty_rules_are_explicit() -> None:
    with pytest.raises(ValidationError, match="unsupported extrapolation"):
        _record(extrapolation_classification=MMMExtrapolationClassification.UNSUPPORTED_EXTRAPOLATION)
    with pytest.raises(ValidationError, match="governed evidence"):
        _record(extrapolation_classification=MMMExtrapolationClassification.LIMITED_EXTRAPOLATION)
    with pytest.raises(ValidationError, match="artifact reference"):
        _record(uncertainty_available=True)
    boundary = _record(range_relation=MMMRangeRelation.AT_UPPER_BOUNDARY, extrapolation_classification=MMMExtrapolationClassification.BOUNDARY, restrictions=[_restriction("upper_boundary")])
    assert boundary.extrapolation_classification == MMMExtrapolationClassification.BOUNDARY


def test_manifest_and_export_boundary_link_range_evidence_additively() -> None:
    context = MMMExportRuntimeContext(model_run_id="run-range-001", training_data_fingerprint="sha256:panel", model_artifact_fingerprint="sha256:config", generated_at="2026-07-15T12:00:00Z", package_version="0.1.0", git_commit="abc1234", model_form="ridge", estimand="modeled_revenue", time_window="2026-01-01/2026-06-30", geo_scope="national", channel_scope=("search",), outcome_metric="revenue", spend_metric="spend", currency="USD")
    result = adapt_runtime_artifacts_to_export_manifest_outcome(context=context, manifest_id="manifest-range-001", created_at=NOW, supported_range_evidence=_evidence(), extension_report={})
    assert result.run_manifest.supported_range_evidence_id == "range-evidence-001"
    assert result.supported_range_evidence_id == "range-evidence-001"


@pytest.mark.parametrize("fixture_name", [
    "available_ridge.json", "observed_only_unavailable.json", "restricted_boundary.json", "unsupported_extrapolation.json", "pre_model_insufficient_evidence.json", "bayesian_research_only.json", "multi_channel_distinct.json",
])
def test_deterministic_supported_range_fixtures_parse_and_round_trip(fixture_name: str) -> None:
    payload = json.loads((FIXTURES / fixture_name).read_text(encoding="utf-8"))
    evidence = MMMSupportedRangeEvidence.model_validate(payload)
    serialized = evidence.to_json()
    assert MMMSupportedRangeEvidence.from_json(serialized) == evidence
    assert not any(value in serialized.lower() for value in ("/users/", "traceback", "stack trace", "dataframe", "secret", "password"))
