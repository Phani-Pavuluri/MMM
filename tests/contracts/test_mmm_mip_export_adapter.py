"""MMM-EXPORT-003 runtime adapter tests."""

from __future__ import annotations

import pytest

from mmm.contracts.mip_export import (
    MMMChannelROIArtifact,
    parse_export_artifact,
    validate_mmm_export_bundle,
)
from mmm.contracts.mip_export_adapter import (
    MMMExportAdapterError,
    MMMExportRuntimeContext,
    adapt_runtime_artifacts_to_export_bundle,
)


def _context(*, training_data_fingerprint: str = "sha256:panel") -> MMMExportRuntimeContext:
    return MMMExportRuntimeContext(
        model_run_id="run-003",
        training_data_fingerprint=training_data_fingerprint,
        model_artifact_fingerprint="sha256:model",
        generated_at="2026-07-13T12:00:00Z",
        package_version="0.1.0",
        git_commit="abc1234",
        model_form="semi_log",
        estimand="modeled_outcome",
        time_window="2026-01-01/2026-06-30",
        geo_scope="national",
        channel_scope=("search", "social"),
        outcome_metric="revenue",
        spend_metric="spend",
        currency="USD",
    )


def test_adapts_only_artifact_families_present_and_validates() -> None:
    bundle = adapt_runtime_artifacts_to_export_bundle(
        context=_context(),
        extension_report={
            "ridge_fit_summary": {"coef": [0.2, 0.1]},
            "panel_qa": {"status": "pass"},
        },
    )

    assert validate_mmm_export_bundle(bundle) == []
    assert [item["artifact_type"] for item in bundle.artifacts] == [
        "MMMModelFitArtifact",
        "MMMModelDiagnosticArtifact",
    ]
    assert bundle.production_claim_allowed is False
    assert bundle.recommendation_allowed is False


def test_absent_optional_artifacts_are_not_invented() -> None:
    bundle = adapt_runtime_artifacts_to_export_bundle(context=_context(), extension_report={})
    assert bundle.artifacts == []
    assert bundle.inventory_status.value == "EXISTS_PARTIAL_NOT_CONSUMABLE"


def test_present_roi_curve_and_optimizer_stay_blocked() -> None:
    bundle = adapt_runtime_artifacts_to_export_bundle(
        context=_context(),
        extension_report={"roi_summary": {"search": 1.4}, "curve_bundles": [{"channel": "search"}]},
        optimizer_result={"solution_spend": {"search": 120.0}},
    )
    by_type = {item["artifact_type"]: item for item in bundle.artifacts}
    assert set(by_type) == {
        "MMMChannelROIArtifact",
        "MMMResponseCurveArtifact",
        "MMMOptimizerResultArtifact",
    }
    roi = parse_export_artifact(by_type["MMMChannelROIArtifact"])
    assert isinstance(roi, MMMChannelROIArtifact)
    assert roi.roi_by_channel == {}
    assert by_type["MMMOptimizerResultArtifact"]["has_recommendation_contract"] is False
    assert all(item["production_claim_allowed"] is False for item in bundle.artifacts)
    assert all(item["recommendation_allowed"] is False for item in bundle.artifacts)


def test_rejects_missing_lineage_instead_of_guessing() -> None:
    with pytest.raises(MMMExportAdapterError, match="training_data_fingerprint"):
        adapt_runtime_artifacts_to_export_bundle(
            context=_context(training_data_fingerprint=""),
            extension_report={"ridge_fit_summary": {}},
        )


def test_rejects_ungoverned_recommendation_input() -> None:
    with pytest.raises(MMMExportAdapterError, match="refusing to invent"):
        adapt_runtime_artifacts_to_export_bundle(
            context=_context(),
            recommendation_contract={"proposed_budget_shifts": []},
        )
