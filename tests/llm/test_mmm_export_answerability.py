"""MIP answerability gates for externally produced MMMExportBundle payloads."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from mmm.contracts.mmm_export_bundle import parse_mmm_export_bundle
from mmm.llm.mmm_export_answerability import MMMIntent, evaluate_mmm_export_answerability

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "mmm_export"


def _payload(name: str) -> dict:
    return json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))


def _evaluate(name: str, intent: MMMIntent, **kwargs: bool):
    return evaluate_mmm_export_answerability(parse_mmm_export_bundle(_payload(name)), intent, **kwargs)


def test_parser_normalizes_required_contract_fields() -> None:
    bundle = parse_mmm_export_bundle(_payload("readiness_only_bundle.json"))
    assert bundle.schema_version == "mmm_mip_export_v1"
    assert bundle.bundle_id == "bundle-readiness-001"
    assert bundle.model_run_id == "run-readiness-001"
    assert bundle.lineage["producer"] == "mmm"
    assert bundle.artifacts[0].artifact_type == "MMMModelFitArtifact"
    assert bundle.artifacts[0].diagnostic_status == "pass"


def test_readiness_only_bundle_allows_readiness_explanation() -> None:
    result = _evaluate("readiness_only_bundle.json", MMMIntent.READINESS)
    assert result.allowed is True
    assert result.scope == "explanation_only"


def test_diagnostic_explanation_is_explicitly_gated() -> None:
    result = _evaluate("readiness_only_bundle.json", MMMIntent.DIAGNOSTICS)
    assert result.allowed is True
    assert result.reason_code == "diagnostic_explanation_allowed"


def test_diagnostic_roi_bundle_blocks_production_roi_claim() -> None:
    result = _evaluate("diagnostic_roi_blocked_bundle.json", MMMIntent.ROI)
    assert result.allowed is False
    assert "forbidden claims override" in result.cannot_say_reason


def test_demo_roi_bundle_allows_only_synthetic_explanation() -> None:
    result = _evaluate("demo_fixture_roi_bundle.json", MMMIntent.ROI)
    assert result.allowed is True
    assert result.scope == "demo_only"
    assert any("synthetic/demo" in item for item in result.required_disclosures)
    assert any("production" in item for item in result.required_disclosures)


def test_demo_scope_does_not_conflict_with_forbidden_production_roi_claims() -> None:
    data = _payload("demo_fixture_roi_bundle.json")
    data["forbidden_claims"].extend(["channel_roi_ranking", "incremental_roi_truth"])
    data["artifacts"][0]["forbidden_claims"].extend(["channel_roi_ranking", "incremental_roi_truth"])
    result = evaluate_mmm_export_answerability(parse_mmm_export_bundle(data), MMMIntent.ROI)
    assert result.allowed is True
    assert result.scope == "demo_only"


def test_blocked_budget_bundle_blocks_budget_shift() -> None:
    result = _evaluate("blocked_budget_recommendation_bundle.json", MMMIntent.BUDGET_RECOMMENDATION)
    assert result.allowed is False
    assert result.cannot_say_reason.startswith("Cannot say:")


def test_recommendation_contract_shape_with_false_gate_still_blocks() -> None:
    result = _evaluate(
        "valid_recommendation_contract_shape_blocked_fixture.json", MMMIntent.BUDGET_RECOMMENDATION
    )
    assert result.allowed is False
    assert "recommendation_allowed" in result.cannot_say_reason


def test_missing_safety_fields_default_to_blocked() -> None:
    bundle = parse_mmm_export_bundle(
        {
            "schema_version": "mmm_mip_export_v1",
            "bundle_id": "missing-safety",
            "model_run_id": "run-missing-safety",
            "artifacts": [{"artifact_type": "MMMChannelContributionArtifact"}],
        }
    )
    artifact = bundle.artifacts[0]
    assert bundle.llm_exposure_allowed is False
    assert artifact.recommendation_allowed is False
    assert artifact.diagnostic_status == "unknown"
    assert artifact.artifact_safety_status == "blocked"
    result = evaluate_mmm_export_answerability(bundle, MMMIntent.CONTRIBUTION)
    assert result.allowed is False


def test_forbidden_claim_overrides_allowed_claim() -> None:
    data = deepcopy(_payload("readiness_only_bundle.json"))
    data["forbidden_claims"].append("readiness_explanation_allowed")
    result = evaluate_mmm_export_answerability(parse_mmm_export_bundle(data), MMMIntent.READINESS)
    assert result.allowed is False
    assert "forbidden claims override" in result.cannot_say_reason


def test_optimizer_artifact_cannot_answer_recommendation() -> None:
    result = _evaluate("blocked_budget_recommendation_bundle.json", MMMIntent.BUDGET_RECOMMENDATION)
    assert result.allowed is False
    assert result.reason_code == "required_artifact_missing"


def test_simulation_artifact_cannot_imply_recommendation() -> None:
    data = {
        "llm_exposure_allowed": True,
        "planning_allowed": True,
        "allowed_claims": ["simulation_result_explanation_allowed"],
        "artifacts": [
            {
                "artifact_type": "MMMSimulationResultArtifact",
                "llm_exposure_allowed": True,
                "planning_allowed": True,
                "allowed_claims": ["simulation_result_explanation_allowed"],
            }
        ],
    }
    result = evaluate_mmm_export_answerability(
        parse_mmm_export_bundle(data), MMMIntent.SIMULATION, implies_recommendation=True
    )
    assert result.allowed is False
    assert result.reason_code == "recommendation_implication_blocked"


def test_response_curve_cannot_imply_recommendation() -> None:
    data = {
        "artifacts": [{"artifact_type": "MMMResponseCurveArtifact"}],
    }
    result = evaluate_mmm_export_answerability(
        parse_mmm_export_bundle(data), MMMIntent.RESPONSE_CURVE, implies_recommendation=True
    )
    assert result.allowed is False
    assert "do not grant recommendation authority" in result.cannot_say_reason


def test_governed_contribution_requires_all_production_gates() -> None:
    claim = "channel_contribution_allowed"
    data = {
        "llm_exposure_allowed": True,
        "production_claim_allowed": True,
        "allowed_claims": [claim],
        "forbidden_claims": [],
        "artifact_safety_status": "production_safe",
        "artifacts": [
            {
                "artifact_type": "MMMChannelContributionArtifact",
                "llm_exposure_allowed": True,
                "production_claim_allowed": True,
                "allowed_claims": [claim],
                "forbidden_claims": [],
                "promotion_status": "approved_for_prod",
                "uncertainty_status": "present",
                "artifact_safety_status": "production_safe",
            }
        ],
    }
    result = evaluate_mmm_export_answerability(parse_mmm_export_bundle(data), MMMIntent.CONTRIBUTION)
    assert result.allowed is True
    assert result.scope == "production"


def test_blocked_result_has_verifier_friendly_shape() -> None:
    result = _evaluate("diagnostic_roi_blocked_bundle.json", MMMIntent.ROI)
    payload = result.as_dict()
    assert payload["allowed"] is False
    assert payload["reason_code"]
    assert payload["reasons"]
    assert str(payload["cannot_say_reason"]).startswith("Cannot say:")
