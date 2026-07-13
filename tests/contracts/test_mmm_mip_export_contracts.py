"""MMM-EXPORT-002: typed MIP export schemas and fixture claim-safety tests."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from mmm.contracts.mip_export import (
    ArtifactSafetyStatus,
    MMMChannelROIArtifact,
    MMMExportBundle,
    MMMOptimizerResultArtifact,
    MMMRecommendationContract,
    PromotionStatus,
    artifact_is_demo_safe,
    artifact_is_mip_exposable,
    artifact_is_readiness_exposable,
    bundle_is_mip_consumable,
    parse_export_artifact,
    parse_export_bundle,
    roi_is_blocked_until_uncertainty,
    validate_claim_safety,
    validate_mmm_export_artifact,
    validate_mmm_export_bundle,
    validate_recommendation_contract,
)

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "mip_export"


def _load(name: str) -> dict:
    return json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))


def test_readiness_only_bundle_validates_and_is_readiness_exposable() -> None:
    raw = _load("readiness_only_bundle.json")
    errors = validate_mmm_export_bundle(raw)
    assert errors == [], errors
    bundle = parse_export_bundle(raw)
    assert not bundle_is_mip_consumable(bundle)
    fit = parse_export_artifact(bundle.artifacts[0])
    assert artifact_is_readiness_exposable(fit)
    assert not artifact_is_mip_exposable(fit)
    assert fit.production_claim_allowed is False
    assert fit.recommendation_allowed is False


def test_missing_lineage_fails_validation() -> None:
    raw = _load("readiness_only_bundle.json")
    art = copy.deepcopy(raw["artifacts"][0])
    art["training_data_fingerprint"] = ""
    art["model_artifact_fingerprint"] = ""
    errors = validate_mmm_export_artifact(art)
    assert any("lineage" in e for e in errors)


def test_missing_schema_version_fails_validation() -> None:
    raw = _load("readiness_only_bundle.json")
    art = copy.deepcopy(raw["artifacts"][0])
    art["schema_version"] = ""
    with pytest.raises(ValidationError):
        parse_export_artifact(art)
    data = raw["artifacts"][0].copy()
    data["model_run_id"] = ""
    errors2 = validate_mmm_export_artifact(data)
    assert any("model_run_id" in e for e in errors2)


def test_roi_without_uncertainty_is_blocked() -> None:
    raw = _load("diagnostic_roi_blocked_bundle.json")
    assert validate_mmm_export_bundle(raw) == []
    roi = parse_export_artifact(raw["artifacts"][0])
    assert isinstance(roi, MMMChannelROIArtifact)
    assert roi_is_blocked_until_uncertainty(roi)
    assert not artifact_is_mip_exposable(roi)
    assert "blocked_until_uncertainty" in roi.forbidden_claims


def test_roi_demo_fixture_cannot_allow_production_claim() -> None:
    raw = _load("demo_fixture_roi_bundle.json")
    assert validate_mmm_export_bundle(raw) == []
    roi = parse_export_artifact(raw["artifacts"][0])
    assert artifact_is_demo_safe(roi)
    assert roi.demo_fixture_allowed is True
    assert roi.production_claim_allowed is False
    bad = copy.deepcopy(raw["artifacts"][0])
    bad["production_claim_allowed"] = True
    bad["promotion_status"] = "approved_for_prod"
    errors = validate_mmm_export_artifact(bad)
    assert any("demo_fixture" in e for e in errors)


def test_response_curve_alone_cannot_allow_recommendation() -> None:
    base = _load("readiness_only_bundle.json")["artifacts"][0]
    curve = {
        **{k: base[k] for k in base if k not in {"artifact_type", "framework", "model_release_state"}},
        "artifact_type": "MMMResponseCurveArtifact",
        "channels_with_curves": ["meta"],
        "recommendation_allowed": True,
        "allowed_claims": ["readiness_explanation_allowed"],
        "forbidden_claims": ["budget_shift_recommendation"],
        "artifact_safety_status": "diagnostic_only",
        "production_claim_allowed": False,
        "llm_exposure_allowed": False,
        "demo_fixture_allowed": False,
    }
    errors = validate_mmm_export_artifact(curve)
    assert any("ResponseCurveArtifact" in e and "recommendation" in e for e in errors)


def test_optimizer_without_recommendation_contract_cannot_allow_budget_recommendation() -> None:
    raw = _load("blocked_budget_recommendation_bundle.json")
    assert validate_mmm_export_bundle(raw) == []
    opt = parse_export_artifact(raw["artifacts"][0])
    assert isinstance(opt, MMMOptimizerResultArtifact)
    assert opt.recommendation_allowed is False
    assert opt.has_recommendation_contract is False
    bad = copy.deepcopy(raw["artifacts"][0])
    bad["recommendation_allowed"] = True
    bad["allowed_claims"] = ["budget_shift_recommendation"]
    errors = validate_mmm_export_artifact(bad)
    assert any("RecommendationContract" in e for e in errors)


def test_recommendation_contract_with_demo_source_blocked_from_production() -> None:
    raw = _load("valid_recommendation_contract_shape_blocked_fixture.json")
    assert validate_mmm_export_bundle(raw) == []
    rec = next(a for a in raw["artifacts"] if a["artifact_type"] == "MMMRecommendationContract")
    art = parse_export_artifact(rec)
    assert isinstance(art, MMMRecommendationContract)
    assert art.recommendation_allowed is False
    assert art.production_claim_allowed is False
    assert not artifact_is_mip_exposable(art)
    bad = copy.deepcopy(rec)
    bad["production_claim_allowed"] = True
    bad["recommendation_allowed"] = True
    bad["promotion_status"] = "approved_for_prod"
    bad["allowed_claims"] = ["budget_shift_recommendation", "demo_fixture_only"]
    # still demo_fixture_allowed=true → must fail
    errors = validate_recommendation_contract(bad)
    assert errors
    assert any("demo" in e.lower() or "production" in e.lower() for e in errors)


def test_research_only_bayes_artifact_cannot_be_production_safe() -> None:
    base = _load("readiness_only_bundle.json")["artifacts"][0]
    bayes = copy.deepcopy(base)
    bayes["framework"] = "bayesian"
    bayes["research_lane"] = "bayes_h5"
    bayes["promotion_status"] = "research_only"
    bayes["production_claim_allowed"] = True
    bayes["artifact_safety_status"] = "production_safe"
    bayes["allowed_claims"] = ["readiness_explanation_allowed"]
    errors = validate_mmm_export_artifact(bayes)
    assert any("research" in e.lower() or "Bayes" in e or "production" in e.lower() for e in errors)


def test_bundle_with_unsafe_artifact_not_mip_consumable() -> None:
    raw = _load("diagnostic_roi_blocked_bundle.json")
    assert not bundle_is_mip_consumable(raw)
    raw2 = _load("blocked_budget_recommendation_bundle.json")
    assert not bundle_is_mip_consumable(raw2)


def test_forbidden_claims_required_for_llm_exposable_artifact() -> None:
    art = copy.deepcopy(_load("readiness_only_bundle.json")["artifacts"][0])
    art["forbidden_claims"] = []
    art["llm_exposure_allowed"] = True
    errors = validate_mmm_export_artifact(art)
    assert any("forbidden_claims" in e for e in errors)


def test_allowed_claims_cannot_include_roi_or_recommendation_when_blocked() -> None:
    art = copy.deepcopy(_load("diagnostic_roi_blocked_bundle.json")["artifacts"][0])
    art["allowed_claims"] = ["channel_roi_ranking", "readiness_explanation_allowed"]
    errors = validate_claim_safety(art)
    assert any("ROI" in e or "roi" in e for e in errors)

    opt = copy.deepcopy(_load("blocked_budget_recommendation_bundle.json")["artifacts"][0])
    opt["allowed_claims"] = ["budget_shift_recommendation"]
    opt["recommendation_allowed"] = False
    errors2 = validate_claim_safety(opt)
    assert any("recommendation" in e for e in errors2)


def test_all_shipped_fixtures_parse_as_bundles() -> None:
    names = [
        "readiness_only_bundle.json",
        "diagnostic_roi_blocked_bundle.json",
        "demo_fixture_roi_bundle.json",
        "blocked_budget_recommendation_bundle.json",
        "valid_recommendation_contract_shape_blocked_fixture.json",
    ]
    for name in names:
        bundle = parse_export_bundle(_load(name))
        assert isinstance(bundle, MMMExportBundle)
        assert validate_mmm_export_bundle(bundle) == [], (name, validate_mmm_export_bundle(bundle))
        assert bundle.production_claim_allowed is False
        assert bundle.recommendation_allowed is False
        assert ArtifactSafetyStatus(bundle.artifact_safety_status) != ArtifactSafetyStatus.PRODUCTION_SAFE


def test_promotion_required_for_production_claim() -> None:
    art = copy.deepcopy(_load("readiness_only_bundle.json")["artifacts"][0])
    art["production_claim_allowed"] = True
    art["promotion_status"] = PromotionStatus.DIAGNOSTIC_ONLY.value
    art["llm_exposure_allowed"] = False
    errors = validate_claim_safety(art)
    assert any("approved_for_prod" in e for e in errors)
