"""Objective normalization audit trail."""

from __future__ import annotations

from mmm.config.schema import NormalizationProfile
from mmm.evaluation.normalization_policy import describe_objective_normalization, normalize_objective_vector


def test_describe_research_normalization_is_decoupled_piecewise() -> None:
    raw = (0.2, 0.4, 0.05, 0.1, 0.3)
    norm = normalize_objective_vector(raw, NormalizationProfile.RESEARCH, baseline_predictive=0.2)
    rep = describe_objective_normalization(
        raw,
        NormalizationProfile.RESEARCH,
        baseline_predictive=0.2,
        calibration_details=None,
        normalized=norm,
    )
    assert rep["profile"] == "research"
    assert "decoupled_piecewise" in rep["rule"]
    assert rep.get("normalization_version")
    norm_strict = normalize_objective_vector(
        raw, NormalizationProfile.STRICT_PROD, baseline_predictive=0.2, calibration_details=None
    )
    assert norm == norm_strict


def test_describe_strict_prod_documents_piecewise_rules() -> None:
    raw = (0.1, 0.5, 0.02, 0.0, 1.0)
    norm = normalize_objective_vector(
        raw,
        NormalizationProfile.STRICT_PROD,
        baseline_predictive=0.2,
        calibration_details={"mean_lift_se": 0.5},
    )
    rep = describe_objective_normalization(
        raw,
        NormalizationProfile.STRICT_PROD,
        baseline_predictive=0.2,
        calibration_details={"mean_lift_se": 0.5},
        normalized=norm,
    )
    assert rep["profile"] == "strict_prod"
    assert "piecewise" in rep
