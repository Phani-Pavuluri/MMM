"""Tests for H5n shadow-policy recommender."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mmm.research.bayes_h3_sandbox.h5_shadow_policy_recommender import (
    CHANNEL_COMPOSITE,
    CHANNEL_DROP_COLLINEAR,
    CHANNEL_DO_NOT_RUN,
    CHANNEL_EXTERNAL_CALIBRATION,
    CHANNEL_KEEP_ALL_WEAK_ID,
    GEOM_ABLATION_BENCHMARK,
    H5ShadowPolicyRecommenderError,
    ShadowPolicyRecommendationInput,
    STATUS_BLOCKED,
    STATUS_DO_NOT_RUN,
    STATUS_RECOMMENDED,
    STATUS_REQUIRES_EXTERNAL_CALIBRATION,
    build_sample_panel_recommendation,
    recommend_shadow_policy,
    validate_recommendation_artifact,
    write_sample_panel_recommendation_artifact,
)

POLICY_PATH = Path("docs/06_investigations/h5m_sample_panel_shadow_policy.json")


def _base_collinearity(max_corr: float, *, groups: list | None = None) -> dict:
    return {
        "max_abs_correlation": max_corr,
        "collinear_groups": groups or [],
        "pairwise_correlations": {},
    }


def _inp(
    *,
    max_corr: float = 0.5,
    experiments: list[dict] | None = None,
    business: dict | None = None,
    frozen: dict | None = None,
    sparsity: dict | None = None,
) -> ShadowPolicyRecommendationInput:
    return ShadowPolicyRecommendationInput(
        panel_id="test_panel",
        dataset_snapshot_id="snap-1",
        panel_schema={"media_columns": ["search", "social", "tv"], "geo_column": "g"},
        collinearity_diagnostics=_base_collinearity(max_corr),
        sparsity_diagnostics=sparsity or {"by_channel": {}},
        convergence_experiment_results=experiments or [],
        business_metadata=business,
        frozen_policy_reference=frozen,
    )


def test_recommends_keep_all_when_no_collinearity() -> None:
    art = recommend_shadow_policy(
        _inp(
            max_corr=0.4,
            experiments=[
                {
                    "variant_id": "ok",
                    "convergence_status": "converged_diagnostic_only",
                    "hierarchy_faithful": True,
                }
            ],
        )
    )
    assert art["recommended_shadow_policy"]["channel_recommendation_id"] == CHANNEL_KEEP_ALL_WEAK_ID
    assert art["recommended_shadow_policy"]["status"] == STATUS_RECOMMENDED
    assert art["approved_for_prod"] is False


def test_recommends_drop_collinear_when_governed_drop_converged() -> None:
    frozen = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    art = recommend_shadow_policy(
        _inp(
            max_corr=0.99,
            frozen=frozen,
            experiments=[
                {
                    "variant_id": "H5M-REPLAY",
                    "convergence_status": "converged_diagnostic_only",
                    "channel_policy": frozen["channel_policy"],
                    "hierarchy_faithful": True,
                }
            ],
        )
    )
    rec = art["recommended_shadow_policy"]
    assert rec["channel_recommendation_id"] == CHANNEL_DROP_COLLINEAR
    assert rec["status"] == STATUS_RECOMMENDED
    assert rec["channel_policy"]["dropped_channels"] == ["tv"]
    assert rec["channel_policy"]["kept_channels"] == ["search", "social"]
    assert art["evidence_status"]["evidence_promotion_allowed"] is True


def test_recommends_composite_when_channels_inseparable() -> None:
    art = recommend_shadow_policy(
        _inp(
            max_corr=0.99,
            business={"channels_strategically_inseparable": [["social", "tv"]]},
            experiments=[],
        )
    )
    ids = [a["recommendation_id"] for a in art["allowed_alternatives"]]
    assert CHANNEL_COMPOSITE in ids


def test_requires_external_calibration_when_business_critical() -> None:
    art = recommend_shadow_policy(
        _inp(
            max_corr=0.99,
            business={"channel_separation_business_critical": True},
            experiments=[],
        )
    )
    assert art["recommended_shadow_policy"]["status"] == STATUS_REQUIRES_EXTERNAL_CALIBRATION
    assert any(
        b["recommendation_id"] == CHANNEL_EXTERNAL_CALIBRATION
        for b in art["blocked_options"] + art["allowed_alternatives"]
    )


def test_blocks_keep_all_when_collinearity_and_keep_all_failed() -> None:
    art = recommend_shadow_policy(
        _inp(
            max_corr=0.99,
            experiments=[
                {
                    "variant_id": "H5J-A",
                    "convergence_status": "failed_convergence",
                    "channel_config": {"mode": "keep_all_channels"},
                }
            ],
        )
    )
    blocked = [b for b in art["blocked_options"] + art["allowed_alternatives"] if b.get("recommendation_id") == CHANNEL_KEEP_ALL_WEAK_ID]
    assert any(b["status"] == STATUS_BLOCKED for b in blocked)


def test_blocks_promotion_when_only_ablation_converged() -> None:
    art = recommend_shadow_policy(
        _inp(
            max_corr=0.99,
            experiments=[
                {
                    "variant_id": "H5K-E-POOLED",
                    "convergence_status": "converged_diagnostic_only",
                    "geometry_config": {"hierarchy_policy": "pooled_channel_effects_ablation"},
                    "ablation_only": True,
                }
            ],
        )
    )
    assert art["evidence_status"]["evidence_promotion_allowed"] is False
    assert art["recommended_shadow_policy"]["status"] == STATUS_DO_NOT_RUN
    assert any(
        a.get("recommendation_id") == GEOM_ABLATION_BENCHMARK
        for a in art["allowed_alternatives"]
    )


def test_dropped_channel_forbidden_claim() -> None:
    frozen = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    art = recommend_shadow_policy(_inp(max_corr=0.99, frozen=frozen, experiments=[]))
    assert any("tv" in c.lower() for c in art["forbidden_claims"] + art["interpretation_changes"])
    assert any("Dropped channel" in c for c in art["interpretation_changes"])


def test_composite_interpretation_combined_effect() -> None:
    art = recommend_shadow_policy(
        _inp(
            max_corr=0.99,
            business={"channels_strategically_inseparable": [["social", "tv"]]},
        )
    )
    comp = next(a for a in art["allowed_alternatives"] if a["recommendation_id"] == CHANNEL_COMPOSITE)
    text = " ".join(art["interpretation_changes"] + [comp["rationale"]])
    assert "combined-media-block" in text or "combined" in text.lower()


def test_production_flags_always_false() -> None:
    art = recommend_shadow_policy(_inp(max_corr=0.3))
    assert art["approved_for_prod"] is False
    assert art["prod_decisioning_allowed"] is False
    assert art["hard_gate"] is False


def test_no_optimizer_decision_surface_on_artifact() -> None:
    art = recommend_shadow_policy(_inp(max_corr=0.3))
    assert "optimizer" in art["excluded_fields"]
    assert "DecisionSurface" in art["excluded_fields"]
    assert art.get("decision_surface") is None


def test_sample_panel_recommendation_matches_h5m_policy() -> None:
    if not POLICY_PATH.is_file():
        pytest.skip("H5m policy missing")
    art = build_sample_panel_recommendation()
    validate_recommendation_artifact(art)
    rec = art["recommended_shadow_policy"]
    assert rec["status"] == STATUS_RECOMMENDED
    assert rec["channel_policy"]["dropped_channels"] == ["tv"]
    assert rec["channel_policy"]["kept_channels"] == ["search", "social"]
    assert rec["h5_geometry_config"]["sigma_policy"] == "sigma_floor"
    assert art["recommended_frozen_policy_id"] == "bayes_h5m_sample_panel_shadow_policy_v1"
    assert art["evidence_status"]["evidence_promotion_allowed"] is True


def test_missing_collinearity_fails_closed() -> None:
    with pytest.raises(H5ShadowPolicyRecommenderError, match="collinearity_diagnostics"):
        recommend_shadow_policy(
            ShadowPolicyRecommendationInput(
                panel_id="p",
                dataset_snapshot_id="s",
                panel_schema={"media_columns": ["a"]},
                collinearity_diagnostics={},
            )
        )


def test_write_artifact_file(tmp_path) -> None:
    if not Path("examples/sample_panel.csv").is_file():
        pytest.skip("sample panel missing")
    out = tmp_path / "rec.json"
    art = write_sample_panel_recommendation_artifact(out)
    assert out.is_file()
    assert art["artifact_id"]
