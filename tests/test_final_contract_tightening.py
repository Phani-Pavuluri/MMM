"""Regression tests for final contract tightening (typed sections, allowlist, curve export gate, waivers)."""

from __future__ import annotations

import pytest

from mmm.config.schema import DataConfig, Framework, MMMConfig, ModelForm, RunEnvironment
from mmm.contracts.business_surface import enrich_decision_simulation_json
from mmm.contracts.quantity_models import (
    ROIApproxQuantityResult,
    validate_typed_approximate_artifact_section,
)
from mmm.decision.optimize_enrichment import (
    SIMULATION_AT_RECOMMENDATION_FORBIDDEN_TOP_LEVEL_KEYS,
    apply_simulation_at_recommendation_allowlist,
    apply_simulation_at_recommendation_allowlist_post_enrich,
)
from mmm.decomposition.curve_export_gate import validate_curve_bundle_typed_curve_quantity
from mmm.governance.identifiability_waiver import IdentifiabilityWaiverArtifact, validate_waiver_for_run
from mmm.governance.policy import PolicyError


def test_validate_typed_approximate_artifact_section_rejects_plain_roi_bridge() -> None:
    with pytest.raises(PolicyError, match="quantity_contract_version"):
        validate_typed_approximate_artifact_section(
            {"semantics": "roi_first_order", "roi_bridge": {}},
            section_name="roi_bridge",
        )


def test_validate_typed_approximate_artifact_section_accepts_typed_roi() -> None:
    sec = ROIApproxQuantityResult().section_dict()
    validate_typed_approximate_artifact_section(sec, section_name="roi_bridge")


def test_curve_export_gate_requires_typed_curve_quantity() -> None:
    with pytest.raises(PolicyError, match="typed_curve_quantity"):
        validate_curve_bundle_typed_curve_quantity(
            {"channel": "c", "spend_grid": [1.0], "response_on_modeling_scale": [1.0]},
            context="test",
        )


def test_optimize_allowlist_strips_unknown_in_research() -> None:
    cfg = MMMConfig(
        framework=Framework.RIDGE_BO,
        model_form=ModelForm.SEMI_LOG,
        data=DataConfig(path=None, channel_columns=["a"], target_column="y"),
        run_environment=RunEnvironment.RESEARCH,
    )
    sim = {"delta_mu": 1.0, "posterior_exploration_quantity_injected": {"x": 1}}
    filtered, audit = apply_simulation_at_recommendation_allowlist(sim, cfg=cfg, context="t")
    assert "posterior_exploration_quantity_injected" not in filtered
    assert "removed_keys" in audit


def test_optimize_allowlist_forbidden_raises() -> None:
    cfg = MMMConfig(
        framework=Framework.RIDGE_BO,
        model_form=ModelForm.SEMI_LOG,
        data=DataConfig(path=None, channel_columns=["a"], target_column="y"),
        run_environment=RunEnvironment.RESEARCH,
    )
    bad = {k: 1 for k in sorted(SIMULATION_AT_RECOMMENDATION_FORBIDDEN_TOP_LEVEL_KEYS)[:1]}
    with pytest.raises(PolicyError, match="forbidden"):
        apply_simulation_at_recommendation_allowlist(bad, cfg=cfg, context="t")


def test_post_enrich_allowlist_accepts_enriched_payload() -> None:
    base = {
        "delta_mu": 1.0,
        "baseline_mu": 0.0,
        "plan_mu": 1.0,
        "delta_spend": 0.0,
        "roi": None,
        "mroas": None,
        "baseline_type": "bau",
        "baseline_definition": "bau",
        "uncertainty_mode": "point",
        "decision_safe": True,
        "economics_version": "v",
        "planner_mode": "full_model",
        "canonical_quantity": "q",
        "mean_kpi_level_baseline": None,
        "mean_kpi_level_plan": None,
        "delta_kpi_level": None,
        "disclosure": "",
        "p10": None,
        "p50": None,
        "p90": None,
        "horizon_weeks": None,
        "candidate_plan_type": "constant_channel_levels",
        "counterfactual_construction_method": "m",
        "spend_path_assumption": "s",
        "aggregation_semantics": "mean_mu_over_all_panel_rows_equal_weight",
        "kpi_column": "y",
    }
    cfg = MMMConfig(
        framework=Framework.RIDGE_BO,
        model_form=ModelForm.SEMI_LOG,
        data=DataConfig(path=None, channel_columns=["a"], target_column="y"),
        run_environment=RunEnvironment.RESEARCH,
    )
    pre, _ = apply_simulation_at_recommendation_allowlist(base, cfg=cfg, context="t")
    enriched = enrich_decision_simulation_json(
        pre,
        cfg=cfg,
        unsupported_questions=[],
        governance_gate_allowed=True,
    )
    apply_simulation_at_recommendation_allowlist_post_enrich(enriched, context="t2")


def test_waiver_rejects_dataset_snapshot_mismatch() -> None:
    w = IdentifiabilityWaiverArtifact(
        waiver_id="w1234",
        created_at="2025-01-01T00:00:00+00:00",
        reason="integration test waiver reason long enough",
        affected_dataset_snapshot_id="snap-a",
        max_identifiability_score_waived=1.0,
    )
    with pytest.raises(PolicyError, match="dataset_snapshot"):
        validate_waiver_for_run(
            w,
            identifiability_score=0.9,
            model_release_id=None,
            config_fingerprint_sha256=None,
            dataset_snapshot_id="snap-b",
        )
