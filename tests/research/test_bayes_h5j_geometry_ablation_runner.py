"""Tests for H5j geometry ablation runner."""

from __future__ import annotations

from mmm.research.bayes_h3_sandbox.h5_trust_diagnostics import classify_convergence_status
from mmm.research.bayes_h3_sandbox.h5j_geometry_ablation_runner import (
    build_geometry_ablation_artifact,
    default_ablation_specs,
    validate_geometry_ablation_artifact,
)


def test_artifact_schema_valid_without_fit() -> None:
    artifact = build_geometry_ablation_artifact(execute_fit=False)
    validate_geometry_ablation_artifact(artifact)
    assert artifact["approved_for_prod"] is False
    assert artifact["prod_decisioning_allowed"] is False
    assert artifact["hard_gate"] is False
    assert "media_correlation_matrix" in artifact
    assert len(artifact["ablation_results"]) == len(default_ablation_specs())


def test_production_flags_false_on_all_ablations() -> None:
    artifact = build_geometry_ablation_artifact(execute_fit=False)
    for row in artifact["ablation_results"]:
        assert row["approved_for_prod"] is False
        assert row["evidence_promotion_allowed"] is False


def test_no_optimizer_fields_on_artifact() -> None:
    artifact = build_geometry_ablation_artifact(execute_fit=False)
    for forbidden in ("decision_surface", "optimizer_ready_curves", "budget_recommendation"):
        assert artifact.get(forbidden) is None


def test_failed_convergence_not_promotable() -> None:
    assert classify_convergence_status(rhat_max=1.5, divergence_count=10) == "failed_convergence"


def test_pooled_variant_not_implemented() -> None:
    artifact = build_geometry_ablation_artifact(execute_fit=False)
    pooled = next(r for r in artifact["ablation_results"] if r["variant_id"] == "H5J-F-POOLED-CHANNEL-EFFECTS")
    assert pooled["convergence_status"] == "not_executed"
    assert pooled["recommendation"] == "not_implemented"


def test_ablation_specs_cover_required_variants() -> None:
    ids = {s.variant_id for s in default_ablation_specs()}
    assert "H5J-A-BASELINE-REPLAY" in ids
    assert "H5J-B-PRESCALE-EXTENDED" in ids
    assert "H5J-C-SINGLE-SEARCH-PRESCALE-EXTENDED" in ids
    assert "H5J-D-DROP-COLLINEAR-PRESCALE-EXTENDED" in ids
    assert "H5J-E-COMPOSITE-SOCIAL-TV-PRESCALE-EXTENDED" in ids
