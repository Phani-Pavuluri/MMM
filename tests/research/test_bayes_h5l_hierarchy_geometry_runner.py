"""Tests for H5l hierarchy-faithful geometry refinement runner."""

from __future__ import annotations

import pytest

from mmm.research.bayes_h3_sandbox.h5_geometry_config import (
    H5GeometryConfigError,
    HIERARCHY_FIXED_TAU,
    HIERARCHY_FULL_GEO_CHANNEL,
    HIERARCHY_POOLED_CHANNEL,
    HIERARCHY_STRENGTH_FIXED_TAU,
    PARAMETERIZATION_NON_CENTERED,
    SIGMA_POLICY_FLOOR,
    TAU_PARAM_CURRENT,
    TAU_PARAM_LOG_TAU,
    evidence_promotion_for_geometry,
    is_ablation_only_geometry,
    is_hierarchy_faithful_geometry,
    validate_geometry_config,
)
from mmm.research.bayes_h3_sandbox.h5l_hierarchy_geometry_runner import (
    build_hierarchy_geometry_artifact,
    default_hierarchy_specs,
    validate_hierarchy_geometry_artifact,
)
from mmm.research.bayes_h3_sandbox.h5_trust_diagnostics import classify_convergence_status


def _full_geom(**kwargs: object) -> dict[str, object]:
    base = {
        "parameterization": PARAMETERIZATION_NON_CENTERED,
        "likelihood_scale_policy": "prescaled_log_outcome",
        "hierarchy_policy": HIERARCHY_FULL_GEO_CHANNEL,
        "tau_parameterization": TAU_PARAM_CURRENT,
        "sigma_policy": "current_default",
        "beta_prior_policy": "current_default",
        "hierarchy_strength_policy": "learned_tau",
    }
    base.update(kwargs)
    return base


def test_hierarchy_faithful_vs_ablation_labels() -> None:
    faithful = _full_geom()
    pooled = _full_geom(hierarchy_policy=HIERARCHY_POOLED_CHANNEL)
    fixed = _full_geom(
        hierarchy_policy=HIERARCHY_FIXED_TAU,
        hierarchy_strength_policy=HIERARCHY_STRENGTH_FIXED_TAU,
        fixed_tau_value=0.2,
    )
    assert is_hierarchy_faithful_geometry(faithful) is True
    assert is_ablation_only_geometry(pooled) is True
    assert is_ablation_only_geometry(fixed) is True


def test_ablation_cannot_be_evidence_promoted_even_if_converged() -> None:
    geom = _full_geom(
        hierarchy_policy=HIERARCHY_POOLED_CHANNEL,
    )
    status = classify_convergence_status(rhat_max=1.0, divergence_count=0)
    assert status == "converged_diagnostic_only"
    assert evidence_promotion_for_geometry(status, geom) is False


def test_faithful_can_be_promoted_when_converged() -> None:
    geom = _full_geom()
    status = classify_convergence_status(rhat_max=1.0, divergence_count=0)
    assert evidence_promotion_for_geometry(status, geom) is True


def test_unsupported_tau_parameterization_fails_closed() -> None:
    with pytest.raises(H5GeometryConfigError, match="tau_parameterization"):
        validate_geometry_config(_full_geom(tau_parameterization="horseshoe"))


def test_unsupported_sigma_policy_fails_closed() -> None:
    with pytest.raises(H5GeometryConfigError, match="sigma_policy"):
        validate_geometry_config(_full_geom(sigma_policy="inverse_gamma"))


def test_sigma_floor_requires_value() -> None:
    with pytest.raises(H5GeometryConfigError, match="sigma_floor"):
        validate_geometry_config(_full_geom(sigma_policy=SIGMA_POLICY_FLOOR))


def test_artifact_schema_valid_without_fit() -> None:
    artifact = build_hierarchy_geometry_artifact(execute_fit=False)
    validate_hierarchy_geometry_artifact(artifact)
    assert artifact["approved_for_prod"] is False
    assert artifact["prod_decisioning_allowed"] is False
    assert artifact["hard_gate"] is False
    assert artifact["any_hierarchy_faithful_converged_diagnostic_only"] is False
    assert "ablation_variants_not_promotion_evidence" in artifact


def test_production_flags_false_on_all_variants() -> None:
    artifact = build_hierarchy_geometry_artifact(execute_fit=False)
    for row in artifact["variants"]:
        assert row["approved_for_prod"] is False
        assert row["evidence_promotion_allowed"] is False


def test_no_optimizer_fields_on_artifact() -> None:
    artifact = build_hierarchy_geometry_artifact(execute_fit=False)
    for forbidden in ("decision_surface", "optimizer_ready_curves", "budget_recommendation"):
        assert artifact.get(forbidden) is None


def test_specs_label_benchmarks() -> None:
    specs = default_hierarchy_specs()
    fixed = next(s for s in specs if s.variant_id == "H5L-J-FIXED-TAU-BENCHMARK")
    pooled = next(s for s in specs if s.variant_id == "H5L-K-POOLED-BENCHMARK")
    assert fixed.ablation_only is True
    assert fixed.hierarchy_faithful is False
    assert pooled.ablation_only is True


def test_specs_cover_faithful_variants() -> None:
    ids = {s.variant_id for s in default_hierarchy_specs() if s.hierarchy_faithful}
    assert "H5L-A-H5K-FULL-HIERARCHY-REPLAY" in ids
    assert "H5L-B-NC-SIGMA-FLOOR" in ids
    assert "H5L-F-LOG-TAU-REPARAM" in ids


def test_log_tau_config_valid() -> None:
    validate_geometry_config(_full_geom(tau_parameterization=TAU_PARAM_LOG_TAU))
