"""Semantic validation for planning_assumptions on decision-tier prod bundles."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from mmm.artifacts.decision_bundle import build_decision_bundle, validate_prod_decision_bundle
from mmm.config.schema import MMMConfig, RunEnvironment
from mmm.data.schema import PanelSchema
from mmm.planning.assumption_contract import (
    parse_planning_assumptions,
    planning_assumptions_enum_errors,
    validate_planning_assumptions_semantics,
)
from mmm.planning.assumptions import build_planning_assumptions, minimal_planning_assumptions_optimize


def _fp() -> dict:
    return {"sha256_panel_keycols_sorted_csv": "x" * 64, "sha256_schema_json": "y" * 64, "n_rows": 1}


def _prod_bundle(
    *,
    planning_assumptions: dict | None,
    scenario_lineage: dict | None = None,
    optimize: bool = False,
) -> dict:
    cfg = MMMConfig(
        run_environment=RunEnvironment.PROD,
        prod_canonical_modeling_contract_id="ridge_bo_semi_log_calendar_cv_v1",
        data={"channel_columns": ["a"], "control_columns": []},
        cv={"mode": "rolling"},
        objective={"normalization_profile": "strict_prod", "named_profile": "ridge_bo_standard_v1"},
    )
    schema = PanelSchema("g", "w", "y", ("a",))
    sim_contract = (
        {"source": "full_model_simulation_slsqp", "objective": "delta_mu"}
        if optimize
        else {"source": "decision_service_simulate", "objective": "delta_mu"}
    )
    return build_decision_bundle(
        config=cfg,
        schema=schema,
        governance={"approved_for_optimization": True},
        simulation_contract=sim_contract,
        data_fingerprint=_fp(),
        economics_surface="full_model_simulation",
        decision_safe=True,
        artifact_tier="decision",
        governance_passed=True,
        optimizer_success=True if optimize else None,
        extension_report={"ridge_fit_summary": {"coef": [0.1]}},
        simulation_json={"aggregation_semantics": "mean_mu_over_all_panel_rows_equal_weight"},
        runtime_policy_hash="a" * 16,
        planning_assumptions=planning_assumptions,
        scenario_lineage=scenario_lineage,
    )


@pytest.mark.parametrize(
    ("controls", "media", "world"),
    [
        ("observed", "optimized", "historical_panel"),
        ("overlay", "piecewise_path", "explicit_scenario"),
        ("frozen_scenario", "optimized", "explicit_scenario"),
    ],
)
def test_valid_semantic_combinations_pass(controls: str, media: str, world: str) -> None:
    lineage = None
    if world == "explicit_scenario":
        lineage = {
            "scenario_id": "s1",
            "scenario_hash": "a" * 64,
            "plan_overlay_spec_sha256": "b" * 64 if controls == "overlay" else None,
        }
        if controls == "frozen_scenario":
            lineage["non_media_overlay_applied"] = True
    elif controls == "frozen_scenario":
        lineage = {"scenario_id": "s1", "scenario_hash": "a" * 64, "non_media_overlay_applied": True}
    pa = build_planning_assumptions(
        controls_assumption=controls,  # type: ignore[arg-type]
        media_assumption=media,  # type: ignore[arg-type]
        world_assumption=world,  # type: ignore[arg-type]
    )
    issues = validate_planning_assumptions_semantics(pa, scenario_lineage=lineage, strict=True)
    assert issues == []


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("controls_assumption", "observeddd"),
        ("media_assumption", "optimised"),
        ("world_assumption", "history"),
    ],
)
def test_invalid_enum_values_fail(field: str, bad: str) -> None:
    pa = minimal_planning_assumptions_optimize()
    pa[field] = bad
    issues = planning_assumptions_enum_errors(pa)
    assert any(field in i and "invalid_enum" in i for i in issues)


def test_malformed_not_dict() -> None:
    assert "planning_assumptions_must_be_dict" in planning_assumptions_enum_errors([])


def test_missing_required_field() -> None:
    pa = minimal_planning_assumptions_optimize()
    del pa["media_assumption"]
    errs = planning_assumptions_enum_errors(pa)
    assert any("media_assumption" in i for i in errs)


def test_explicit_scenario_without_lineage_fails() -> None:
    pa = build_planning_assumptions(
        controls_assumption="overlay",
        media_assumption="constant",
        world_assumption="explicit_scenario",
    )
    issues = validate_planning_assumptions_semantics(pa, scenario_lineage={}, strict=True)
    assert "explicit_scenario_requires_scenario_lineage_id_and_hash" in issues


def test_frozen_scenario_without_metadata_fails() -> None:
    pa = build_planning_assumptions(
        controls_assumption="frozen_scenario",
        media_assumption="optimized",
        world_assumption="historical_panel",
    )
    issues = validate_planning_assumptions_semantics(pa, scenario_lineage={}, strict=True)
    assert "frozen_scenario_requires_scenario_lineage_or_overlay_metadata" in issues


def test_optimized_media_on_simulate_bundle_fails() -> None:
    pa = build_planning_assumptions(
        controls_assumption="observed",
        media_assumption="optimized",
        world_assumption="historical_panel",
    )
    bundle = _prod_bundle(planning_assumptions=pa, optimize=False)
    issues = validate_planning_assumptions_semantics(
        pa, scenario_lineage=bundle.get("scenario_lineage"), bundle=bundle, strict=True
    )
    assert "optimized_media_assumption_requires_optimize_bundle_context" in issues


def test_optimize_bundle_requires_media_optimized() -> None:
    pa = build_planning_assumptions(
        controls_assumption="observed",
        media_assumption="constant",
        world_assumption="historical_panel",
    )
    bundle = _prod_bundle(planning_assumptions=pa, optimize=True)
    issues = validate_planning_assumptions_semantics(
        pa, scenario_lineage=bundle.get("scenario_lineage"), bundle=bundle, strict=True
    )
    assert "optimize_bundle_requires_media_assumption_optimized" in issues


def test_multi_world_rejected_in_strict_prod() -> None:
    pa = build_planning_assumptions(
        controls_assumption="observed",
        media_assumption="constant",
        world_assumption="multi_world",
    )
    issues = validate_planning_assumptions_semantics(pa, strict=True)
    assert "multi_world_not_implemented_for_decision_bundles" in issues


def test_parse_planning_assumptions_valid() -> None:
    raw = minimal_planning_assumptions_optimize()
    c = parse_planning_assumptions(raw)
    assert c.media_assumption == "optimized"


def test_parse_rejects_typo() -> None:
    raw = minimal_planning_assumptions_optimize()
    raw["controls_assumption"] = "observeddd"
    with pytest.raises(ValidationError):
        parse_planning_assumptions(raw)


def test_backward_compatible_valid_optimize_bundle(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MMM_GIT_SHA", "a" * 40)
    pa = minimal_planning_assumptions_optimize()
    bundle = _prod_bundle(planning_assumptions=pa, optimize=True)
    bundle["dataset_snapshot_id"] = bundle.get("dataset_snapshot_id") or "snap-1"
    bundle["semantic_contract"] = bundle.get("semantic_contract") or {
        "estimand": "delta_mu_full_panel",
        "aggregation": "mean_mu_over_all_panel_rows_equal_weight",
        "scale": "x",
        "baseline_definition": "bau",
    }
    bundle["unsupported_questions"] = bundle.get("unsupported_questions") or []
    miss = validate_prod_decision_bundle(bundle, run_environment=RunEnvironment.PROD, decision_cli_surface=True)
    planning_miss = [m for m in miss if m.startswith("planning_")]
    assert planning_miss == []
