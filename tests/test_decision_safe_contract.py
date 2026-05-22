"""Canonical decision_safe contract (audit fix P2)."""

from __future__ import annotations

import pytest

from mmm.artifacts.schema import SimulationDecisionResult
from mmm.config.schema import Framework, MMMConfig, RunEnvironment
from mmm.contracts.business_surface import enrich_decision_simulation_json
from mmm.governance.decision_safe_contract import (
    canonical_decision_safe,
    scenario_decision_safe_from_simulation,
)


def _sim_base(*, decision_safe: bool = True, baseline_type: str = "bau") -> dict:
    return {
        "baseline_mu": 1.0,
        "plan_mu": 1.1,
        "delta_mu": 0.1,
        "baseline_type": baseline_type,
        "baseline_suitable_for_decisioning": baseline_type == "bau",
        "decision_safe": decision_safe,
        "uncertainty_mode": "point",
    }


def test_bau_and_gates_pass_safe() -> None:
    sim = _sim_base(decision_safe=True)
    assert scenario_decision_safe_from_simulation(sim) is True
    cfg = MMMConfig(framework=Framework.RIDGE_BO, data={"channel_columns": ["a"]})
    out = enrich_decision_simulation_json(
        sim, cfg=cfg, unsupported_questions=[], governance_gate_allowed=True
    )
    assert out["decision_safe"] is True
    assert out["scenario_decision_safe"] is True


def test_locked_baseline_unsafe_despite_gates() -> None:
    sim = _sim_base(decision_safe=False, baseline_type="locked_plan")
    sim["baseline_suitable_for_decisioning"] = False
    assert scenario_decision_safe_from_simulation(sim) is False
    cfg = MMMConfig(framework=Framework.RIDGE_BO, data={"channel_columns": ["a"]})
    out = enrich_decision_simulation_json(
        sim, cfg=cfg, unsupported_questions=[], governance_gate_allowed=True
    )
    assert out["decision_safe"] is False


def test_bau_gates_fail_unsafe() -> None:
    sim = _sim_base(decision_safe=True)
    cfg = MMMConfig(framework=Framework.RIDGE_BO, data={"channel_columns": ["a"]})
    out = enrich_decision_simulation_json(
        sim, cfg=cfg, unsupported_questions=[], governance_gate_allowed=False
    )
    assert out["decision_safe"] is False


def test_missing_decision_safe_scenario_unsafe() -> None:
    sim = {"baseline_mu": 1.0, "plan_mu": 1.1, "delta_mu": 0.1, "baseline_type": "bau"}
    assert scenario_decision_safe_from_simulation(sim) is False


def test_simulation_decision_result_preserves_decision_safe() -> None:
    sim = _sim_base(decision_safe=False)
    cfg = MMMConfig(framework=Framework.RIDGE_BO, data={"channel_columns": ["a"]})
    enriched = enrich_decision_simulation_json(
        sim, cfg=cfg, unsupported_questions=[], governance_gate_allowed=True
    )
    canon = SimulationDecisionResult.from_simulation_json(
        enriched, governance_refs={}, lineage_refs={}
    )
    assert canon.as_result_dict()["decision_safe"] is False


def test_from_simulation_json_requires_bool_decision_safe() -> None:
    with pytest.raises(ValueError, match="decision_safe"):
        SimulationDecisionResult.from_simulation_json(
            {"baseline_mu": 1.0, "plan_mu": 2.0, "delta_mu": 1.0},
            governance_refs={},
            lineage_refs={},
        )


def test_locked_baseline_blocks_prod_decision_tier_even_if_gates_pass() -> None:
    cfg = MMMConfig(
        framework=Framework.RIDGE_BO,
        data={"channel_columns": ["a"]},
        run_environment=RunEnvironment.RESEARCH,
    )
    sim = _sim_base(decision_safe=False, baseline_type="locked_plan")
    sim["baseline_suitable_for_decisioning"] = False
    out = enrich_decision_simulation_json(
        sim,
        cfg=cfg,
        unsupported_questions=["q"],
        governance_gate_allowed=True,
    )
    assert out["decision_safe"] is False
    assert out["artifact_tier"] == "research"
    assert out["scenario_decision_safe"] is False


def test_canonical_helper_optimizer_failure() -> None:
    assert (
        canonical_decision_safe(
            scenario_safe=True,
            governance_gate_allowed=True,
            optimizer_internal_safe=False,
            optimizer_success=True,
        )
        is False
    )


def test_non_bau_blocks_decision_safe() -> None:
    sim = _sim_base(decision_safe=True, baseline_type="locked_plan")
    sim["baseline_suitable_for_decisioning"] = False
    cfg = MMMConfig(
        run_environment=RunEnvironment.RESEARCH,
        data={"channel_columns": ["c1"], "control_columns": []},
    )
    out = enrich_decision_simulation_json(
        sim, cfg=cfg, unsupported_questions=[], governance_gate_allowed=True
    )
    assert out["decision_safe"] is False
    assert scenario_decision_safe_from_simulation(sim) is False
