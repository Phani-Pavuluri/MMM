"""Unified decision_safe semantics across surfaces."""

from __future__ import annotations

from mmm.config.schema import MMMConfig, RunEnvironment
from mmm.contracts.business_surface import enrich_decision_simulation_json
from mmm.governance.decision_safe_contract import compute_decision_safe


def test_same_scenario_same_flag_across_surfaces() -> None:
    cfg = MMMConfig(
        run_environment=RunEnvironment.RESEARCH,
        data={"channel_columns": ["c1"], "control_columns": []},
    )
    sim = {
        "baseline_mu": 10.0,
        "plan_mu": 11.0,
        "delta_mu": 1.0,
        "baseline_type": "bau",
        "scenario_suitable_for_decisioning": True,
        "uncertainty_mode": "point",
    }
    expected = compute_decision_safe(
        governance_gate_allowed=True,
        scenario_suitable_for_decisioning=True,
        baseline_is_bau=True,
        run_environment=cfg.run_environment,
    )
    enriched = enrich_decision_simulation_json(
        dict(sim), cfg=cfg, unsupported_questions=[], governance_gate_allowed=True
    )
    assert enriched["decision_safe"] is expected
    assert compute_decision_safe(
        governance_gate_allowed=False,
        scenario_suitable_for_decisioning=True,
        baseline_is_bau=True,
    ) is False


def test_non_bau_blocks_decision_safe() -> None:
    assert (
        compute_decision_safe(
            governance_gate_allowed=True,
            scenario_suitable_for_decisioning=True,
            baseline_is_bau=False,
        )
        is False
    )
