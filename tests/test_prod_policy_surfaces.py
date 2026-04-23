"""Prod policy: decision bundles, model_release gates, optimization safety gate."""

from __future__ import annotations

from mmm.artifacts.decision_bundle import build_decision_bundle, validate_prod_decision_bundle
from mmm.config.extensions import ExtensionSuiteConfig, GovernanceConfig, OptimizationGateConfig
from mmm.config.schema import MMMConfig, RunEnvironment
from mmm.data.schema import PanelSchema
from mmm.governance.model_release import ModelReleaseState, prod_release_allows_decision_cli
from mmm.optimization.safety_gate import OptimizationSafetyGate


def _minimal_config() -> MMMConfig:
    return MMMConfig(
        run_environment=RunEnvironment.PROD,
        prod_canonical_modeling_contract_id="ridge_bo_semi_log_calendar_cv_v1",
        data={
            "path": "dummy.csv",
            "geo_column": "g",
            "week_column": "w",
            "target_column": "y",
            "channel_columns": ["c1"],
        },
        cv={"mode": "rolling"},
        objective={
            "normalization_profile": "strict_prod",
            "named_profile": "ridge_bo_standard_v1",
        },
        extensions=ExtensionSuiteConfig(
            governance=GovernanceConfig(bayesian_max_mean_abs_ppc_gap=0.5),
            optimization_gates=OptimizationGateConfig(
                enabled=True,
                prod_block_on_panel_qa_block=True,
            ),
        ),
    )


def test_validate_prod_decision_bundle_requires_core_fields() -> None:
    cfg = _minimal_config()
    schema = PanelSchema("g", "w", "y", ("c1",))
    fp = {
        "sha256_panel_keycols_sorted_csv": "0" * 64,
        "sha256_schema_json": "1" * 64,
        "n_rows": 10,
    }
    bad = build_decision_bundle(
        config=cfg,
        schema=schema,
        governance={"approved_for_optimization": True},
        simulation_contract={"source": "t", "objective": "delta_mu"},
        data_fingerprint=fp,
        economics_surface="full_model_simulation",
        decision_safe=True,
        governance_passed=True,
        extension_report={"ridge_fit_summary": {"coef": [1.0]}},
    )
    bad.pop("governance", None)
    miss = validate_prod_decision_bundle(
        bad, run_environment=RunEnvironment.PROD, decision_cli_surface=False
    )
    assert any("governance" in m for m in miss)


def test_prod_release_blocks_optimize_unless_planning_allowed() -> None:
    ok, _ = prod_release_allows_decision_cli(
        {"state": ModelReleaseState.REPORTING_ALLOWED.value},
        surface="optimize_budget",
        run_environment=RunEnvironment.PROD,
    )
    assert not ok
    ok2, _ = prod_release_allows_decision_cli(
        {"state": ModelReleaseState.PLANNING_ALLOWED.value},
        surface="optimize_budget",
        run_environment=RunEnvironment.PROD,
    )
    assert ok2


def test_optimization_gate_prod_blocks_panel_qa_block() -> None:
    cfg = _minimal_config()
    gate = OptimizationSafetyGate(cfg.extensions.optimization_gates)
    gr = gate.check(
        governance={"approved_for_optimization": True},
        response_diag={"safe_for_optimization": True},
        identifiability_score=0.0,
        run_environment=RunEnvironment.PROD,
        extension_report_present=True,
        panel_qa={"max_severity": "block", "issues": []},
    )
    assert not gr.allowed
