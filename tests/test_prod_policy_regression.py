"""Regression tests for PROD policy (must fail closed on accidental weakening)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from mmm.artifacts.decision_bundle import build_decision_bundle, validate_prod_decision_bundle
from mmm.config.schema import CVSplitAxis, Framework, MMMConfig, RunEnvironment
from mmm.governance.policy import PolicyError
from mmm.contracts.runtime_validation import SemanticContractError, assert_decision_artifact_tier
from mmm.data.schema import PanelSchema
from mmm.decision.core import finalize_and_validate_cli_decision_bundle


def _prod_ext_planning_allowed() -> dict:
    return {
        "ridge_fit_summary": {
            "best_params": {"decay": 0.5, "hill_half": 1.0, "hill_slope": 2.0},
            "coef": [0.05, 0.04],
            "intercept": [3.0],
        },
        "governance": {"approved_for_optimization": True},
        "response_diagnostics": {"safe_for_optimization": True},
        "identifiability": {"identifiability_score": 0.5},
        "panel_qa": {"max_severity": "info", "issues": []},
        "model_release": {"state": "planning_allowed", "reasons": [], "triggers": {}},
        "experiment_matching": {"replay_ok": True},
    }


def _prod_yaml(tmp_path: Path, csv_path: Path) -> Path:
    y = tmp_path / "prod.yaml"
    y.write_text(
        f"""
run_environment: prod
allow_unsafe_decision_apis: false
cv:
  mode: rolling
objective:
  normalization_profile: strict_prod
  named_profile: ridge_bo_standard_v1
prod_canonical_modeling_contract_id: ridge_bo_semi_log_calendar_cv_v1
data:
  path: {csv_path}
  geo_column: geo
  week_column: week
  target_column: revenue
  channel_columns: [c1, c2]
  control_columns: []
  data_version_id: regtest-dataset-snapshot
budget:
  total_budget: 100
extensions:
  optimization_gates:
    enabled: true
""",
        encoding="utf-8",
    )
    return y


def _tiny_panel(tmp_path: Path) -> Path:
    p = tmp_path / "panel.csv"
    p.write_text("geo,week,c1,c2,revenue\nG1,1,10,8,100\n", encoding="utf-8")
    return p


def test_prod_simulate_rejects_reporting_allowed_model_release(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MMM_GIT_SHA", "regtestsha")
    from mmm.cli import main as cli_main

    csv = _tiny_panel(tmp_path)
    ext = tmp_path / "ext.json"
    er = _prod_ext_planning_allowed()
    er["model_release"] = {"state": "reporting_allowed", "reasons": [], "triggers": {}}
    ext.write_text(json.dumps(er), encoding="utf-8")
    yaml = _prod_yaml(tmp_path, csv)
    scen = tmp_path / "scen.yaml"
    scen.write_text("candidate_spend:\n  c1: 12.0\n  c2: 10.0\n", encoding="utf-8")
    out = tmp_path / "out.json"
    r = CliRunner().invoke(
        cli_main.app,
        ["decide", "simulate", str(yaml), "--scenario", str(scen), "--extension-report", str(ext), "--out", str(out)],
    )
    assert r.exit_code == 2
    assert "planning_allowed" in (r.stderr or "") + (r.stdout or "")


def test_prod_optimize_rejects_reporting_allowed_model_release(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MMM_GIT_SHA", "regtestsha")
    from mmm.cli import main as cli_main

    csv = _tiny_panel(tmp_path)
    ext = tmp_path / "ext.json"
    er = _prod_ext_planning_allowed()
    er["model_release"] = {"state": "reporting_allowed", "reasons": [], "triggers": {}}
    ext.write_text(json.dumps(er), encoding="utf-8")
    yaml = _prod_yaml(tmp_path, csv)
    out = tmp_path / "opt.json"
    r = CliRunner().invoke(
        cli_main.app,
        ["decide", "optimize-budget", str(yaml), "--extension-report", str(ext), "--out", str(out)],
    )
    assert r.exit_code == 2
    assert "planning_allowed" in (r.stderr or "") + (r.stdout or "")


def test_prod_finalize_rejects_non_decision_tier(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MMM_GIT_SHA", "abc")
    cfg = MMMConfig(
        run_environment=RunEnvironment.PROD,
        prod_canonical_modeling_contract_id="ridge_bo_semi_log_calendar_cv_v1",
        data={"channel_columns": ["c1"], "control_columns": []},
        cv={"mode": "rolling"},
        objective={
            "normalization_profile": "strict_prod",
            "named_profile": "ridge_bo_standard_v1",
        },
    )
    schema = PanelSchema("g", "w", "y", ("c1",))
    fp = {"sha256_panel_keycols_sorted_csv": "x" * 64, "sha256_schema_json": "y" * 64, "n_rows": 1}
    bundle = build_decision_bundle(
        config=cfg,
        schema=schema,
        governance={"approved_for_optimization": True},
        simulation_contract={"source": "t", "objective": "delta_mu"},
        data_fingerprint=fp,
        economics_surface="full_model_simulation",
        decision_safe=True,
        governance_passed=True,
        extension_report={"ridge_fit_summary": {"coef": [0.1]}},
        simulation_json={"aggregation_semantics": "mean_mu_over_all_panel_rows_equal_weight"},
        artifact_tier="research",
    )
    with pytest.raises(SemanticContractError, match="artifact_tier"):
        assert_decision_artifact_tier(bundle, run_environment=RunEnvironment.PROD)


def test_prod_bayesian_missing_ppc_gate_fails_at_config_parse() -> None:
    with pytest.raises(ValueError, match="bayesian_max_mean_abs_ppc_gap"):
        MMMConfig(
            run_environment=RunEnvironment.PROD,
            framework=Framework.BAYESIAN,
            data={"channel_columns": ["c1"], "control_columns": []},
            cv={"mode": "rolling"},
            objective={"normalization_profile": "strict_prod"},
            bayesian={"posterior_predictive_draws": 100},
        )


def test_prod_geo_blocked_cv_rejected() -> None:
    with pytest.raises(PolicyError, match="calendar_week"):
        MMMConfig(
            run_environment=RunEnvironment.PROD,
            prod_canonical_modeling_contract_id="ridge_bo_semi_log_calendar_cv_v1",
            data={"channel_columns": ["c1"], "control_columns": []},
            cv={"mode": "rolling", "split_axis": CVSplitAxis.GEO_BLOCKED.value},
            objective={
                "normalization_profile": "strict_prod",
                "named_profile": "ridge_bo_standard_v1",
            },
        )


def test_finalize_rejects_incomplete_bundle_after_semantic_strip(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MMM_GIT_SHA", "abc")
    cfg = MMMConfig(
        run_environment=RunEnvironment.PROD,
        prod_canonical_modeling_contract_id="ridge_bo_semi_log_calendar_cv_v1",
        data={"channel_columns": ["c1"], "control_columns": []},
        cv={"mode": "rolling"},
        objective={
            "normalization_profile": "strict_prod",
            "named_profile": "ridge_bo_standard_v1",
        },
    )
    schema = PanelSchema("g", "w", "y", ("c1",))
    fp = {"sha256_panel_keycols_sorted_csv": "x" * 64, "sha256_schema_json": "y" * 64, "n_rows": 1}
    bundle = build_decision_bundle(
        config=cfg,
        schema=schema,
        governance={"approved_for_optimization": True},
        simulation_contract={"source": "t", "objective": "delta_mu"},
        data_fingerprint=fp,
        economics_surface="full_model_simulation",
        decision_safe=True,
        governance_passed=True,
        extension_report={"ridge_fit_summary": {"coef": [0.1]}},
        simulation_json={"aggregation_semantics": "mean_mu_over_all_panel_rows_equal_weight"},
        runtime_policy_hash="a" * 16,
    )
    bundle["dataset_snapshot_id"] = bundle.get("dataset_snapshot_id") or "test-snapshot"
    bundle.pop("semantic_contract", None)
    sim = {"aggregation_semantics": "mean_mu_over_all_panel_rows_equal_weight"}
    with pytest.raises(SemanticContractError, match="semantic_contract"):
        finalize_and_validate_cli_decision_bundle(bundle, cfg, simulation_json=sim)
