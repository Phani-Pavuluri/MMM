"""Decision bundle hardening: tiers, semantics, CV prod gates, lineage."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from mmm.artifacts.decision_bundle import build_decision_bundle, validate_prod_decision_bundle
from mmm.config.schema import CVSplitAxis, MMMConfig, RunEnvironment
from mmm.governance.policy import PolicyError
from mmm.contracts.runtime_validation import SemanticContractError, validate_semantic_contract
from mmm.data.schema import PanelSchema
from mmm.reporting.safe_language import assert_safe_reporting_language


def test_prod_config_rejects_geo_rank_cv() -> None:
    with pytest.raises(PolicyError, match="calendar_week"):
        MMMConfig(
            run_environment=RunEnvironment.PROD,
            prod_canonical_modeling_contract_id="ridge_bo_semi_log_calendar_cv_v1",
            data={"channel_columns": ["c1"], "control_columns": []},
            cv={"mode": "rolling", "split_axis": CVSplitAxis.GEO_RANK.value},
            objective={
                "normalization_profile": "strict_prod",
                "named_profile": "ridge_bo_standard_v1",
            },
        )


def test_validate_prod_cli_bundle_requires_lineage(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MMM_GIT_SHA", "abc123")
    cfg = MMMConfig(
        run_environment=RunEnvironment.PROD,
        prod_canonical_modeling_contract_id="ridge_bo_semi_log_calendar_cv_v1",
        data={
            "path": "p.csv",
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
    )
    schema = PanelSchema("g", "w", "y", ("c1",))
    fp = {"sha256_panel_keycols_sorted_csv": "x" * 64, "sha256_schema_json": "y" * 64, "n_rows": 1}
    b = build_decision_bundle(
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
    )
    b2 = dict(b)
    b2.pop("git_sha")
    miss = validate_prod_decision_bundle(b2, run_environment=RunEnvironment.PROD, decision_cli_surface=True)
    assert any("git_sha" in m for m in miss)


def test_semantic_contract_rejects_baseline_mismatch() -> None:
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
        simulation_contract={"objective": "delta_mu"},
        data_fingerprint=fp,
        economics_surface="full_model_simulation",
        decision_safe=True,
        governance_passed=True,
        baseline_type="bau",
        extension_report={"ridge_fit_summary": {"coef": [0.1]}},
        simulation_json={"aggregation_semantics": "mean_mu_over_all_panel_rows_equal_weight"},
    )
    sim_json = {"aggregation_semantics": "mean_mu_over_all_panel_rows_equal_weight"}
    bundle["semantic_contract"] = dict(bundle["semantic_contract"])
    bundle["semantic_contract"]["baseline_definition"] = "mismatch"
    with pytest.raises(SemanticContractError):
        validate_semantic_contract(bundle, simulation_json=sim_json)


def test_safe_language_blocks_true_roi_phrase() -> None:
    with pytest.raises(ValueError, match="forbidden"):
        assert_safe_reporting_language("This is the true ROI for the brand", context="roi_table")


def test_cli_decide_simulate_matches_api_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MMM_GIT_SHA", "testsha")
    from mmm.cli import main as cli_main
    from mmm.decision.api import run_decision_simulation

    csv_path = tmp_path / "panel.csv"
    csv_path.write_text("geo,week,c1,c2,revenue\nG1,1,10,8,100\n", encoding="utf-8")
    ext = tmp_path / "ext.json"
    ext.write_text(
        json.dumps(
            {
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
        ),
        encoding="utf-8",
    )
    yaml = tmp_path / "c.yaml"
    yaml.write_text(
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
  data_version_id: test-dataset-snapshot-1
budget:
  total_budget: 100
extensions:
  optimization_gates:
    enabled: true
""",
        encoding="utf-8",
    )
    scen = tmp_path / "scen.yaml"
    scen.write_text("candidate_spend:\n  c1: 12.0\n  c2: 10.0\n", encoding="utf-8")
    out_cli = tmp_path / "cli.json"
    out_api = tmp_path / "api.json"
    runner = CliRunner()
    argv = [
        "decide",
        "simulate",
        str(yaml),
        "--scenario",
        str(scen),
        "--extension-report",
        str(ext),
        "--out",
        str(out_cli),
    ]
    r = runner.invoke(cli_main.app, argv)
    assert r.exit_code == 0, r.stdout + (r.stderr or "")
    run_decision_simulation(config=yaml, scenario=scen, extension_report=ext, out=out_api)
    cli_js = json.loads(out_cli.read_text(encoding="utf-8"))
    api_js = json.loads(out_api.read_text(encoding="utf-8"))

    def _scrub_nondeterministic(d: dict) -> None:
        db = d.get("decision_bundle")
        if isinstance(db, dict):
            db.pop("created_at", None)
            db.pop("python_version", None)

    _scrub_nondeterministic(cli_js)
    _scrub_nondeterministic(api_js)
    assert cli_js == api_js


def test_diagnostic_tier_rejected_for_cli_validation() -> None:
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
    b = build_decision_bundle(
        config=cfg,
        schema=schema,
        governance={"approved_for_optimization": True},
        simulation_contract={"objective": "delta_mu"},
        data_fingerprint=fp,
        economics_surface="curve_diagnostic",
        decision_safe=False,
        governance_passed=True,
        extension_report={"ridge_fit_summary": {"coef": [0.1]}},
        artifact_tier="diagnostic",
    )
    miss = validate_prod_decision_bundle(b, run_environment=RunEnvironment.PROD, decision_cli_surface=True)
    assert any("artifact_tier_must_be_decision" in m for m in miss)
