"""Multi-channel curve bundles: I/O, optimizer, extension report shape."""

from __future__ import annotations

import json

import numpy as np
import pytest
from typer.testing import CliRunner

from mmm.config.schema import CVConfig, DataConfig, Framework, MMMConfig, ModelForm, RunEnvironment
from mmm.diagnostics.curve_optimizer import optimize_budget_from_curve_bundles
from mmm.economics.canonical import build_economics_contract
from mmm.evaluation.extension_runner import run_post_fit_extensions
from mmm.models.ridge_bo.trainer import RidgeBOMMMTrainer
from mmm.optimization.budget.curve_bundles_io import gather_curve_bundles_from_dict
from mmm.utils.synthetic import SyntheticGeoPanelSpec, generate_geo_panel


def test_gather_curve_bundles_single_and_list():
    b = {
        "channel": "c1",
        "spend_grid": [1.0, 2.0, 3.0],
        "response_on_modeling_scale": [0.1, 0.2, 0.25],
    }
    g1 = gather_curve_bundles_from_dict({"curve_bundle": b})
    assert g1 is not None
    assert g1[0] == ["c1"]
    g2 = gather_curve_bundles_from_dict({"curve_bundles": [b, {**b, "channel": "c2"}]})
    assert g2 is not None
    assert g2[0] == ["c1", "c2"]


def _curve_research_cfg() -> MMMConfig:
    return MMMConfig(
        framework=Framework.RIDGE_BO,
        model_form=ModelForm.SEMI_LOG,
        data=DataConfig(path=None, channel_columns=["a", "b"], target_column="y"),
        allow_unsafe_decision_apis=True,
        run_environment=RunEnvironment.RESEARCH,
    )


def test_multichannel_curve_optimizer_respects_budget():
    names = ["a", "b"]
    bundles = [
        {
            "channel": "a",
            "spend_grid": [1.0, 10.0, 20.0],
            "response_on_modeling_scale": [0.0, 1.0, 1.2],
        },
        {
            "channel": "b",
            "spend_grid": [1.0, 10.0, 20.0],
            "response_on_modeling_scale": [0.0, 0.5, 0.55],
        },
    ]
    n = 2
    res = optimize_budget_from_curve_bundles(
        names,
        bundles,
        config=_curve_research_cfg(),
        current_spend=np.array([50.0, 50.0]),
        total_budget=100.0,
        channel_min=np.zeros(n),
        channel_max=np.ones(n) * 200.0,
    )
    assert res["success"]
    assert abs(sum(res["optimal_spend"].values()) - 100.0) < 0.05


def test_multichannel_optimizer_rejects_objective_key_not_in_contract() -> None:
    names = ["a", "b"]
    bundles = [
        {
            "channel": "a",
            "spend_grid": [1.0, 10.0, 20.0],
            "response_on_modeling_scale": [0.0, 1.0, 1.2],
        },
        {
            "channel": "b",
            "spend_grid": [1.0, 10.0, 20.0],
            "response_on_modeling_scale": [0.0, 0.5, 0.55],
        },
    ]
    ec = build_economics_contract(
        MMMConfig(
            framework=Framework.RIDGE_BO,
            model_form=ModelForm.SEMI_LOG,
            data=DataConfig(path=None, channel_columns=["a", "b"], target_column="y"),
        )
    )
    n = 2
    with pytest.raises(ValueError, match="objective_value_key"):
        optimize_budget_from_curve_bundles(
            names,
            bundles,
            config=_curve_research_cfg(),
            current_spend=np.array([50.0, 50.0]),
            total_budget=100.0,
            channel_min=np.zeros(n),
            channel_max=np.ones(n) * 200.0,
            objective_value_key="not_allowed",
            economics_contract=ec,
        )


def test_extension_report_has_curve_bundles_per_channel():
    df, schema = generate_geo_panel(
        SyntheticGeoPanelSpec(n_geos=2, n_weeks=40, channels=("s", "t"), betas=(0.5, 0.5))
    )
    cfg = MMMConfig(
        framework=Framework.RIDGE_BO,
        model_form=ModelForm.SEMI_LOG,
        data={
            "path": None,
            "geo_column": schema.geo_column,
            "week_column": schema.week_column,
            "target_column": schema.target_column,
            "channel_columns": list(schema.channel_columns),
        },
        cv=CVConfig(mode="rolling", n_splits=2, min_train_weeks=12, horizon_weeks=3),
        ridge_bo={"n_trials": 2},
    )
    tr = RidgeBOMMMTrainer(cfg, schema)
    fit = tr.fit(df)
    yhat = tr.predict(df)
    rep = run_post_fit_extensions(panel=df, schema=schema, config=cfg, fit_out=fit, yhat=yhat, store=None)
    assert "curve_bundles" in rep
    assert len(rep["curve_bundles"]) == len(schema.channel_columns)
    assert rep["curve_bundles"][0]["channel"] == schema.channel_columns[0]
    assert "economics_contract" in rep["curve_bundles"][0]
    assert "economics_contract" in rep
    assert "decision_bundle" in rep
    assert rep["decision_bundle"]["bundle_version"] == "mmm_decision_bundle_v1"
    assert "economics_output_metadata" in rep["decision_bundle"]
    assert "resolved_config_snapshot" in rep["decision_bundle"]
    assert "economics_output_metadata" in rep
    db_em = rep["decision_bundle"]["economics_output_metadata"]
    assert db_em["economics_version"] and db_em["computation_mode"] and db_em["baseline_type"] != "unspecified"
    assert "transform_policy" in rep
    assert rep["transform_policy"]["policy_version"]


def test_prod_yaml_rejects_disabled_optimization_gates(tmp_path):
    from mmm.config.load import load_config

    yaml = tmp_path / "c.yaml"
    yaml.write_text(
        """
run_environment: prod
allow_unsafe_decision_apis: false
data:
  channel_columns: [c1]
  control_columns: []
extensions:
  optimization_gates:
    enabled: false
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="optimization_gates"):
        load_config(yaml)


def test_prod_yaml_rejects_allow_unsafe(tmp_path):
    from mmm.config.load import load_config

    yaml = tmp_path / "c.yaml"
    yaml.write_text(
        """
run_environment: prod
allow_unsafe_decision_apis: true
data:
  channel_columns: [c1]
  control_columns: []
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="allow_unsafe_decision_apis"):
        load_config(yaml)


def test_cli_curve_bundle_file_runs(tmp_path):
    from mmm.cli import main as cli_main

    ec = build_economics_contract(
        MMMConfig(
            framework=Framework.RIDGE_BO,
            model_form=ModelForm.SEMI_LOG,
            data=DataConfig(
                path=None,
                channel_columns=["c1"],
                control_columns=[],
                target_column="revenue",
            ),
        )
    )
    curve_path = tmp_path / "curves.json"
    curve_path.write_text(
        json.dumps(
            {
                "curve_bundles": [
                    {
                        "channel": "c1",
                        "spend_grid": [1.0, 5.0, 10.0],
                        "response_on_modeling_scale": [0.1, 0.4, 0.45],
                        "economics_contract": ec,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    yaml = tmp_path / "c.yaml"
    yaml.write_text(
        """
run_environment: research
allow_unsafe_decision_apis: true
data:
  channel_columns: [c1]
  control_columns: []
budget:
  total_budget: 100
extensions:
  optimization_gates:
    enabled: false
""",
        encoding="utf-8",
    )
    runner = CliRunner()
    result = runner.invoke(
        cli_main.app,
        [
            "optimize-budget",
            str(yaml),
            "--allow-unsafe-decision-apis",
            "--legacy-diagnostic-curve-optimizer",
            "--curve-bundle",
            str(curve_path),
        ],
    )
    assert result.exit_code == 0, result.stdout + (result.stderr or "")
    assert "optimal_spend" in (result.stdout or "")


def test_cli_simulate_prod_requires_economics_contract_on_bundles(tmp_path):
    from mmm.cli import main as cli_main

    scen = tmp_path / "scen.yaml"
    scen.write_text(
        """
baseline_spend:
  c1: 5.0
proposed_spend:
  c1: 8.0
""",
        encoding="utf-8",
    )
    curve_path = tmp_path / "curves.json"
    curve_path.write_text(
        json.dumps(
            {
                "curve_bundles": [
                    {
                        "channel": "c1",
                        "spend_grid": [1.0, 5.0, 10.0],
                        "response_on_modeling_scale": [0.1, 0.4, 0.45],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    yaml = tmp_path / "c.yaml"
    yaml.write_text(
        """
run_environment: prod
allow_unsafe_decision_apis: false
data:
  channel_columns: [c1]
  control_columns: []
""",
        encoding="utf-8",
    )
    runner = CliRunner()
    result = runner.invoke(
        cli_main.app,
        [
            "simulate-diagnostic-curves",
            str(yaml),
            "--scenario",
            str(scen),
            "--curve-bundle",
            str(curve_path),
        ],
    )
    assert result.exit_code == 2
    err = (result.stderr or "") + (result.stdout or "")
    assert "contract_version" in err.lower() or "economics_contract" in err.lower()

    ec = build_economics_contract(
        MMMConfig(
            framework=Framework.RIDGE_BO,
            model_form=ModelForm.SEMI_LOG,
            data=DataConfig(
                path=None,
                channel_columns=["c1"],
                control_columns=[],
                target_column="revenue",
            ),
        )
    )
    curve_path.write_text(
        json.dumps(
            {
                "curve_bundles": [
                    {
                        "channel": "c1",
                        "spend_grid": [1.0, 5.0, 10.0],
                        "response_on_modeling_scale": [0.1, 0.4, 0.45],
                        "economics_contract": ec,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    result_ok = runner.invoke(
        cli_main.app,
        [
            "simulate-diagnostic-curves",
            str(yaml),
            "--scenario",
            str(scen),
            "--curve-bundle",
            str(curve_path),
        ],
    )
    assert result_ok.exit_code == 0, result_ok.stdout + (result_ok.stderr or "")
    assert "curve_bundle_spend_shift" in (result_ok.stdout or "")


def test_cli_prod_optimize_budget_full_model_smoke(tmp_path):
    """Prod path: panel CSV + extension ridge_fit_summary → SLSQP on simulate() Δμ."""
    from mmm.cli import main as cli_main

    csv_path = tmp_path / "panel.csv"
    rows = []
    for g in ("G1", "G2"):
        for w in range(1, 16):
            rows.append(
                {
                    "geo": g,
                    "week": w,
                    "c1": float(10 + w * 0.1),
                    "c2": float(8 + w * 0.05),
                    "revenue": float(100 + w + (1 if g == "G1" else 0)),
                }
            )
    import pandas as pd

    pd.DataFrame(rows).to_csv(csv_path, index=False)

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
            }
        ),
        encoding="utf-8",
    )
    yaml = tmp_path / "c.yaml"
    yaml.write_text(
        f"""
run_environment: prod
allow_unsafe_decision_apis: false
data:
  path: {csv_path}
  geo_column: geo
  week_column: week
  target_column: revenue
  channel_columns: [c1, c2]
  control_columns: []
budget:
  total_budget: 100
extensions:
  optimization_gates:
    enabled: true
""",
        encoding="utf-8",
    )
    runner = CliRunner()
    result = runner.invoke(
        cli_main.app,
        [
            "optimize-budget",
            str(yaml),
            "--extension-report",
            str(ext),
        ],
    )
    assert result.exit_code == 0, result.stdout + (result.stderr or "")
    out = (result.stdout or "") + (result.stderr or "")
    assert "full_model_simulation_slsqp" in out
    assert "recommended_spend_plan" in out
