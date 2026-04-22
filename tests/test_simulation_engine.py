from mmm.config.schema import DataConfig, Framework, MMMConfig, ModelForm
from mmm.economics.canonical import build_economics_contract
from mmm.simulation.engine import SpendScenario, run_curve_bundle_scenario, run_stepped_scenario

_EC = build_economics_contract(
    MMMConfig(
        framework=Framework.RIDGE_BO,
        model_form=ModelForm.SEMI_LOG,
        data=DataConfig(path=None, channel_columns=["a", "b"], target_column="rev"),
    )
)


def test_run_curve_bundle_scenario_delta():
    curves = [
        {
            "channel": "a",
            "spend_grid": [1.0, 10.0, 20.0],
            "response_on_modeling_scale": [0.0, 1.0, 1.2],
            "economics_contract": _EC,
        },
        {
            "channel": "b",
            "spend_grid": [1.0, 10.0, 20.0],
            "response_on_modeling_scale": [0.0, 0.5, 0.6],
            "economics_contract": _EC,
        },
    ]
    scen = SpendScenario(
        baseline_spend={"a": 5.0, "b": 5.0},
        proposed_spend={"a": 10.0, "b": 5.0},
        y_level_scale=100.0,
    )
    out = run_curve_bundle_scenario(curves, scen)
    assert out["delta_response_modeling_sum"] != 0.0
    assert "incremental_kpi_level_sum_small_delta_proxy" in out
    assert out["economics_contract"]["contract_version"]


def test_run_stepped_scenario_trajectory():
    ec_a = build_economics_contract(
        MMMConfig(
            framework=Framework.RIDGE_BO,
            model_form=ModelForm.SEMI_LOG,
            data=DataConfig(path=None, channel_columns=["a"], target_column="rev"),
        )
    )
    curves = [
        {
            "channel": "a",
            "spend_grid": [1.0, 50.0],
            "response_on_modeling_scale": [0.0, 2.0],
            "economics_contract": ec_a,
        },
    ]
    steps = [{"week": 0, "spend": {"a": 5.0}}, {"week": 1, "spend": {"a": 15.0}}]
    out = run_stepped_scenario(curves, steps=steps, y_level_scale=50.0)
    assert len(out["trajectory"]) == 2
    assert out["trajectory"][1]["incremental_kpi_level_small_delta_proxy_vs_prev"] is not None
    assert out["economics_contract"]["contract_version"]


def test_spend_scenario_yaml_roundtrip(tmp_path):
    p = tmp_path / "s.yaml"
    p.write_text(
        """
baseline_spend:
  a: 10.0
proposed_spend:
  a: 12.0
y_level_scale: 100.0
""",
        encoding="utf-8",
    )
    s = SpendScenario.from_yaml(p)
    assert s.baseline_spend["a"] == 10.0
    assert s.resolved_proposed()["a"] == 12.0
