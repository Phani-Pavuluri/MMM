from mmm.config.schema import DataConfig, Framework, MMMConfig, ModelForm
from mmm.economics.canonical import build_economics_contract
from mmm.simulation.engine import SpendPlan, simulate


def test_simulate_spend_plan_point():
    ec = build_economics_contract(
        MMMConfig(
            framework=Framework.RIDGE_BO,
            model_form=ModelForm.SEMI_LOG,
            data=DataConfig(path=None, channel_columns=["a"], target_column="rev"),
        )
    )
    curves = [
        {
            "channel": "a",
            "spend_grid": [1.0, 10.0],
            "response_on_modeling_scale": [0.0, 1.0],
            "economics_contract": ec,
        },
    ]
    plan = SpendPlan(
        horizon_weeks=2,
        aggregate_steps=[
            {"week": 0, "spend": {"a": 2.0}},
            {"week": 1, "spend": {"a": 5.0}},
        ],
        y_level_scale=50.0,
    )
    out = simulate(plan, curves, uncertainty_mode="point")
    assert out["kind"] == "simulate_point"
    assert "aggregate" in out
    assert out.get("economics_contract", {}).get("contract_version")
    assert out["aggregate"].get("economics_contract", {}).get("contract_version")
