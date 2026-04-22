import numpy as np

from mmm.decomposition.curves import build_curve_for_channel
from mmm.optimization.budget.curve_optimizer import optimize_spend_from_curve_bundle


def test_curve_horizon_changes_saturation_level():
    grid = np.array([10.0, 15.0, 20.0])
    c_short = build_curve_for_channel(
        grid, decay=0.5, hill_half=1.0, hill_slope=2.0, beta=1.0, model_form="semi_log", horizon_weeks=8
    )
    c_long = build_curve_for_channel(
        grid, decay=0.5, hill_half=1.0, hill_slope=2.0, beta=1.0, model_form="semi_log", horizon_weeks=52
    )
    assert not np.allclose(c_short.response, c_long.response)


def test_curve_optimizer_runs():
    bundle = {
        "spend_grid": [1.0, 5.0, 10.0],
        "response_on_modeling_scale": [0.1, 0.5, 0.9],
        "marginal_roi_modeling_scale": [0.1, 0.1, 0.1],
    }
    out = optimize_spend_from_curve_bundle(
        bundle,
        current_spend=5.0,
        total_budget=100.0,
        spend_min=1.0,
        spend_max=10.0,
    )
    assert "optimal_spend" in out
