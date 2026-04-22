import numpy as np

from mmm.optimization.budget.optimizer import BudgetOptimizer


def test_budget_optimizer_runs():
    opt = BudgetOptimizer(
        channel_names=["a", "b"],
        marginal_roi=np.array([1.0, 2.0]),
        channel_min=np.array([0.0, 0.0]),
        channel_max=np.array([100.0, 100.0]),
    )
    res = opt.optimize(np.array([50.0, 50.0]), total_budget=100.0)
    assert res["success"]
    assert abs(sum(res["optimal_spend"].values()) - 100.0) < 1e-3
