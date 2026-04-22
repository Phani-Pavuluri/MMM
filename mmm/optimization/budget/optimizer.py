"""Constrained budget allocation on smooth response approximations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


class BudgetOptimizerBase(ABC):
    @abstractmethod
    def optimize(self, current_spend: np.ndarray, total_budget: float) -> dict:
        raise NotImplementedError


@dataclass
class BudgetOptimizer(BudgetOptimizerBase):
    """Maximize sum_i mroi_i * sqrt(spend_i) proxy (concave) under linear budget — illustrative production pattern."""

    channel_names: list[str]
    marginal_roi: np.ndarray  # positive weights
    channel_min: np.ndarray
    channel_max: np.ndarray

    def optimize(self, current_spend: np.ndarray, total_budget: float) -> dict:
        x0 = np.clip(current_spend, self.channel_min, self.channel_max)

        def neg_value(x: np.ndarray) -> float:
            return -float(np.sum(self.marginal_roi * np.sqrt(np.maximum(x, 1e-9))))

        cons = [{"type": "eq", "fun": lambda x: float(np.sum(x)) - float(total_budget)}]
        bounds = [(lo, hi) for lo, hi in zip(self.channel_min, self.channel_max, strict=True)]
        res = minimize(neg_value, x0=x0, method="SLSQP", bounds=bounds, constraints=cons)
        opt_x = res.x
        return {
            "success": res.success,
            "message": res.message,
            "optimal_spend": {c: float(x) for c, x in zip(self.channel_names, opt_x, strict=True)},
            "expected_incremental_value": float(-res.fun),
            "marginal_roi": {
                c: float(m) for c, m in zip(self.channel_names, self.marginal_roi, strict=True)
            },
        }
