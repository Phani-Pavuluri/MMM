"""What-if spend scenarios on simple response surface."""

from __future__ import annotations

import numpy as np


def simulate_spend_scenario(
    base_spend: np.ndarray,
    delta_pct: np.ndarray,
    elasticities: np.ndarray,
) -> np.ndarray:
    """log outcome delta ~ elasticity * log spend delta."""
    return base_spend * (1 + delta_pct) ** elasticities
