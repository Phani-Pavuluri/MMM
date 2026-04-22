"""Coefficient-vs-experiment mismatch (research only — not experiment incremental lift)."""

from __future__ import annotations

import numpy as np

from mmm.calibration.matching import MatchedExperiment


def implied_channel_weights_from_coef(best_coef: np.ndarray, channel_names: list[str]) -> dict[str, float]:
    """Map leading media coefficients to channel keys — NOT aligned to experiment estimands."""
    out: dict[str, float] = {}
    for i, ch in enumerate(channel_names):
        if i < len(best_coef):
            out[ch] = float(best_coef[i])
    return out


def calibration_mismatch_loss(
    matched: list[MatchedExperiment],
    implied_lift_by_channel: dict[str, float],
) -> float:
    """
    Weighted squared mismatch vs experiment ``lift`` values.

    ``implied_lift_by_channel`` must be in the **same units and estimand** as ``obs.lift``.
    Mapping raw model coefficients into this dict is **not** experiment incremental lift.
    """
    if not matched:
        return 0.0
    losses: list[float] = []
    for m in matched:
        imp = implied_lift_by_channel.get(m.obs.channel)
        if imp is None:
            continue
        diff = imp - m.obs.lift
        se = float(m.obs.lift_se) if m.obs.lift_se is not None and m.obs.lift_se > 0 else None
        denom = (se * se + 1e-12) if se is not None else 1.0
        losses.append(m.weight * (diff**2 / denom))
    return float(np.mean(losses)) if losses else 0.0
