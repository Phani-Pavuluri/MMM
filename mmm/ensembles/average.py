"""E14: lightweight ensemble summaries from BO leaderboard (metrics only)."""

from __future__ import annotations

from typing import Any


def leaderboard_metric_average(leaderboard: list[dict[str, Any]], k: int = 3) -> dict[str, float]:
    """Average top-k trial scalar objectives (proxy for ensemble quality spread)."""
    if not leaderboard:
        return {}
    top = sorted(leaderboard, key=lambda d: d.get("total", 1e9))[:k]
    totals = [float(d.get("total", 0.0)) for d in top]
    return {"mean_topk_objective": float(sum(totals) / len(totals)), "k": float(len(totals))}
