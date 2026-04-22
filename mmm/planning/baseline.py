"""Baseline spend policy for planning, simulation, and optimization."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np
import pandas as pd

from mmm.data.panel_order import sort_panel_for_modeling
from mmm.data.schema import PanelSchema


class BaselineType(StrEnum):
    """Supported baseline spend constructions."""

    BAU = "bau"
    HISTORICAL_AVERAGE = "historical_average"
    ZERO_SPEND = "zero_spend"
    LOCKED_PLAN = "locked_plan"
    EXPERIMENT_SPECIFIC = "experiment_specific"


@dataclass
class BaselinePlan:
    """
    Explicit baseline for Δμ = μ(plan) − μ(baseline) on the **Gaussian mean** (modeling) scale.

    ``spend_by_channel`` holds **global** channel levels (same across geos) when
    ``spend_by_geo_channel`` is absent.

    When ``spend_by_geo_channel`` is set, counterfactual media uses **per-geo** levels; ``spend_by_channel``
    should still hold cross-geo means (or another summary) for reporting and backward compatibility.
    """

    baseline_type: BaselineType
    spend_by_channel: dict[str, float]
    baseline_definition: str
    baseline_plan_source: str
    suitable_for_decisioning: bool
    disclosure: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    #: Optional per-geo channel spends (geo id string → channel → level).
    spend_by_geo_channel: dict[str, dict[str, float]] | None = None


def bau_baseline_from_panel(panel: pd.DataFrame, schema: PanelSchema) -> BaselinePlan:
    """
    Primary default: **current / BAU** — mean channel spend in the last calendar week across geos.

    Suitable for decisioning when used as the reference for budget scenarios.
    """
    df = sort_panel_for_modeling(panel, schema)
    wcol = schema.week_column
    wmax = df[wcol].max()
    tail = df[df[wcol] == wmax]
    spend = {ch: float(tail[ch].mean()) for ch in schema.channel_columns}
    return BaselinePlan(
        baseline_type=BaselineType.BAU,
        spend_by_channel=spend,
        baseline_definition="Mean channel spend in the last observed calendar week, averaged across geos.",
        baseline_plan_source="panel:last_week_mean_across_geos",
        suitable_for_decisioning=True,
        disclosure="",
        metadata={"last_week_value": str(wmax)},
        spend_by_geo_channel=None,
    )


def bau_baseline_per_geo_from_panel(panel: pd.DataFrame, schema: PanelSchema) -> BaselinePlan:
    """
    BAU with **per-geo** last-week channel means (same last week as :func:`bau_baseline_from_panel`).

    ``spend_by_channel`` stores the cross-geo mean of those per-geo values for compatibility.
    """
    df = sort_panel_for_modeling(panel, schema)
    wcol = schema.week_column
    gcol = schema.geo_column
    wmax = df[wcol].max()
    tail = df[df[wcol] == wmax]
    by_geo: dict[str, dict[str, float]] = {}
    for g, sub in tail.groupby(gcol, sort=False):
        gid = str(g)
        by_geo[gid] = {ch: float(sub[ch].mean()) for ch in schema.channel_columns}
    spend_mean = {
        ch: float(np.mean([by_geo[g][ch] for g in by_geo])) if by_geo else 0.0 for ch in schema.channel_columns
    }
    return BaselinePlan(
        baseline_type=BaselineType.BAU,
        spend_by_channel=spend_mean,
        baseline_definition="Per-geo mean channel spend in the last observed calendar week.",
        baseline_plan_source="panel:last_week_mean_per_geo",
        suitable_for_decisioning=True,
        disclosure="",
        metadata={"last_week_value": str(wmax), "n_geos": len(by_geo)},
        spend_by_geo_channel=by_geo,
    )


def historical_average_baseline_from_panel(panel: pd.DataFrame, schema: PanelSchema) -> BaselinePlan:
    """Historical average spend — benchmarking / normalized reporting only (not default for optimization)."""
    df = sort_panel_for_modeling(panel, schema)
    spend = {ch: float(df[ch].mean()) for ch in schema.channel_columns}
    return BaselinePlan(
        baseline_type=BaselineType.HISTORICAL_AVERAGE,
        spend_by_channel=spend,
        baseline_definition="Time mean of observed channel spend over the full panel window.",
        baseline_plan_source="panel:full_window_mean",
        suitable_for_decisioning=False,
        disclosure="historical_average baseline is for benchmarking only; do not treat as current BAU.",
        spend_by_geo_channel=None,
    )


def zero_spend_baseline(schema: PanelSchema) -> BaselinePlan:
    """Zero media spend — analytical contribution studies only."""
    spend = {ch: 0.0 for ch in schema.channel_columns}
    return BaselinePlan(
        baseline_type=BaselineType.ZERO_SPEND,
        spend_by_channel=spend,
        baseline_definition="All modeled channel spends set to zero (lower bound).",
        baseline_plan_source="policy:zero_spend",
        suitable_for_decisioning=False,
        disclosure="zero_spend is for contribution-style analysis only, not BAU planning.",
        spend_by_geo_channel=None,
    )


def locked_plan_baseline(spend_by_channel: dict[str, float], *, source: str, notes: str = "") -> BaselinePlan:
    """User-supplied fixed reference plan."""
    return BaselinePlan(
        baseline_type=BaselineType.LOCKED_PLAN,
        spend_by_channel=dict(spend_by_channel),
        baseline_definition=notes or "Locked / reference spend vector supplied by operator.",
        baseline_plan_source=source,
        suitable_for_decisioning=False,
        disclosure="locked_plan baseline: verify suitability before decisioning; default prod optimizer uses BAU.",
        metadata={},
        spend_by_geo_channel=None,
    )


def locked_geo_plan_baseline(
    spend_by_geo_channel: dict[str, dict[str, float]],
    *,
    source: str,
    notes: str = "",
) -> BaselinePlan:
    """Locked per-geo reference plan; ``spend_by_channel`` is set to cross-geo means for summaries."""
    if not spend_by_geo_channel:
        raise ValueError("spend_by_geo_channel must be non-empty")
    geos = list(spend_by_geo_channel.keys())
    all_ch: set[str] = set()
    for row in spend_by_geo_channel.values():
        all_ch.update(str(k) for k in row.keys())
    mean_ch = {
        ch: float(np.mean([float(spend_by_geo_channel[g].get(ch, 0.0)) for g in geos])) for ch in sorted(all_ch)
    }
    return BaselinePlan(
        baseline_type=BaselineType.LOCKED_PLAN,
        spend_by_channel=mean_ch,
        baseline_definition=notes or "Locked per-geo reference spend supplied by operator.",
        baseline_plan_source=source,
        suitable_for_decisioning=False,
        disclosure="locked_plan per-geo baseline: verify suitability before decisioning.",
        metadata={},
        spend_by_geo_channel={str(k): dict(v) for k, v in spend_by_geo_channel.items()},
    )


def experiment_baseline_from_spend(
    spend_by_channel: dict[str, float],
    *,
    source: str,
    definition: str,
) -> BaselinePlan:
    """Replay / calibration counterfactual reference (not default for planning unless overridden)."""
    return BaselinePlan(
        baseline_type=BaselineType.EXPERIMENT_SPECIFIC,
        spend_by_channel=dict(spend_by_channel),
        baseline_definition=definition,
        baseline_plan_source=source,
        suitable_for_decisioning=False,
        disclosure="experiment_specific baseline: intended for calibration/replay unless explicitly overridden.",
        spend_by_geo_channel=None,
    )


def disclosure_for_non_bau_optimization(plan: BaselinePlan) -> str:
    """Prominent disclosure when optimization uses a non-BAU baseline."""
    if plan.baseline_type == BaselineType.BAU:
        return ""
    return (
        f"OPTIMIZATION BASELINE IS NOT BAU ({plan.baseline_type.value}): {plan.baseline_definition}. "
        "Interpret budget results relative to this explicit reference, not current BAU."
    )


def total_spend_vector(schema: PanelSchema, spend: dict[str, float]) -> float:
    return float(sum(float(spend.get(ch, 0.0)) for ch in schema.channel_columns))


def spend_delta_l1(schema: PanelSchema, base: dict[str, float], plan: dict[str, float]) -> float:
    return float(sum(abs(float(plan.get(ch, 0.0)) - float(base.get(ch, 0.0))) for ch in schema.channel_columns))


def total_spend_geo_plan(schema: PanelSchema, spend_by_geo: dict[str, dict[str, float]]) -> float:
    """Sum of all channel spends across all geos in the plan."""
    t = 0.0
    for g, row in spend_by_geo.items():
        _ = g
        for ch in schema.channel_columns:
            t += float(row.get(ch, 0.0))
    return float(t)


def spend_delta_l1_geo(
    schema: PanelSchema,
    base: dict[str, dict[str, float]],
    plan: dict[str, dict[str, float]],
    geos: list[str],
) -> float:
    """Sum of L1 deltas over channels for each geo."""
    s = 0.0
    for g in geos:
        br, pr = base.get(g, {}), plan.get(g, {})
        for ch in schema.channel_columns:
            s += abs(float(pr.get(ch, 0.0)) - float(br.get(ch, 0.0)))
    return float(s)


def channel_means_from_geo_plan(
    spend_by_geo: dict[str, dict[str, float]],
    schema: PanelSchema,
    geos: list[str],
) -> dict[str, float]:
    """Cross-geo arithmetic mean spend per channel (for scalar summaries)."""
    if not geos:
        return {ch: 0.0 for ch in schema.channel_columns}
    out: dict[str, float] = {}
    for ch in schema.channel_columns:
        out[ch] = float(np.mean([float(spend_by_geo.get(g, {}).get(ch, 0.0)) for g in geos]))
    return out
