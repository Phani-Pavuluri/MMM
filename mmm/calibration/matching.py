"""Match experiments to model scopes — all ``match_levels`` entries are enforced when present on observations."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from mmm.calibration.quality import experiment_quality_score
from mmm.calibration.schema import ExperimentObservation


@dataclass
class MatchedExperiment:
    obs: ExperimentObservation
    weight: float
    quality_score: float = 1.0


def _to_ts(v: str | None) -> pd.Timestamp | None:
    if v is None or str(v).strip() == "":
        return None
    t = pd.to_datetime(v, errors="coerce")
    if pd.isna(t):
        return None
    return t  # type: ignore[return-value]


def _time_window_ok(
    ex: ExperimentObservation,
    *,
    panel_week_min: pd.Timestamp | None,
    panel_week_max: pd.Timestamp | None,
) -> bool:
    if ex.start_week is None and ex.end_week is None:
        return True
    if panel_week_min is None or panel_week_max is None:
        return False
    obs_start = _to_ts(ex.start_week) if ex.start_week else panel_week_min
    obs_end = _to_ts(ex.end_week) if ex.end_week else panel_week_max
    if obs_start is None or obs_end is None:
        return False
    return not (obs_end < panel_week_min or obs_start > panel_week_max)


def match_experiments(
    experiments: list[ExperimentObservation],
    *,
    available_geos: set[str] | None,
    available_channels: set[str],
    match_levels: list[str],
    apply_quality: bool = True,
    panel_week_min: pd.Timestamp | None = None,
    panel_week_max: pd.Timestamp | None = None,
    allowed_devices: set[str] | None = None,
    allowed_products: set[str] | None = None,
) -> list[MatchedExperiment]:
    """
    Filter experiments to those compatible with the training scope.

    - ``geo``: if ``geo`` in ``match_levels`` and ``ex.geo_id`` is set, it must be in ``available_geos``.
    - ``time_window``: if in ``match_levels`` and the observation has ``start_week`` / ``end_week``,
      overlap with ``[panel_week_min, panel_week_max]`` is required (strict; missing panel bounds ⇒ no match).
    - ``device`` / ``product``: if in ``match_levels`` and the observation sets the field, it must be in the
      corresponding allowed set (strict; missing allowed set when the field is set ⇒ no match).
    """
    matched: list[MatchedExperiment] = []
    for ex in experiments:
        if ex.channel not in available_channels:
            continue
        if "geo" in match_levels and ex.geo_id and available_geos and ex.geo_id not in available_geos:
            continue
        if "time_window" in match_levels:
            if not _time_window_ok(ex, panel_week_min=panel_week_min, panel_week_max=panel_week_max):
                continue
        if "device" in match_levels and ex.device:
            if allowed_devices is None:
                continue
            if ex.device not in allowed_devices:
                continue
        if "product" in match_levels and ex.product:
            if allowed_products is None:
                continue
            if ex.product not in allowed_products:
                continue
        se = ex.lift_se if ex.lift_se and ex.lift_se > 0 else None
        inv_se = 1.0 / (se if se else 1.0)
        q = experiment_quality_score(ex) if apply_quality else 1.0
        weight = float(inv_se * q)
        matched.append(MatchedExperiment(obs=ex, weight=weight, quality_score=q))
    return matched
