"""Match experiments to model scopes with graceful degradation."""

from __future__ import annotations

from dataclasses import dataclass

from mmm.calibration.quality import experiment_quality_score
from mmm.calibration.schema import ExperimentObservation


@dataclass
class MatchedExperiment:
    obs: ExperimentObservation
    weight: float
    quality_score: float = 1.0


def match_experiments(
    experiments: list[ExperimentObservation],
    *,
    available_geos: set[str] | None,
    available_channels: set[str],
    match_levels: list[str],
    apply_quality: bool = True,
) -> list[MatchedExperiment]:
    matched: list[MatchedExperiment] = []
    for ex in experiments:
        if ex.channel not in available_channels:
            continue
        if "geo" in match_levels and ex.geo_id and available_geos and ex.geo_id not in available_geos:
            continue
        se = ex.lift_se if ex.lift_se and ex.lift_se > 0 else None
        inv_se = 1.0 / (se if se else 1.0)
        q = experiment_quality_score(ex) if apply_quality else 1.0
        weight = float(inv_se * q)
        matched.append(MatchedExperiment(obs=ex, weight=weight, quality_score=q))
    return matched
