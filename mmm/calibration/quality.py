"""E8: experiment quality scores for weighting calibration / likelihood."""

from __future__ import annotations

import contextlib

from mmm.calibration.schema import ExperimentObservation


def experiment_quality_score(obs: ExperimentObservation) -> float:
    """Return multiplier in (0, 1]; higher = more trustworthy."""
    se = obs.lift_se if obs.lift_se and obs.lift_se > 0 else None
    prec = 1.0 / (1.0 + float(se)) if se else 0.5
    meta_bonus = 0.0
    if obs.metadata.get("design_strength"):
        with contextlib.suppress(TypeError, ValueError):
            meta_bonus += 0.1 * float(obs.metadata["design_strength"])
    if obs.metadata.get("spillover_risk", "").lower() in {"high", "h"}:
        prec *= 0.6
    score = min(1.0, prec + meta_bonus)
    return float(max(0.05, score))
