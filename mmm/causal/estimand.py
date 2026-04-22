"""E2: estimand definitions and experiment ↔ MMM alignment checks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel

from mmm.calibration.schema import ExperimentObservation


class EstimandConfig(BaseModel):
    """Canonical estimand; should match experiment readouts."""

    target_kpi: str = "revenue"
    horizon: Literal["short_term", "long_term"] = "short_term"
    adjustment: Literal["immediate", "lag_adjusted", "long_run"] = "lag_adjusted"
    aggregation: str = "geo_week"
    mismatch_policy: Literal["warn", "error"] = "warn"

    model_config = {"extra": "forbid"}


@dataclass
class EstimandValidationResult:
    ok: bool
    messages: list[str] = field(default_factory=list)


class EstimandValidator:
    """Validate experiment metadata vs MMM estimand (KPI, window, geo)."""

    def __init__(self, estimand: EstimandConfig, calibration_target_kpi: str | None) -> None:
        self.estimand = estimand
        self.calibration_target_kpi = calibration_target_kpi

    def validate_experiments(self, experiments: list[ExperimentObservation]) -> EstimandValidationResult:
        msgs: list[str] = []
        ok = True
        if self.calibration_target_kpi and self.calibration_target_kpi != self.estimand.target_kpi:
            m = (
                f"experiment target_kpi ({self.calibration_target_kpi}) "
                f"!= estimand.target_kpi ({self.estimand.target_kpi})"
            )
            msgs.append(m)
            if self.estimand.mismatch_policy == "error":
                ok = False
        for ex in experiments:
            if ex.start_week and ex.end_week and ex.start_week > ex.end_week:
                msgs.append(f"experiment {ex.experiment_id}: inverted time window")
                ok = False
        if self.estimand.horizon == "long_term" and self.estimand.adjustment == "immediate":
            msgs.append("long_term horizon with immediate adjustment is ambiguous")
        return EstimandValidationResult(ok=ok, messages=msgs)
