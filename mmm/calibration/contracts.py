"""Path-level calibration units for replay (roadmap §2)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import pandas as pd

EffectHorizon = Literal["short_run", "long_run"]


@dataclass
class CalibrationUnit:
    """
    One matched experiment with explicit treatment paths for replay.

    **Estimand (Sprint 3):** ``observed_lift`` is the experiment estimand on the KPI scale;
    model replay compares ``predict(observed_frame)`` vs ``predict(counterfactual_frame)`` and
    maps the difference into the same units before loss — here we use mean(pred_obs - pred_cf)
    as ``implied_delta`` on the **prediction (level revenue)** scale unless you transform in JSON.

    ``observed_spend_frame`` / ``counterfactual_spend_frame`` are optional until
    loaders populate them; matching on geo/channel/time alone remains supported
    via :class:`mmm.calibration.schema.ExperimentObservation`.
    """

    unit_id: str
    treated_channel_names: list[str] = field(default_factory=list)
    observed_spend_frame: pd.DataFrame | None = None
    counterfactual_spend_frame: pd.DataFrame | None = None
    post_window_weeks: tuple[int, int] | None = None
    ramp_window_weeks: tuple[int, int] | None = None
    washout_window_weeks: tuple[int, int] | None = None
    effect_horizon: EffectHorizon = "short_run"
    treatment_intensity_note: str = ""
    #: Experiment estimand label (e.g. ``ATT_geo_time``); required for prod replay.
    estimand: str = ""
    #: Units of ``observed_lift`` / ``lift_se`` (e.g. ``mean_kpi_level_delta``).
    lift_scale: str = ""
    observed_lift: float | None = None
    lift_se: float | None = None
    target_kpi: str = ""
    geo_ids: list[str] = field(default_factory=list)
    #: Serialized :class:`mmm.calibration.replay_estimand.ReplayEstimandSpec` — **required** for replay loss.
    replay_estimand: dict[str, Any] | None = None
    #: Immutable experiment identifier from the experimentation platform (recommended in prod).
    experiment_id: str = ""
    #: Semantic version of the payload contract (ingestion / ETL).
    payload_version: str = ""
    #: Optional SHA-256 over a canonical JSON serialization of the signed experiment payload.
    payload_sha256: str = ""
    #: Operator workflow: ``pending`` | ``approved`` | ``rejected`` | ``unknown``.
    calibration_readiness: str = ""
