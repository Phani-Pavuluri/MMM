"""Calibration engine — loads experiments and produces matched sets."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from mmm.data.schema import PanelSchema

from mmm.calibration.matching import MatchedExperiment, match_experiments
from mmm.calibration.schema import ExperimentObservation


class CalibrationEngineBase(ABC):
    @abstractmethod
    def load(self) -> list[ExperimentObservation]:
        raise NotImplementedError


class CalibrationEngine(CalibrationEngineBase):
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> list[ExperimentObservation]:
        if self.path.suffix.lower() == ".csv":
            df = pd.read_csv(self.path)
            rows = []
            for _, r in df.iterrows():
                rows.append(
                    ExperimentObservation(
                        experiment_id=str(r.get("experiment_id", "exp")),
                        geo_id=r.get("geo_id"),
                        channel=str(r["channel"]),
                        start_week=str(r["start_week"]) if "start_week" in r else None,
                        end_week=str(r["end_week"]) if "end_week" in r else None,
                        lift=float(r["lift"]),
                        lift_se=float(r["lift_se"]) if "lift_se" in r and pd.notna(r["lift_se"]) else None,
                        device=r.get("device"),
                        product=r.get("product"),
                    )
                )
            return rows
        data = json.loads(self.path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [ExperimentObservation.model_validate(x) for x in data]
        raise ValueError("Unsupported experiments file")

    def match(
        self,
        experiments: list[ExperimentObservation],
        *,
        geos: set[str] | None,
        channels: set[str],
        levels: list[str],
        apply_quality: bool = True,
        panel: pd.DataFrame | None = None,
        schema: "PanelSchema | None" = None,
        allowed_devices: set[str] | None = None,
        allowed_products: set[str] | None = None,
    ) -> list[MatchedExperiment]:
        from mmm.data.schema import PanelSchema as _PS

        p_min: pd.Timestamp | None = None
        p_max: pd.Timestamp | None = None
        if panel is not None and schema is not None and "time_window" in levels:
            if not isinstance(schema, _PS):
                raise TypeError("schema must be a PanelSchema when panel is provided for time_window matching")
            wt = pd.to_datetime(panel[schema.week_column], errors="coerce")
            if wt.isna().all():
                p_min, p_max = None, None
            else:
                p_min = pd.Timestamp(wt.min())
                p_max = pd.Timestamp(wt.max())
        return match_experiments(
            experiments,
            available_geos=geos,
            available_channels=channels,
            match_levels=levels,
            apply_quality=apply_quality,
            panel_week_min=p_min,
            panel_week_max=p_max,
            allowed_devices=allowed_devices,
            allowed_products=allowed_products,
        )
