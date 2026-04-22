"""Experiment / lift observation schema."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ExperimentObservation(BaseModel):
    """GeoX / CLS-style row."""

    experiment_id: str
    geo_id: str | None = None
    channel: str
    start_week: str | None = None
    end_week: str | None = None
    lift: float
    lift_se: float | None = None
    device: str | None = None
    product: str | None = None
    metadata: dict[str, str] = Field(default_factory=dict)
