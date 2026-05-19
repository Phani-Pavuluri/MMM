"""Experiment prioritization requests — diagnostic contract only (no test design)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

PriorityTier = Literal["high", "medium", "low"]
SuggestedTestType = Literal[
    "geo_holdout",
    "heavy_up",
    "spend_shock",
    "incrementality_test",
]


class ExperimentRequest(BaseModel):
    """
    Prioritized experiment **request** for an external geo-experimentation system.

    Does not specify treatment assignment, geo selection, MDE, duration, or estimators.
    """

    request_id: str
    channel_or_group: str
    reason: str
    uncertainty_source: str
    priority_score: float = Field(ge=0.0, le=1.0)
    priority_tier: PriorityTier
    business_importance: float = Field(ge=0.0, le=1.0)
    required_estimand: str = "incremental_lift_on_target_kpi"
    required_kpi: str = "target_column_from_schema"
    suggested_test_type: SuggestedTestType = "geo_holdout"
    preferred_geo_level: str = "geo_panel_unit"
    notes: list[str] = Field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return self.model_dump(mode="json")
