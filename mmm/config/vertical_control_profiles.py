"""Vertical control profiles for Ridge diagnostics and H6 synthetic worlds."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class VerticalControlProfile:
    """Recommended controls for a vertical."""

    vertical_id: str
    label: str
    required_controls: tuple[str, ...]
    optional_controls: tuple[str, ...]
    control_effects_log: dict[str, float] = field(default_factory=dict)
    notes: str = ""

    @property
    def all_controls(self) -> tuple[str, ...]:
        return self.required_controls + self.optional_controls


VERTICAL_RETAIL = VerticalControlProfile(
    vertical_id="retail",
    label="Retail / omnichannel",
    required_controls=("promo_flag", "holiday", "unemployment_index"),
    optional_controls=("competitor_price_index", "weather_index"),
    control_effects_log={
        "promo_flag": 0.12,
        "holiday": 0.08,
        "unemployment_index": -0.05,
        "competitor_price_index": -0.03,
        "weather_index": 0.02,
    },
)

VERTICAL_CPG = VerticalControlProfile(
    vertical_id="cpg",
    label="CPG / FMCG",
    required_controls=("promo_depth", "distribution_index", "cpi_food"),
    optional_controls=("competitive_sov",),
    control_effects_log={
        "promo_depth": 0.15,
        "distribution_index": 0.10,
        "cpi_food": 0.04,
        "competitive_sov": -0.06,
    },
)

VERTICAL_AUTO = VerticalControlProfile(
    vertical_id="auto",
    label="Automotive",
    required_controls=("incentive_index", "fuel_price", "unemployment_index"),
    optional_controls=("competitor_conquest_spend", "interest_rate_proxy"),
    control_effects_log={
        "incentive_index": 0.14,
        "fuel_price": -0.07,
        "unemployment_index": -0.09,
        "competitor_conquest_spend": -0.04,
        "interest_rate_proxy": -0.03,
    },
)

VERTICAL_PROFILES: dict[str, VerticalControlProfile] = {
    "retail": VERTICAL_RETAIL,
    "cpg": VERTICAL_CPG,
    "auto": VERTICAL_AUTO,
}


def get_vertical_profile(vertical_id: str) -> VerticalControlProfile:
    if vertical_id not in VERTICAL_PROFILES:
        raise KeyError(f"unknown vertical: {vertical_id!r}; known: {sorted(VERTICAL_PROFILES)}")
    return VERTICAL_PROFILES[vertical_id]


def control_truth_for_profile(
    profile: VerticalControlProfile,
    *,
    active_controls: tuple[str, ...] | None = None,
    effect_overrides: dict[str, float] | None = None,
) -> dict[str, Any]:
    active = active_controls if active_controls is not None else profile.all_controls
    effects = {c: profile.control_effects_log.get(c, 0.0) for c in active}
    if effect_overrides:
        effects.update(effect_overrides)
    return {
        "vertical_id": profile.vertical_id,
        "required_controls": list(profile.required_controls),
        "optional_controls": list(profile.optional_controls),
        "active_controls": list(active),
        "control_effects_log": effects,
        "omitted_controls": [c for c in profile.all_controls if c not in active],
    }
