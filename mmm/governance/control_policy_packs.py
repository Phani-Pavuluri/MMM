"""Domain control guidance packs (recommendations only — never applied to training)."""

from __future__ import annotations

from typing import Any

PolicyPackId = str

# Each pack lists illustrative control themes — not training defaults.
POLICY_PACKS: dict[PolicyPackId, dict[str, Any]] = {
    "generic": {
        "label": "Generic",
        "themes": ["seasonality", "macro_indicators", "promotions", "pricing", "events"],
        "recommended_controls": [
            "holiday_indicator",
            "promo_intensity",
            "pricing_index",
            "macro_sentiment_index",
            "major_event_indicator",
        ],
    },
    "b2b": {
        "label": "B2B",
        "themes": ["equity_markets", "product_launches", "fiscal_calendar", "sales_cycles"],
        "recommended_controls": [
            "sp500_index",
            "product_launch",
            "fiscal_quarter_indicator",
            "enterprise_pipeline_index",
            "unemployment_rate",
        ],
    },
    "retail": {
        "label": "Retail",
        "themes": ["holidays", "promotions", "pricing", "inventory"],
        "recommended_controls": [
            "holiday_indicator",
            "discount_rate",
            "competitor_promo",
            "inventory_level",
            "weather_index",
        ],
    },
    "subscription": {
        "label": "Subscription / SaaS",
        "themes": ["churn_drivers", "renewals", "seasonality"],
        "recommended_controls": [
            "churn_risk_index",
            "renewal_cycle_indicator",
            "trial_volume_index",
            "pricing_index",
            "holiday_indicator",
        ],
    },
    "geo_experimentation": {
        "label": "Geo experimentation",
        "themes": ["weather", "local_events", "regional_macro"],
        "recommended_controls": [
            "weather_index",
            "local_event_indicator",
            "regional_unemployment",
            "regional_fuel_price",
            "holiday_indicator",
        ],
    },
}


def list_policy_pack_ids() -> list[str]:
    return list(POLICY_PACKS.keys())


def policy_pack_recommendations(
    pack_id: PolicyPackId,
    *,
    configured_controls: list[str],
) -> dict[str, Any]:
    """Return pack metadata and controls not yet configured."""
    pack = POLICY_PACKS.get(pack_id)
    if pack is None:
        raise KeyError(f"Unknown policy pack: {pack_id!r}")
    rec = list(pack["recommended_controls"])
    missing = [c for c in rec if c not in configured_controls]
    return {
        "pack_id": pack_id,
        "label": pack["label"],
        "themes": list(pack["themes"]),
        "recommended_controls": rec,
        "missing_from_config": missing,
        "guidance_only": True,
    }


def all_pack_summaries(*, configured_controls: list[str]) -> list[dict[str, Any]]:
    return [policy_pack_recommendations(pid, configured_controls=configured_controls) for pid in POLICY_PACKS]
