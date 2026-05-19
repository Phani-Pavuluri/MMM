"""
Control-variable CSV scaffolds for onboarding.

Illustrative values only — never training defaults or priors.
"""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

POLICY_VERSION = "control_template_v1"


class ControlDomain(str, Enum):
    GENERIC = "generic"
    B2B = "b2b"
    ECOMMERCE = "ecommerce"
    RETAIL = "retail"
    SAAS = "saas"
    TRAVEL = "travel"


class ControlFrequency(str, Enum):
    WEEKLY = "weekly"
    MONTHLY = "monthly"


_PERIOD_COLUMN: dict[ControlFrequency, str] = {
    ControlFrequency.WEEKLY: "week",
    ControlFrequency.MONTHLY: "month",
}

# Columns after the period key — illustrative names only.
_DOMAIN_CONTROL_COLUMNS: dict[ControlDomain, list[str]] = {
    ControlDomain.GENERIC: [
        "holiday_indicator",
        "promo_intensity",
        "pricing_index",
        "product_launch",
        "site_outage",
    ],
    ControlDomain.B2B: [
        "holiday_indicator",
        "promo_intensity",
        "pricing_index",
        "sp500_index",
        "unemployment_rate",
        "product_launch",
    ],
    ControlDomain.ECOMMERCE: [
        "holiday_indicator",
        "promo_intensity",
        "discount_rate",
        "site_traffic_index",
        "cart_abandonment_rate",
        "competitor_price_index",
    ],
    ControlDomain.RETAIL: [
        "holiday_indicator",
        "discount_rate",
        "inventory_level",
        "weather_index",
        "competitor_promo",
    ],
    ControlDomain.SAAS: [
        "holiday_indicator",
        "promo_intensity",
        "pricing_index",
        "trial_volume_index",
        "churn_risk_index",
        "product_launch",
    ],
    ControlDomain.TRAVEL: [
        "holiday_indicator",
        "weather_index",
        "fuel_price",
        "consumer_sentiment",
    ],
}


def template_metadata(
    *,
    domain: ControlDomain,
    frequency: ControlFrequency,
    n_rows: int,
    period_column: str,
    control_columns: list[str],
) -> dict[str, Any]:
    return {
        "policy_version": POLICY_VERSION,
        "template_only": True,
        "illustrative_values_only": True,
        "modeling_default": False,
        "replace_with_real_historical_data": True,
        "diagnostic_only": True,
        "auto_model_integration_forbidden": True,
        "domain": domain.value,
        "frequency": frequency.value,
        "n_rows": n_rows,
        "period_column": period_column,
        "control_columns": control_columns,
        "value_note": (
            "SYNTHETIC_ILLUSTRATIVE — replace every cell with real historical controls "
            "before training or planning. These values are not priors, defaults, or "
            "recommended modeling assumptions."
        ),
    }


def _period_labels(n_rows: int, frequency: ControlFrequency) -> list[str]:
    if frequency == ControlFrequency.WEEKLY:
        return [f"2024-W{i:02d}" for i in range(1, n_rows + 1)]
    return [f"2024-{i:02d}" for i in range(1, n_rows + 1)]


def _column_seed(domain: ControlDomain, frequency: ControlFrequency, column: str) -> int:
    raw = f"{domain.value}|{frequency.value}|{column}|{POLICY_VERSION}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:8]
    return int(digest, 16) % (2**32)


def _illustrative_series(
    column: str,
    n_rows: int,
    *,
    domain: ControlDomain,
    frequency: ControlFrequency,
) -> list[float | int]:
    """Deterministic synthetic values — scale hints by column name."""
    seed = _column_seed(domain, frequency, column)
    rng = np.random.default_rng(seed)
    t = np.arange(n_rows, dtype=float)

    if column == "holiday_indicator" or column == "product_launch" or column == "site_outage":
        base = ((np.sin(2 * np.pi * t / max(n_rows, 4)) > 0.55).astype(int)).tolist()
        return [int(x) for x in base]

    if column in ("promo_intensity", "discount_rate", "competitor_promo", "cart_abandonment_rate", "churn_risk_index"):
        vals = 0.15 + 0.35 * (0.5 + 0.5 * np.sin(2 * np.pi * t / 13.0 + seed % 7))
        return [round(float(v), 4) for v in vals]

    if column == "pricing_index" or column.endswith("_index") and column != "weather_index":
        vals = 100.0 + 4.0 * np.sin(2 * np.pi * t / 11.0) + rng.normal(0, 0.4, size=n_rows)
        return [round(float(v), 3) for v in vals]

    if column == "weather_index" or column == "inventory_level":
        vals = 0.45 + 0.25 * np.sin(2 * np.pi * t / 9.0)
        return [round(float(v), 4) for v in vals]

    if column == "sp500_index":
        vals = 4200.0 + 180.0 * np.sin(2 * np.pi * t / 26.0) + rng.normal(0, 8.0, size=n_rows)
        return [round(float(v), 2) for v in vals]

    if column == "unemployment_rate":
        vals = 4.2 + 0.6 * np.sin(2 * np.pi * t / 18.0)
        return [round(float(v), 3) for v in vals]

    if column == "fuel_price":
        vals = 3.4 + 0.5 * np.sin(2 * np.pi * t / 15.0)
        return [round(float(v), 3) for v in vals]

    if column == "consumer_sentiment":
        vals = 82.0 + 8.0 * np.sin(2 * np.pi * t / 12.0)
        return [round(float(v), 2) for v in vals]

  # fallback smooth 0-1
    vals = 0.5 + 0.3 * np.sin(2 * np.pi * t / 10.0 + seed)
    return [round(float(v), 4) for v in vals]


def build_control_template_dataframe(
    *,
    domain: ControlDomain,
    frequency: ControlFrequency,
    n_rows: int,
) -> tuple[pd.DataFrame, str, list[str]]:
    if n_rows < 1:
        raise ValueError("n_rows must be >= 1")
    period_col = _PERIOD_COLUMN[frequency]
    controls = list(_DOMAIN_CONTROL_COLUMNS[domain])
    data: dict[str, Any] = {period_col: _period_labels(n_rows, frequency)}
    for col in controls:
        data[col] = _illustrative_series(col, n_rows, domain=domain, frequency=frequency)
    return pd.DataFrame(data), period_col, controls


def metadata_path_for_csv(csv_path: Path) -> Path:
    return csv_path.with_name(f"{csv_path.stem}_template.meta.json")


def generate_control_template(
    *,
    domain: str | ControlDomain,
    frequency: str | ControlFrequency,
    n_rows: int,
    out: Path,
    write_metadata: bool = True,
) -> dict[str, Any]:
    """
  Write CSV scaffold + optional sidecar metadata JSON.

  Returns paths and metadata dict (does not touch MMM config or training).
    """
    dom = ControlDomain(domain) if not isinstance(domain, ControlDomain) else domain
    freq = ControlFrequency(frequency) if not isinstance(frequency, ControlFrequency) else frequency
    df, period_col, controls = build_control_template_dataframe(
        domain=dom, frequency=freq, n_rows=n_rows
    )
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    meta = template_metadata(
        domain=dom,
        frequency=freq,
        n_rows=n_rows,
        period_column=period_col,
        control_columns=controls,
    )
    meta_path: Path | None = None
    if write_metadata:
        meta_path = metadata_path_for_csv(out)
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return {
        "csv_path": str(out),
        "metadata_path": str(meta_path) if meta_path else None,
        "metadata": meta,
        "columns": [period_col, *controls],
    }


def parse_domain(value: str) -> ControlDomain:
    try:
        return ControlDomain(value.lower())
    except ValueError as e:
        allowed = ", ".join(d.value for d in ControlDomain)
        raise ValueError(f"Unknown domain {value!r}; allowed: {allowed}") from e


def parse_frequency(value: str) -> ControlFrequency:
    try:
        return ControlFrequency(value.lower())
    except ValueError as e:
        allowed = ", ".join(f.value for f in ControlFrequency)
        raise ValueError(f"Unknown frequency {value!r}; allowed: {allowed}") from e
