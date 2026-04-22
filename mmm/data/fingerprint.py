"""Dataset + schema fingerprints for reproducibility (Sprint 11)."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import pandas as pd

from mmm.data.schema import PanelSchema


def fingerprint_panel(df: pd.DataFrame, schema: PanelSchema) -> dict[str, Any]:
    """Stable hash over sorted (geo, week) rows of key columns."""
    key_cols = [schema.geo_column, schema.week_column, schema.target_column, *schema.channel_columns]
    sub = df[key_cols].sort_values([schema.geo_column, schema.week_column])
    payload = sub.to_csv(index=False).encode("utf-8")
    h = hashlib.sha256(payload).hexdigest()
    schema_blob = json.dumps(
        {
            "geo": schema.geo_column,
            "week": schema.week_column,
            "target": schema.target_column,
            "channels": list(schema.channel_columns),
            "controls": list(schema.control_columns),
        },
        sort_keys=True,
    ).encode()
    return {
        "sha256_panel_keycols_sorted_csv": h,
        "sha256_schema_json": hashlib.sha256(schema_blob).hexdigest(),
        "n_rows": int(len(df)),
    }
