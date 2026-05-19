"""Dataset + schema + config fingerprints for reproducibility."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import pandas as pd

from mmm.config.schema import MMMConfig
from mmm.data.schema import PanelSchema

FINGERPRINT_VERSION = "fingerprint_v2"
CONFIG_FINGERPRINT_SCHEMA_VERSION = "mmm_config_v1"

# Volatile / non-reproducibility fields intentionally excluded from the combined hash.
OMITTED_FIELDS = (
    "run_id",
    "created_at",
    "generated_id",
    "timestamp",
    "mlflow_run_id",
    "artifact_uri",
)


def _hash_json_blob(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def fingerprint_panel(
    df: pd.DataFrame,
    schema: PanelSchema,
    *,
    config: MMMConfig | None = None,
    seed_resolution: dict[str, Any] | None = None,
    planning_assumptions: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Stable fingerprint over panel key columns, schema, modeling config, seeds, and planning assumptions.

    When ``config`` is omitted, only panel + schema columns are hashed (legacy behavior).
    """
    key_cols = [schema.geo_column, schema.week_column, schema.target_column, *schema.channel_columns]
    if schema.control_columns:
        key_cols.extend(schema.control_columns)
    sub = df[key_cols].sort_values([schema.geo_column, schema.week_column])
    panel_payload = sub.to_csv(index=False).encode("utf-8")
    h_panel = hashlib.sha256(panel_payload).hexdigest()

    schema_blob = {
        "geo": schema.geo_column,
        "week": schema.week_column,
        "target": schema.target_column,
        "channels": list(schema.channel_columns),
        "controls": list(schema.control_columns),
    }
    h_schema = hashlib.sha256(json.dumps(schema_blob, sort_keys=True).encode()).hexdigest()

    included: list[str] = [
        "geo_column",
        "week_column",
        "target_column",
        "channel_columns",
        "control_columns",
        "panel_keycols_sorted_csv",
        "schema_json",
    ]
    combined_parts: dict[str, Any] = {
        "fingerprint_version": FINGERPRINT_VERSION,
        "panel": h_panel,
        "schema": h_schema,
    }

    if config is not None:
        d = config.data
        combined_parts["model_form"] = config.model_form.value
        combined_parts["framework"] = config.framework.value
        combined_parts["transforms"] = config.transforms.model_dump(mode="json")
        combined_parts["config_schema_version"] = CONFIG_FINGERPRINT_SCHEMA_VERSION
        combined_parts["data_version_id"] = d.data_version_id
        included.extend(
            [
                "model_form",
                "framework",
                "transforms",
                "config_schema_version",
                "data_version_id",
            ]
        )
        if seed_resolution is not None:
            combined_parts["resolved_seeds"] = seed_resolution.get("resolved_child_seeds") or {}
            combined_parts["master_seed"] = seed_resolution.get("master_seed")
            included.append("resolved_seeds")
            included.append("master_seed")
        else:
            combined_parts["resolved_seeds"] = {
                "random_seed": int(config.random_seed),
                "ridge_bo.sampler_seed": int(config.ridge_bo.sampler_seed or config.random_seed),
                "cv.geo_blocked_seed": int(config.cv.geo_blocked_seed or config.random_seed),
                "bootstrap_seed": int(config.bootstrap_seed or config.random_seed),
                "extension_seed": int(config.extension_seed or config.random_seed),
                "experiment_scheduler_seed": int(
                    config.experiment_scheduler_seed or config.random_seed
                ),
                "simulation_seed": int(config.simulation_seed or config.random_seed),
            }
            included.append("resolved_seeds_fallback")

    if planning_assumptions is not None:
        combined_parts["planning_assumptions"] = {
            k: planning_assumptions.get(k)
            for k in ("controls_assumption", "media_assumption", "world_assumption")
        }
        included.append("planning_assumptions")

    h_combined = _hash_json_blob(combined_parts)

    return {
        "fingerprint_version": FINGERPRINT_VERSION,
        "sha256_combined": h_combined,
        "sha256_panel_keycols_sorted_csv": h_panel,
        "sha256_schema_json": h_schema,
        "n_rows": int(len(df)),
        "fingerprint_details": {
            "included_fields": included,
            "fingerprint_version": FINGERPRINT_VERSION,
            "omitted_fields": list(OMITTED_FIELDS),
        },
    }
