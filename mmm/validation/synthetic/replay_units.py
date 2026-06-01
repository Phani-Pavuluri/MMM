"""Replay unit materialization from experiment_truth (no DGP)."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from mmm.calibration.replay_estimand import REPLAY_TRANSFORM_MODE_FULL_PANEL
from mmm.economics.canonical import REPLAY_LIFT_SCALES_KPI_LEVEL

SUPPORTED_REPLAY_TRANSFORM_MODES = frozenset({REPLAY_TRANSFORM_MODE_FULL_PANEL})


def _unit_experiment_id(unit: dict[str, Any], *, fallback: str) -> str:
    return str(unit.get("experiment_id") or unit.get("unit_id") or fallback)


def _unit_geo_scope(unit: dict[str, Any]) -> str:
    gs = str(unit.get("geo_scope", "listed"))
    return gs if gs in ("all", "listed") else "listed"


def build_replay_units_payload(truth: dict[str, Any]) -> list[dict[str, Any]] | None:
    """
    Render ``experiment_truth.units`` into JSON rows for ``load_calibration_units_from_json``.

    Derived artifact only — authoritative lift and windows remain in ``world_truth.json``.
    """
    units = (truth.get("experiment_truth") or {}).get("units") or []
    if not units:
        return None

    meta = truth["metadata"]
    outcome = truth["outcome_truth"]
    target_kpi = str(outcome["target_column"])
    world_id = str(meta["world_id"])
    default_transform = REPLAY_TRANSFORM_MODE_FULL_PANEL

    payload: list[dict[str, Any]] = []
    for u in units:
        lift = u.get("lift_definition") or {}
        unc = u.get("uncertainty") or {}
        channel = str(u["channel"])
        geos = list(u["geos"])
        week_start = str(u["week_start"])
        week_end = str(u["week_end"])
        lift_scale = str(lift.get("scale", "mean_kpi_level_delta"))
        estimand = str(u.get("estimand", "geo_time_ATT"))
        transform_mode = str(u.get("replay_transform_mode", default_transform))

        replay_estimand: dict[str, Any] = {
            "aggregation": str(u.get("aggregation", "mean")),
            "geo_ids": geos,
            "geo_scope": _unit_geo_scope(u),
            "lift_scale": lift_scale,
            "notes": f"materialized from {world_id} experiment_truth",
            "replay_transform_mode": transform_mode,
            "target_kpi_column": target_kpi,
            "week_end": week_end,
            "week_start": week_start,
        }

        observed_lift = float(lift["value"])
        lift_se = float(unc.get("se", 0.0))

        entry: dict[str, Any] = {
            "unit_id": str(u["unit_id"]),
            "world_id": world_id,
            "experiment_id": _unit_experiment_id(u, fallback=str(u["unit_id"])),
            "channel": channel,
            "treated_channel_names": [channel],
            "geo_scope": _unit_geo_scope(u),
            "time_window": {"week_start": week_start, "week_end": week_end},
            "lift": observed_lift,
            "standard_error": lift_se,
            "observed_lift": observed_lift,
            "lift_se": lift_se,
            "target_kpi": target_kpi,
            "geo_ids": geos,
            "estimand": estimand,
            "lift_scale": lift_scale,
            "replay_transform_mode": transform_mode,
            "replay_estimand": replay_estimand,
            "post_window_weeks": None,
            "observed_spend_frame": None,
            "counterfactual_spend_frame": None,
            "payload_version": "synthetic_world_v1",
            "calibration_readiness": str(u.get("calibration_readiness", "approved")),
        }
        payload.append(entry)

    return payload


def parse_iso_date(value: str) -> datetime:
    return datetime.strptime(str(value)[:10], "%Y-%m-%d")


def week_window_inside_time_truth(
    *,
    week_start: str,
    week_end: str,
    time_truth: dict[str, Any],
) -> bool:
    start = parse_iso_date(str(time_truth["start_date"]))
    end = parse_iso_date(str(time_truth["end_date"]))
    ws = parse_iso_date(week_start)
    we = parse_iso_date(week_end)
    return start <= ws <= end and start <= we <= end and ws <= we


def lift_scale_supported(scale: str) -> bool:
    return scale in REPLAY_LIFT_SCALES_KPI_LEVEL
