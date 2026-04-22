"""Load replay calibration units from JSON."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from mmm.calibration.contracts import CalibrationUnit


def load_calibration_units_from_json(path: str | Path) -> list[CalibrationUnit]:
    p = Path(path)
    raw = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Replay units file must be a JSON list")
    out: list[CalibrationUnit] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        obs = _df_optional(item.get("observed_spend_frame"))
        cf = _df_optional(item.get("counterfactual_spend_frame"))
        out.append(
            CalibrationUnit(
                unit_id=str(item.get("unit_id", "unit")),
                treated_channel_names=list(item.get("treated_channel_names", [])),
                observed_spend_frame=obs,
                counterfactual_spend_frame=cf,
                post_window_weeks=tuple(item["post_window_weeks"]) if item.get("post_window_weeks") else None,
                observed_lift=float(item["observed_lift"]) if item.get("observed_lift") is not None else None,
                lift_se=float(item["lift_se"]) if item.get("lift_se") is not None else None,
                target_kpi=str(item.get("target_kpi", "")),
                geo_ids=list(item.get("geo_ids", [])),
                estimand=str(item.get("estimand", "") or ""),
                lift_scale=str(item.get("lift_scale", "") or ""),
                replay_estimand=item.get("replay_estimand") if isinstance(item.get("replay_estimand"), dict) else None,
                experiment_id=str(item.get("experiment_id", "") or ""),
                payload_version=str(item.get("payload_version", "") or ""),
                payload_sha256=str(item.get("payload_sha256", "") or ""),
                calibration_readiness=str(item.get("calibration_readiness", "") or ""),
            )
        )
    return out


def write_calibration_units_to_json(units: list[CalibrationUnit], path: str | Path) -> None:
    """Serialize replay units (including embedded frames as record lists) for ``load_calibration_units_from_json``."""
    p = Path(path)
    rows: list[dict[str, Any]] = []
    for u in units:
        rows.append(
            {
                "unit_id": u.unit_id,
                "treated_channel_names": list(u.treated_channel_names),
                "observed_spend_frame": (
                    u.observed_spend_frame.to_dict(orient="records") if u.observed_spend_frame is not None else None
                ),
                "counterfactual_spend_frame": (
                    u.counterfactual_spend_frame.to_dict(orient="records")
                    if u.counterfactual_spend_frame is not None
                    else None
                ),
                "post_window_weeks": list(u.post_window_weeks) if u.post_window_weeks else None,
                "observed_lift": u.observed_lift,
                "lift_se": u.lift_se,
                "target_kpi": u.target_kpi,
                "geo_ids": list(u.geo_ids),
                "estimand": u.estimand,
                "lift_scale": u.lift_scale,
                "replay_estimand": u.replay_estimand,
                "experiment_id": u.experiment_id,
                "payload_version": u.payload_version,
                "payload_sha256": u.payload_sha256,
                "calibration_readiness": u.calibration_readiness,
            }
        )
    p.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")


def _df_optional(obj: Any) -> pd.DataFrame | None:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return pd.DataFrame(obj)
    if isinstance(obj, list):
        return pd.DataFrame(obj)
    return None
