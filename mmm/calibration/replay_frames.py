"""Shared full-panel replay frame construction (legacy + evidence-registry)."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd

from mmm.calibration.contracts import CalibrationUnit
from mmm.calibration.replay_estimand import (
    REPLAY_TRANSFORM_MODE_FULL_PANEL,
    ReplayEstimandSpec,
    eval_mask_for_replay,
)
from mmm.data.panel_order import sort_panel_for_modeling
from mmm.data.schema import PanelSchema

LEGACY_REPLAY_DEPRECATED_WARNING = "legacy_replay_deprecated_use_evidence_registry"


def panel_time_mask(series: pd.Series, start: Any, end: Any) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return (series >= float(start)) & (series <= float(end))
    s = pd.to_datetime(series, errors="coerce")
    a = pd.to_datetime(start, errors="coerce")
    b = pd.to_datetime(end, errors="coerce")
    return (s >= a) & (s <= b)


def build_full_panel_replay_frames(
    panel: pd.DataFrame,
    schema: PanelSchema,
    spec: ReplayEstimandSpec,
    channel: str,
    spend_multiplier: float,
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """
    Full-panel spend paths with counterfactual change only inside the experiment estimand mask.

    Adstock/saturation run on the full sorted panel so pre-window carryover is preserved;
    implied lift is aggregated only over ``eval_mask_for_replay`` rows.
    """
    full = sort_panel_for_modeling(panel.copy(), schema)
    mask = eval_mask_for_replay(full, schema, spec)
    if not np.any(mask):
        return None
    obs = full.copy()
    cf = full.copy()
    cf.loc[mask, channel] = cf.loc[mask, channel].astype(float) * float(spend_multiplier)
    return obs.reset_index(drop=True), cf.reset_index(drop=True)


def _ensure_replay_estimand_dict(
    re_dict: dict[str, Any],
    *,
    schema: PanelSchema,
    geo_ids: list[str],
    week_start: Any,
    week_end: Any,
    lift_scale: str = "mean_kpi_level_delta",
    notes: str = "",
) -> dict[str, Any]:
    out = dict(re_dict)
    out.setdefault("geo_scope", "listed" if geo_ids else "all")
    out.setdefault("geo_ids", list(geo_ids))
    out.setdefault("week_start", week_start)
    out.setdefault("week_end", week_end)
    out.setdefault("aggregation", out.get("aggregation", "mean"))
    out.setdefault("target_kpi_column", schema.target_column)
    out.setdefault("lift_scale", lift_scale or "mean_kpi_level_delta")
    out.setdefault("notes", notes)
    out["replay_transform_mode"] = REPLAY_TRANSFORM_MODE_FULL_PANEL
    ReplayEstimandSpec.from_dict(out)
    return out


def build_calibration_unit_from_shift(
    panel: pd.DataFrame,
    schema: PanelSchema,
    *,
    unit_id: str,
    channel: str,
    geo_ids: list[str],
    week_start: Any,
    week_end: Any,
    spend_multiplier: float,
    observed_lift: float | None = None,
    lift_se: float | None = None,
    target_kpi: str | None = None,
    estimand: str = "",
    lift_scale: str = "",
    replay_estimand: dict[str, Any] | None = None,
    experiment_id: str = "",
    calibration_readiness: str = "",
) -> CalibrationUnit | None:
    """Build a replay unit using the same full-panel transform path as evidence-registry replay."""
    tk = target_kpi or schema.target_column
    re_dict = _ensure_replay_estimand_dict(
        replay_estimand
        or {
            "geo_scope": "listed",
            "geo_ids": list(geo_ids),
            "week_start": week_start,
            "week_end": week_end,
            "aggregation": "mean",
            "target_kpi_column": tk,
            "lift_scale": lift_scale or "mean_kpi_level_delta",
            "notes": "legacy_spend_shift_full_panel",
        },
        schema=schema,
        geo_ids=geo_ids,
        week_start=week_start,
        week_end=week_end,
        lift_scale=lift_scale or "mean_kpi_level_delta",
        notes=str((replay_estimand or {}).get("notes", "legacy_spend_shift_full_panel")),
    )
    spec = ReplayEstimandSpec.from_dict(re_dict)
    gcol, wcol = schema.geo_column, schema.week_column
    window_mask = (
        panel[gcol].astype(str).isin(set(spec.geo_ids))
        if spec.geo_scope == "listed" and spec.geo_ids
        else pd.Series(True, index=panel.index)
    ) & panel_time_mask(panel[wcol], spec.week_start, spec.week_end)
    if not bool(window_mask.any()):
        return None
    frames = build_full_panel_replay_frames(panel, schema, spec, channel, spend_multiplier)
    if frames is None:
        return None
    obs, cf = frames
    return CalibrationUnit(
        unit_id=unit_id,
        treated_channel_names=[channel],
        observed_spend_frame=obs,
        counterfactual_spend_frame=cf,
        observed_lift=observed_lift,
        lift_se=lift_se,
        target_kpi=tk,
        geo_ids=list(spec.geo_ids) if spec.geo_ids else list(geo_ids),
        estimand=estimand,
        lift_scale=str(re_dict.get("lift_scale", "")),
        replay_estimand=re_dict,
        experiment_id=experiment_id,
        calibration_readiness=calibration_readiness,
    )


def _full_panel_row_count(panel: pd.DataFrame, schema: PanelSchema) -> int:
    return len(sort_panel_for_modeling(panel, schema))


def normalize_replay_units_to_full_panel(
    panel: pd.DataFrame,
    schema: PanelSchema,
    units: list[CalibrationUnit],
) -> tuple[list[CalibrationUnit], list[str]]:
    """
    Upgrade stored window-slice units to full-panel frames when ``replay_estimand`` is present.
    """
    n_full = _full_panel_row_count(panel, schema)
    warnings: list[str] = []
    out: list[CalibrationUnit] = []
    for u in units:
        if u.observed_spend_frame is None or u.counterfactual_spend_frame is None:
            out.append(u)
            continue
        if not u.replay_estimand:
            if len(u.observed_spend_frame) < n_full:
                warnings.append(
                    f"{u.unit_id}: {LEGACY_REPLAY_DEPRECATED_WARNING} (missing replay_estimand; "
                    "window-slice frames cannot be upgraded)"
                )
            out.append(u)
            continue
        if len(u.observed_spend_frame) >= n_full:
            re = dict(u.replay_estimand)
            re["replay_transform_mode"] = REPLAY_TRANSFORM_MODE_FULL_PANEL
            out.append(replace(u, replay_estimand=re))
            continue
        spec = ReplayEstimandSpec.from_dict(u.replay_estimand)
        ch = u.treated_channel_names[0] if u.treated_channel_names else ""
        if not ch:
            out.append(u)
            continue
        obs_spend = u.observed_spend_frame
        cf_spend = u.counterfactual_spend_frame
        if ch in obs_spend.columns and ch in cf_spend.columns:
            mult = float(cf_spend[ch].astype(float).mean() / max(obs_spend[ch].astype(float).mean(), 1e-9))
        else:
            mult = 1.0
        rebuilt = build_calibration_unit_from_shift(
            panel,
            schema,
            unit_id=u.unit_id,
            channel=ch,
            geo_ids=list(u.geo_ids) or list(spec.geo_ids),
            week_start=spec.week_start,
            week_end=spec.week_end,
            spend_multiplier=mult,
            observed_lift=u.observed_lift,
            lift_se=u.lift_se,
            target_kpi=u.target_kpi or schema.target_column,
            estimand=u.estimand,
            lift_scale=u.lift_scale,
            replay_estimand=u.replay_estimand,
            experiment_id=u.experiment_id,
            calibration_readiness=u.calibration_readiness,
        )
        if rebuilt is None:
            warnings.append(f"{u.unit_id}: could not rebuild full-panel replay frames")
            out.append(u)
        else:
            warnings.append(f"{u.unit_id}: upgraded window-slice replay to full_panel_transform_estimand_mask")
            out.append(rebuilt)
    return out, warnings
