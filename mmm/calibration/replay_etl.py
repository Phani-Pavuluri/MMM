"""Build replay :class:`CalibrationUnit` objects from panel data + spend-shift specs (production ETL)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from mmm.calibration.contracts import CalibrationUnit
from mmm.calibration.replay_estimand import ReplayEstimandSpec
from mmm.data.schema import PanelSchema


@dataclass
class SpendShiftSpec:
    unit_id: str
    channel: str
    spend_multiplier: float
    geo_ids: list[str]
    week_start: Any
    week_end: Any
    observed_lift: float | None = None
    lift_se: float | None = None
    #: Experiment-reported KPI name; must match MMM ``target_column`` (or calibration override).
    target_kpi: str | None = None
    estimand: str = ""
    lift_scale: str = ""
    #: Serialized :class:`mmm.calibration.replay_estimand.ReplayEstimandSpec` (required for replay loss).
    replay_estimand: dict[str, Any] | None = None


def load_spend_shift_specs(path: str | Path) -> list[SpendShiftSpec]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    items = raw.get("shifts") if isinstance(raw, dict) else raw
    if not isinstance(items, list):
        raise ValueError("YAML must contain a list or mapping with key 'shifts'")
    out: list[SpendShiftSpec] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        g = it.get("geo_ids")
        if not isinstance(g, list):
            g = [str(g)] if g is not None else []
        out.append(
            SpendShiftSpec(
                unit_id=str(it.get("unit_id", "unit")),
                channel=str(it["channel"]),
                spend_multiplier=float(it["spend_multiplier"]),
                geo_ids=[str(x) for x in g],
                week_start=it["week_start"],
                week_end=it["week_end"],
                observed_lift=float(it["observed_lift"]) if it.get("observed_lift") is not None else None,
                lift_se=float(it["lift_se"]) if it.get("lift_se") is not None else None,
                target_kpi=str(it["target_kpi"]) if it.get("target_kpi") else None,
                estimand=str(it.get("estimand", "") or ""),
                lift_scale=str(it.get("lift_scale", "") or ""),
                replay_estimand=it["replay_estimand"] if isinstance(it.get("replay_estimand"), dict) else None,
            )
        )
    return out


def _panel_time_mask(series: pd.Series, start: Any, end: Any) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return (series >= float(start)) & (series <= float(end))
    s = pd.to_datetime(series, errors="coerce")
    a = pd.to_datetime(start, errors="coerce")
    b = pd.to_datetime(end, errors="coerce")
    return (s >= a) & (s <= b)


def _build_one_replay_unit(
    panel: pd.DataFrame,
    schema: PanelSchema,
    sp: SpendShiftSpec,
    *,
    target_kpi: str,
) -> CalibrationUnit | None:
    gcol, wcol = schema.geo_column, schema.week_column
    m = panel[gcol].isin(sp.geo_ids) & _panel_time_mask(panel[wcol], sp.week_start, sp.week_end)
    obs = panel.loc[m].copy()
    if obs.empty:
        return None
    if sp.channel not in obs.columns:
        raise ValueError(f"channel {sp.channel!r} not in panel columns")
    cf = obs.copy()
    cf[sp.channel] = cf[sp.channel].astype(float) * float(sp.spend_multiplier)
    tk = sp.target_kpi or target_kpi
    re_dict = sp.replay_estimand or {
        "geo_scope": "listed",
        "geo_ids": list(sp.geo_ids),
        "week_start": sp.week_start,
        "week_end": sp.week_end,
        "aggregation": "mean",
        "target_kpi_column": tk,
        "lift_scale": sp.lift_scale or "mean_kpi_level_delta",
        "notes": "derived_from_spend_shift_window",
    }
    ReplayEstimandSpec.from_dict(re_dict)
    return CalibrationUnit(
        unit_id=sp.unit_id,
        treated_channel_names=[sp.channel],
        observed_spend_frame=obs.reset_index(drop=True),
        counterfactual_spend_frame=cf.reset_index(drop=True),
        observed_lift=sp.observed_lift,
        lift_se=sp.lift_se,
        target_kpi=tk,
        geo_ids=list(sp.geo_ids),
        estimand=sp.estimand,
        lift_scale=sp.lift_scale,
        replay_estimand=re_dict,
    )


def build_replay_units_from_panel_shifts(
    panel: pd.DataFrame,
    schema: PanelSchema,
    shifts: list[SpendShiftSpec],
    *,
    target_kpi: str,
) -> list[CalibrationUnit]:
    """
    For each shift, slice ``panel`` to geo × week window, duplicate row set, and scale treated
    channel spend in the counterfactual frame by ``spend_multiplier``.
    """
    units: list[CalibrationUnit] = []
    for sp in shifts:
        u = _build_one_replay_unit(panel, schema, sp, target_kpi=target_kpi)
        if u is not None:
            units.append(u)
    return units


def ingest_validate_and_build_replay_units(
    panel: pd.DataFrame,
    schema: PanelSchema,
    shifts: list[SpendShiftSpec],
    *,
    target_kpi: str,
    expected_target_kpi: str | None = None,
) -> tuple[list[CalibrationUnit], list[dict]]:
    """
    Validate each shift vs panel scope (**reject** invalid experiments — no unit built), then build units.

    Returns ``(units, validation_reports)`` for audit / governance.
    """
    from mmm.calibration.experiment_validation import validate_spend_shift_against_panel

    exp_kpi = expected_target_kpi or target_kpi
    units: list[CalibrationUnit] = []
    reports: list[dict] = []
    for sp in shifts:
        rep = validate_spend_shift_against_panel(
            sp,
            panel,
            schema,
            expected_target_kpi=exp_kpi,
            unit_kpi=sp.target_kpi or target_kpi,
        )
        if not rep.accepted:
            reports.append(rep.to_json())
            continue
        u = _build_one_replay_unit(panel, schema, sp, target_kpi=target_kpi)
        if u is None:
            rep.add_error("empty_window", "no panel rows matched geo × week filter")
            reports.append(rep.to_json())
            continue
        reports.append(rep.to_json())
        units.append(u)
    return units, reports
