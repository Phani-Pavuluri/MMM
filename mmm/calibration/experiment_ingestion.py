"""Load experiment / replay shift rows from CSV (Tier 1 production ingestion)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from mmm.calibration.replay_etl import SpendShiftSpec
from mmm.calibration.schema import ExperimentObservation


def load_experiment_observations_csv(path: str | Path) -> list[ExperimentObservation]:
    """
    CSV columns (flexible names mapped):

    - ``experiment_id`` (or ``exp_id``)
    - ``geo_id`` (optional)
    - ``channel``
    - ``start_week``, ``end_week`` (optional)
    - ``lift``, ``lift_se`` (optional)
    """
    p = Path(path)
    df = pd.read_csv(p)
    colmap = {c.lower().strip(): c for c in df.columns}

    def col(*names: str) -> str | None:
        for n in names:
            if n in colmap:
                return colmap[n]
        return None

    c_exp = col("experiment_id", "exp_id", "experiment")
    c_geo = col("geo_id", "geo")
    c_ch = col("channel")
    if not c_ch:
        raise ValueError("CSV must contain a channel column")
    c_sw = col("start_week", "week_start")
    c_ew = col("end_week", "week_end")
    c_lift = col("lift", "observed_lift")
    out: list[ExperimentObservation] = []
    for _, row in df.iterrows():
        eid = str(row[c_exp]) if c_exp else f"row_{len(out)}"
        meta: dict[str, str] = {}
        c_kpi = col("target_kpi", "kpi", "kpi_name")
        if c_kpi and pd.notna(row.get(c_kpi)):
            meta["target_kpi"] = str(row[c_kpi])
        lift = float(row[c_lift]) if c_lift and pd.notna(row.get(c_lift)) else 0.0
        se_col = col("lift_se", "se")
        lift_se = float(row[se_col]) if se_col and pd.notna(row.get(se_col)) else None
        sw = str(row[c_sw]) if c_sw and pd.notna(row.get(c_sw)) else None
        ew = str(row[c_ew]) if c_ew and pd.notna(row.get(c_ew)) else None
        out.append(
            ExperimentObservation(
                experiment_id=eid,
                geo_id=str(row[c_geo]) if c_geo and pd.notna(row.get(c_geo)) else None,
                channel=str(row[c_ch]),
                start_week=sw,
                end_week=ew,
                lift=lift,
                lift_se=lift_se,
                metadata=meta,
            )
        )
    return out


def load_spend_shift_specs_csv(path: str | Path) -> list[SpendShiftSpec]:
    """
    CSV for replay ETL: ``unit_id, geo_id(s), channel, week_start, week_end, spend_multiplier``,
    optional ``observed_lift, lift_se``.

    ``geo_ids`` can be ``geo_id`` single or ``geo_ids`` semicolon-separated.
    """
    df = pd.read_csv(Path(path))
    rows: list[SpendShiftSpec] = []
    for _, r in df.iterrows():
        gid = r.get("geo_ids") if "geo_ids" in df.columns else r.get("geo_id")
        if pd.isna(gid):
            geos: list[str] = []
        elif isinstance(gid, str) and ";" in gid:
            geos = [x.strip() for x in gid.split(";") if x.strip()]
        else:
            geos = [str(gid).strip()]
        tk = str(r["target_kpi"]) if "target_kpi" in df.columns and pd.notna(r.get("target_kpi")) else None
        est = str(r["estimand"]) if "estimand" in df.columns and pd.notna(r.get("estimand")) else ""
        ls = str(r["lift_scale"]) if "lift_scale" in df.columns and pd.notna(r.get("lift_scale")) else ""
        rows.append(
            SpendShiftSpec(
                unit_id=str(r.get("unit_id", "unit")),
                channel=str(r["channel"]),
                spend_multiplier=float(r["spend_multiplier"]),
                geo_ids=geos,
                week_start=r["week_start"],
                week_end=r["week_end"],
                observed_lift=float(r["observed_lift"]) if pd.notna(r.get("observed_lift")) else None,
                lift_se=float(r["lift_se"]) if pd.notna(r.get("lift_se")) else None,
                target_kpi=tk,
                estimand=est,
                lift_scale=ls,
            )
        )
    return rows


def spend_shift_specs_to_jsonable(specs: list[SpendShiftSpec]) -> list[dict[str, Any]]:
    return [
        {
            "unit_id": s.unit_id,
            "channel": s.channel,
            "spend_multiplier": s.spend_multiplier,
            "geo_ids": s.geo_ids,
            "week_start": s.week_start,
            "week_end": s.week_end,
            "observed_lift": s.observed_lift,
            "lift_se": s.lift_se,
            "estimand": s.estimand,
            "lift_scale": s.lift_scale,
        }
        for s in specs
    ]
