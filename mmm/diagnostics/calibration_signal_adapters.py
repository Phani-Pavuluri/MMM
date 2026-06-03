"""MIP-C3 — GeoX/CLS export → CalibrationSignal adapter (no live APIs)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from mmm.diagnostics.calibration_signal_attachment import FORBIDDEN_ATTACHMENT_OUTPUT_FIELDS

ADAPTER_VERSION = "mip_geox_cls_calibration_signal_adapter_v1"

# Platform estimand ids (MMM contract); exports may use vendor aliases.
ESTIMAND_ALIASES: dict[str, str] = {
    "incremental_sales": "incremental_sales",
    "incremental_roi": "incremental_sales",
    "sales_lift": "incremental_sales",
    "brand_lift": "brand_lift",
    "awareness_lift": "brand_lift",
}

FRESHNESS_STALE_DAYS = 180


def _parse_date(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    text = str(value).strip()
    if not text:
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            dt = datetime.strptime(text.replace("Z", ""), fmt.replace("Z", ""))
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def _freshness_from_as_of(as_of: Any, *, reference: datetime | None = None) -> str:
    ref = reference or datetime.now(timezone.utc)
    dt = _parse_date(as_of)
    if dt is None:
        return "unknown"
    age_days = (ref - dt).days
    if age_days > FRESHNESS_STALE_DAYS:
        return "stale"
    if age_days < 0:
        return "unknown"
    return "fresh"


def _map_estimand(raw: Any) -> tuple[str | None, bool]:
    if raw is None or raw == "":
        return None, False
    key = str(raw).strip().lower().replace(" ", "_")
    mapped = ESTIMAND_ALIASES.get(key, key)
    known = mapped in ESTIMAND_ALIASES.values()
    return mapped, known


def _geo_scope_from_geox(record: dict[str, Any]) -> dict[str, Any]:
    dma_ids = record.get("dma_ids") or record.get("geo_ids") or record.get("treated_dma_ids")
    if isinstance(dma_ids, list) and dma_ids:
        return {"kind": "dma", "ids": [str(x) for x in dma_ids]}
    region = record.get("region_id")
    if region:
        return {"kind": "region", "ids": [str(region)]}
    scope = record.get("geo_scope")
    if isinstance(scope, dict):
        return scope
    return {"kind": "national"}


def _time_window(record: dict[str, Any]) -> dict[str, str] | None:
    start = record.get("window_start") or record.get("time_window_start") or record.get("start_date")
    end = record.get("window_end") or record.get("time_window_end") or record.get("end_date")
    if start and end:
        return {"start": str(start), "end": str(end)}
    tw = record.get("time_window")
    if isinstance(tw, dict) and tw.get("start") and tw.get("end"):
        return {"start": str(tw["start"]), "end": str(tw["end"])}
    return None


def _uncertainty_from_record(
    record: dict[str, Any],
) -> tuple[float | None, float | None, list[float] | None, bool]:
    se = record.get("standard_error")
    if se is None:
        se = record.get("incremental_lift_se") or record.get("lift_se") or record.get("posterior_se")
    interval = record.get("interval") or record.get("credible_interval")
    if isinstance(interval, (list, tuple)) and len(interval) >= 2:
        try:
            interval_f = [float(interval[0]), float(interval[1])]
        except (TypeError, ValueError):
            interval_f = None
    else:
        interval_f = None
    effect = record.get("effect_estimate")
    if effect is None:
        effect = record.get("incremental_lift") or record.get("lift") or record.get("point_estimate")
    try:
        effect_f = float(effect) if effect is not None else None
    except (TypeError, ValueError):
        effect_f = None
    try:
        se_f = float(se) if se is not None else None
    except (TypeError, ValueError):
        se_f = None
    has_uncertainty = se_f is not None or interval_f is not None
    return effect_f, se_f, interval_f, has_uncertainty


def geox_record_to_calibration_signal(record: dict[str, Any]) -> dict[str, Any]:
    """Map a GeoX export row to MIP-C2 CalibrationSignal JSON."""
    experiment_id = str(
        record.get("geox_experiment_id")
        or record.get("experiment_id")
        or record.get("experiment_key")
        or "geox-unknown"
    )
    signal_id = str(record.get("signal_id") or f"geox-{experiment_id}")
    channel = str(record.get("channel") or record.get("media_channel") or "").strip()
    estimand_raw = record.get("estimand_id") or record.get("estimand_type") or record.get("estimand")
    estimand_id, estimand_known = _map_estimand(estimand_raw)
    effect_f, se_f, interval_f, has_uncertainty = _uncertainty_from_record(record)

    instrument = str(
        record.get("measurement_instrument_id")
        or record.get("geox_export_version")
        or record.get("export_version")
        or "geox_export_v1"
    )
    as_of = record.get("evidence_as_of") or record.get("as_of_date") or record.get("export_timestamp")
    freshness = record.get("freshness_status") or _freshness_from_as_of(as_of)

    eligibility = str(record.get("eligibility_status") or "eligible")
    adapter_notes: list[str] = []
    if not has_uncertainty:
        eligibility = "blocked"
        adapter_notes.append("missing_uncertainty:geox_export")
    if estimand_id and not estimand_known:
        adapter_notes.append(f"estimand_unmapped:{estimand_raw}")
    if record.get("estimand_mismatch_flag"):
        adapter_notes.append("estimand_mismatch_flagged_at_source")

    return {
        "signal_id": signal_id,
        "source_system": "geox",
        "source_modality": str(record.get("source_modality") or "geo_experiment"),
        "experiment_id": experiment_id,
        "channel": channel,
        "geo_scope": _geo_scope_from_geox(record),
        "time_window": _time_window(record),
        "estimand_id": estimand_id,
        "measurement_instrument_id": instrument,
        "lift_scale": str(record.get("lift_scale") or record.get("lift_metric") or "incremental_sales"),
        "effect_estimate": effect_f,
        "standard_error": se_f,
        "interval": interval_f,
        "freshness_status": str(freshness),
        "eligibility_status": eligibility,
        "adapter_metadata": {
            "adapter_version": ADAPTER_VERSION,
            "source_system": "geox",
            "source_lineage": {
                "geox_experiment_id": experiment_id,
                "geox_export_version": instrument,
                "source_record_id": record.get("source_record_id"),
            },
            "adapter_notes": adapter_notes,
            "estimand_source_value": estimand_raw,
            "trust_report_boundary": "GeoX export is context-only; TrustReport governs promotion.",
        },
    }


def cls_record_to_calibration_signal(record: dict[str, Any]) -> dict[str, Any]:
    """Map a CLS readout row to MIP-C2 CalibrationSignal JSON."""
    study_id = str(record.get("cls_study_id") or record.get("study_id") or record.get("study_key") or "cls-unknown")
    signal_id = str(record.get("signal_id") or f"cls-{study_id}")
    channel = str(record.get("channel") or record.get("media_channel") or "").strip()
    estimand_raw = record.get("estimand_id") or record.get("estimand_type") or record.get("kpi_type")
    estimand_id, estimand_known = _map_estimand(estimand_raw)
    effect_f, se_f, interval_f, has_uncertainty = _uncertainty_from_record(record)

    readout_status = str(record.get("readout_status") or record.get("cls_readout_status") or "eligible").lower()
    eligibility = str(record.get("eligibility_status") or "eligible")
    if readout_status in ("inconclusive", "failed", "underpowered"):
        eligibility = "inconclusive"
    elif readout_status in ("excluded", "blocked"):
        eligibility = "excluded"
    if not has_uncertainty:
        eligibility = "blocked"
    adapter_notes: list[str] = []
    if not has_uncertainty:
        adapter_notes.append("missing_uncertainty:cls_readout")
    if estimand_id and not estimand_known:
        adapter_notes.append(f"estimand_unmapped:{estimand_raw}")

    as_of = record.get("as_of_date") or record.get("readout_date") or record.get("evidence_as_of")
    freshness = record.get("freshness_status") or _freshness_from_as_of(as_of)
    if record.get("stale_flag"):
        freshness = "stale"

    instrument = str(
        record.get("measurement_instrument_id")
        or record.get("cls_readout_version")
        or record.get("readout_id")
        or "cls_readout_v1"
    )

    geo_scope = record.get("geo_scope")
    if not isinstance(geo_scope, dict):
        geo_scope = {"kind": str(record.get("geo_scope_kind") or "national")}

    return {
        "signal_id": signal_id,
        "source_system": "cls",
        "source_modality": str(record.get("source_modality") or "cls_readout"),
        "study_id": study_id,
        "experiment_id": str(record.get("experiment_id") or study_id),
        "channel": channel,
        "geo_scope": geo_scope,
        "time_window": _time_window(record),
        "estimand_id": estimand_id,
        "measurement_instrument_id": instrument,
        "lift_scale": str(record.get("lift_scale") or record.get("kpi_metric") or "incremental_sales"),
        "effect_estimate": effect_f,
        "standard_error": se_f,
        "interval": interval_f,
        "freshness_status": str(freshness),
        "eligibility_status": eligibility,
        "adapter_metadata": {
            "adapter_version": ADAPTER_VERSION,
            "source_system": "cls",
            "source_lineage": {
                "cls_study_id": study_id,
                "cls_readout_id": record.get("cls_readout_id") or record.get("readout_id"),
                "readout_status": readout_status,
            },
            "adapter_notes": adapter_notes,
            "estimand_source_value": estimand_raw,
            "trust_report_boundary": "CLS readout is context-only; inconclusive readouts route to TrustReport.",
        },
    }


def validate_adapter_output(signal: dict[str, Any]) -> tuple[bool, list[str]]:
    """Validate adapted signal is C2-ingestible; return (ok, errors)."""
    errors: list[str] = []
    if not isinstance(signal, dict):
        return False, ["signal must be object"]
    if not signal.get("signal_id"):
        errors.append("missing signal_id")
    if not signal.get("source_system"):
        errors.append("missing source_system")
    if not signal.get("channel"):
        errors.append("missing channel")
    for forbidden in FORBIDDEN_ATTACHMENT_OUTPUT_FIELDS:
        if forbidden in signal:
            errors.append(f"forbidden_field:{forbidden}")
    meta = signal.get("adapter_metadata")
    if not isinstance(meta, dict) or meta.get("adapter_version") != ADAPTER_VERSION:
        errors.append("missing_or_invalid adapter_metadata")
    return len(errors) == 0, errors


def normalize_calibration_signal_batch(
    records: list[Any],
    source_system: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    """
    Convert a batch of GeoX or CLS export records to CalibrationSignal list.

    ``records`` may be raw export dicts, or wrappers ``{"source_system", "record"}``.
    """
    system = source_system.lower().strip()
    if system not in ("geox", "cls"):
        return [], [f"unsupported source_system:{source_system}"]

    adapted: list[dict[str, Any]] = []
    errors: list[str] = []
    for i, row in enumerate(records):
        if isinstance(row, dict) and "record" in row and "source_system" in row:
            sys_row = str(row["source_system"]).lower()
            rec = row["record"]
            if not isinstance(rec, dict):
                errors.append(f"batch[{i}]: record must be object")
                continue
            if sys_row == "geox":
                sig = geox_record_to_calibration_signal(rec)
            elif sys_row == "cls":
                sig = cls_record_to_calibration_signal(rec)
            else:
                errors.append(f"batch[{i}]: unsupported source_system:{sys_row}")
                continue
        elif isinstance(row, dict):
            if system == "geox":
                sig = geox_record_to_calibration_signal(row)
            else:
                sig = cls_record_to_calibration_signal(row)
        else:
            errors.append(f"batch[{i}]: expected object, got {type(row).__name__}")
            continue
        ok, val_errors = validate_adapter_output(sig)
        if not ok:
            errors.extend([f"batch[{i}]:{e}" for e in val_errors])
            continue
        adapted.append(sig)
    return adapted, errors


def adapt_mixed_batch_export(data: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str], dict[str, Any]]:
    """Adapt ``{"geox": [...], "cls": [...]}`` or ``{"records": [...]}`` export bundle."""
    errors: list[str] = []
    signals: list[dict[str, Any]] = []
    lineage: dict[str, Any] = {
        "adapter_version": ADAPTER_VERSION,
        "geox_count": 0,
        "cls_count": 0,
    }
    if "records" in data:
        mixed = data["records"]
        if not isinstance(mixed, list):
            return [], ["records must be array"], lineage
        for row in mixed:
            if not isinstance(row, dict):
                errors.append("mixed record must be object")
                continue
            sys_name = str(row.get("source_system", "")).lower()
            rec = row.get("record")
            if sys_name == "geox" and isinstance(rec, dict):
                sig = geox_record_to_calibration_signal(rec)
                lineage["geox_count"] += 1
            elif sys_name == "cls" and isinstance(rec, dict):
                sig = cls_record_to_calibration_signal(rec)
                lineage["cls_count"] += 1
            else:
                errors.append(f"unsupported mixed row source_system:{sys_name}")
                continue
            ok, val_errs = validate_adapter_output(sig)
            if ok:
                signals.append(sig)
            else:
                errors.extend(val_errs)
        return signals, errors, lineage

    for key, adapter_fn, count_key in (
        ("geox", geox_record_to_calibration_signal, "geox_count"),
        ("cls", cls_record_to_calibration_signal, "cls_count"),
    ):
        rows = data.get(key)
        if rows is None:
            continue
        if not isinstance(rows, list):
            errors.append(f"{key} must be array")
            continue
        for i, rec in enumerate(rows):
            if not isinstance(rec, dict):
                errors.append(f"{key}[{i}]: must be object")
                continue
            sig = adapter_fn(rec)
            ok, val_errs = validate_adapter_output(sig)
            if ok:
                signals.append(sig)
                lineage[count_key] += 1
            else:
                errors.extend([f"{key}[{i}]:{e}" for e in val_errs])
    return signals, errors, lineage
