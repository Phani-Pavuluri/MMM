"""MIP-C2 — CalibrationSignal ingestion at Ridge train/extension diagnostic boundary."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mmm.diagnostics.calibration_signal_attachment import attach_calibration_evidence_context

REQUIRED_SIGNAL_KEYS = ("signal_id",)


def _validate_signal_record(raw: Any, *, index: int) -> tuple[dict[str, Any] | None, str | None]:
    if not isinstance(raw, dict):
        return None, f"signals[{index}]: expected object, got {type(raw).__name__}"
    signal_id = raw.get("signal_id")
    if not signal_id or not isinstance(signal_id, str):
        return None, f"signals[{index}]: missing or invalid signal_id"
    return raw, None


def parse_calibration_signals_payload(data: Any) -> tuple[list[dict[str, Any]], list[str]]:
    """Parse JSON payload into signal list; return (valid_signals, errors)."""
    errors: list[str] = []
    if isinstance(data, list):
        rows = data
    elif isinstance(data, dict):
        if "signals" in data:
            rows = data["signals"]
            if not isinstance(rows, list):
                return [], ["payload.signals must be a JSON array"]
        else:
            return [], ["payload must be a JSON array or object with signals array"]
    else:
        return [], [f"payload must be array or object, got {type(data).__name__}"]

    valid: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        sig, err = _validate_signal_record(row, index=i)
        if err:
            errors.append(err)
        elif sig is not None:
            valid.append(sig)
    return valid, errors


def load_calibration_signals_from_path(path: str | Path) -> tuple[list[dict[str, Any]], list[str], dict[str, Any]]:
    """Load CalibrationSignal records from a JSON file (fail-closed on parse errors)."""
    p = Path(path)
    meta: dict[str, Any] = {"source_type": "file", "source_path": str(p)}
    if not p.is_file():
        return [], [f"calibration_signals_file_not_found:{p}"], meta
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [], [f"calibration_signals_invalid_json:{exc}"], meta
    if isinstance(data, dict) and data.get("source_ref"):
        meta["source_ref"] = str(data["source_ref"])
    valid, errors = parse_calibration_signals_payload(data)
    return valid, errors, meta


def load_calibration_signals(
    *,
    signals: list[dict[str, Any]] | None = None,
    signals_path: str | Path | None = None,
) -> tuple[list[dict[str, Any]], list[str], dict[str, Any]]:
    """Resolve signals from in-memory list and/or file path (path loads first if both set)."""
    if signals_path is not None:
        return load_calibration_signals_from_path(signals_path)
    if signals is not None:
        valid, errors = parse_calibration_signals_payload(signals)
        return valid, errors, {"source_type": "list", "source_ref": "in_memory"}
    return [], [], {"source_type": "none"}


def build_evidence_attachment_lineage(
    report: dict[str, Any],
    *,
    attempted: bool = False,
    source_type: str = "none",
    source_path: str | None = None,
    source_ref: str | None = None,
    signals_count: int = 0,
    attached_count: int = 0,
    rejected_count: int = 0,
    attachment_errors: list[str] | None = None,
) -> dict[str, Any]:
    """Governed lineage block for MIP-C2 train/extension boundary."""
    cal_ctx = report.get("calibration_evidence_context")
    col = report.get("collinearity") or {}
    errors = list(attachment_errors or [])
    return {
        "milestone": "MIP-C2",
        "attempted": attempted,
        "source_type": source_type,
        "source_path": source_path,
        "source_ref": source_ref,
        "signals_count": signals_count,
        "attached_count": attached_count,
        "rejected_count": rejected_count,
        "attachment_errors": errors,
        "context_only": True,
        "optimizer_unchanged": True,
        "decision_surface_unchanged": True,
        "recommendations_unchanged": True,
        "calibration_evidence_context_present": bool(cal_ctx),
        "calibration_signal_count": len((cal_ctx or {}).get("signals") or []),
        "mip_c1_attachment_wired": bool(cal_ctx),
        "collinearity_calibration_evidence_available": bool(col.get("calibration_evidence_available")),
        "note": (
            "CalibrationSignal ingestion is diagnostic context only; does not refit Ridge, "
            "override coefficients, or feed optimizer/DecisionSurface/recommendations."
        ),
    }


def ingest_calibration_signals_into_report(
    report: dict[str, Any],
    *,
    signals: list[dict[str, Any]] | None = None,
    signals_path: str | Path | None = None,
) -> dict[str, Any]:
    """
    Optionally attach CalibrationSignal context to a composed Ridge diagnostic report.

    Does not mutate fit artifacts or trainer state.
    """
    if signals is None and signals_path is None:
        out = dict(report)
        out["evidence_attachment_lineage"] = build_evidence_attachment_lineage(
            out, attempted=False, source_type="none"
        )
        return out

    valid, errors, meta = load_calibration_signals(signals=signals, signals_path=signals_path)
    signals_count = len(valid) + len(errors)
    rejected_count = len(errors)

    if signals_path is not None and not valid and errors:
        out = dict(report)
        out["evidence_attachment_lineage"] = build_evidence_attachment_lineage(
            out,
            attempted=True,
            source_type=str(meta.get("source_type", "file")),
            source_path=meta.get("source_path"),
            source_ref=meta.get("source_ref"),
            signals_count=0,
            attached_count=0,
            rejected_count=rejected_count,
            attachment_errors=errors,
        )
        out["warnings"] = sorted(set(list(out.get("warnings") or []) + [f"mip_c2:{e}" for e in errors]))
        return out

    if not valid:
        out = dict(report)
        out["evidence_attachment_lineage"] = build_evidence_attachment_lineage(
            out,
            attempted=True,
            source_type=str(meta.get("source_type", "none")),
            source_path=meta.get("source_path"),
            source_ref=meta.get("source_ref"),
            signals_count=0,
            attached_count=0,
            rejected_count=rejected_count,
            attachment_errors=errors,
        )
        if errors:
            out["warnings"] = sorted(set(list(out.get("warnings") or []) + [f"mip_c2:{e}" for e in errors]))
        return out

    out = attach_calibration_evidence_context(report, valid)
    included = (out.get("calibration_evidence_context") or {}).get("included_signals") or []
    out["evidence_attachment_lineage"] = build_evidence_attachment_lineage(
        out,
        attempted=True,
        source_type=str(meta.get("source_type", "file")),
        source_path=meta.get("source_path"),
        source_ref=meta.get("source_ref"),
        signals_count=signals_count,
        attached_count=len(included),
        rejected_count=rejected_count,
        attachment_errors=errors,
    )
    if errors:
        out["warnings"] = sorted(set(list(out.get("warnings") or []) + [f"mip_c2:{e}" for e in errors]))
    return out
