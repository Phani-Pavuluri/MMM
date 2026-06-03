"""MIP-C4 — GeoX/CLS CalibrationSignal ETL dry-run (export files → C2 JSON artifact)."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from mmm.diagnostics.calibration_signal_adapters import (
    ADAPTER_VERSION,
    adapt_mixed_batch_export,
    validate_adapter_output,
)
from mmm.diagnostics.calibration_signal_attachment import FORBIDDEN_ATTACHMENT_OUTPUT_FIELDS
from mmm.diagnostics.calibration_signal_ingestion import (
    ingest_calibration_signals_into_report,
    parse_calibration_signals_payload,
)
from mmm.diagnostics.ridge_diagnostics import FORBIDDEN_OUTPUT_FIELDS

ETL_ARTIFACT_TYPE = "calibration_signal_etl_dry_run"
ETL_VERSION = "mip_calibration_signal_etl_dry_run_v1"


def load_geox_cls_export(path: str | Path) -> dict[str, Any]:
    """Load offline GeoX/CLS export JSON (supports MIP-C3 fixture wrapper)."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"export file not found: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"export must be JSON object at {p}")
    if "export_bundle" in data and isinstance(data["export_bundle"], dict):
        return data["export_bundle"]
    return data


def adapt_export_to_signals(
    export: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str], dict[str, Any]]:
    """Adapt export bundle to CalibrationSignal list via MIP-C3 adapters."""
    return adapt_mixed_batch_export(export)


def _count_signal_dispositions(signals: list[dict[str, Any]]) -> dict[str, int]:
    blocked = inconclusive = stale = 0
    for sig in signals:
        if sig.get("eligibility_status") == "blocked":
            blocked += 1
        if sig.get("eligibility_status") == "inconclusive":
            inconclusive += 1
        if sig.get("freshness_status") == "stale":
            stale += 1
    return {
        "blocked_count": blocked,
        "inconclusive_count": inconclusive,
        "stale_count": stale,
    }


def build_etl_lineage(
    *,
    input_path: str | Path,
    output_path: str | Path,
    source_systems: list[str],
    records_seen: int,
    signals_written: int,
    adapter_errors: list[str],
    adapter_lineage: dict[str, Any],
    disposition_counts: dict[str, int],
) -> dict[str, Any]:
    """Build governed ETL lineage block for dry-run artifact."""
    inp = Path(input_path)
    content_hash = hashlib.sha256(inp.read_bytes()).hexdigest() if inp.is_file() else None
    return {
        "etl_version": ETL_VERSION,
        "adapter_version": ADAPTER_VERSION,
        "input_source": str(inp),
        "input_content_sha256": content_hash,
        "output_path": str(output_path),
        "source_systems": sorted(set(source_systems)),
        "records_seen": records_seen,
        "signals_written": signals_written,
        "adapter_errors": list(adapter_errors),
        "adapter_lineage": adapter_lineage,
        **disposition_counts,
        "context_only": True,
        "optimizer_unchanged": True,
        "decision_surface_unchanged": True,
        "recommendations_unchanged": True,
        "live_api": False,
        "dry_run": True,
        "note": "Dry-run ETL only — not production scheduling or evidence approval.",
    }


def build_signal_artifact(
    signals: list[dict[str, Any]],
    *,
    etl_lineage: dict[str, Any],
    source_ref: str | None = None,
) -> dict[str, Any]:
    """Assemble full dry-run artifact envelope."""
    return {
        "artifact_type": ETL_ARTIFACT_TYPE,
        "artifact_version": ETL_VERSION,
        "source_ref": source_ref or etl_lineage.get("input_source"),
        "signals": signals,
        "etl_lineage": etl_lineage,
        "production_flags": {
            "approved_for_prod": False,
            "decisioning_allowed": False,
            "bayes_h5_research_only": True,
            "etl_is_dry_run": True,
        },
    }


def validate_signal_artifact(artifact: dict[str, Any]) -> tuple[bool, list[str]]:
    """Validate dry-run artifact is C2-consumable and decision-safe."""
    errors: list[str] = []
    if artifact.get("artifact_type") != ETL_ARTIFACT_TYPE:
        errors.append(f"artifact_type must be {ETL_ARTIFACT_TYPE}")
    signals = artifact.get("signals")
    if not isinstance(signals, list):
        errors.append("signals must be array")
    else:
        valid, parse_errors = parse_calibration_signals_payload({"signals": signals})
        if parse_errors:
            errors.extend(parse_errors)
        if len(valid) != len(signals):
            errors.append("one or more signals failed C2 ingest validation")
        for i, sig in enumerate(signals):
            if not isinstance(sig, dict):
                errors.append(f"signals[{i}] must be object")
                continue
            ok, val_errs = validate_adapter_output(sig)
            if not ok:
                errors.extend([f"signals[{i}]:{e}" for e in val_errs])
            for key in FORBIDDEN_OUTPUT_FIELDS | FORBIDDEN_ATTACHMENT_OUTPUT_FIELDS:
                if key in sig:
                    errors.append(f"signals[{i}]:forbidden_field:{key}")
    lineage = artifact.get("etl_lineage")
    if not isinstance(lineage, dict):
        errors.append("etl_lineage must be object")
    else:
        for req in (
            "input_source",
            "source_systems",
            "records_seen",
            "signals_written",
            "context_only",
            "optimizer_unchanged",
        ):
            if req not in lineage:
                errors.append(f"etl_lineage missing {req}")
        if lineage.get("context_only") is not True:
            errors.append("etl_lineage.context_only must be true")
    flags = artifact.get("production_flags") or {}
    if flags.get("approved_for_prod") is not False:
        errors.append("production_flags.approved_for_prod must be false")
    if flags.get("decisioning_allowed") is not False:
        errors.append("production_flags.decisioning_allowed must be false")
    return len(errors) == 0, errors


def write_signal_artifact(artifact: dict[str, Any], output_path: str | Path) -> Path:
    """Write validated artifact JSON."""
    ok, errors = validate_signal_artifact(artifact)
    if not ok:
        raise ValueError(f"artifact validation failed: {errors}")
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2, default=str), encoding="utf-8")
    return out


def run_dry_run_etl(input_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    """Execute dry-run ETL: load export → adapt → validate → write."""
    export = load_geox_cls_export(input_path)
    signals, adapter_errors, adapter_lineage = adapt_export_to_signals(export)

    records_seen = int(adapter_lineage.get("geox_count", 0)) + int(
        adapter_lineage.get("cls_count", 0)
    )
    if "records" in export and isinstance(export["records"], list):
        records_seen = max(records_seen, len(export["records"]))
    for key in ("geox", "cls"):
        if isinstance(export.get(key), list):
            records_seen = max(records_seen, len(export[key]))

    source_systems: list[str] = []
    if adapter_lineage.get("geox_count"):
        source_systems.append("geox")
    if adapter_lineage.get("cls_count"):
        source_systems.append("cls")
    for sig in signals:
        sys_name = sig.get("source_system")
        if sys_name:
            source_systems.append(str(sys_name))

    disposition = _count_signal_dispositions(signals)
    lineage = build_etl_lineage(
        input_path=input_path,
        output_path=output_path,
        source_systems=source_systems,
        records_seen=records_seen,
        signals_written=len(signals),
        adapter_errors=adapter_errors,
        adapter_lineage=adapter_lineage,
        disposition_counts=disposition,
    )
    artifact = build_signal_artifact(
        signals,
        etl_lineage=lineage,
        source_ref=export.get("source_ref"),
    )
    write_signal_artifact(artifact, output_path)
    return artifact


def prove_c2_ingest(artifact: dict[str, Any], ridge_report_stub: dict[str, Any] | None = None) -> dict[str, Any]:
    """Prove artifact signals ingest through MIP-C2 (diagnostic context only)."""
    stub = ridge_report_stub or {
        "coefficient_stability": {
            "media_coef_by_channel": {"search": 0.05, "social": 0.03, "tv": 0.02}
        },
        "forbidden_claims": [],
        "world_metadata": {"estimand_id": "incremental_sales"},
    }
    return ingest_calibration_signals_into_report(stub, signals=artifact.get("signals"))


def build_train_consumption_archive(
    extension_report: dict[str, Any],
    *,
    etl_artifact_path: str,
    bundle_id: str = "MIP-C4-DRY-RUN-TRAIN-PROOF",
) -> dict[str, Any]:
    """Redacted archive proving train consumed ETL artifact (diagnostic only)."""
    report = extension_report.get("ridge_production_diagnostics_report") or {}
    redacted = json.loads(json.dumps(report, default=str))
    coef = (redacted.get("coefficient_stability") or {}).get("media_coef_by_channel")
    if isinstance(coef, dict):
        for ch in coef:
            coef[ch] = "[redacted]"
    return {
        "milestone": "MIP-C4",
        "bundle_id": bundle_id,
        "etl_artifact_path": etl_artifact_path,
        "ridge_production_diagnostics_report": redacted,
        "evidence_attachment_lineage": redacted.get("evidence_attachment_lineage"),
        "calibration_evidence_context_present": bool(redacted.get("calibration_evidence_context")),
        "production_boundary": {
            "ridge_fitting_unchanged": True,
            "optimizer_unchanged": True,
            "decision_surface_unchanged": True,
            "context_only": True,
        },
    }


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="MIP-C4 dry-run: GeoX/CLS export → C2 CalibrationSignal JSON artifact."
    )
    parser.add_argument("--input", required=True, help="Path to GeoX/CLS export JSON")
    parser.add_argument("--output", required=True, help="Path to write signals artifact JSON")
    args = parser.parse_args(argv)
    try:
        artifact = run_dry_run_etl(args.input, args.output)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ETL dry-run failed: {exc}", file=sys.stderr)
        return 1
    lineage = artifact["etl_lineage"]
    print(
        f"Wrote {args.output}: {lineage['signals_written']} signal(s) "
        f"from {lineage['records_seen']} record(s); "
        f"blocked={lineage.get('blocked_count', 0)} "
        f"inconclusive={lineage.get('inconclusive_count', 0)} "
        f"stale={lineage.get('stale_count', 0)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
