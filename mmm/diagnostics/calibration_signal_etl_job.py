"""MIP-C5 — Scheduled drop-zone wrapper for CalibrationSignal ETL (file workflow only)."""

from __future__ import annotations

import argparse
import fnmatch
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mmm.diagnostics.calibration_signal_etl import (
    ETL_ARTIFACT_TYPE,
    build_train_consumption_archive,
    prove_c2_ingest,
    run_dry_run_etl,
    validate_signal_artifact,
)

JOB_VERSION = "mip_calibration_signal_etl_job_v1"
MANIFEST_ARTIFACT_TYPE = "calibration_signal_etl_job_manifest"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def discover_export_files(input_dir: str | Path, pattern: str) -> list[Path]:
    """Discover export JSON files in drop-zone directory matching pattern."""
    root = Path(input_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"input_dir not found: {root}")

    if any(ch in pattern for ch in "*?[]"):
        matches = sorted(root.glob(pattern))
    else:
        candidate = root / pattern
        matches = [candidate] if candidate.is_file() else []

    return [p for p in matches if p.is_file() and p.suffix.lower() == ".json"]


def _output_signals_path(output_dir: Path, run_id: str, input_path: Path) -> Path:
    stem = input_path.stem.replace(" ", "_")
    return output_dir / f"{run_id}_{stem}_signals.json"


def run_etl_for_file(
    input_path: str | Path,
    output_dir: str | Path,
    run_id: str,
) -> dict[str, Any]:
    """Run MIP-C4 ETL for one export file; return per-file result row."""
    inp = Path(input_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = _output_signals_path(out_dir, run_id, inp)
    row: dict[str, Any] = {
        "input_path": str(inp),
        "output_path": str(out_path),
        "success": False,
        "error": None,
        "records_seen": 0,
        "signals_written": 0,
        "blocked_count": 0,
        "inconclusive_count": 0,
        "stale_count": 0,
    }
    try:
        artifact = run_dry_run_etl(inp, out_path)
        lineage = artifact.get("etl_lineage") or {}
        row.update(
            {
                "success": True,
                "records_seen": lineage.get("records_seen", 0),
                "signals_written": lineage.get("signals_written", 0),
                "blocked_count": lineage.get("blocked_count", 0),
                "inconclusive_count": lineage.get("inconclusive_count", 0),
                "stale_count": lineage.get("stale_count", 0),
                "source_systems": lineage.get("source_systems", []),
            }
        )
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        row["error"] = str(exc)
    return row


def write_job_manifest(manifest: dict[str, Any], output_dir: str | Path, run_id: str) -> Path:
    """Write job manifest JSON to output directory."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{run_id}_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return path


def validate_job_outputs(manifest: dict[str, Any], output_dir: str | Path) -> tuple[bool, list[str]]:
    """Validate manifest and each listed output artifact."""
    errors: list[str] = []
    if manifest.get("artifact_type") != MANIFEST_ARTIFACT_TYPE:
        errors.append(f"manifest artifact_type must be {MANIFEST_ARTIFACT_TYPE}")
    for req in (
        "run_id",
        "input_dir",
        "output_dir",
        "files_seen",
        "files_processed",
        "context_only",
        "live_api_used",
        "production_scheduler",
    ):
        if req not in manifest:
            errors.append(f"manifest missing {req}")
    if manifest.get("live_api_used") is not False:
        errors.append("live_api_used must be false")
    if manifest.get("production_scheduler") is not False:
        errors.append("production_scheduler must be false")

    for entry in manifest.get("output_artifacts") or []:
        if not entry.get("success"):
            continue
        out_path = entry.get("output_path")
        if not out_path:
            errors.append("output_artifact missing output_path")
            continue
        p = Path(out_path)
        if not p.is_file():
            errors.append(f"output file missing: {p}")
            continue
        try:
            artifact = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            errors.append(f"invalid json {p}: {exc}")
            continue
        ok, art_errs = validate_signal_artifact(artifact)
        if not ok:
            errors.extend([f"{p.name}:{e}" for e in art_errs])
    return len(errors) == 0, errors


def summarize_job_run(manifest: dict[str, Any]) -> str:
    """Markdown summary for operators."""
    lines = [
        f"# CalibrationSignal ETL job — {manifest.get('run_id')}",
        "",
        f"**Started:** {manifest.get('started_at')}",
        f"**Completed:** {manifest.get('completed_at')}",
        f"**Input dir:** `{manifest.get('input_dir')}`",
        f"**Pattern:** `{manifest.get('pattern')}`",
        f"**Output dir:** `{manifest.get('output_dir')}`",
        "",
        "## Counts",
        f"- Files seen: {manifest.get('files_seen')}",
        f"- Files processed: {manifest.get('files_processed')}",
        f"- Files failed: {manifest.get('files_failed')}",
        f"- Records seen: {manifest.get('records_seen')}",
        f"- Signals written: {manifest.get('signals_written')}",
        f"- Blocked: {manifest.get('blocked_count')}",
        f"- Inconclusive: {manifest.get('inconclusive_count')}",
        f"- Stale: {manifest.get('stale_count')}",
        "",
        "## Boundaries",
        f"- Context only: **{manifest.get('context_only')}**",
        f"- Live API used: **{manifest.get('live_api_used')}**",
        f"- Production scheduler: **{manifest.get('production_scheduler')}**",
        "",
        "## Output artifacts",
    ]
    for entry in manifest.get("output_artifacts") or []:
        status = "ok" if entry.get("success") else "FAILED"
        lines.append(f"- [{status}] `{entry.get('input_path')}` → `{entry.get('output_path')}`")
        if entry.get("error"):
            lines.append(f"  - error: {entry['error']}")
    errors = manifest.get("errors") or []
    lines.extend(["", "## Errors"])
    if errors:
        for err in errors:
            lines.append(f"- {err}")
    else:
        lines.append("- none")
    lines.append("")
    lines.append("_Scheduled drop-zone workflow only — not production deployment or evidence approval._")
    return "\n".join(lines) + "\n"


def run_scheduled_etl_job(
    *,
    input_dir: str | Path,
    output_dir: str | Path,
    run_id: str,
    pattern: str = "*.json",
    continue_on_error: bool = True,
) -> dict[str, Any]:
    """Execute drop-zone ETL for all matching exports; write manifest + summary."""
    started_at = _utc_now_iso()
    input_root = Path(input_dir)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    discovered = discover_export_files(input_root, pattern)
    file_results: list[dict[str, Any]] = []
    job_errors: list[str] = []

    for path in discovered:
        row = run_etl_for_file(path, output_root, run_id)
        file_results.append(row)
        if not row["success"]:
            job_errors.append(f"{path.name}:{row['error']}")
            if not continue_on_error:
                break

    files_processed = sum(1 for r in file_results if r["success"])
    files_failed = sum(1 for r in file_results if not r["success"])

    manifest: dict[str, Any] = {
        "artifact_type": MANIFEST_ARTIFACT_TYPE,
        "job_version": JOB_VERSION,
        "run_id": run_id,
        "started_at": started_at,
        "completed_at": _utc_now_iso(),
        "input_dir": str(input_root.resolve()),
        "output_dir": str(output_root.resolve()),
        "pattern": pattern,
        "files_seen": len(discovered),
        "files_processed": files_processed,
        "files_failed": files_failed,
        "records_seen": sum(r.get("records_seen", 0) for r in file_results),
        "signals_written": sum(r.get("signals_written", 0) for r in file_results),
        "blocked_count": sum(r.get("blocked_count", 0) for r in file_results),
        "inconclusive_count": sum(r.get("inconclusive_count", 0) for r in file_results),
        "stale_count": sum(r.get("stale_count", 0) for r in file_results),
        "errors": job_errors,
        "output_artifacts": file_results,
        "context_only": True,
        "optimizer_unchanged": True,
        "decision_surface_unchanged": True,
        "recommendations_unchanged": True,
        "live_api_used": False,
        "production_scheduler": False,
        "continue_on_error": continue_on_error,
        "production_flags": {
            "approved_for_prod": False,
            "decisioning_allowed": False,
            "bayes_h5_research_only": True,
        },
    }

    manifest_path = write_job_manifest(manifest, output_root, run_id)
    summary_path = output_root / f"{run_id}_summary.md"
    summary_path.write_text(summarize_job_run(manifest), encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path)
    manifest["summary_path"] = str(summary_path)
    return manifest


def prove_train_consumption(
    signals_path: str | Path,
    *,
    config_path: str = "examples/minimal_train.yaml",
) -> dict[str, Any]:
    """Run attach-only train proof using ETL output (full train optional)."""
    from mmm.config.load import load_config
    from mmm.data.loader import DatasetBuilder
    from mmm.diagnostics.ridge_diagnostics import attach_ridge_diagnostics_to_extension_report
    from mmm.models.ridge_bo.trainer import RidgeBOMMMTrainer

    artifact = json.loads(Path(signals_path).read_text(encoding="utf-8"))
    config = load_config(config_path)
    config.ridge_bo.n_trials = 2
    builder = DatasetBuilder(config.data)
    panel = builder.build()
    schema = builder.schema()
    trainer = RidgeBOMMMTrainer(config, schema)
    fit_out = trainer.fit(panel)
    fit_out["_ridge_trainer"] = trainer
    ext = attach_ridge_diagnostics_to_extension_report(
        {},
        panel,
        schema,
        config,
        fit_out,
        trainer=trainer,
        calibration_signals_path=str(signals_path),
    )
    return build_train_consumption_archive(
        ext,
        etl_artifact_path=str(signals_path),
        bundle_id="MIP-C5-SCHEDULED-ETL-TRAIN-PROOF",
    )


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="MIP-C5 scheduled drop-zone CalibrationSignal ETL (no live API)."
    )
    parser.add_argument("--input-dir", required=True, help="Drop-zone directory with export JSON files")
    parser.add_argument(
        "--pattern",
        default="*.json",
        help="Filename or glob pattern within input-dir (e.g. mixed_batch.json)",
    )
    parser.add_argument("--output-dir", required=True, help="Directory for signals JSON + manifest")
    parser.add_argument("--run-id", required=True, help="Run identifier for output filenames")
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop batch on first file failure (default: continue)",
    )
    args = parser.parse_args(argv)
    try:
        manifest = run_scheduled_etl_job(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            run_id=args.run_id,
            pattern=args.pattern,
            continue_on_error=not args.fail_fast,
        )
    except FileNotFoundError as exc:
        print(f"ETL job failed: {exc}", file=sys.stderr)
        return 1

    ok, val_errors = validate_job_outputs(manifest, args.output_dir)
    if not ok:
        print(f"Job output validation warnings: {val_errors}", file=sys.stderr)

    print(
        f"Job {args.run_id}: processed {manifest['files_processed']}/{manifest['files_seen']} file(s); "
        f"signals_written={manifest['signals_written']}; manifest={manifest['manifest_path']}"
    )
    return 0 if manifest["files_failed"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(_main())
