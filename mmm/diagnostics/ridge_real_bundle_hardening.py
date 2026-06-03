"""H11 — Ridge diagnostics on real/realistic training bundles (diagnostic hardening only)."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd

from mmm.config.schema import CVConfig, Framework, MMMConfig, ModelForm, PoolingMode, RunEnvironment
from mmm.data.schema import PanelSchema
from mmm.diagnostics.ridge_diagnostic_summary import (
    export_ridge_diagnostic_artifacts,
    format_ridge_diagnostics_cli_block,
    format_ridge_diagnostics_markdown,
    summarize_ridge_diagnostics,
)
from mmm.diagnostics.ridge_diagnostics import (
    FORBIDDEN_OUTPUT_FIELDS,
    attach_ridge_diagnostics_to_extension_report,
    compose_ridge_diagnostic_report,
)
from mmm.models.ridge_bo.trainer import RidgeBOMMMTrainer

H11_MILESTONE = "H11"
BUNDLE_BENCHMARK_GEO_PANEL_V1 = "MMM-BENCHMARK-GEO-PANEL-V1"

BENCHMARK_BUNDLE_SPEC: dict[str, Any] = {
    "bundle_id": BUNDLE_BENCHMARK_GEO_PANEL_V1,
    "panel_id": "examples_mmm_benchmark_geo_panel_v1",
    "dataset_snapshot_id": "mmm-examples-benchmark-geo-panel-frozen-2022-v1",
    "panel_path": "examples/benchmark_geo_panel_v1.csv",
    "vertical_id": "retail",
    "vertical_assumption": "Illustrative retail profile applied; panel has no control columns in schema.",
    "privacy_status": "public_in_repo_illustrative",
    "calibration_evidence_available": False,
    "calibration_signals": [],
    "expected_diagnostic_risks": [
        "omitted_required_retail_controls",
        "weak_identification_if_high_media_correlation",
        "no_calibration_evidence_context_on_default_run",
    ],
}


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_ridge_config_for_bundle(spec: dict[str, Any]) -> MMMConfig:
    """Research Ridge config shaped like prod template (reduced trials for CI)."""
    return MMMConfig(
        framework=Framework.RIDGE_BO,
        model_form=ModelForm.SEMI_LOG,
        pooling=PoolingMode.PARTIAL,
        random_seed=42,
        run_environment=RunEnvironment.RESEARCH,
        data={
            "path": spec["panel_path"],
            "geo_column": "geo_id",
            "week_column": "week_start_date",
            "target_column": "revenue",
            "channel_columns": list(spec.get("channel_columns") or ("search", "social", "tv")),
            "control_columns": list(spec.get("control_columns") or ()),
            "data_version_id": spec["dataset_snapshot_id"],
        },
        transforms={
            "adstock": "geometric",
            "saturation": "hill",
            "adstock_params": {"decay": 0.5},
            "saturation_params": {"half_max": 1.0, "slope": 2.0},
        },
        cv=CVConfig(mode="auto", n_splits=3, min_train_weeks=20, horizon_weeks=4),
        ridge_bo={"n_trials": int(spec.get("n_trials", 4)), "sampler_seed": 1},
        calibration={"enabled": False},
    )


def load_bundle_panel(spec: dict[str, Any]) -> tuple[pd.DataFrame, PanelSchema, MMMConfig]:
    path = Path(spec["panel_path"])
    if not path.is_file():
        raise FileNotFoundError(f"H11 bundle panel not found: {path}")
    config = build_ridge_config_for_bundle(spec)
    panel = pd.read_csv(path)
    panel[config.data.week_column] = pd.to_datetime(panel[config.data.week_column])
    schema = PanelSchema(
        geo_column=config.data.geo_column,
        week_column=config.data.week_column,
        target_column=config.data.target_column,
        channel_columns=tuple(config.data.channel_columns),
        control_columns=tuple(config.data.control_columns),
    )
    return panel, schema, config


def bundle_panel_lineage(panel: pd.DataFrame, schema: PanelSchema, spec: dict[str, Any]) -> dict[str, Any]:
    week_col = schema.week_column
    dates = panel[week_col]
    return {
        "bundle_id": spec["bundle_id"],
        "panel_id": spec.get("panel_id"),
        "dataset_snapshot_id": spec["dataset_snapshot_id"],
        "panel_path": spec["panel_path"],
        "panel_content_sha256": _file_sha256(Path(spec["panel_path"])),
        "vertical_id": spec.get("vertical_id"),
        "vertical_assumption": spec.get("vertical_assumption"),
        "date_range": {
            "start": str(dates.min().date()),
            "end": str(dates.max().date()),
        },
        "geo_grain": schema.geo_column,
        "geo_count": int(panel[schema.geo_column].nunique()),
        "row_count": int(len(panel)),
        "outcome_column": schema.target_column,
        "media_columns": list(schema.channel_columns),
        "control_columns": list(schema.control_columns),
        "privacy_status": spec.get("privacy_status"),
        "calibration_evidence_available_config": bool(spec.get("calibration_evidence_available")),
        "calibration_signals_declared": list(spec.get("calibration_signals") or []),
    }


def run_real_bundle_ridge_diagnostics(
    spec: dict[str, Any],
    *,
    fit_result_override: dict[str, Any] | None = None,
    skip_fit: bool = False,
) -> dict[str, Any]:
    """Fit Ridge (unless skipped) and compose full diagnostic + extension artifacts."""
    panel, schema, config = load_bundle_panel(spec)
    lineage = bundle_panel_lineage(panel, schema, spec)

    fit_out: dict[str, Any] = fit_result_override or {}
    trainer: RidgeBOMMMTrainer | None = None
    if not skip_fit:
        trainer = RidgeBOMMMTrainer(config, schema)
        fit_out = trainer.fit(panel)
        fit_out["_ridge_trainer"] = trainer

    world_metadata = {
        "h11_bundle_id": spec["bundle_id"],
        "h11_milestone": H11_MILESTONE,
        "panel_lineage": lineage,
        "synthetic_world": False,
    }
    report = compose_ridge_diagnostic_report(
        panel,
        schema,
        config,
        fit_out,
        trainer=trainer,
        vertical_id=spec.get("vertical_id"),
        model_id="ridge_bo",
        run_id=spec["dataset_snapshot_id"],
        dataset_snapshot_id=spec["dataset_snapshot_id"],
        calibration_evidence_available=bool(spec.get("calibration_evidence_available")),
        world_metadata=world_metadata,
    )
    extension_report = attach_ridge_diagnostics_to_extension_report(
        {"governance_stub": True, "h11_bundle_id": spec["bundle_id"]},
        panel,
        schema,
        config,
        fit_out,
        trainer=trainer,
        vertical_id=spec.get("vertical_id"),
        calibration_evidence_available=bool(spec.get("calibration_evidence_available")),
    )
    summary = summarize_ridge_diagnostics(report)
    extension_report["ridge_production_diagnostics_summary"] = summary
    markdown = format_ridge_diagnostics_markdown(report, summary=summary)
    cli_lines = format_ridge_diagnostics_cli_block(report)

    return {
        "bundle_id": spec["bundle_id"],
        "lineage": lineage,
        "report": report,
        "extension_report": extension_report,
        "summary": summary,
        "markdown": markdown,
        "cli_lines": cli_lines,
        "completeness": validate_h11_artifact_completeness(report, extension_report),
    }


def validate_h11_artifact_completeness(
    report: dict[str, Any],
    extension_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """H11 checklist — explicit pass/fail per required artifact field."""
    ext = extension_report or {}
    embedded = ext.get("ridge_production_diagnostics_report") or {}
    lineage = report.get("evidence_attachment_lineage") or {}
    standalone = extension_report is None
    checks: dict[str, bool] = {
        "ridge_production_diagnostics_report_exists": bool(report),
        "severity_present": "severity" in report,
        "output_eligibility_present": "output_eligibility" in report,
        "transform_diagnostics_present": "transform_diagnostics" in report,
        "vertical_controls_evaluated": "control_completeness" in report,
        "sparse_channels_evaluated": "sparse_channels" in report,
        "collinearity_evaluated": "collinearity" in report,
        "forbidden_claims_present": "forbidden_claims" in report,
        "evidence_attachment_lineage_present": bool(lineage),
        "calibration_context_absence_explicit": lineage.get("calibration_evidence_context_present") is False
        or lineage.get("mip_c1_attachment_wired") is True,
        "extension_report_embeds_report": standalone or bool(embedded),
        "extension_report_embeds_summary": standalone
        or "ridge_production_diagnostics_summary" in ext
        or embedded.get("severity") is not None,
        "no_optimizer_fields": not any(k in report for k in FORBIDDEN_OUTPUT_FIELDS),
    }
    transform = report.get("transform_diagnostics") or {}
    checks["transform_metadata_rendered"] = "transform_config" in transform and "warnings" in transform
    checks["transform_warnings_when_incomplete"] = (
        transform.get("metadata_complete") is True
        or "ridge_transform:missing_best_params_metadata" in (transform.get("warnings") or [])
        or bool(transform.get("warnings"))
    )
    return {
        "all_passed": all(checks.values()),
        "checks": checks,
    }


def redact_report_for_archive(report: dict[str, Any]) -> dict[str, Any]:
    """Redact coefficient magnitudes for archive; keep diagnostic structure."""
    out = deepcopy(report)
    coef = out.get("coefficient_stability") or {}
    if isinstance(coef.get("media_coef_by_channel"), dict):
        coef["media_coef_by_channel"] = {
            ch: "[redacted]" for ch in coef["media_coef_by_channel"]
        }
        coef["redaction_note"] = "H11 archive — coef signs/magnitudes withheld; structure retained."
        out["coefficient_stability"] = coef
    return out


def write_h11_archive_artifacts(
    result: dict[str, Any],
    *,
    archive_dir: Path | str = "docs/05_validation/archives",
    date_suffix: str = "20260601",
) -> dict[str, str]:
    """Write H11 JSON + Markdown archives for a bundle run."""
    archive_dir = Path(archive_dir)
    archive_dir.mkdir(parents=True, exist_ok=True)
    bundle_id = result["bundle_id"]
    safe_id = bundle_id.replace("-", "_")
    json_path = archive_dir / f"H11_RIDGE_DIAGNOSTICS_REAL_BUNDLE_{safe_id}_{date_suffix}.json"
    md_path = archive_dir / f"H11_RIDGE_DIAGNOSTICS_REAL_BUNDLE_{safe_id}_SUMMARY_{date_suffix}.md"

    payload = {
        "h11_milestone": H11_MILESTONE,
        "bundle_id": bundle_id,
        "lineage": result["lineage"],
        "completeness": result["completeness"],
        "ridge_production_diagnostics_report": redact_report_for_archive(result["report"]),
        "ridge_production_diagnostics_summary": result["summary"],
        "cli_lines": result["cli_lines"],
        "production_boundary": {
            "ridge_fitting_unchanged": True,
            "optimizer_unchanged": True,
            "decision_surface_unchanged": True,
            "mip_c1_context_only": True,
            "bayes_h5_research_only": True,
        },
    }
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    md_path.write_text(result["markdown"], encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path)}


def run_h11_benchmark_bundle_archive() -> dict[str, Any]:
    """Default H11 entry: benchmark geo panel v1."""
    result = run_real_bundle_ridge_diagnostics(BENCHMARK_BUNDLE_SPEC)
    paths = write_h11_archive_artifacts(result)
    result["archive_paths"] = paths
    return result
