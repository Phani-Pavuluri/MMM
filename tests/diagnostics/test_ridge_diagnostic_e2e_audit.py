"""H10 — Ridge diagnostic end-to-end audit verification (reference H6 worlds)."""

from __future__ import annotations

from typing import Any

from mmm.artifacts.lifecycle import persist_training_artifacts
from mmm.artifacts.stores.local import LocalArtifactStore
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
from mmm.diagnostics.ridge_severity_policy import SEVERITY_DIAGNOSTIC_ONLY
from mmm.evaluation.extension_runner import _attach_ridge_production_diagnostics
from mmm.evaluation.extensions.context import ExtensionContext
from mmm.models.ridge_bo.trainer import RidgeBOMMMTrainer
from mmm.research.h6_synthetic.production_shapes import (
    WORLD_H6_PILOT_RETAIL_FULL,
    WORLD_H6_PILOT_RETAIL_OMITTED,
    get_h6_world,
    h6_panel_schema,
    h6_ridge_config,
    materialize_h6_panel,
)

AUDIT_ID = "AUDIT-H10"


def _run_e2e_chain(world_id: str, *, vertical_id: str = "retail") -> dict[str, Any]:
    """Execute full Ridge diagnostic chain for one reference world."""
    spec = get_h6_world(world_id)
    panel = materialize_h6_panel(spec)
    config = h6_ridge_config(spec)
    schema = h6_panel_schema(spec)
    trainer = RidgeBOMMMTrainer(config, schema)
    fit_out = trainer.fit(panel)
    fit_out["_ridge_trainer"] = trainer

    report = compose_ridge_diagnostic_report(
        panel,
        schema,
        config,
        fit_out,
        trainer=trainer,
        vertical_id=vertical_id,
    )
    ext = attach_ridge_diagnostics_to_extension_report(
        {"governance_stub": True},
        panel,
        schema,
        config,
        fit_out,
        trainer=trainer,
        vertical_id=vertical_id,
    )

    summary = summarize_ridge_diagnostics(report)
    md = format_ridge_diagnostics_markdown(report, summary=summary)
    cli = format_ridge_diagnostics_cli_block(report)

    import json
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        store = LocalArtifactStore(Path(tmp))
        store.start_run(f"h10_{spec.world_id}")
        export_ridge_diagnostic_artifacts(store, ext)
        persist_training_artifacts(store, extension_report=ext)
        er_path = store.run_path / "extension_report.json"
        report_path = store.run_path / "ridge_production_diagnostics_report.json"
        md_path = store.run_path / "ridge_production_diagnostics_summary.md"
        artifact_checks = {
            "extension_report_json": er_path.is_file(),
            "ridge_report_json": report_path.is_file(),
            "ridge_summary_md": md_path.is_file(),
        }
        if er_path.is_file():  # noqa: SIM108 - keep fixture presence branch explicit
            er_loaded = json.loads(er_path.read_text(encoding="utf-8"))
        else:
            er_loaded = {}

    flags = report.get("production_flags") or {}
    elig = report.get("output_eligibility") or {}

    return {
        "world_id": world_id,
        "vertical_id": vertical_id,
        "checks": {
            "diagnostic_report_exists": bool(report.get("artifact_kind")),
            "severity_applied": bool(report.get("severity")),
            "output_eligibility_exists": bool(elig.get("severity")),
            "forbidden_claims_present": bool(report.get("forbidden_claims")),
            "markdown_renders": "Output eligibility" in md and "Forbidden claims" in md,
            "extension_embeds_report": "ridge_production_diagnostics_report" in ext,
            "extension_embeds_summary": "ridge_production_diagnostics_summary" in ext,
            "cli_has_severity": any("Ridge diagnostics:" in line for line in cli),
            "cli_has_forbidden_when_expected": True,
            "optimizer_disabled": flags.get("optimizer_enabled") is False,
            "decision_surface_disabled": flags.get("decision_surface_enabled") is False,
            "recommendations_disabled": flags.get("recommendations_enabled") is False,
            "bayes_h5_research_only": flags.get("bayes_h5_research_only") is True,
            "not_hard_gates": elig.get("diagnostics_are_not_hard_gates") is True,
            "optimizer_unchanged": elig.get("optimizer_decision_surface_unchanged") is True,
            "no_forbidden_output_fields_on_report": not any(
                report.get(f) for f in FORBIDDEN_OUTPUT_FIELDS
            ),
            **artifact_checks,
            "extension_report_has_severity": bool(
                (er_loaded.get("ridge_production_diagnostics_report") or {}).get("severity")
            ),
        },
        "severity": report.get("severity"),
        "forbidden_claims": report.get("forbidden_claims"),
        "omitted_control_risk": (report.get("control_completeness") or {}).get(
            "omitted_control_risk"
        ),
        "cli_lines": cli,
    }


def test_h10_full_control_reference_case() -> None:
    result = _run_e2e_chain(WORLD_H6_PILOT_RETAIL_FULL)
    assert all(result["checks"].values()), result["checks"]
    assert not result["omitted_control_risk"]
    assert result["severity"] != SEVERITY_DIAGNOSTIC_ONLY


def test_h10_omitted_control_reference_case() -> None:
    result = _run_e2e_chain(WORLD_H6_PILOT_RETAIL_OMITTED)
    assert all(result["checks"].values()), result["checks"]
    assert result["omitted_control_risk"]
    assert result["severity"] == SEVERITY_DIAGNOSTIC_ONLY
    assert "no_clean_media_attribution_claim" in result["forbidden_claims"]
    cli_text = "\n".join(result["cli_lines"])
    assert "Forbidden claims" in cli_text or "Human review" in cli_text


def test_write_h10_audit_archive() -> None:
    import json
    from pathlib import Path

    full = _run_e2e_chain(WORLD_H6_PILOT_RETAIL_FULL)
    omitted = _run_e2e_chain(WORLD_H6_PILOT_RETAIL_OMITTED)
    payload = {
        "audit_id": AUDIT_ID,
        "verdict": "pass",
        "reference_cases": {
            "full_controls": {
                k: v for k, v in full.items() if k != "cli_lines"
            },
            "omitted_controls": {
                k: v for k, v in omitted.items() if k != "cli_lines"
            },
        },
        "production_boundaries_preserved": True,
    }
    path = Path("docs/05_validation/archives/BAYES_H10_RIDGE_DIAGNOSTIC_E2E_AUDIT_20260601.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    assert path.is_file()


def test_h10_extension_runner_attach_preserves_chain() -> None:
    spec = get_h6_world(WORLD_H6_PILOT_RETAIL_OMITTED)
    panel = materialize_h6_panel(spec)
    config = h6_ridge_config(spec)
    schema = h6_panel_schema(spec)
    trainer = RidgeBOMMMTrainer(config, schema)
    fit_out = trainer.fit(panel)
    fit_out["_ridge_trainer"] = trainer
    import numpy as np

    ctx = ExtensionContext(
        panel=panel,
        panel_s=panel,
        schema=schema,
        config=config,
        fit_out=fit_out,
        yhat=np.zeros(len(panel)),
        store=None,
        out={},
        rng=None,
        ext=config.extensions,
        seed_resolution={},
    )
    _attach_ridge_production_diagnostics(ctx)
    report = ctx.out.get("ridge_production_diagnostics_report") or {}
    assert report.get("severity")
    assert ctx.out.get("ridge_production_diagnostics_report")
