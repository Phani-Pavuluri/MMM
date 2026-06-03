"""H8 — operator-facing Ridge diagnostic summaries and artifact export."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mmm.artifacts.base import ArtifactStoreBase
from mmm.diagnostics.calibration_signal_attachment import build_calibration_evidence_summary

FORBIDDEN_SUMMARY_TERMS = (
    "optimizer",
    "decision_surface",
    "decision surface",
    "budget_recommendation",
    "budget recommendation",
    "recommendations_enabled",
)


def extract_forbidden_claims(report: dict[str, Any] | None) -> list[str]:
    if not report:
        return []
    return list(report.get("forbidden_claims") or [])


def extract_top_warnings(report: dict[str, Any] | None, *, limit: int = 12) -> list[str]:
    if not report:
        return []
    warnings = list(report.get("warnings") or [])
    for key in (
        "transform_diagnostics",
        "control_completeness",
        "collinearity",
        "sparse_channels",
        "fold_stability",
        "coefficient_stability",
        "sign_plausibility",
        "response_curve_plausibility",
    ):
        block = report.get(key) or {}
        if isinstance(block, dict):
            warnings.extend(block.get("warnings") or [])
    deduped: list[str] = []
    seen: set[str] = set()
    for w in warnings:
        if w not in seen:
            seen.add(w)
            deduped.append(w)
    return deduped[:limit]


def severity_badge(report: dict[str, Any] | None) -> str:
    if not report or report.get("status") == "unavailable":
        return "UNAVAILABLE"
    policy = str(report.get("severity") or "").lower()
    policy_badges = {
        "clean": "OK",
        "info": "INFO",
        "warning": "WARNING",
        "restricted_interpretation": "RESTRICTED",
        "diagnostic_only": "DIAGNOSTIC_ONLY",
        "blocked_for_decision_use": "BLOCKED",
    }
    if policy in policy_badges:
        return policy_badges[policy]
    severity = str(report.get("diagnostic_severity") or "none").lower()
    return {
        "none": "OK",
        "low": "INFO",
        "medium": "WARNING",
        "high": "CRITICAL",
    }.get(severity, severity.upper())


def summarize_ridge_diagnostics(report: dict[str, Any] | None) -> dict[str, Any]:
    """Structured operator summary from ``ridge_production_diagnostics_report``."""
    if not report or report.get("status") == "unavailable":
        return {
            "status": "unavailable",
            "severity_badge": "UNAVAILABLE",
            "overall_severity": None,
        }

    control = report.get("control_completeness") or {}
    collinearity = report.get("collinearity") or {}
    sparse = report.get("sparse_channels") or {}
    transform = report.get("transform_diagnostics") or {}
    fold = report.get("fold_stability") or {}
    coef = report.get("coefficient_stability") or {}
    flags = report.get("production_flags") or {}
    eligibility = report.get("output_eligibility") or {}

    sparse_extreme = list(sparse.get("sparse_channel_extreme") or [])
    collinear_groups = [
        g.get("channels") for g in (collinearity.get("collinear_channel_groups") or []) if g.get("channels")
    ]

    evidence_lineage = report.get("evidence_attachment_lineage") or {}

    return {
        "status": "ok",
        "severity_badge": severity_badge(report),
        "severity": report.get("severity"),
        "overall_severity": report.get("severity") or report.get("diagnostic_severity"),
        "legacy_diagnostic_severity": report.get("diagnostic_severity"),
        "allowed_uses": list(eligibility.get("allowed_uses") or []),
        "forbidden_uses": list(eligibility.get("forbidden_uses") or []),
        "human_review_required": bool(eligibility.get("human_review_required")),
        "diagnostic_only_reason": eligibility.get("diagnostic_only_reason"),
        "classification_triggers": list(eligibility.get("classification_triggers") or []),
        "run_id": report.get("run_id"),
        "dataset_snapshot_id": report.get("dataset_snapshot_id"),
        "vertical_id": control.get("vertical_id"),
        "missing_required_controls": list(control.get("missing_required_controls") or []),
        "missing_controls": list(control.get("missing_controls") or []),
        "omitted_control_risk": bool(control.get("omitted_control_risk")),
        "media_correlated_controls": bool(control.get("media_correlated_controls")),
        "sparse_channels_extreme": sparse_extreme,
        "sparse_near_zero_threshold": sparse.get("near_zero_threshold"),
        "max_abs_correlation": collinearity.get("max_abs_correlation"),
        "weak_identification_risk": bool(collinearity.get("weak_identification_risk")),
        "collinear_channel_groups": collinear_groups,
        "transform_metadata_complete": bool(transform.get("metadata_complete")),
        "transform_warnings": list(transform.get("warnings") or []),
        "selected_adstock_saturation": transform.get("selected_adstock_saturation"),
        "fold_stability_ok": fold.get("fold_stability_ok"),
        "geo_fold_rmse_mean": fold.get("geo_fold_rmse_mean"),
        "coefficient_stability_available": bool(coef.get("available")),
        "forbidden_claims": extract_forbidden_claims(report),
        "top_warnings": extract_top_warnings(report),
        "calibration_evidence_context_present": bool(
            evidence_lineage.get("calibration_evidence_context_present")
            or report.get("calibration_evidence_context")
        ),
        "mip_c1_attachment_wired": bool(evidence_lineage.get("mip_c1_attachment_wired")),
        "production_boundary": {
            "ridge_remains_production_baseline": flags.get("ridge_remains_production_baseline"),
            "bayes_h5_research_only": flags.get("bayes_h5_research_only"),
            "approved_for_prod": flags.get("approved_for_prod"),
            "optimizer_enabled": flags.get("optimizer_enabled"),
            "decision_surface_enabled": flags.get("decision_surface_enabled"),
            "recommendations_enabled": flags.get("recommendations_enabled"),
            "diagnostics_are_not_hard_gates": flags.get("diagnostics_are_not_hard_gates"),
        },
    }


def format_ridge_diagnostics_markdown(
    report: dict[str, Any] | None,
    *,
    summary: dict[str, Any] | None = None,
) -> str:
    """Human-readable Markdown summary for operator review."""
    summary = summary or summarize_ridge_diagnostics(report)
    if summary.get("status") == "unavailable":
        return "# Ridge production diagnostics\n\n**Status:** unavailable\n"

    lines = [
        "# Ridge production diagnostics",
        "",
        f"**Severity:** {summary['severity_badge']} (`{summary.get('severity') or summary.get('overall_severity')}`)",
        "",
        "## Output eligibility (H9)",
        f"- Allowed uses: {summary.get('allowed_uses') or 'none'}",
        f"- Forbidden uses: {summary.get('forbidden_uses') or 'none'}",
        f"- Human review required: **{summary.get('human_review_required')}**",
    ]
    if summary.get("diagnostic_only_reason"):
        lines.append(f"- Diagnostic-only reason: {summary['diagnostic_only_reason']}")
    if summary.get("classification_triggers"):
        lines.append(f"- Classification triggers: {summary['classification_triggers']}")
    lines.extend(
        [
        "",
        "## Control completeness",
        f"- Vertical: `{summary.get('vertical_id') or 'not specified'}`",
        f"- Missing required controls: {summary.get('missing_required_controls') or 'none'}",
        f"- Omitted control risk: **{summary.get('omitted_control_risk')}**",
        f"- Media-correlated controls: {summary.get('media_correlated_controls')}",
        "",
        "## Sparse channels",
        f"- Extreme sparse (near_zero ≥ {summary.get('sparse_near_zero_threshold')}): "
        f"{summary.get('sparse_channels_extreme') or 'none'}",
        "",
        "## Collinearity",
        f"- Max |ρ|: {summary.get('max_abs_correlation')}",
        f"- Weak identification risk: **{summary.get('weak_identification_risk')}**",
        f"- Correlated groups: {summary.get('collinear_channel_groups') or 'none'}",
        "",
        "## Transform",
        f"- Metadata complete: **{summary.get('transform_metadata_complete')}**",
        f"- Selected params: `{summary.get('selected_adstock_saturation')}`",
        ],
    )
    if summary.get("transform_warnings"):
        lines.append(f"- Transform warnings: {summary['transform_warnings']}")

    lines.extend(
        [
            "",
            "## Stability",
            f"- Fold stability OK: {summary.get('fold_stability_ok')}",
            f"- Geo-fold RMSE mean: {summary.get('geo_fold_rmse_mean')}",
            f"- Coefficient stability available: {summary.get('coefficient_stability_available')}",
            "",
            "## Forbidden claims",
        ]
    )
    for claim in summary.get("forbidden_claims") or []:
        lines.append(f"- `{claim}`")
    if not summary.get("forbidden_claims"):
        lines.append("- none")

    lines.extend(
        [
            "",
            "## Top warnings",
        ]
    )
    for w in summary.get("top_warnings") or []:
        lines.append(f"- {w}")
    if not summary.get("top_warnings"):
        lines.append("- none")

    evidence_lineage = (report or {}).get("evidence_attachment_lineage") or {}
    if evidence_lineage and not evidence_lineage.get("calibration_evidence_context_present"):
        lines.extend(
            [
                "",
                "## Calibration evidence (MIP-C1)",
                "- **CalibrationSignal context:** not attached on this run (explicit).",
                f"- Collinearity replay flag: `{evidence_lineage.get('collinearity_calibration_evidence_available')}`",
            ]
        )

    cal_ctx = (report or {}).get("calibration_evidence_context")
    if cal_ctx:
        cal_sum = cal_ctx.get("summary") or build_calibration_evidence_summary(cal_ctx)
        lines.extend(
            [
                "",
                "## Calibration evidence context (MIP-C1)",
                f"- {cal_sum.get('headline') or cal_ctx.get('headline')}",
                f"- Signals: {cal_sum.get('signal_count', 0)} "
                f"(aligned={cal_sum.get('aligned_count', 0)}, "
                f"conflicts={cal_sum.get('directional_conflict_count', 0)})",
                "- **Context only** — does not override Ridge coefficients or optimizer.",
            ]
        )
        for w in (cal_sum.get("warnings") or cal_ctx.get("warnings") or [])[:6]:
            lines.append(f"- {w}")

    boundary = summary.get("production_boundary") or {}
    lines.extend(
        [
            "",
            "## Production boundary",
            f"- Ridge remains production baseline: {boundary.get('ridge_remains_production_baseline')}",
            f"- Bayes-H5 research-only: {boundary.get('bayes_h5_research_only')}",
            f"- Diagnostics are not hard gates: {boundary.get('diagnostics_are_not_hard_gates')}",
            f"- Optimizer enabled: {boundary.get('optimizer_enabled')} (must be false)",
            f"- DecisionSurface enabled: {boundary.get('decision_surface_enabled')} (must be false)",
            f"- Recommendations enabled: {boundary.get('recommendations_enabled')} (must be false)",
            "",
            "_Diagnostic only — not budget optimization or Bayes promotion._",
        ]
    )
    return "\n".join(lines) + "\n"


def format_ridge_diagnostics_cli_block(report: dict[str, Any] | None) -> list[str]:
    """Short CLI lines for ``mmm train`` operator printout."""
    summary = summarize_ridge_diagnostics(report)
    if summary.get("status") == "unavailable":
        return ["Ridge diagnostics: UNAVAILABLE"]

    lines = [
        f"Ridge diagnostics: {summary['severity_badge']} ({summary.get('severity') or 'unknown'})"
    ]
    if summary.get("human_review_required"):
        lines.append("- Human review required before business interpretation")
    allowed = summary.get("allowed_uses") or []
    if allowed:
        lines.append(f"- Allowed uses: {', '.join(allowed[:4])}{'...' if len(allowed) > 4 else ''}")
    forbidden_uses = summary.get("forbidden_uses") or []
    if forbidden_uses:
        lines.append(f"- Forbidden uses: {', '.join(forbidden_uses[:3])}{'...' if len(forbidden_uses) > 3 else ''}")
    if summary.get("diagnostic_only_reason"):
        lines.append(f"- Diagnostic-only: {summary['diagnostic_only_reason']}")
    missing = summary.get("missing_required_controls") or []
    if missing:
        lines.append(f"- Missing required {summary.get('vertical_id') or ''} controls: {', '.join(missing)}")
    for ch in summary.get("sparse_channels_extreme") or []:
        lines.append(f"- Sparse channel: {ch} (near_zero_share ≥ {summary.get('sparse_near_zero_threshold')})")
    if summary.get("weak_identification_risk"):
        lines.append(
            f"- Collinearity weak-ID risk: max_abs_corr={summary.get('max_abs_correlation')}"
        )
    if not summary.get("transform_metadata_complete"):
        lines.append("- Transform metadata incomplete (see ridge_production_diagnostics_report.json)")
    if summary.get("fold_stability_ok") is False:
        lines.append("- Fold stability: review geo-fold RMSE variance")
    forbidden = summary.get("forbidden_claims") or []
    if forbidden:
        lines.append(f"- Forbidden claims: {'; '.join(forbidden)}")
    cal_ctx = (report or {}).get("calibration_evidence_context")
    if cal_ctx:
        cal_sum = cal_ctx.get("summary") or build_calibration_evidence_summary(cal_ctx)
        lines.append(
            f"- Calibration context: {cal_sum.get('headline') or 'attached'} (MMM unchanged)"
        )
        if cal_sum.get("directional_conflict_count"):
            lines.append(
                f"- External/MMM conflict(s): {cal_sum.get('conflict_signal_ids')} — review TrustReport"
            )
    return lines


def export_ridge_diagnostic_artifacts(
    store: ArtifactStoreBase,
    extension_report: dict[str, Any],
) -> dict[str, str]:
    """
    Write standalone Ridge diagnostic JSON + Markdown summary to the run bundle.

    Does not alter optimizer, DecisionSurface, or recommendation behavior.
    """
    report = extension_report.get("ridge_production_diagnostics_report")
    if not report or report.get("status") == "unavailable":
        return {}

    store.log_dict("ridge_production_diagnostics_report", report)
    summary = summarize_ridge_diagnostics(report)
    extension_report["ridge_production_diagnostics_summary"] = summary

    md = format_ridge_diagnostics_markdown(report, summary=summary)
    md_path = store.run_path / "ridge_production_diagnostics_summary.md"
    md_path.write_text(md, encoding="utf-8")
    store.log_artifact("ridge_production_diagnostics_summary", md_path)

    return {
        "ridge_production_diagnostics_report": "ridge_production_diagnostics_report.json",
        "ridge_production_diagnostics_summary": "ridge_production_diagnostics_summary.md",
    }


def write_ridge_diagnostic_archive_files(
    report: dict[str, Any],
    *,
    json_path: Path,
    markdown_path: Path | None = None,
) -> None:
    """Write reference archive copies (e.g. under docs/05_validation/archives/)."""
    import json

    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    if markdown_path is not None:
        markdown_path.write_text(format_ridge_diagnostics_markdown(report), encoding="utf-8")
