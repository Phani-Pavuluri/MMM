"""H9 — governed Ridge diagnostic severity and output eligibility."""

from __future__ import annotations

from typing import Any

from mmm.diagnostics.ridge_diagnostics import FORBIDDEN_OUTPUT_FIELDS

SEVERITY_POLICY_VERSION = "mmm_ridge_severity_policy_v1"

SEVERITY_CLEAN = "clean"
SEVERITY_INFO = "info"
SEVERITY_WARNING = "warning"
SEVERITY_RESTRICTED = "restricted_interpretation"
SEVERITY_DIAGNOSTIC_ONLY = "diagnostic_only"
SEVERITY_BLOCKED = "blocked_for_decision_use"

SEVERITY_ORDER: tuple[str, ...] = (
    SEVERITY_CLEAN,
    SEVERITY_INFO,
    SEVERITY_WARNING,
    SEVERITY_RESTRICTED,
    SEVERITY_DIAGNOSTIC_ONLY,
    SEVERITY_BLOCKED,
)

_LEGACY_FROM_POLICY: dict[str, str] = {
    SEVERITY_CLEAN: "none",
    SEVERITY_INFO: "low",
    SEVERITY_WARNING: "medium",
    SEVERITY_RESTRICTED: "high",
    SEVERITY_DIAGNOSTIC_ONLY: "high",
    SEVERITY_BLOCKED: "high",
}

_BASE_ALLOWED_CLEAN = [
    "model_fit_review",
    "coefficient_review",
    "aggregate_performance_review",
    "planning_input_with_standard_caveats",
]
_BASE_FORBIDDEN_BUSINESS = [
    "clean_channel_attribution",
    "clean_channel_lift_claim",
    "budget_reallocation_claim",
    "production_incrementality_claim",
    "clean_media_attribution",
    "channel_level_causal_claim",
]

_ELIGIBILITY_BY_SEVERITY: dict[str, dict[str, Any]] = {
    SEVERITY_CLEAN: {
        "allowed_uses": list(_BASE_ALLOWED_CLEAN),
        "forbidden_uses": [],
        "human_review_required": False,
    },
    SEVERITY_INFO: {
        "allowed_uses": list(_BASE_ALLOWED_CLEAN),
        "forbidden_uses": ["unsupported_strong_causal_claim"],
        "human_review_required": False,
    },
    SEVERITY_WARNING: {
        "allowed_uses": [
            "model_fit_review",
            "coefficient_review",
            "aggregate_performance_review",
        ],
        "forbidden_uses": [
            "clean_channel_attribution",
            "budget_reallocation_claim",
        ],
        "human_review_required": True,
    },
    SEVERITY_RESTRICTED: {
        "allowed_uses": [
            "model_fit_review",
            "aggregate_diagnostic_review",
            "collinearity_audit",
        ],
        "forbidden_uses": list(_BASE_FORBIDDEN_BUSINESS) + ["isolated_sparse_channel_claim"],
        "human_review_required": True,
    },
    SEVERITY_DIAGNOSTIC_ONLY: {
        "allowed_uses": [
            "model_fit_review",
            "qa_regression_review",
            "methodology_benchmark",
        ],
        "forbidden_uses": list(_BASE_FORBIDDEN_BUSINESS),
        "human_review_required": True,
    },
    SEVERITY_BLOCKED: {
        "allowed_uses": ["engineering_debug_only"],
        "forbidden_uses": ["all_business_and_planning_claims"],
        "human_review_required": True,
    },
}


def _severity_rank(level: str) -> int:
    try:
        return SEVERITY_ORDER.index(level)
    except ValueError:
        return 0


def _worst_severity(*levels: str) -> str:
    return max(levels, key=_severity_rank)


def _forbidden_production_fields_on_report(report: dict[str, Any]) -> list[str]:
    found: list[str] = []
    for field in FORBIDDEN_OUTPUT_FIELDS:
        if report.get(field) is True:
            found.append(field)
    flags = report.get("production_flags") or {}
    if flags.get("optimizer_enabled") is True:
        found.append("optimizer_enabled")
    if flags.get("decision_surface_enabled") is True:
        found.append("decision_surface_enabled")
    if flags.get("recommendations_enabled") is True:
        found.append("recommendations_enabled")
    return found


def _collect_classification_triggers(report: dict[str, Any]) -> list[str]:
    triggers: list[str] = []
    if report.get("status") == "unavailable":
        triggers.append("report:unavailable")
        return triggers

    control = report.get("control_completeness") or {}
    if control.get("omitted_control_risk"):
        triggers.append("control:missing_required_vertical_controls")
    missing_opt = control.get("missing_optional_controls") or []
    if missing_opt and not control.get("omitted_control_risk"):
        triggers.append("control:missing_optional_controls")

    col = report.get("collinearity") or {}
    if col.get("weak_identification_risk") and not col.get("calibration_evidence_available"):
        triggers.append("collinearity:weak_identification_without_calibration")
    if control.get("media_correlated_controls"):
        triggers.append("control:media_correlated_confounding")

    sparse = report.get("sparse_channels") or {}
    for ch in sparse.get("sparse_channel_extreme") or []:
        triggers.append(f"sparse:extreme_near_zero:{ch}")

    transform = report.get("transform_diagnostics") or {}
    if not transform.get("metadata_complete"):
        triggers.append("transform:missing_metadata")

    fold = report.get("fold_stability") or {}
    if fold.get("fold_stability_ok") is False:
        triggers.append("fold:severe_instability")
    elif fold.get("warnings"):
        triggers.append("fold:stability_warning")

    forbidden_fields = _forbidden_production_fields_on_report(report)
    for f in forbidden_fields:
        triggers.append(f"artifact:forbidden_field:{f}")

    if report.get("forbidden_claims"):
        triggers.append("claims:forbidden_claims_present")

    return triggers


def classify_ridge_diagnostic_severity(report: dict[str, Any] | None) -> dict[str, Any]:
    """
    Map a Ridge diagnostic report to governed severity and output eligibility.

    Does not alter optimizer, DecisionSurface, or recommendation behavior.
    """
    if not report:
        level = SEVERITY_DIAGNOSTIC_ONLY
        triggers = ["report:missing"]
        reason = "Diagnostics report missing."
        return _build_eligibility(level, triggers, reason)

    if report.get("status") == "unavailable":
        return _build_eligibility(
            SEVERITY_DIAGNOSTIC_ONLY,
            ["report:unavailable"],
            "Ridge diagnostics could not be produced for this run.",
        )

    forbidden_fields = _forbidden_production_fields_on_report(report)
    if forbidden_fields:
        return _build_eligibility(
            SEVERITY_BLOCKED,
            [f"artifact:forbidden_field:{f}" for f in forbidden_fields],
            f"Forbidden production fields on diagnostic artifact: {forbidden_fields}",
        )

    triggers = _collect_classification_triggers(report)
    control = report.get("control_completeness") or {}
    col = report.get("collinearity") or {}
    sparse = report.get("sparse_channels") or {}
    transform = report.get("transform_diagnostics") or {}
    fold = report.get("fold_stability") or {}

    level = SEVERITY_CLEAN

    if control.get("omitted_control_risk"):
        level = _worst_severity(level, SEVERITY_DIAGNOSTIC_ONLY)
    if fold.get("fold_stability_ok") is False:
        level = _worst_severity(level, SEVERITY_DIAGNOSTIC_ONLY)

    restricted = False
    if col.get("weak_identification_risk") and not col.get("calibration_evidence_available"):
        restricted = True
    if sparse.get("sparse_channel_extreme"):
        restricted = True
    if control.get("media_correlated_controls"):
        restricted = True
    if report.get("forbidden_claims") and not control.get("omitted_control_risk"):
        restricted = True
    if restricted:
        level = _worst_severity(level, SEVERITY_RESTRICTED)

    if control.get("missing_optional_controls") and not control.get("omitted_control_risk"):
        level = _worst_severity(level, SEVERITY_WARNING)
    if not transform.get("metadata_complete"):
        level = _worst_severity(level, SEVERITY_WARNING)

    warnings = report.get("warnings") or []
    if warnings and level == SEVERITY_CLEAN:
        level = SEVERITY_INFO

    reason: str | None = None
    if level == SEVERITY_DIAGNOSTIC_ONLY:
        if control.get("omitted_control_risk"):
            reason = "Required vertical controls missing — diagnostic QA only."
        elif fold.get("fold_stability_ok") is False:
            reason = "Severe geo-fold instability — not decision-grade."
        else:
            reason = "Run is diagnostic-only per severity policy."
    elif level == SEVERITY_RESTRICTED:
        reason = "Channel-level and budget claims require caveats or external calibration."
    elif level == SEVERITY_BLOCKED:
        reason = "Artifact integrity failure — engineering debug only."

    return _build_eligibility(level, triggers, reason)


def _build_eligibility(
    level: str,
    triggers: list[str],
    diagnostic_only_reason: str | None,
) -> dict[str, Any]:
    template = dict(_ELIGIBILITY_BY_SEVERITY.get(level, _ELIGIBILITY_BY_SEVERITY[SEVERITY_WARNING]))
    forbidden_uses = list(template.get("forbidden_uses") or [])
    for t in triggers:
        if t.startswith("sparse:extreme_near_zero:"):
            ch = t.split(":")[-1]
            forbidden_uses.append(f"isolated_channel_claim:{ch}")

    return {
        "severity_policy_version": SEVERITY_POLICY_VERSION,
        "severity": level,
        "allowed_uses": list(template.get("allowed_uses") or []),
        "forbidden_uses": sorted(set(forbidden_uses)),
        "human_review_required": bool(template.get("human_review_required")),
        "diagnostic_only_reason": diagnostic_only_reason,
        "classification_triggers": sorted(set(triggers)),
        "optimizer_decision_surface_unchanged": True,
        "diagnostics_are_not_hard_gates": True,
        "legacy_diagnostic_severity": _LEGACY_FROM_POLICY.get(level, "medium"),
    }


def apply_severity_policy_to_report(report: dict[str, Any]) -> dict[str, Any]:
    """Attach ``output_eligibility`` and policy ``severity`` to an H7 report."""
    eligibility = classify_ridge_diagnostic_severity(report)
    out = dict(report)
    out["severity_policy_version"] = SEVERITY_POLICY_VERSION
    out["severity"] = eligibility["severity"]
    out["output_eligibility"] = eligibility
    out["diagnostic_severity"] = eligibility["legacy_diagnostic_severity"]
    return out
