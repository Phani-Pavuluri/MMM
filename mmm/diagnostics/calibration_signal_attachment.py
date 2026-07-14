"""MIP-C1 — CalibrationSignal attachment to Ridge diagnostics (context-only).

Does not mutate coefficients, fit objects, optimizer inputs, or DecisionSurface inputs.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

ATTACHMENT_VERSION = "mip_calibration_signal_attachment_v1"

FORBIDDEN_ATTACHMENT_OUTPUT_FIELDS = frozenset(
    {
        "decision_surface",
        "optimizer_ready_curves",
        "budget_recommendation",
        "recommendation",
        "production_decision_surface",
        "optimizer_output",
        "optimizer_input",
        "decision_surface_input",
        "ridge_refit_input",
        "coefficient_override",
        "posterior_draws",
        "fitted_beta",
    }
)

CALIBRATION_USE_FORBIDDEN_WITHOUT_UNCERTAINTY = frozenset(
    {
        "calibration_informed_prior",
        "automatic_recalibration",
        "coef_override",
        "silent_mmm_override",
    }
)

STALE_FRESHNESS = frozenset({"stale", "expired", "unknown"})
INCONCLUSIVE_ELIGIBILITY = frozenset({"inconclusive", "excluded", "blocked"})


def _channel_coef_sign(report: dict[str, Any], channel: str) -> str | None:
    coef = (report.get("coefficient_stability") or {}).get("media_coef_by_channel") or {}
    if channel not in coef:
        return None
    v = float(coef[channel])
    if v > 0:
        return "positive"
    if v < 0:
        return "negative"
    return "neutral"


def _effect_direction(signal: dict[str, Any]) -> str | None:
    est = signal.get("effect_estimate")
    if est is None:
        claimed = signal.get("claimed_direction")
        if claimed in ("positive", "negative", "neutral"):
            return claimed
        return None
    try:
        v = float(est)
    except (TypeError, ValueError):
        return None
    if v > 0:
        return "positive"
    if v < 0:
        return "negative"
    return "neutral"


def _has_uncertainty(signal: dict[str, Any]) -> bool:
    if signal.get("standard_error") is not None:
        return True
    interval = signal.get("interval") or signal.get("credible_interval")
    if isinstance(interval, (list, tuple)) and len(interval) >= 2:  # noqa: SIM103 - keep gate readable
        return True
    return False


def _estimand_matches(signal: dict[str, Any], report: dict[str, Any]) -> bool:
    estimand = signal.get("estimand_id")
    if not estimand:
        return True
    report_estimand = (report.get("world_metadata") or {}).get("estimand_id")
    if report_estimand and report_estimand != estimand:
        return False
    allowed = signal.get("estimand_allowlist")
    if isinstance(allowed, list) and estimand not in allowed:  # noqa: SIM103 - keep gate readable
        return False
    return True


def evaluate_signal_alignment(signal: dict[str, Any], report: dict[str, Any]) -> str:
    """Return alignment_status: aligned | misaligned | inconclusive | not_applicable."""
    channel = signal.get("channel")
    if not channel:
        return "not_applicable"
    if signal.get("freshness_status") in STALE_FRESHNESS:
        return "not_applicable"
    if not _estimand_matches(signal, report):
        return "not_applicable"
    if not _has_uncertainty(signal):
        return "inconclusive"
    ext_dir = _effect_direction(signal)
    mmm_dir = _channel_coef_sign(report, channel)
    if ext_dir is None or mmm_dir is None:
        return "inconclusive"
    if ext_dir == mmm_dir or ext_dir == "neutral" or mmm_dir == "neutral":
        return "aligned"
    return "misaligned"


def evaluate_signal_conflict(signal: dict[str, Any], report: dict[str, Any]) -> str:
    """Return conflict_status: none | directional_conflict | scope_mismatch | trust_report_only."""
    if signal.get("eligibility_status") in INCONCLUSIVE_ELIGIBILITY:
        return "trust_report_only"
    if not _estimand_matches(signal, report):
        return "scope_mismatch"
    alignment = evaluate_signal_alignment(signal, report)
    if alignment == "misaligned":
        return "directional_conflict"
    return "none"


def _trust_report_disposition(signal: dict[str, Any], report: dict[str, Any]) -> str:
    if not _estimand_matches(signal, report):
        return "trust_report_only"
    if signal.get("freshness_status") in STALE_FRESHNESS:
        return "context_only_stale"
    if not _has_uncertainty(signal):
        return "trust_report_only"
    conflict = evaluate_signal_conflict(signal, report)
    if conflict == "directional_conflict":
        return "trust_report_and_human_review"
    if conflict in ("scope_mismatch", "trust_report_only"):
        return "trust_report_only"
    return "diagnostic_context"


def _allowed_use(signal: dict[str, Any], report: dict[str, Any]) -> list[str]:
    disposition = _trust_report_disposition(signal, report)
    if disposition == "trust_report_only":
        return ["trust_report_disclosure", "operator_context"]
    if disposition == "context_only_stale":
        return ["operator_context_stale_warning"]
    if disposition == "trust_report_and_human_review":
        return ["trust_report_disclosure", "operator_context", "conflict_flag"]
    return ["operator_context", "ridge_diagnostic_interpretation"]


def _forbidden_claims_for_signal(signal: dict[str, Any], report: dict[str, Any]) -> list[str]:
    claims: list[str] = [
        "external_evidence_overrides_mmm_coefficients",
        "external_evidence_authorizes_optimizer_use",
        "external_evidence_authorizes_budget_recommendation",
        "geox_or_cls_silently_overrides_mmm",
    ]
    channel = signal.get("channel")
    sparse = set((report.get("sparse_channels") or {}).get("sparse_channel_extreme") or [])
    if channel and channel in sparse:
        claims.append(f"no_clean_mmm_only_channel_claim_for_{channel}_despite_external_signal")
    if not _has_uncertainty(signal):
        claims.extend(
            [
                "calibration_informed_attribution_without_uncertainty",
                "production_calibration_use_without_se",
            ]
        )
    if evaluate_signal_conflict(signal, report) == "directional_conflict":
        claims.extend(
            [
                "mmm_direction_validated_by_external_evidence",
                "single_source_attribution_without_caveat",
            ]
        )
    if signal.get("freshness_status") in STALE_FRESHNESS:
        claims.append("fresh_calibration_evidence_claim")
    if not _estimand_matches(signal, report):
        claims.append("cross_estimand_calibration_claim")
    return sorted(set(claims))


def normalize_signal_attachment(signal: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    """Build governed attachment record for one CalibrationSignal."""
    alignment = evaluate_signal_alignment(signal, report)
    conflict = evaluate_signal_conflict(signal, report)
    disposition = _trust_report_disposition(signal, report)
    return {
        "signal_id": signal.get("signal_id"),
        "source_system": signal.get("source_system"),
        "source_modality": signal.get("source_modality"),
        "experiment_id": signal.get("experiment_id") or signal.get("study_id"),
        "study_id": signal.get("study_id"),
        "channel": signal.get("channel"),
        "geo_scope": signal.get("geo_scope"),
        "time_window": signal.get("time_window"),
        "estimand_id": signal.get("estimand_id"),
        "measurement_instrument_id": signal.get("measurement_instrument_id"),
        "lift_scale": signal.get("lift_scale"),
        "effect_estimate": signal.get("effect_estimate"),
        "standard_error": signal.get("standard_error"),
        "interval": signal.get("interval"),
        "freshness_status": signal.get("freshness_status"),
        "eligibility_status": signal.get("eligibility_status"),
        "alignment_status": alignment,
        "conflict_status": conflict,
        "trust_report_disposition": disposition,
        "allowed_use": _allowed_use(signal, report),
        "forbidden_claims": _forbidden_claims_for_signal(signal, report),
        "included_in_context": signal.get("eligibility_status") != "excluded",
    }


def build_calibration_forbidden_claims(context: dict[str, Any]) -> list[str]:
    """Aggregate forbidden claims from calibration evidence context."""
    claims: list[str] = list(context.get("global_forbidden_claims") or [])
    for row in context.get("signals") or []:
        claims.extend(row.get("forbidden_claims") or [])
    return sorted(set(claims))


def build_calibration_evidence_summary(context: dict[str, Any]) -> dict[str, Any]:
    """Operator-facing summary block for Markdown / CLI."""
    signals = context.get("signals") or []
    aligned = sum(1 for s in signals if s.get("alignment_status") == "aligned")
    conflicts = [s for s in signals if s.get("conflict_status") == "directional_conflict"]
    stale = [s for s in signals if s.get("freshness_status") in STALE_FRESHNESS]
    trust_only = [s for s in signals if s.get("trust_report_disposition") == "trust_report_only"]
    return {
        "signal_count": len(signals),
        "aligned_count": aligned,
        "directional_conflict_count": len(conflicts),
        "stale_count": len(stale),
        "trust_report_only_count": len(trust_only),
        "conflict_signal_ids": [s.get("signal_id") for s in conflicts],
        "stale_signal_ids": [s.get("signal_id") for s in stale],
        "headline": context.get("headline"),
        "warnings": list(context.get("warnings") or []),
        "context_only": True,
        "mmm_coefficients_unchanged": True,
    }


def attach_calibration_evidence_context(
    report: dict[str, Any],
    signals: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    """Attach governed calibration evidence context to a Ridge diagnostic report."""
    out = deepcopy(report)
    if not signals:
        return out

    attachments = [normalize_signal_attachment(s, out) for s in signals]
    included = [a for a in attachments if a.get("included_in_context")]

    warnings: list[str] = []
    for a in attachments:
        if a.get("conflict_status") == "directional_conflict":
            warnings.append(
                f"CalibrationSignal {a.get('signal_id')}: directional conflict with MMM "
                f"(channel={a.get('channel')}); context only — does not override Ridge."
            )
        if a.get("freshness_status") in STALE_FRESHNESS:
            warnings.append(
                f"CalibrationSignal {a.get('signal_id')}: stale evidence — context-only."
            )
        if a.get("trust_report_disposition") == "trust_report_only":
            warnings.append(
                f"CalibrationSignal {a.get('signal_id')}: TrustReport-only — not attached "
                "as calibration-informed Ridge interpretation."
            )

    global_forbidden = [
        "automatic_mmm_recalibration_from_calibration_signal",
        "ridge_coefficient_override_from_external_evidence",
        "optimizer_input_from_calibration_signal",
        "decision_surface_input_from_calibration_signal",
        "budget_recommendation_from_calibration_signal",
    ]

    context: dict[str, Any] = {
        "attachment_version": ATTACHMENT_VERSION,
        "milestone": "MIP-C1",
        "signals": attachments,
        "included_signals": included,
        "global_forbidden_claims": global_forbidden,
        "warnings": sorted(set(warnings)),
        "trust_report_boundary": (
            "CalibrationSignal may inform TrustReport and operator context only; "
            "must not bypass TrustReport, override MMM coefficients, or feed optimizer/DecisionSurface."
        ),
        "headline": _context_headline(attachments),
        "context_only": True,
        "bayes_h5_research_only": True,
    }
    context["summary"] = build_calibration_evidence_summary(context)
    context["forbidden_claims"] = build_calibration_forbidden_claims(context)

    out["calibration_evidence_context"] = context
    existing = list(out.get("forbidden_claims") or [])
    out["forbidden_claims"] = sorted(set(existing) | set(context["forbidden_claims"]))
    out["warnings"] = sorted(set(list(out.get("warnings") or []) + warnings))

    col = out.get("collinearity") or {}
    if isinstance(col, dict) and included:
        col = dict(col)
        col["calibration_evidence_available"] = True
        out["collinearity"] = col

    for forbidden in FORBIDDEN_ATTACHMENT_OUTPUT_FIELDS:
        if forbidden in out and out.get(forbidden):
            raise ValueError(f"calibration attachment must not emit {forbidden!r}")
    if isinstance(out.get("calibration_evidence_context"), dict):
        for forbidden in FORBIDDEN_ATTACHMENT_OUTPUT_FIELDS:
            if forbidden in out["calibration_evidence_context"]:
                raise ValueError(
                    f"calibration_evidence_context must not contain {forbidden!r}"
                )

    return out


def _context_headline(attachments: list[dict[str, Any]]) -> str:
    if not attachments:
        return "No calibration signals attached."
    conflicts = sum(1 for a in attachments if a.get("conflict_status") == "directional_conflict")
    aligned = sum(1 for a in attachments if a.get("alignment_status") == "aligned")
    if conflicts:
        return (
            f"{len(attachments)} external signal(s): {aligned} aligned, "
            f"{conflicts} directional conflict(s) — MMM coefficients unchanged."
        )
    return f"{len(attachments)} external signal(s) attached as diagnostic context only."
