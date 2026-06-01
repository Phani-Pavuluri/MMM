"""Phase 5E — TrustReport interpretation from ReliabilityScorecard and trust modifiers."""

from __future__ import annotations

from typing import Any, Literal

TrustGrade = Literal["high", "moderate", "low", "insufficient"]
DriftSeverityLevel = Literal["none", "minor", "moderate", "severe"]

INTERPRETATION_MATRIX: tuple[dict[str, str], ...] = (
    {
        "condition": "structural fail or contract fail",
        "decision_usable": "no",
        "attribution_safe": "no",
        "optimization_blocked": "yes",
        "release_gate": "block",
    },
    {
        "condition": "decision_grade high, trust none/minor, structural pass",
        "decision_usable": "yes (decision metrics)",
        "attribution_safe": "only if attribution_diagnostic high",
        "optimization_blocked": "no",
        "release_gate": "conditional (TBD_v1 not approved)",
    },
    {
        "condition": "decision_grade high, coef diagnostic low",
        "decision_usable": "yes",
        "attribution_safe": "no",
        "optimization_blocked": "no",
        "release_gate": "conditional",
    },
    {
        "condition": "trust_modifier moderate (drift/identifiability)",
        "decision_usable": "caution",
        "attribution_safe": "no",
        "optimization_blocked": "conditional",
        "release_gate": "warn",
    },
    {
        "condition": "trust_modifier severe/degraded",
        "decision_usable": "no or heavy caution",
        "attribution_safe": "no",
        "optimization_blocked": "yes",
        "release_gate": "block or warn",
    },
    {
        "condition": "decision_grade low",
        "decision_usable": "no",
        "attribution_safe": "no",
        "optimization_blocked": "yes",
        "release_gate": "block",
    },
)


def _max_drift_severity(reports: dict[str, dict[str, Any]]) -> DriftSeverityLevel:
    order = {"none": 0, "minor": 1, "moderate": 2, "severe": 3}
    best: DriftSeverityLevel = "none"
    for report in reports.values():
        for row in report.get("recovery_validation_results") or []:
            if str(row.get("check_id", "")) != "REC-4B5-DRIFT":
                continue
            details = row.get("details") if isinstance(row.get("details"), dict) else {}
            sev = str(details.get("drift_severity_level", "none"))
            if sev in order and order[sev] > order[best]:
                best = sev  # type: ignore[assignment]
    return best


def _collect_modifier_signals(
    scorecard: dict[str, Any],
    world_reports: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    trust = scorecard.get("trust_modifier_status") or {}
    signals: dict[str, Any] = {
        "drift_severity": "none",
        "identifiability_status": "not_evaluated",
        "replay_status": "not_evaluated",
        "calibration_freshness": "not_evaluated",
        "optimizer_stability": "not_evaluated",
    }
    if world_reports:
        signals["drift_severity"] = _max_drift_severity(world_reports)

    cap = scorecard.get("capability_summary") or {}
    id_failures = (cap.get("identifiability_behavior") or {}).get("failures") or []
    id_partials = (cap.get("identifiability_behavior") or {}).get("partials") or []
    if id_failures:
        signals["identifiability_status"] = "degraded"
    elif id_partials:
        signals["identifiability_status"] = "caution"
    elif (cap.get("identifiability_behavior") or {}).get("n_scored"):
        signals["identifiability_status"] = "acceptable"

    replay_failures = (cap.get("replay_recovery") or {}).get("failures") or []
    if replay_failures:
        signals["replay_status"] = "degraded"
    elif (cap.get("replay_recovery") or {}).get("n_scored"):
        signals["replay_status"] = "acceptable"

    drift_failures = (cap.get("drift_behavior") or {}).get("failures") or []
    if drift_failures:
        signals["drift_status"] = "degraded"
    elif trust.get("status") == "caution":
        signals["drift_status"] = "caution"
    else:
        signals["drift_status"] = trust.get("status", "not_evaluated")

    opt_failures = (cap.get("optimizer_recovery") or {}).get("failures") or []
    if opt_failures:
        signals["optimizer_stability"] = "degraded"
    elif (cap.get("optimizer_recovery") or {}).get("n_scored"):
        signals["optimizer_stability"] = "acceptable"

    return signals


def build_trust_report_interpretation(
    scorecard: dict[str, Any],
    *,
    world_reports: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Unified TrustReport-style interpretation for synthetic ReliabilityScorecard evidence.

    Attribution metrics are not primary trust signals (Phase 5D/5E policy).
    """
    structural = scorecard.get("structural_reliability_score")
    decision = scorecard.get("decision_reliability_score")
    attribution = scorecard.get("attribution_diagnostic_score")
    trust_mod = scorecard.get("trust_modifier_status") or {}
    signals = _collect_modifier_signals(scorecard, world_reports)

    structural_ok = structural is not None and float(structural) >= 0.75
    decision_high = decision is not None and float(decision) >= 0.85
    decision_low = decision is None or float(decision) < 0.75
    attribution_high = attribution is not None and float(attribution) >= 0.85
    trust_status = str(trust_mod.get("status", "not_evaluated"))
    drift_sev = str(signals.get("drift_severity", "none"))

    if not structural_ok or decision_low:
        grade: TrustGrade = "insufficient" if decision is None else "low"
        decision_usable = False
        opt_block = True
    elif trust_status == "degraded" or drift_sev == "severe":
        grade = "low"
        decision_usable = False
        opt_block = True
    elif trust_status == "caution" or drift_sev == "moderate":
        grade = "moderate"
        decision_usable = decision_high
        opt_block = drift_sev in ("moderate", "severe")
    elif decision_high and structural_ok:
        grade = "high"
        decision_usable = True
        opt_block = False
    else:
        grade = "moderate"
        decision_usable = decision is not None and float(decision) >= 0.75
        opt_block = False

    return {
        "semantics_version": "trust_report_semantics_v1.0.0",
        "trust_grade": grade,
        "decision_usable": decision_usable,
        "attribution_safe": attribution_high,
        "optimization_blocked": opt_block,
        "release_gate_recommendation": (
            "block"
            if grade == "insufficient" or grade == "low"
            else "warn"
            if grade == "moderate"
            else "conditional_not_approved"
        ),
        "primary_signals": {
            "decision_reliability_score": decision,
            "structural_reliability_score": structural,
            "attribution_diagnostic_score": attribution,
            "trust_modifier_status": trust_mod,
        },
        "modifier_signals": signals,
        "interpretation_notes": [
            "decision_usable_may_be_true_when_attribution_diagnostic_score_is_low",
            "attribution_safe_requires_high_attribution_diagnostic_score",
            "severe_drift_downgrades_trust_even_when_delta_mu_passes",
            "TBD_v1_runtime_thresholds_are_not_production_gates",
        ],
        "interpretation_matrix": list(INTERPRETATION_MATRIX),
    }


def enrich_scorecard_with_trust_report(
    scorecard: dict[str, Any],
    *,
    world_reports: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Attach ``trust_report_interpretation`` to an existing scorecard dict (in place)."""
    scorecard["trust_report_interpretation"] = build_trust_report_interpretation(
        scorecard, world_reports=world_reports
    )
    return scorecard
