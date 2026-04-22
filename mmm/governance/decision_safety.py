"""Decision-facing safety defaults and messaging."""

from __future__ import annotations

from typing import Any

# User-facing strings (reports, CLI, artifacts)
MSG_ANALYSIS_ONLY = (
    "This build defaults to analysis-only posture for decision-facing outputs until replay calibration, "
    "governance approval for optimization, and full-panel Δμ workflows are satisfied. "
    "Coefficient-to-experiment calibration has been removed from training paths."
)

MSG_CALIBRATION_NOT_DECISION_SAFE = (
    "Calibration vs experiments is only decision-meaningful under replay with explicit estimands "
    "and aligned spend frames; treat any other score as diagnostic."
)

MSG_OPTIMIZATION_BLOCKED = (
    "Budget optimization is blocked: pass governance + extension_report for full-panel Δμ, "
    "or (non-prod only) enable allow_unsafe_decision_apis for legacy diagnostic curve paths."
)


def decision_safety_artifact(*, allow_unsafe_decision_apis: bool) -> dict[str, Any]:
    return {
        "allow_unsafe_decision_apis": allow_unsafe_decision_apis,
        "analysis_only_default": not allow_unsafe_decision_apis,
        "calibration_uses_replay_estimand": False,
        "coefficient_aligned_experiment_lift": False,
        "messages": {
            "summary": (
                MSG_ANALYSIS_ONLY
                if not allow_unsafe_decision_apis
                else (
                    "allow_unsafe_decision_apis is true: legacy diagnostic optimizers may run in non-prod; "
                    "production still forbids unsafe APIs and curve optimizers."
                )
            ),
            "calibration": MSG_CALIBRATION_NOT_DECISION_SAFE,
            "optimization": MSG_OPTIMIZATION_BLOCKED if not allow_unsafe_decision_apis else None,
        },
    }


def report_decision_safety_section(*, allow_unsafe_decision_apis: bool) -> dict[str, Any]:
    """Section payload for ReportBuilder / JSON reports."""
    return {
        "decision_safe_for_budgeting": False,
        "decision_safe_for_experiment_calibration": False,
        "labels": {
            "calibration": (
                "not_decision_safe_yet (replay-based estimand required)"
                if not allow_unsafe_decision_apis
                else "experimental_only_replay_required_for_decision_grade"
            ),
            "optimization": (
                "not_decision_safe_yet (blocked by default)"
                if not allow_unsafe_decision_apis
                else "experimental_requires_gates_and_model_curves"
            ),
        },
        "artifact_contract": decision_safety_artifact(allow_unsafe_decision_apis=allow_unsafe_decision_apis),
    }
