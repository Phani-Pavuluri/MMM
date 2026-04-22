"""Phase 0: analysis-only defaults until replay calibration and model-bound optimization ship."""

from __future__ import annotations

from typing import Any

# User-facing strings (reports, CLI, artifacts)
MSG_ANALYSIS_ONLY = (
    "This build is in analysis-only mode for decision-facing outputs: experiment calibration, "
    "governance approval for optimization, and budget optimization are not decision-safe until "
    "replay-based calibration and model-implied economics are wired. "
    "Ridge coefficients are not experiment-aligned incremental lift."
)

MSG_CALIBRATION_NOT_DECISION_SAFE = (
    "Calibration vs experiments is not decision-safe: do not interpret any score as validating "
    "incremental lift until replay-based model_implied_lift is used. "
    "Coefficient-based calibration is disabled unless allow_unsafe_decision_apis is true."
)

MSG_OPTIMIZATION_BLOCKED = (
    "Budget optimization is blocked under the safety freeze. "
    "Set allow_unsafe_decision_apis: true in YAML and pass --allow-unsafe-decision-apis on the CLI "
    "to run the experimental optimizer path only after explicit review."
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
                    "allow_unsafe_decision_apis is true: experimental paths may run; "
                    "coefficient-based calibration is still not replay-validated as experiment lift."
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
                else "experimental_only_not_replay_validated_do_not_treat_as_experiment_lift"
            ),
            "optimization": (
                "not_decision_safe_yet (blocked by default)"
                if not allow_unsafe_decision_apis
                else "experimental_requires_gates_and_model_curves"
            ),
        },
        "artifact_contract": decision_safety_artifact(allow_unsafe_decision_apis=allow_unsafe_decision_apis),
    }
