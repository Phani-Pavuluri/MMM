"""Prod replay / experiment-matching evidence checks (fail-closed semantics)."""

from __future__ import annotations

from typing import Any


def replay_calibration_active(calibration_summary: dict[str, Any] | None) -> bool:
    if not isinstance(calibration_summary, dict):
        return False
    return bool(calibration_summary.get("replay_calibration_active"))


def experiment_matching_satisfies_prod(em: dict[str, Any] | None) -> bool:
    """
    True when matching produced real experiments (not skipped / empty).

    ``{"skipped": true}`` and other non-match payloads must not satisfy prod gates.
    """
    if not isinstance(em, dict) or not em:
        return False
    if bool(em.get("skipped")):
        return False
    n = em.get("n_matched")
    if n is not None:
        try:
            return int(n) >= 1
        except (TypeError, ValueError):
            return False
    return False


def prod_replay_evidence_ok(
    calibration_summary: dict[str, Any] | None,
    experiment_matching: dict[str, Any] | None,
) -> tuple[bool, str]:
    """
    Prod decision paths require replay calibration **or** at least one matched experiment.

    Returns ``(ok, reason_code)``.
    """
    if replay_calibration_active(calibration_summary):
        return True, "replay_calibration_active"
    if experiment_matching_satisfies_prod(experiment_matching):
        return True, "experiment_matching_n_matched"
    if isinstance(experiment_matching, dict) and bool(experiment_matching.get("skipped")):
        return False, "experiment_matching_skipped"
    if (
        isinstance(calibration_summary, dict)
        and calibration_summary
        and not calibration_summary.get("replay_calibration_active")
    ):
        return False, "calibration_summary_without_active_replay"
    return False, "missing_replay_or_matched_experiments"


def prod_replay_evidence_failure_message(
    calibration_summary: dict[str, Any] | None,
    experiment_matching: dict[str, Any] | None,
) -> str:
    ok, code = prod_replay_evidence_ok(calibration_summary, experiment_matching)
    if ok:
        return ""
    em_skipped = isinstance(experiment_matching, dict) and bool(experiment_matching.get("skipped"))
    em_n = experiment_matching.get("n_matched") if isinstance(experiment_matching, dict) else None
    return (
        "prod requires replay calibration evidence: extension_report.calibration_summary with "
        "replay_calibration_active=true, or experiment_matching with n_matched>=1 (not skipped). "
        f"diagnosis={code}; experiment_matching.skipped={em_skipped}; n_matched={em_n!r}."
    )
