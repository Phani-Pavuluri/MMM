"""Calibration freshness and coefficient drift readiness (warnings or planning block)."""

from __future__ import annotations

from datetime import date
from typing import Any, Literal

import numpy as np

from mmm.config.schema import MMMConfig
from mmm.evaluation.drift_history import load_historical_reference

REPORT_VERSION = "mmm_calibration_readiness_v1"

RecommendedAction = Literal[
    "monitor",
    "recalibration_recommended",
    "experiment_refresh_required",
    "model_review_required",
]

GOVERNANCE_WARNINGS: tuple[str, ...] = (
    "Calibration readiness affects planning_allowed only when governance.require_review_on_drift=true.",
    "No automatic retraining, coefficient changes, or budget updates are performed.",
    "Drift signals do not prove causal invalidity — they flag operational review need.",
)


def _parse_date(value: Any) -> date | None:
    if value is None:
        return None
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def _coef_vector(extension_report: dict[str, Any]) -> np.ndarray | None:
    rfs = extension_report.get("ridge_fit_summary")
    if isinstance(rfs, dict) and rfs.get("coef") is not None:
        return np.asarray(rfs["coef"], dtype=float).ravel()
    return None


def _reference_coef(reference: dict[str, Any] | None) -> np.ndarray | None:
    if not reference:
        return None
    fit = reference.get("fit_summary") or reference.get("ridge_fit_summary") or {}
    if isinstance(fit, dict) and fit.get("coef") is not None:
        return np.asarray(fit["coef"], dtype=float).ravel()
    er = reference.get("extension_report") or reference
    if isinstance(er, dict):
        return _coef_vector(er)
    return None


def _coefficient_shift(
    current: np.ndarray | None,
    reference: np.ndarray | None,
    *,
    threshold: float,
) -> tuple[float, list[dict[str, Any]], list[dict[str, Any]]]:
    if current is None or reference is None or current.size == 0 or reference.size == 0:
        return 0.0, [], []
    n = min(current.size, reference.size)
    cur = current[:n]
    ref = reference[:n]
    denom = np.maximum(np.abs(ref), 1e-9)
    rel = np.abs(cur - ref) / denom
    score = float(np.max(rel))
    direction_changes = []
    rank_changes = []
    for i in range(n):
        if ref[i] * cur[i] < 0 and abs(ref[i]) > 1e-12 and abs(cur[i]) > 1e-12:
            direction_changes.append({"index": i, "reference": float(ref[i]), "current": float(cur[i])})
    ref_order = np.argsort(-np.abs(ref))
    cur_order = np.argsort(-np.abs(cur))
    if not np.array_equal(ref_order, cur_order):
        rank_changes.append(
            {
                "reference_top_index": int(ref_order[0]),
                "current_top_index": int(cur_order[0]),
                "threshold_exceeded": score >= threshold,
            }
        )
    return score, direction_changes, rank_changes


def _replay_miss_trend(continuous_validation: dict[str, Any] | None, *, threshold: float) -> dict[str, Any]:
    if not isinstance(continuous_validation, dict):
        return {"replay_miss_rate": None, "replay_trend": "unknown", "n_evaluated": 0}
    rows = continuous_validation.get("experiment_comparisons") or continuous_validation.get("comparisons") or []
    if not isinstance(rows, list) or not rows:
        return {"replay_miss_rate": None, "replay_trend": "unknown", "n_evaluated": 0}
    misses = 0
    evaluated = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        cls = str(row.get("classification", row.get("status", "")))
        if cls == "not_evaluable":
            continue
        evaluated += 1
        if cls in ("mild_miss", "severe_miss"):
            misses += 1
    rate = float(misses / evaluated) if evaluated else None
    trend = "stable"
    if rate is not None and rate >= threshold:
        trend = "degrading"
    elif rate is not None and rate > 0:
        trend = "monitor"
    return {
        "replay_miss_rate": rate,
        "replay_trend": trend,
        "n_evaluated": evaluated,
    }


def _stale_calibration(
    extension_report: dict[str, Any],
    *,
    max_age_days: int,
) -> tuple[bool, str, int | None]:
    cont = extension_report.get("continuous_validation_report")
    if isinstance(cont, dict):
        ef = cont.get("evidence_freshness_report") or {}
        max_age = ef.get("max_age_days")
        if max_age is not None and int(max_age) > max_age_days:
            return True, f"last experiment evidence age {max_age}d exceeds {max_age_days}d", int(max_age)
    cal = extension_report.get("calibration_summary") or {}
    if isinstance(cal, dict) and cal.get("replay_calibration_active") and cal.get("replay_train_loss") is None:
        return True, "replay calibration active but no replay loss recorded", None
    return False, "", None


def build_calibration_readiness_report(
    config: MMMConfig,
    extension_report: dict[str, Any],
    *,
    historical_reference: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assess calibration freshness and drift vs promoted/accepted reference."""
    gov = config.governance
    max_age = int(gov.calibration_max_age_days)
    coef_thr = float(gov.coefficient_shift_threshold)
    replay_thr = float(gov.replay_miss_threshold)

    stale, stale_msg, max_evidence_age = _stale_calibration(extension_report, max_age_days=max_age)

    if historical_reference is None and config.extensions.drift_historical.registry_dir:
        from mmm.evaluation.run_registry import AcceptedRunRegistry

        reg = AcceptedRunRegistry(config.extensions.drift_historical.registry_dir)
        latest = reg.latest_accepted()
        if latest is not None:
            historical_reference = load_historical_reference(latest.run_dir)

    cur_coef = _coef_vector(extension_report)
    ref_coef = _reference_coef(historical_reference)
    shift_score, dir_changes, rank_changes = _coefficient_shift(
        cur_coef, ref_coef, threshold=coef_thr
    )

    cont_val = extension_report.get("continuous_validation_report")
    replay_trend = _replay_miss_trend(
        cont_val if isinstance(cont_val, dict) else None,
        threshold=replay_thr,
    )

    recommended: RecommendedAction = "monitor"
    warnings: list[str] = list(GOVERNANCE_WARNINGS)
    stale_warning = None
    if stale:
        stale_warning = stale_msg
        warnings.append(stale_msg)
        recommended = "experiment_refresh_required"
    if shift_score >= coef_thr:
        warnings.append(f"coefficient_shift_score={shift_score:.4f} >= threshold {coef_thr}")
        recommended = "model_review_required"
    miss_rate = replay_trend.get("replay_miss_rate")
    if miss_rate is not None and float(miss_rate) >= replay_thr:
        warnings.append(f"replay_miss_rate={miss_rate:.4f} >= threshold {replay_thr}")
        if recommended == "monitor":
            recommended = "recalibration_recommended"

    blocks_planning = bool(gov.require_review_on_drift) and recommended in (
        "recalibration_recommended",
        "experiment_refresh_required",
        "model_review_required",
    )

    return {
        "report_version": REPORT_VERSION,
        "diagnostic_only": not gov.require_review_on_drift,
        "stale_calibration_warning": stale_warning,
        "last_experiment_max_age_days": max_evidence_age,
        "calibration_max_age_days": max_age,
        "coefficient_shift_score": shift_score,
        "coefficient_shift_threshold": coef_thr,
        "coefficient_direction_changes": dir_changes[:10],
        "coefficient_rank_changes": rank_changes,
        "replay_miss_rate": replay_trend.get("replay_miss_rate"),
        "replay_miss_threshold": replay_thr,
        "replay_trend": replay_trend.get("replay_trend"),
        "recommended_action": recommended,
        "require_review_on_drift": gov.require_review_on_drift,
        "blocks_planning_allowed": blocks_planning,
        "warnings": warnings,
        "reference_available": historical_reference is not None,
    }


def apply_calibration_readiness_to_model_release(
    config: MMMConfig,
    extension_report: dict[str, Any],
    readiness: dict[str, Any],
) -> dict[str, Any] | None:
    """Re-infer model_release when drift review blocks planning (no auto-retrain)."""
    if not readiness.get("blocks_planning_allowed"):
        return None
    from mmm.governance.model_release import infer_model_release_state

    mr = extension_report.get("model_release") or {}
    inv = list(mr.get("invalidation_reasons") or mr.get("reasons") or [])
    if "calibration_drift_review_required" not in inv:
        inv.append("calibration_drift_review_required")
    gov = extension_report.get("governance") or {}
    pq = extension_report.get("panel_qa") or {}
    return infer_model_release_state(
        config=config,
        panel_qa_max_severity=str(pq.get("max_severity", "info")),
        governance_approved_for_optimization=bool(gov.get("approved_for_optimization")),
        governance_approved_for_reporting=bool(gov.get("approved_for_reporting")),
        ridge_fit_summary_present=bool(extension_report.get("ridge_fit_summary")),
        invalidation_reasons=inv,
        post_fit_validation=extension_report.get("post_fit_validation"),
        operational_health=extension_report.get("operational_health"),
    )
