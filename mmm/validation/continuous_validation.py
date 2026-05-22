"""Continuous validation: prior model predictions vs arriving experiment evidence."""

from __future__ import annotations

import json
from datetime import date
from typing import Any, Literal

import numpy as np

from mmm.config.extensions import ContinuousValidationConfig
from mmm.config.schema import MMMConfig
from mmm.experiments.evidence import ApprovalStatus, ExperimentEvidence
from mmm.validation.registry_readers import (
    _parse_date,
    extract_predicted_lift_for_experiment,
    filter_runs_before,
    load_accepted_run_registry,
    load_experiment_evidence_list,
    load_extension_report,
)

REPORT_VERSION = "mmm_continuous_validation_v1"

Classification = Literal["aligned", "mild_miss", "severe_miss", "not_evaluable"]
RecommendedAction = Literal[
    "monitor",
    "recalibrate_recommended",
    "experiment_refresh_recommended",
    "model_review_required",
]

GOVERNANCE_WARNINGS: tuple[str, ...] = (
    "Continuous validation is diagnostic only — no automatic retraining or registry promotion.",
    "Prediction vs experiment comparison does not establish causal proof beyond experiment design.",
    "Prior-run predicted lifts may be missing if accepted-run registry is incomplete.",
)


def _classify_error(z: float | None, *, has_se: bool) -> Classification:
    if z is None:
        return "not_evaluable"
    if not has_se:
        return "not_evaluable"
    az = abs(z)
    if az < 1.0:
        return "aligned"
    if az < 2.0:
        return "mild_miss"
    return "severe_miss"


def _freshness_report(evidence_list: list[ExperimentEvidence], *, stale_days: int) -> dict[str, Any]:
    today = date.today()
    stale = 0
    ages: list[int] = []
    for ev in evidence_list:
        fd = _parse_date(ev.freshness_date)
        if fd is None:
            continue
        age = (today - fd).days
        ages.append(age)
        if age > stale_days:
            stale += 1
    return {
        "n_evidence": len(evidence_list),
        "stale_count": stale,
        "stale_after_days": stale_days,
        "mean_age_days": float(np.mean(ages)) if ages else None,
        "max_age_days": int(max(ages)) if ages else None,
    }


def _pick_prior_run(
    runs: list[dict[str, Any]],
    experiment: ExperimentEvidence,
    *,
    lookback_days: int,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Latest prior run before experiment freshness date."""
    ev_date = _parse_date(experiment.freshness_date)
    if ev_date is None or not runs:
        return None, None
    prior = filter_runs_before(runs, before=ev_date, lookback_days=lookback_days)
    if not prior:
        return None, None
    run = prior[-1]
    ext_path = run.get("extension_report_path")
    ext: dict[str, Any] | None = None
    if ext_path:
        try:
            ext = load_extension_report(ext_path)
        except (OSError, json.JSONDecodeError, ValueError):
            ext = None
    return run, ext


def _recommended_action(
    *,
    n_aligned: int,
    n_mild: int,
    n_severe: int,
    n_total: int,
    stale_frac: float,
) -> RecommendedAction:
    if n_total == 0:
        return "monitor"
    severe_rate = n_severe / n_total
    miss_rate = (n_mild + n_severe) / n_total
    if severe_rate >= 0.25:
        return "model_review_required"
    if miss_rate >= 0.5:
        return "recalibrate_recommended"
    if stale_frac >= 0.4:
        return "experiment_refresh_recommended"
    return "monitor"


def build_continuous_validation_report(
    config: MMMConfig,
    *,
    current_extension_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Compare prior accepted-run predictions to experiment evidence (local registry only).
    """
    cv: ContinuousValidationConfig = config.extensions.continuous_validation
    base: dict[str, Any] = {
        "report_version": REPORT_VERSION,
        "enabled": bool(cv.enabled),
        "diagnostic_only": True,
        "research_only": True,
        "prod_decisioning_allowed": False,
        "auto_retrain": False,
        "auto_registry_promotion": False,
        "auto_budget_change": False,
        "governance_warnings": list(GOVERNANCE_WARNINGS),
        "warnings": list(GOVERNANCE_WARNINGS),
    }
    if not cv.enabled:
        base["skipped"] = True
        base["reason"] = "continuous_validation_disabled"
        return base

    evidence_path = cv.experiment_registry_path or config.calibration.evidence_registry_path
    if not evidence_path:
        base["skipped"] = True
        base["reason"] = "experiment_registry_path_not_set"
        return base

    try:
        evidence_list = load_experiment_evidence_list(evidence_path)
    except (OSError, ValueError, FileNotFoundError) as exc:
        base["skipped"] = True
        base["reason"] = "experiment_registry_unavailable"
        base["error"] = str(exc)
        return base

    runs: list[dict[str, Any]] = []
    if cv.registry_dir:
        runs = load_accepted_run_registry(cv.registry_dir)
    if not runs:
        base["skipped"] = True
        base["reason"] = "no_accepted_run_registry"
        base["n_experiments_in_registry"] = len(evidence_list)
        return base

    accepted = [
        ev
        for ev in evidence_list
        if ev.approval_status in (ApprovalStatus.ACCEPTED, ApprovalStatus.PENDING)
    ]
    per_experiment: list[dict[str, Any]] = []
    counts = {"aligned": 0, "mild_miss": 0, "severe_miss": 0, "not_evaluable": 0}
    drift_errors: list[float] = []
    trust_penalty = 0.0

    for ev in accepted:
        run, ext = _pick_prior_run(runs, ev, lookback_days=int(cv.lookback_days))
        predicted = extract_predicted_lift_for_experiment(
            ev.experiment_id, run_record=run, extension_report=ext
        )
        observed = float(ev.lift_estimate)
        se = float(ev.standard_error) if ev.standard_error is not None else None
        has_se = se is not None and se > 0
        if (cv.require_experiment_se and not has_se) or predicted is None or not has_se:
            classification: Classification = "not_evaluable"
            z = None
        else:
            z = float((predicted - observed) / se)
            classification = _classify_error(z, has_se=True)
            drift_errors.append(predicted - observed)

        counts[classification] += 1
        if classification == "severe_miss":
            trust_penalty += 0.12
        elif classification == "mild_miss":
            trust_penalty += 0.05

        per_experiment.append(
            {
                "experiment_id": ev.experiment_id,
                "channel": ev.channel,
                "kpi": ev.kpi,
                "geo_scope": list(ev.geo_scope),
                "freshness_date": str(ev.freshness_date),
                "predicted_lift": predicted,
                "experiment_lift": observed,
                "standard_error": se,
                "standardized_error": z,
                "classification": classification,
                "prior_run_id": run.get("run_id") if run else None,
                "not_evaluable_reason": (
                    None
                    if classification != "not_evaluable"
                    else (
                        "missing_se"
                        if cv.require_experiment_se and not has_se
                        else "missing_prior_prediction"
                        if predicted is None
                        else "missing_se"
                    )
                ),
            }
        )

    n_eval = sum(counts[k] for k in ("aligned", "mild_miss", "severe_miss"))
    freshness = _freshness_report(evidence_list, stale_days=365)
    stale_frac = (
        float(freshness["stale_count"]) / max(int(freshness["n_evidence"]), 1)
    )
    model_trust = float(np.clip(1.0 - trust_penalty, 0.0, 1.0))
    action = _recommended_action(
        n_aligned=counts["aligned"],
        n_mild=counts["mild_miss"],
        n_severe=counts["severe_miss"],
        n_total=n_eval,
        stale_frac=stale_frac,
    )

    base.update(
        {
            "skipped": False,
            "n_experiments_evaluated": n_eval,
            "n_aligned": counts["aligned"],
            "n_mild_miss": counts["mild_miss"],
            "n_severe_miss": counts["severe_miss"],
            "n_not_evaluable": counts["not_evaluable"],
            "per_experiment_results": per_experiment,
            "calibration_drift": {
                "mean_signed_error": float(np.mean(drift_errors)) if drift_errors else None,
                "mean_abs_error": float(np.mean(np.abs(drift_errors))) if drift_errors else None,
                "n_with_error": len(drift_errors),
            },
            "evidence_freshness_report": freshness,
            "model_trust_score": model_trust,
            "recommended_action": action,
            "n_prior_runs_indexed": len(runs),
            "lookback_days": int(cv.lookback_days),
        }
    )
    if current_extension_report:
        base["current_run_fingerprint"] = current_extension_report.get("data_fingerprint")
    return base
