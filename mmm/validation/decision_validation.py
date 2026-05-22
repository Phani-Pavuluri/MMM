"""Decision validation: prior recommendations vs later experiment evidence."""

from __future__ import annotations

import json
from typing import Any, Literal

import numpy as np

from mmm.config.extensions import DecisionValidationConfig
from mmm.config.schema import MMMConfig
from mmm.experiments.evidence import ApprovalStatus, ExperimentEvidence
from mmm.validation.registry_readers import (
    _parse_date,
    extract_channel_ranking_from_bundle,
    extract_recommended_allocation,
    load_decision_bundle,
    load_decision_registry,
    load_experiment_evidence_list,
)

REPORT_VERSION = "mmm_decision_validation_v1"

GOVERNANCE_WARNINGS: tuple[str, ...] = (
    "Decision validation is diagnostic only — no automatic budget or optimizer changes.",
    "Observational outcomes must not be treated as experiment truth for causal validation.",
    "Recommendation quality is not causal proof unless validation design supports it.",
    "Do not auto-update optimizer settings from this report.",
)


def _is_observational_only(ev: ExperimentEvidence) -> bool:
    meta = ev.metadata or {}
    return (
        meta.get("validation_design") == "observational"
        or meta.get("evidence_source") == "observational"
        or meta.get("counts_as_experiment_validation") is False
    )


def _ranking_stability(rank_a: list[str], rank_b: list[str]) -> dict[str, Any]:
    if not rank_a or not rank_b:
        return {"overlap_top3": 0.0, "spearman_proxy": None}
    common = [c for c in rank_a if c in rank_b]
    if not common:
        return {"overlap_top3": 0.0, "spearman_proxy": 0.0}
    overlap = len(set(rank_a[:3]) & set(rank_b[:3])) / 3.0
    ra = {c: i for i, c in enumerate(rank_a)}
    rb = {c: i for i, c in enumerate(rank_b)}
    diffs = [ra[c] - rb[c] for c in common]
    denom = max(len(common) - 1, 1)
    spearman_proxy = 1.0 - float(np.mean(np.abs(diffs))) / denom
    return {"overlap_top3": float(overlap), "spearman_proxy": float(np.clip(spearman_proxy, -1, 1))}


def _allocation_regret_proxy(
    recommended: dict[str, float],
    channel_lifts: dict[str, float],
) -> float:
    if not recommended or not channel_lifts:
        return 0.0
    total_spend = sum(recommended.values()) + 1e-9
    lift_pos = {k: max(v, 0.0) for k, v in channel_lifts.items()}
    total_lift = sum(lift_pos.values()) + 1e-9
    regret = 0.0
    for ch, spend in recommended.items():
        opt_share = lift_pos.get(ch, 0.0) / total_lift
        act_share = spend / total_spend
        regret += abs(act_share - opt_share)
    return float(regret)


def build_decision_validation_report(config: MMMConfig) -> dict[str, Any]:
    dv: DecisionValidationConfig = config.extensions.decision_validation
    report: dict[str, Any] = {
        "report_version": REPORT_VERSION,
        "enabled": bool(dv.enabled),
        "diagnostic_only": True,
        "research_only": True,
        "prod_decisioning_allowed": False,
        "decision_safe": False,
        "auto_retrain": False,
        "auto_budget_change": False,
        "auto_optimizer_change": False,
        "governance_warnings": list(GOVERNANCE_WARNINGS),
        "warnings": list(GOVERNANCE_WARNINGS),
    }
    if not dv.enabled:
        report["skipped"] = True
        report["reason"] = "decision_validation_disabled"
        return report

    exp_path = dv.experiment_registry_path or config.calibration.evidence_registry_path
    if not exp_path:
        report["skipped"] = True
        report["reason"] = "experiment_registry_path_not_set"
        return report

    try:
        evidence_list = load_experiment_evidence_list(exp_path)
    except (OSError, ValueError, FileNotFoundError) as exc:
        report["skipped"] = True
        report["reason"] = "experiment_registry_unavailable"
        report["error"] = str(exc)
        return report

    if not dv.decision_registry_dir:
        report["skipped"] = True
        report["reason"] = "decision_registry_dir_not_set"
        return report

    decisions = load_decision_registry(dv.decision_registry_dir)
    if not decisions:
        report["skipped"] = True
        report["reason"] = "no_decision_registry_entries"
        return report

    per_results: list[dict[str, Any]] = []
    pred_errors: list[float] = []
    regrets: list[float] = []
    ranking_overlaps: list[float] = []
    n_eval = 0
    n_not_eval = 0

    for dec in decisions:
        dec_id = str(dec.get("decision_id", "unknown"))
        dec_date = _parse_date(dec.get("decided_at") or dec.get("decision_date"))
        if dec_date is None:
            n_not_eval += 1
            per_results.append(
                {
                    "decision_id": dec_id,
                    "classification": "not_evaluable",
                    "not_evaluable_reason": "missing_decision_date",
                }
            )
            continue

        bundle_path = dec.get("decision_bundle_path")
        bundle: dict[str, Any] | None = None
        if bundle_path:
            try:
                bundle = load_decision_bundle(bundle_path)
            except (OSError, json.JSONDecodeError, ValueError):
                bundle = None

        recommended = dec.get("recommended_allocation")
        if not isinstance(recommended, dict) and bundle:
            recommended = extract_recommended_allocation(bundle)
        if not isinstance(recommended, dict):
            recommended = {}

        predicted_lifts = dec.get("predicted_lifts_by_channel")
        if not isinstance(predicted_lifts, dict):
            predicted_lifts = {}

        rank_prior = dec.get("channel_ranking")
        if not isinstance(rank_prior, list) and bundle:
            rank_prior = extract_channel_ranking_from_bundle(bundle)
        if not isinstance(rank_prior, list):
            rank_prior = []

        conf = dec.get("decision_confidence")
        if conf is None and bundle:
            conf = bundle.get("decision_safe")

        matched_evidence = False
        for ev in evidence_list:
            if ev.approval_status not in (ApprovalStatus.ACCEPTED, ApprovalStatus.PENDING):
                continue
            ev_date = _parse_date(ev.freshness_date)
            if ev_date is None or ev_date <= dec_date:
                continue
            if (ev_date - dec_date).days > int(dv.lookback_days):
                continue
            if ev.channel not in recommended and ev.channel not in predicted_lifts:
                continue

            matched_evidence = True
            if _is_observational_only(ev):
                n_not_eval += 1
                per_results.append(
                    {
                        "decision_id": dec_id,
                        "experiment_id": ev.experiment_id,
                        "classification": "not_evaluable",
                        "not_evaluable_reason": "observational_evidence_not_experiment_validation",
                    }
                )
                continue

            predicted = predicted_lifts.get(ev.channel)
            if predicted is None:
                predicted = dec.get("predicted_lift")
            observed = float(ev.lift_estimate)
            se = float(ev.standard_error) if ev.standard_error else None
            err = float(predicted - observed) if predicted is not None else None
            if err is not None:
                pred_errors.append(err)

            lift_by_ch = {ev.channel: observed}
            regret = _allocation_regret_proxy(recommended, lift_by_ch)
            regrets.append(regret)
            rank_ev = sorted(lift_by_ch.keys(), key=lambda c: -lift_by_ch[c])
            stab = _ranking_stability(list(rank_prior), rank_ev)
            ranking_overlaps.append(float(stab["overlap_top3"]))
            n_eval += 1
            per_results.append(
                {
                    "decision_id": dec_id,
                    "experiment_id": ev.experiment_id,
                    "channel": ev.channel,
                    "predicted_lift": predicted,
                    "realized_experimental_lift": observed,
                    "prediction_error": err,
                    "standard_error": se,
                    "allocation_regret_proxy": regret,
                    "ranking_stability": stab,
                    "decision_confidence": conf,
                    "classification": "evaluated",
                }
            )

        if not matched_evidence:
            n_not_eval += 1
            per_results.append(
                {
                    "decision_id": dec_id,
                    "classification": "not_evaluable",
                    "not_evaluable_reason": "no_subsequent_experiment_evidence_in_lookback",
                }
            )

    trust = 1.0
    if pred_errors:
        trust -= min(0.4, float(np.mean(np.abs(pred_errors))))
    if regrets:
        trust -= min(0.3, float(np.mean(regrets)))
    trust = float(np.clip(trust, 0.0, 1.0))

    action: Literal[
        "monitor",
        "decision_policy_review",
        "calibration_review",
        "experiment_refresh_recommended",
    ] = "monitor"
    if trust < 0.45:
        action = "decision_policy_review"
    elif pred_errors and float(np.mean(np.abs(pred_errors))) > 1.5:
        action = "calibration_review"
    elif n_not_eval > max(n_eval, 1):
        action = "experiment_refresh_recommended"

    report.update(
        {
            "skipped": False,
            "n_decisions_evaluated": len(
                {r["decision_id"] for r in per_results if r.get("classification") == "evaluated"}
            ),
            "n_decision_experiment_pairs": n_eval,
            "n_not_evaluable": n_not_eval,
            "prediction_error_summary": {
                "mean_error": float(np.mean(pred_errors)) if pred_errors else None,
                "mean_abs_error": float(np.mean(np.abs(pred_errors))) if pred_errors else None,
                "n_with_error": len(pred_errors),
            },
            "ranking_stability": {
                "mean_top3_overlap": float(np.mean(ranking_overlaps)) if ranking_overlaps else None,
                "n_pairs": len(ranking_overlaps),
            },
            "allocation_regret_proxy": {
                "mean": float(np.mean(regrets)) if regrets else None,
                "n_pairs": len(regrets),
            },
            "recommendation_trust_score": trust,
            "per_decision_results": per_results,
            "recommended_action": action,
            "lookback_days": int(dv.lookback_days),
        }
    )
    return report
