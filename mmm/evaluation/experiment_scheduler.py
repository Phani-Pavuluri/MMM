"""
Experiment prioritization scheduler — diagnostic layer above feature separability.

Prioritizes where to spend experimentation budget; does not design or run experiments.
"""

from __future__ import annotations

import hashlib
from typing import Any, Literal

import numpy as np

from mmm.config.extensions import ExperimentSchedulerConfig, FeatureSeparabilityConfig
from mmm.config.schema import MMMConfig
from mmm.contracts.experiment_request import ExperimentRequest, PriorityTier

SchedulerAction = Literal[
    "no_action",
    "monitor",
    "keep_with_caution",
    "rollup_recommended",
    "experiment_optional",
    "experiment_recommended",
    "experiment_high_priority",
]

_CALIBRATION_EVIDENCE_NUM = {"strong": 0.9, "partial": 0.45, "absent": 0.05}
_GLOBAL_EVIDENCE_STALENESS = {"strong": 0.1, "moderate": 0.35, "weak": 0.65}


def deterministic_request_id(
    *,
    channel_or_group: str,
    reason: str,
    uncertainty_source: str,
) -> str:
    payload = f"{channel_or_group}|{reason}|{uncertainty_source}|experiment_scheduler_v1"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def _priority_tier(score: float, cfg: ExperimentSchedulerConfig) -> PriorityTier:
    if score >= cfg.high_priority_threshold:
        return "high"
    if score <= cfg.low_priority_threshold:
        return "low"
    return "medium"


def curve_optimizer_sensitivity_by_channel(
    curve_bundles: list[dict[str, Any]] | None,
) -> dict[str, float]:
    """Per-channel sensitivity proxy from curve diagnostics (0=stable, 1=sensitive)."""
    out: dict[str, float] = {}
    if not curve_bundles:
        return out
    for bundle in curve_bundles:
        ch = str(bundle.get("channel", ""))
        if not ch:
            continue
        diag = bundle.get("diagnostics") if isinstance(bundle.get("diagnostics"), dict) else {}
        stress = bundle.get("stress") if isinstance(bundle.get("stress"), dict) else {}
        safe = bool(diag.get("safe_for_optimization", True))
        unstable = bool(stress.get("numerically_unstable_for_sqp", False))
        jump = float(diag.get("max_gradient_jump", 0.0) or 0.0)
        jump_risk = _clip01(jump / 50.0) if jump > 0 else 0.0
        sens = 0.0
        if not safe:
            sens = max(sens, 0.75)
        if unstable:
            sens = max(sens, 0.85)
        sens = max(sens, jump_risk * 0.5)
        out[ch] = _clip01(sens)
    return out


def identifiability_channel_risk(
    identifiability_json: dict[str, Any] | None,
    channel: str,
    *,
    vif_warning: float,
) -> float:
    ident = identifiability_json or {}
    vif_map = ident.get("vif_by_channel") if isinstance(ident.get("vif_by_channel"), dict) else {}
    vif = float(vif_map.get(channel, 1.0))
    vif_risk = _clip01((vif - 1.0) / max(vif_warning - 1.0, 1e-6))
    global_inst = float(ident.get("instability_score", 0.0))
    global_id = float(ident.get("identifiability_score", 0.0))
    global_risk = _clip01(max(global_inst, global_id))
    return _clip01(0.7 * vif_risk + 0.3 * global_risk)


def group_identifiability_risk(
    members: list[str],
    identifiability_json: dict[str, Any] | None,
    *,
    vif_warning: float,
) -> float:
    if not members:
        return 0.0
    risks = [identifiability_channel_risk(identifiability_json, m, vif_warning=vif_warning) for m in members]
    return float(max(risks))


def calibration_staleness_score(
    classification: str,
    *,
    sched_cfg: ExperimentSchedulerConfig,
    global_evidence_strength: str | None,
) -> float:
    base = {
        "strong": sched_cfg.staleness_strong_calibration,
        "partial": sched_cfg.staleness_partial_calibration,
        "absent": sched_cfg.staleness_absent_calibration,
    }.get(classification, sched_cfg.staleness_absent_calibration)
    boost = _GLOBAL_EVIDENCE_STALENESS.get(str(global_evidence_strength or ""), 0.0)
    return _clip01(max(base, boost * 0.5))


def uncertainty_score_for_group(
    group: dict[str, Any],
    *,
    identifiability_json: dict[str, Any] | None,
    vif_warning: float,
) -> tuple[float, str]:
    sep = float(group.get("separability_score", 0.5))
    sep_unc = _clip01(1.0 - sep)
    members = list(group.get("member_columns") or [])
    id_risk = group_identifiability_risk(members, identifiability_json, vif_warning=vif_warning)
    contrib = group.get("contribution_stability") if isinstance(group.get("contribution_stability"), dict) else {}
    coef_unstable = bool(group.get("unstable_coefficient_members"))
    contrib_unstable = bool(contrib.get("unstable"))
    effect_pen = 0.85 if coef_unstable else 0.0
    contrib_pen = 0.9 if contrib_unstable else 0.0
    corr_band = str(group.get("correlation_band", "low"))
    corr_pen = {"low": 0.0, "moderate": 0.35, "high": 0.75}.get(corr_band, 0.35)
    score = _clip01(
        0.30 * sep_unc
        + 0.25 * id_risk
        + 0.20 * max(effect_pen, contrib_pen)
        + 0.15 * corr_pen
        + 0.10 * (1.0 - _CALIBRATION_EVIDENCE_NUM.get(
            str((group.get("calibration_evidence") or {}).get("classification", "absent")),
            0.05,
        ))
    )
    sources: list[str] = []
    if sep_unc > 0.5:
        sources.append("low_separability")
    if id_risk > 0.5:
        sources.append("identifiability_stress")
    if coef_unstable or contrib_unstable:
        sources.append("unstable_attribution")
    if corr_pen > 0.4:
        sources.append("high_correlation")
    if not sources:
        sources.append("moderate_uncertainty")
    return score, "+".join(sources)


def business_importance_score_from_group(group: dict[str, Any]) -> float:
    biz = group.get("business_importance") if isinstance(group.get("business_importance"), dict) else {}
    spend = float(biz.get("group_spend_share_of_panel", 0.0))
    raw_contrib = biz.get("contribution_share_by_member")
    contrib_map = raw_contrib if isinstance(raw_contrib, dict) else {}
    max_contrib = max((float(v) for v in contrib_map.values()), default=0.0)
    high_spend = bool(biz.get("material_spend_share"))
    high_contrib = bool(biz.get("material_contribution_share"))
    spend_s = _clip01(spend / 0.15) if spend > 0 else 0.0
    contrib_s = _clip01(max_contrib / 0.20) if max_contrib > 0 else 0.0
    if high_spend or high_contrib:
        return _clip01(max(spend_s, contrib_s, 0.72))
    return _clip01(0.5 * spend_s + 0.5 * contrib_s)


def decision_impact_score_for_group(
    group: dict[str, Any],
    *,
    governance_json: dict[str, Any] | None,
    curve_sensitivity: dict[str, float],
    optimization_gate_allowed: bool | None,
) -> float:
    biz = group.get("business_importance") if isinstance(group.get("business_importance"), dict) else {}
    members = list(group.get("member_columns") or [])
    used_opt = bool(biz.get("used_in_optimization"))
    gov = governance_json or {}
    approved = bool(gov.get("approved_for_optimization"))
    gate = 1.0 if optimization_gate_allowed else 0.35 if optimization_gate_allowed is False else 0.55
    curve_risk = max((curve_sensitivity.get(m, 0.0) for m in members), default=0.0)
    opt_sens = _clip01(curve_risk)
    if used_opt and approved:
        base = 0.85
    elif used_opt or approved:
        base = 0.62
    else:
        base = 0.25
    return _clip01(0.45 * base + 0.35 * opt_sens + 0.20 * gate)


def calibration_evidence_score_for_group(group: dict[str, Any]) -> float:
    cal = group.get("calibration_evidence") if isinstance(group.get("calibration_evidence"), dict) else {}
    cls = str(cal.get("classification", "absent"))
    return _CALIBRATION_EVIDENCE_NUM.get(cls, 0.05)


def experiment_priority_score(
    *,
    uncertainty: float,
    business_importance: float,
    decision_impact: float,
    calibration_evidence: float,
    staleness: float,
) -> float:
    return _clip01(
        0.28 * uncertainty
        + 0.24 * business_importance
        + 0.20 * decision_impact
        + 0.14 * staleness
        + 0.14 * (1.0 - calibration_evidence)
    )


def recommend_scheduler_action(
    *,
    priority_tier: PriorityTier,
    priority_score: float,
    uncertainty: float,
    business_importance: float,
    separability_classification: str,
    calibration_classification: str,
    experiment_eligible: bool,
    optimizer_sensitive: bool,
    separability_recommended_action: str | None,
) -> SchedulerAction:
    if not experiment_eligible:
        if separability_classification == "low" and business_importance < 0.35:
            return "rollup_recommended"
        if separability_classification == "low":
            return "monitor"
        return "no_action"

    if separability_classification == "high" and calibration_classification == "strong" and uncertainty < 0.35:
        return "no_action"

    if separability_classification == "high" and calibration_classification in ("strong", "partial"):
        if priority_tier == "high" and optimizer_sensitive:
            return "experiment_optional"
        return "monitor"

    if separability_classification == "medium":
        if priority_tier == "low":
            return "keep_with_caution"
        if priority_tier == "high" and optimizer_sensitive:
            return "experiment_recommended"
        return "keep_with_caution"

    # low separability + material spend
    if priority_tier == "high" and (optimizer_sensitive or uncertainty >= 0.55):
        return "experiment_high_priority"
    if separability_recommended_action == "experiment_recommended" or priority_tier == "high":
        return "experiment_recommended"
    if priority_tier == "medium":
        return "experiment_optional"
    if business_importance < 0.35:
        return "rollup_recommended"
    return "keep_with_caution"


def _suggested_test_type(
    *,
    separability_classification: str,
    optimizer_sensitive: bool,
) -> str:
    if separability_classification == "low" and optimizer_sensitive:
        return "geo_holdout"
    if optimizer_sensitive:
        return "spend_shock"
    return "incrementality_test"


def _build_request_if_needed(
    *,
    channel_or_group: str,
    action: SchedulerAction,
    reason: str,
    uncertainty_source: str,
    priority_score: float,
    priority_tier: PriorityTier,
    business_importance: float,
    separability_classification: str,
    optimizer_sensitive: bool,
    target_column_hint: str,
) -> ExperimentRequest | None:
    if action in ("no_action", "monitor", "keep_with_caution", "rollup_recommended"):
        return None
    if action == "experiment_optional" and priority_tier == "low":
        return None
    req_id = deterministic_request_id(
        channel_or_group=channel_or_group,
        reason=reason,
        uncertainty_source=uncertainty_source,
    )
    test_type = _suggested_test_type(
        separability_classification=separability_classification,
        optimizer_sensitive=optimizer_sensitive,
    )
    notes = [
        "Diagnostic request only — design and execution belong in the external experiment package.",
        f"Scheduler action: {action}.",
    ]
    return ExperimentRequest(
        request_id=req_id,
        channel_or_group=channel_or_group,
        reason=reason,
        uncertainty_source=uncertainty_source,
        priority_score=priority_score,
        priority_tier=priority_tier,
        business_importance=business_importance,
        required_estimand="incremental_lift_on_target_kpi",
        required_kpi=target_column_hint,
        suggested_test_type=test_type,  # type: ignore[arg-type]
        preferred_geo_level="geo_panel_unit",
        notes=notes,
    )


def score_feature_group_unit(
    group: dict[str, Any],
    *,
    identifiability_json: dict[str, Any] | None,
    governance_json: dict[str, Any] | None,
    curve_sensitivity: dict[str, float],
    experiment_matching_json: dict[str, Any] | None,
    sched_cfg: ExperimentSchedulerConfig,
    sep_cfg: FeatureSeparabilityConfig,
    optimization_gate_allowed: bool | None,
    target_column_hint: str,
) -> dict[str, Any]:
    members = list(group.get("member_columns") or [])
    fg = str(group.get("feature_group", "group"))
    uncertainty, unc_source = uncertainty_score_for_group(
        group, identifiability_json=identifiability_json, vif_warning=sep_cfg.vif_warning
    )
    biz = business_importance_score_from_group(group)
    decision = decision_impact_score_for_group(
        group,
        governance_json=governance_json,
        curve_sensitivity=curve_sensitivity,
        optimization_gate_allowed=optimization_gate_allowed,
    )
    cal_ev = calibration_evidence_score_for_group(group)
    cal_cls = str((group.get("calibration_evidence") or {}).get("classification", "absent"))
    global_strength = None
    if isinstance(experiment_matching_json, dict):
        global_strength = experiment_matching_json.get("evidence_strength")
    staleness = calibration_staleness_score(
        cal_cls, sched_cfg=sched_cfg, global_evidence_strength=str(global_strength) if global_strength else None
    )
    priority = experiment_priority_score(
        uncertainty=uncertainty,
        business_importance=biz,
        decision_impact=decision,
        calibration_evidence=cal_ev,
        staleness=staleness,
    )
    tier = _priority_tier(priority, sched_cfg)
    biz_raw = group.get("business_importance") if isinstance(group.get("business_importance"), dict) else {}
    spend_share = float(biz_raw.get("group_spend_share_of_panel", 0.0))
    experiment_eligible = spend_share >= sep_cfg.experiment_min_group_spend_share
    opt_sens = max((curve_sensitivity.get(m, 0.0) for m in members), default=0.0) > 0.45
    sep_cls = str(group.get("separability_classification", "medium"))
    action = recommend_scheduler_action(
        priority_tier=tier,
        priority_score=priority,
        uncertainty=uncertainty,
        business_importance=biz,
        separability_classification=sep_cls,
        calibration_classification=cal_cls,
        experiment_eligible=experiment_eligible,
        optimizer_sensitive=opt_sens,
        separability_recommended_action=str(group.get("recommended_action") or ""),
    )
    reason = (
        f"Prioritize experimentation for {fg} ({', '.join(members)}): "
        f"uncertainty={uncertainty:.2f}, business_importance={biz:.2f}, "
        f"calibration={cal_cls}, action={action}."
    )
    request = _build_request_if_needed(
        channel_or_group=fg,
        action=action,
        reason=reason,
        uncertainty_source=unc_source,
        priority_score=priority,
        priority_tier=tier,
        business_importance=biz,
        separability_classification=sep_cls,
        optimizer_sensitive=opt_sens,
        target_column_hint=target_column_hint,
    )
    return {
        "unit_type": "feature_group",
        "channel_or_group": fg,
        "member_columns": members,
        "uncertainty_score": uncertainty,
        "uncertainty_source": unc_source,
        "business_importance_score": biz,
        "decision_impact_score": decision,
        "calibration_evidence_score": cal_ev,
        "experiment_staleness_score": staleness,
        "experiment_priority_score": priority,
        "priority_tier": tier,
        "recommended_action": action,
        "experiment_eligible": experiment_eligible,
        "group_spend_share_of_panel": spend_share,
        "optimizer_sensitive": opt_sens,
        "separability_classification": sep_cls,
        "calibration_classification": cal_cls,
        "experiment_request": request.to_json() if request else None,
    }


def score_singleton_channel_unit(
    channel: str,
    *,
    panel_spend_share: float,
    identifiability_json: dict[str, Any] | None,
    governance_json: dict[str, Any] | None,
    curve_sensitivity: dict[str, float],
    experiment_matching_json: dict[str, Any] | None,
    matched_channels: set[str],
    sched_cfg: ExperimentSchedulerConfig,
    sep_cfg: FeatureSeparabilityConfig,
    optimization_gate_allowed: bool | None,
    target_column_hint: str,
) -> dict[str, Any]:
    """Channels outside multi-member separability groups — reuse identifiability + curves only."""
    id_risk = identifiability_channel_risk(
        identifiability_json, channel, vif_warning=sep_cfg.vif_warning
    )
    uncertainty = _clip01(id_risk)
    unc_source = "identifiability_stress" if id_risk > 0.5 else "channel_level_uncertainty"
    spend_s = _clip01(panel_spend_share / 0.15) if panel_spend_share > 0 else 0.0
    biz = spend_s
    opt_sens = curve_sensitivity.get(channel, 0.0) > 0.45
    gov = governance_json or {}
    approved = bool(gov.get("approved_for_optimization"))
    decision = _clip01((0.5 if approved else 0.2) + (0.5 if opt_sens else 0.0))
    cal_cls = "strong" if channel in matched_channels else "absent"
    cal_ev = _CALIBRATION_EVIDENCE_NUM.get(cal_cls, 0.05)
    global_strength = None
    if isinstance(experiment_matching_json, dict):
        global_strength = experiment_matching_json.get("evidence_strength")
    staleness = calibration_staleness_score(
        cal_cls, sched_cfg=sched_cfg, global_evidence_strength=str(global_strength) if global_strength else None
    )
    priority = experiment_priority_score(
        uncertainty=uncertainty,
        business_importance=biz,
        decision_impact=decision,
        calibration_evidence=cal_ev,
        staleness=staleness,
    )
    tier = _priority_tier(priority, sched_cfg)
    experiment_eligible = panel_spend_share >= sep_cfg.experiment_min_group_spend_share
    sep_cls = "low" if uncertainty > 0.65 else "medium" if uncertainty > 0.4 else "high"
    action = recommend_scheduler_action(
        priority_tier=tier,
        priority_score=priority,
        uncertainty=uncertainty,
        business_importance=biz,
        separability_classification=sep_cls,
        calibration_classification=cal_cls,
        experiment_eligible=experiment_eligible,
        optimizer_sensitive=opt_sens,
        separability_recommended_action=None,
    )
    reason = (
        f"Channel {channel}: uncertainty={uncertainty:.2f}, spend_share={panel_spend_share:.3f}, "
        f"calibration={cal_cls}, action={action}."
    )
    request = _build_request_if_needed(
        channel_or_group=channel,
        action=action,
        reason=reason,
        uncertainty_source=unc_source,
        priority_score=priority,
        priority_tier=tier,
        business_importance=biz,
        separability_classification=sep_cls,
        optimizer_sensitive=opt_sens,
        target_column_hint=target_column_hint,
    )
    return {
        "unit_type": "channel",
        "channel_or_group": channel,
        "member_columns": [channel],
        "uncertainty_score": uncertainty,
        "uncertainty_source": unc_source,
        "business_importance_score": biz,
        "decision_impact_score": decision,
        "calibration_evidence_score": cal_ev,
        "experiment_staleness_score": staleness,
        "experiment_priority_score": priority,
        "priority_tier": tier,
        "recommended_action": action,
        "experiment_eligible": experiment_eligible,
        "group_spend_share_of_panel": panel_spend_share,
        "optimizer_sensitive": opt_sens,
        "separability_classification": sep_cls,
        "calibration_classification": cal_cls,
        "experiment_request": request.to_json() if request else None,
    }


def _collect_matched_channels(separability_report: dict[str, Any]) -> set[str]:
    matched: set[str] = set()
    for g in separability_report.get("feature_groups") or []:
        if not isinstance(g, dict):
            continue
        cal = g.get("calibration_evidence") if isinstance(g.get("calibration_evidence"), dict) else {}
        for ch in cal.get("channels_with_experiments") or []:
            matched.add(str(ch))
    return matched


def compute_experiment_scheduler_report(
    *,
    config: MMMConfig,
    extension_report: dict[str, Any],
    channel_columns: list[str],
    channel_spend_share_of_panel: dict[str, float],
    target_column: str = "target",
    optimization_gate_allowed: bool | None = None,
) -> dict[str, Any]:
    """
    Build ``experiment_scheduler_report`` from existing extension outputs.

    Expected upstream keys: ``feature_separability_report``, ``identifiability``,
    ``governance``, ``experiment_matching``, ``curve_bundles``, ``response_diagnostics``.
    """
    sched_cfg = config.extensions.experiment_scheduler
    sep_cfg = config.extensions.feature_separability
    base: dict[str, Any] = {
        "policy_version": "experiment_scheduler_v1",
        "diagnostic_only": True,
        "auto_experiment_execution_forbidden": True,
        "auto_budget_reallocation_forbidden": True,
        "auto_model_retraining_forbidden": True,
        "notes": [
            "Prioritizes experimentation budget; does not design geo tests or assign treatment.",
            "Reuses feature_separability_report, identifiability, calibration matching, governance, and curves.",
            "experiment_recommended actions require spend share >= experiment_min_group_spend_share.",
        ],
    }
    if not sched_cfg.enabled:
        return {**base, "skipped": True, "reason": "experiment_scheduler_disabled"}

    sep_report = extension_report.get("feature_separability_report")
    if not isinstance(sep_report, dict) or sep_report.get("skipped"):
        return {
            **base,
            "skipped": True,
            "reason": "feature_separability_unavailable",
        }

    raw_ident = extension_report.get("identifiability")
    ident = raw_ident if isinstance(raw_ident, dict) else {}
    raw_gov = extension_report.get("governance")
    gov = raw_gov if isinstance(raw_gov, dict) else {}
    raw_em = extension_report.get("experiment_matching")
    em = raw_em if isinstance(raw_em, dict) else {}
    curve_bundles = extension_report.get("curve_bundles")
    bundles_list = curve_bundles if isinstance(curve_bundles, list) else []
    curve_sens = curve_optimizer_sensitivity_by_channel(bundles_list)

    gate_allowed = optimization_gate_allowed

    matched_ch = _collect_matched_channels(sep_report)

    scored: list[dict[str, Any]] = []
    grouped_members: set[str] = set()
    for g in sep_report.get("feature_groups") or []:
        if not isinstance(g, dict):
            continue
        for m in g.get("member_columns") or []:
            grouped_members.add(str(m))
        scored.append(
            score_feature_group_unit(
                g,
                identifiability_json=ident,
                governance_json=gov,
                curve_sensitivity=curve_sens,
                experiment_matching_json=em,
                sched_cfg=sched_cfg,
                sep_cfg=sep_cfg,
                optimization_gate_allowed=gate_allowed,
                target_column_hint=target_column,
            )
        )

    for ch in channel_columns:
        if ch in grouped_members:
            continue
        panel_share = float(channel_spend_share_of_panel.get(ch, 0.0))
        scored.append(
            score_singleton_channel_unit(
                ch,
                panel_spend_share=panel_share,
                identifiability_json=ident,
                governance_json=gov,
                curve_sensitivity=curve_sens,
                experiment_matching_json=em,
                matched_channels=matched_ch,
                sched_cfg=sched_cfg,
                sep_cfg=sep_cfg,
                optimization_gate_allowed=gate_allowed,
                target_column_hint=target_column,
            )
        )

    high_req: list[dict[str, Any]] = []
    med_req: list[dict[str, Any]] = []
    low_req: list[dict[str, Any]] = []
    stale_coverage: list[dict[str, Any]] = []
    unsupported = list(sep_report.get("unsupported_split_level_claims") or [])

    for row in scored:
        req = row.get("experiment_request")
        if isinstance(req, dict):
            bucket = {"tier": row["priority_tier"], "request": req, "unit": row["channel_or_group"]}
            if row["priority_tier"] == "high":
                high_req.append(bucket)
            elif row["priority_tier"] == "medium":
                med_req.append(bucket)
            else:
                low_req.append(bucket)
        if float(row.get("experiment_staleness_score", 0)) >= sched_cfg.staleness_partial_calibration:
            stale_coverage.append(
                {
                    "channel_or_group": row["channel_or_group"],
                    "staleness_score": row["experiment_staleness_score"],
                    "calibration_classification": row["calibration_classification"],
                }
            )

    high_units = [r for r in scored if r.get("priority_tier") == "high"]
    summary = {
        "n_units_scored": len(scored),
        "n_high_priority": len(high_units),
        "n_experiment_requests": len(high_req) + len(med_req) + len(low_req),
        "top_priority_units": sorted(
            scored, key=lambda x: float(x.get("experiment_priority_score", 0)), reverse=True
        )[:5],
    }

    return {
        **base,
        "skipped": False,
        "experiment_priority_summary": summary,
        "scored_units": scored,
        "high_priority_requests": high_req,
        "medium_priority_requests": med_req,
        "low_priority_requests": low_req,
        "stale_experiment_coverage": stale_coverage,
        "unsupported_split_level_claims": unsupported,
        "scheduler_notes": [
            "Scores combine separability, identifiability, calibration staleness, "
            "business importance, and curve sensitivity.",
            "Use high_priority_requests for quarterly experiment portfolio planning.",
        ],
    }
