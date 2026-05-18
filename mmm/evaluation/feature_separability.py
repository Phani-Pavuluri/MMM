"""
Feature separability governance — diagnostic guidance for split media variables.

Does not modify training inputs, optimization, planning, or attribution math.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Literal

import numpy as np
import pandas as pd

from mmm.calibration.engine import CalibrationEngine
from mmm.calibration.matching import match_experiments_with_trace
from mmm.config.extensions import FeatureSeparabilityConfig
from mmm.config.schema import Framework, MMMConfig
from mmm.data.schema import PanelSchema
from mmm.identifiability.engine import _vif_numpy
from mmm.models.ridge_bo.ridge import fit_ridge

CorrelationBand = Literal["low", "moderate", "high"]
VifBand = Literal["healthy", "warning", "high"]
CalibrationEvidence = Literal["strong", "partial", "absent"]
ImportanceBand = Literal["high", "low"]
SeparabilityClass = Literal["high", "medium", "low"]
RecommendedAction = Literal[
    "keep_split",
    "keep_with_caution",
    "rollup_recommended",
    "experiment_recommended",
]


def infer_feature_groups(
    channel_columns: list[str],
    *,
    explicit_groups: dict[str, list[str]] | None,
    auto_group_prefix: bool,
) -> dict[str, list[str]]:
    """Build feature groups from explicit config or first-token prefix (2+ members)."""
    if explicit_groups:
        out: dict[str, list[str]] = {}
        for name, members in explicit_groups.items():
            valid = [c for c in members if c in channel_columns]
            if len(valid) >= 2:
                out[str(name)] = sorted(valid)
        if out:
            return out
    if not auto_group_prefix:
        return {}
    buckets: dict[str, list[str]] = defaultdict(list)
    for col in channel_columns:
        key = col.split("_", 1)[0] if "_" in col else col
        buckets[key].append(col)
    return {k: sorted(v) for k, v in buckets.items() if len(v) >= 2}


def _correlation_band(abs_r: float, *, moderate: float, high: float) -> CorrelationBand:
    if abs_r < moderate:
        return "low"
    if abs_r <= high:
        return "moderate"
    return "high"


def _vif_band(vif: float, *, healthy: float, warning: float) -> VifBand:
    if vif < healthy:
        return "healthy"
    if vif <= warning:
        return "warning"
    return "high"


def pairwise_correlations(panel: pd.DataFrame, columns: list[str]) -> tuple[dict[str, float], float]:
    if len(columns) < 2:
        return {}, 0.0
    sub = panel[list(columns)].astype(float)
    if sub.shape[0] < 3:
        return {}, 0.0
    corr = sub.corr()
    pairs: dict[str, float] = {}
    max_abs = 0.0
    for i, a in enumerate(columns):
        for b in columns[i + 1 :]:
            r = float(corr.loc[a, b])
            if not np.isfinite(r):
                r = 0.0
            pairs[f"{a}|{b}"] = r
            max_abs = max(max_abs, abs(r))
    return pairs, max_abs


def spend_shares(panel: pd.DataFrame, columns: list[str]) -> dict[str, float]:
    totals = {c: float(panel[c].astype(float).clip(lower=0.0).sum()) for c in columns}
    s = sum(totals.values()) + 1e-12
    return {c: totals[c] / s for c in columns}


def _sign_flip_rate(samples: list[float]) -> float:
    if len(samples) < 2:
        return 0.0
    med = float(np.median(samples))
    ref = float(np.sign(med)) if med != 0.0 else 1.0
    flips = sum(1 for x in samples if x != 0.0 and np.sign(x) != ref)
    return float(flips / len(samples))


def _coef_cv(samples: list[float]) -> float:
    if not samples:
        return 0.0
    arr = np.asarray(samples, dtype=float)
    m = float(np.mean(arr))
    if abs(m) < 1e-12:
        return float(np.std(arr))
    return float(np.std(arr) / (abs(m) + 1e-12))


def bootstrap_channel_coefs(
    X_media: np.ndarray,
    y_log: np.ndarray,
    channel_names: list[str],
    *,
    ridge_alpha: float,
    rng: np.random.Generator,
    rounds: int,
    bootstrap_frac: float,
) -> dict[str, list[float]]:
    """Per-channel ridge coefficients across bootstrap refits (diagnostic only)."""
    n, p = X_media.shape
    if p == 0 or n < 4 or rounds <= 0:
        return {c: [] for c in channel_names}
    m = max(int(bootstrap_frac * n), min(p + 2, n))
    idx_map = {name: i for i, name in enumerate(channel_names)}
    out: dict[str, list[float]] = {c: [] for c in channel_names}
    for _ in range(rounds):
        idx = rng.choice(n, size=m, replace=True)
        coef, _ = fit_ridge(X_media[idx], y_log[idx], alpha=ridge_alpha)
        for name, j in idx_map.items():
            if j < len(coef):
                out[name].append(float(coef[j]))
    return out


def feature_column_scales(X_media: np.ndarray, channel_names: list[str]) -> dict[str, float]:
    """Per-channel std of the media design matrix (for standardized effect diagnostics)."""
    scales: dict[str, float] = {}
    for j, name in enumerate(channel_names):
        if j < X_media.shape[1]:
            scales[name] = float(np.std(X_media[:, j]) + 1e-12)
    return scales


def coefficient_stability_metrics(
    boot_coefs: dict[str, list[float]],
    *,
    scale_by_channel: dict[str, float],
    sign_flip_threshold: float,
    coef_cv_threshold: float,
) -> tuple[dict[str, dict[str, float]], list[str], bool]:
    """
    Stability on **standardized effects** (coef × feature scale), not raw coef variance alone.

    Raw coefficients are reported for transparency; instability flags use standardized effects only.
    """
    per: dict[str, dict[str, float]] = {}
    unstable: list[str] = []
    for name, samples in boot_coefs.items():
        scale = float(scale_by_channel.get(name, 1.0))
        raw_mean = float(np.mean(samples)) if samples else 0.0
        raw_std = float(np.std(samples)) if samples else 0.0
        std_effects = [float(s) * scale for s in samples]
        sfr = _sign_flip_rate(std_effects)
        cv = _coef_cv(std_effects)
        per[name] = {
            "raw_coefficient_mean": raw_mean,
            "raw_coefficient_std": raw_std,
            "standardized_effect_mean": float(np.mean(std_effects)) if std_effects else 0.0,
            "standardized_effect_cv": cv,
            "sign_flip_rate_standardized_effect": sfr,
            "feature_scale_std": scale,
        }
        if sfr > sign_flip_threshold or cv > coef_cv_threshold:
            unstable.append(name)
    return per, unstable, bool(unstable)


def contribution_share_series(
    boot_coefs: dict[str, list[float]],
    mean_x: dict[str, float],
    members: list[str],
) -> list[dict[str, float]]:
    """Within-group contribution share per bootstrap draw (abs coef × mean feature)."""
    n_rounds = max((len(boot_coefs.get(m) or []) for m in members), default=0)
    series: list[dict[str, float]] = []
    for r in range(n_rounds):
        weights = {}
        for m in members:
            coefs = boot_coefs.get(m) or []
            if r >= len(coefs):
                continue
            weights[m] = abs(float(coefs[r]) * float(mean_x.get(m, 0.0)))
        total = sum(weights.values()) + 1e-12
        series.append({m: weights.get(m, 0.0) / total for m in members if m in weights})
    return series


def contribution_stability_metrics(
    share_series: list[dict[str, float]],
    members: list[str],
    *,
    variance_threshold: float,
) -> dict[str, Any]:
    if len(share_series) < 2:
        mean_shares = share_series[0] if share_series else {m: 1.0 / max(len(members), 1) for m in members}
        return {
            "mean_shares": mean_shares,
            "share_variance_by_member": {m: 0.0 for m in members},
            "rank_change_rate": 0.0,
            "unstable": False,
        }
    var_by: dict[str, float] = {}
    for m in members:
        vals = [float(s.get(m, 0.0)) for s in share_series if m in s]
        var_by[m] = float(np.var(vals)) if len(vals) >= 2 else 0.0
    rank_changes = 0
    comparisons = 0
    prev_rank: list[str] | None = None
    for s in share_series:
        ranked = sorted(members, key=lambda c: s.get(c, 0.0), reverse=True)
        if prev_rank is not None and ranked != prev_rank:
            rank_changes += 1
        comparisons += 1 if prev_rank is not None else 0
        prev_rank = ranked
    rank_rate = float(rank_changes / comparisons) if comparisons else 0.0
    unstable = any(v > variance_threshold for v in var_by.values()) or rank_rate > 0.35
    mean_shares = {m: float(np.mean([s.get(m, 0.0) for s in share_series])) for m in members}
    return {
        "mean_shares": mean_shares,
        "share_variance_by_member": var_by,
        "rank_change_rate": rank_rate,
        "unstable": unstable,
    }


def matched_experiment_channels(
    config: MMMConfig,
    *,
    panel: pd.DataFrame,
    schema: PanelSchema,
) -> set[str]:
    if not config.calibration.enabled or not config.calibration.experiments_path:
        return set()
    eng = CalibrationEngine(config.calibration.experiments_path)
    exps = eng.load()
    geos = {str(x) for x in panel[schema.geo_column].unique()}
    wk = panel[schema.week_column]
    wmin = pd.to_datetime(wk, errors="coerce").min()
    wmax = pd.to_datetime(wk, errors="coerce").max()
    res = match_experiments_with_trace(
        exps,
        available_geos=geos,
        available_channels=set(schema.channel_columns),
        match_levels=list(config.calibration.match_levels),
        apply_quality=config.calibration.use_quality_weights,
        panel_week_min=wmin if pd.notna(wmin) else None,
        panel_week_max=wmax if pd.notna(wmax) else None,
        run_environment=config.run_environment,
    )
    return {str(m.obs.channel) for m in res.matched}


def calibration_evidence_for_group(
    members: list[str],
    *,
    matched_channels: set[str],
    group_name: str,
) -> dict[str, Any]:
    member_hits = [m for m in members if m in matched_channels]
    aggregate_hit = group_name in matched_channels
    if len(member_hits) == len(members) and len(members) >= 1:
        classification: CalibrationEvidence = "strong"
        rationale = "Split-level experiments matched for all group members."
    elif member_hits or aggregate_hit:
        classification = "partial"
        parts = []
        if member_hits:
            parts.append(f"split-level matches: {member_hits}")
        if aggregate_hit:
            parts.append(f"aggregate '{group_name}' experiment only")
        rationale = "Partial calibration evidence (" + "; ".join(parts) + ")."
    else:
        classification = "absent"
        rationale = "No matched experiments at split or aggregate level for this group."
    return {
        "classification": classification,
        "channels_with_experiments": sorted(member_hits),
        "aggregate_channel_matched": aggregate_hit,
        "rationale": rationale,
    }


def business_importance_for_group(
    members: list[str],
    *,
    panel: pd.DataFrame,
    schema: PanelSchema,
    spend_by_member: dict[str, float],
    contribution_shares: dict[str, float],
    cfg: FeatureSeparabilityConfig,
    governance_approved_for_optimization: bool,
    planner_mode: str,
) -> dict[str, Any]:
    group_spend = sum(spend_by_member.get(m, 0.0) for m in members)
    panel_total = sum(
        float(panel[c].astype(float).clip(lower=0.0).sum())
        for c in schema.channel_columns
        if c in panel.columns
    ) + 1e-12
    group_share = group_spend / panel_total if panel_total > 0 else 0.0
    max_contrib = max((contribution_shares.get(m, 0.0) for m in members), default=0.0)
    used_in_optimization = bool(
        governance_approved_for_optimization and planner_mode == "full_model"
    )
    material_spend = group_share >= cfg.business_importance_high_spend_share
    material_contrib = max_contrib >= cfg.business_importance_high_contribution_share
    high = material_spend or material_contrib
    experiment_eligible = high and group_share >= cfg.experiment_min_group_spend_share
    band: ImportanceBand = "high" if high else "low"
    return {
        "spend_share_by_member": spend_by_member,
        "group_spend_share_of_panel": float(group_share),
        "contribution_share_by_member": contribution_shares,
        "importance_band": band,
        "experiment_eligible": experiment_eligible,
        "material_spend_share": material_spend,
        "material_contribution_share": material_contrib,
        "used_in_optimization": used_in_optimization,
        "decision_relevance_note": (
            "Group spend or contribution is material for budgeting interpretation."
            if high
            else "Group is minor on spend/contribution; rollup or caution-only guidance if separability is weak."
        ),
    }


def separability_score_from_signals(
    *,
    correlation_band: CorrelationBand,
    vif_band: VifBand,
    standardized_effect_unstable: bool,
    contribution_unstable: bool,
    calibration: CalibrationEvidence,
) -> float:
    """
    Weighted score: contribution stability is primary; standardized effects secondary.

    Raw coefficient variance is not scored directly (scale-sensitive).
    """
    corr_s = {"low": 1.0, "moderate": 0.55, "high": 0.15}[correlation_band]
    vif_s = {"healthy": 1.0, "warning": 0.55, "high": 0.2}[vif_band]
    contrib_s = 0.30 if contribution_unstable else 1.0
    effect_s = 0.55 if standardized_effect_unstable else 1.0
    cal_s = {"strong": 1.0, "partial": 0.6, "absent": 0.25}[calibration]
    weights = (0.20, 0.15, 0.35, 0.15, 0.15)
    weighted = (
        weights[0] * corr_s
        + weights[1] * vif_s
        + weights[2] * contrib_s
        + weights[3] * effect_s
        + weights[4] * cal_s
    )
    return float(np.clip(weighted, 0.0, 1.0))


def classify_separability(score: float) -> SeparabilityClass:
    if score >= 0.72:
        return "high"
    if score < 0.45:
        return "low"
    return "medium"


def recommend_action_and_text(
    *,
    classification: SeparabilityClass,
    importance_band: ImportanceBand,
    experiment_eligible: bool,
    feature_group: str,
    member_columns: list[str],
    correlation_band: CorrelationBand,
    standardized_effect_unstable: bool,
    contribution_unstable: bool,
    calibration: CalibrationEvidence,
) -> tuple[RecommendedAction, str]:
    members_label = ", ".join(member_columns)
    if classification == "high":
        return (
            "keep_split",
            f"Keep split for {feature_group} ({members_label}): low collinearity, stable contribution shares "
            "and standardized effects, with calibration evidence supporting independent interpretation.",
        )
    if classification == "medium":
        return (
            "keep_with_caution",
            f"Interpret {feature_group} ({members_label}) cautiously: moderate separability "
            f"(correlation={correlation_band}, calibration={calibration}). Avoid precise split-level ROI claims.",
        )
    if importance_band == "high" and experiment_eligible:
        return (
            "experiment_recommended",
            f"Run a geo experiment or platform incrementality test to separate {members_label} "
            f"before relying on split-level effects for {feature_group}. "
            "Material spend/contribution with low separability.",
        )
    if importance_band == "high":
        return (
            "keep_with_caution",
            f"Low separability for {feature_group} ({members_label}) but spend share is below the "
            "experiment budget threshold — interpret cautiously; experiment optional, not required.",
        )
    return (
        "rollup_recommended",
        f"Roll up {members_label} to a single {feature_group} feature or treat as one block for reporting. "
        "Do not interpret split coefficients independently "
        f"(correlation={correlation_band}, unstable_standardized_effect={standardized_effect_unstable}, "
        f"unstable_contribution={contribution_unstable}).",
    )


def analyze_feature_group(
    feature_group: str,
    members: list[str],
    *,
    panel: pd.DataFrame,
    schema: PanelSchema,
    X_media: np.ndarray,
    channel_names: list[str],
    y_log: np.ndarray,
    vif_by_channel: dict[str, float],
    boot_coefs: dict[str, list[float]],
    matched_channels: set[str],
    cfg: FeatureSeparabilityConfig,
    governance_approved_for_optimization: bool,
    planner_mode: str,
) -> dict[str, Any]:
    pairs, max_corr = pairwise_correlations(panel, members)
    corr_band = _correlation_band(
        max_corr, moderate=cfg.correlation_moderate, high=cfg.correlation_high
    )
    vif_metrics = {m: float(vif_by_channel.get(m, 1.0)) for m in members}
    max_vif = max(vif_metrics.values()) if vif_metrics else 1.0
    vif_band = _vif_band(max_vif, healthy=cfg.vif_healthy, warning=cfg.vif_warning)
    scales = feature_column_scales(X_media, channel_names)
    coef_metrics, unstable_members, effect_unstable = coefficient_stability_metrics(
        {m: boot_coefs.get(m, []) for m in members},
        scale_by_channel=scales,
        sign_flip_threshold=cfg.sign_flip_rate_unstable,
        coef_cv_threshold=cfg.coef_cv_unstable,
    )
    mean_x = {
        m: float(X_media[:, channel_names.index(m)].mean())
        for m in members
        if m in channel_names
    }
    share_series = contribution_share_series(boot_coefs, mean_x, members)
    contrib = contribution_stability_metrics(
        share_series,
        members,
        variance_threshold=cfg.contribution_share_variance_unstable,
    )
    spend_by_member = spend_shares(panel, members)
    contrib_shares = contrib.get("mean_shares") or {}
    cal = calibration_evidence_for_group(
        members, matched_channels=matched_channels, group_name=feature_group
    )
    biz = business_importance_for_group(
        members,
        panel=panel,
        schema=schema,
        spend_by_member=spend_by_member,
        contribution_shares=contrib_shares if isinstance(contrib_shares, dict) else {},
        cfg=cfg,
        governance_approved_for_optimization=governance_approved_for_optimization,
        planner_mode=planner_mode,
    )
    contribution_unstable = bool(contrib.get("unstable"))
    group_effect_unstable = effect_unstable
    score = separability_score_from_signals(
        correlation_band=corr_band,
        vif_band=vif_band,
        standardized_effect_unstable=group_effect_unstable,
        contribution_unstable=contribution_unstable,
        calibration=cal["classification"],
    )
    classification = classify_separability(score)
    action, recommendation = recommend_action_and_text(
        classification=classification,
        importance_band=biz["importance_band"],
        experiment_eligible=bool(biz.get("experiment_eligible")),
        feature_group=feature_group,
        member_columns=members,
        correlation_band=corr_band,
        standardized_effect_unstable=group_effect_unstable,
        contribution_unstable=contribution_unstable,
        calibration=cal["classification"],
    )
    return {
        "feature_group": feature_group,
        "member_columns": members,
        "pairwise_correlations": pairs,
        "max_pairwise_correlation": max_corr,
        "correlation_band": corr_band,
        "vif_metrics": vif_metrics,
        "max_vif": max_vif,
        "vif_band": vif_band,
        "coefficient_stability": coef_metrics,
        "unstable_coefficient_members": unstable_members,
        "contribution_stability": contrib,
        "calibration_evidence": cal,
        "business_importance": biz,
        "separability_score": score,
        "separability_classification": classification,
        "recommendation": recommendation,
        "recommended_action": action,
    }


def build_governance_artifacts(
    groups: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    rollup_recs: list[dict[str, Any]] = []
    experiment_recs: list[dict[str, Any]] = []
    unsupported: list[dict[str, Any]] = []
    release_warnings: list[str] = []
    optimization_warnings: list[str] = []

    for g in groups:
        action = g.get("recommended_action")
        fg = str(g.get("feature_group"))
        members = list(g.get("member_columns") or [])
        biz = g.get("business_importance") or {}
        if action == "rollup_recommended":
            rollup_recs.append(
                {
                    "feature_group": fg,
                    "member_columns": members,
                    "recommendation": g.get("recommendation"),
                    "governance_severity": "warning",
                }
            )
        if action == "experiment_recommended":
            experiment_recs.append(
                {
                    "feature_group": fg,
                    "member_columns": members,
                    "recommendation": g.get("recommendation"),
                    "governance_severity": "warning",
                }
            )
            release_warnings.append(f"{fg}: low separability with high business importance")
            if biz.get("used_in_optimization"):
                optimization_warnings.append(
                    f"{fg}: split-level optimization interpretation unsupported; run experiment or roll up"
                )
        if g.get("separability_classification") == "low":
            unsupported.append(
                {
                    "feature_group": fg,
                    "member_columns": members,
                    "claim": "split_level_attribution_or_incrementality",
                    "reason": "low_separability",
                    "recommended_action": action,
                }
            )
            if biz.get("importance_band") == "low":
                pass  # warning only via rollup_recs
            else:
                release_warnings.append(f"{fg}: split-level reporting not supported at current separability")

    summary = {
        "release_review_warnings": release_warnings,
        "optimization_use_warnings": optimization_warnings,
        "diagnostic_only": True,
        "auto_merge_forbidden": True,
    }
    return rollup_recs, experiment_recs, unsupported, summary


def compute_feature_separability_report(
    *,
    panel: pd.DataFrame,
    schema: PanelSchema,
    config: MMMConfig,
    fit_out: dict[str, Any],
    X_media: np.ndarray,
    identifiability_json: dict[str, Any] | None,
    experiment_matching_json: dict[str, Any] | None,
    rng: np.random.Generator,
    governance_approved_for_optimization: bool = False,
) -> dict[str, Any]:
    """
    Post-fit separability governance report.

    Reuses identifiability VIF when present; optional bootstrap refit uses the same ridge alpha as the fit.
    """
    cfg = config.extensions.feature_separability
    base: dict[str, Any] = {
        "policy_version": "feature_separability_v1",
        "diagnostic_only": True,
        "notes": [
            "This report does not modify training features, optimization, planning, or attribution math.",
            "Use recommendations for taxonomy and experiment design — not automatic column merges.",
            "Coefficient stability flags use standardized effects (coef × feature scale); contribution "
            "share stability is weighted highest in the separability score.",
            "experiment_recommended requires material spend share, not optimization approval alone.",
        ],
    }
    if not cfg.enabled:
        return {**base, "skipped": True, "reason": "feature_separability_disabled"}
    if config.framework != Framework.RIDGE_BO:
        return {
            **base,
            "skipped": True,
            "reason": "ridge_bo_primary_path",
            "feature_groups": [],
        }
    channels = list(schema.channel_columns)
    groups_map = infer_feature_groups(
        channels,
        explicit_groups=cfg.feature_groups or None,
        auto_group_prefix=cfg.auto_group_prefix,
    )
    if not groups_map:
        return {
            **base,
            "skipped": False,
            "feature_groups": [],
            "rollup_recommendations": [],
            "experiment_recommendations": [],
            "unsupported_split_level_claims": [],
            "governance_summary": {"release_review_warnings": [], "optimization_use_warnings": []},
            "note": "no multi_member_feature_groups_detected",
        }

    ident = identifiability_json or {}
    vif_by_channel = dict(ident.get("vif_by_channel") or {})
    if not vif_by_channel and X_media.size:
        vif_by_channel = _vif_numpy(np.asarray(X_media, dtype=float), channels)

    y = panel[schema.target_column].to_numpy(dtype=float)
    y_log = np.log(np.clip(y, 1e-9, None))
    ridge_la = 0.0
    if fit_out.get("artifacts") is not None:
        ridge_la = float(fit_out["artifacts"].best_params.get("log_alpha", 0.0))
    alpha = float(10**ridge_la)
    id_cfg = config.extensions.identifiability
    rounds = id_cfg.bootstrap_rounds if cfg.reuse_identifiability_bootstrap else cfg.bootstrap_rounds
    boot_coefs = bootstrap_channel_coefs(
        np.asarray(X_media, dtype=float),
        y_log,
        channels,
        ridge_alpha=alpha,
        rng=rng,
        rounds=rounds,
        bootstrap_frac=id_cfg.bootstrap_frac,
    )

    matched = matched_experiment_channels(config, panel=panel, schema=schema)
    planner_mode = str(config.extensions.product.planner_mode)

    analyzed: list[dict[str, Any]] = []
    for fg, members in sorted(groups_map.items()):
        analyzed.append(
            analyze_feature_group(
                fg,
                members,
                panel=panel,
                schema=schema,
                X_media=np.asarray(X_media, dtype=float),
                channel_names=channels,
                y_log=y_log,
                vif_by_channel=vif_by_channel,
                boot_coefs=boot_coefs,
                matched_channels=matched,
                cfg=cfg,
                governance_approved_for_optimization=governance_approved_for_optimization,
                planner_mode=planner_mode,
            )
        )

    rollup_recs, experiment_recs, unsupported, gov_summary = build_governance_artifacts(analyzed)
    em = experiment_matching_json or {}
    return {
        **base,
        "skipped": False,
        "experiment_matching_summary": {
            "n_matched": em.get("n_matched"),
            "evidence_strength": em.get("evidence_strength"),
            "skipped": em.get("skipped"),
        },
        "feature_groups": analyzed,
        "rollup_recommendations": rollup_recs,
        "experiment_recommendations": experiment_recs,
        "unsupported_split_level_claims": unsupported,
        "governance_summary": gov_summary,
    }
