"""Weighted Ridge replay calibration from experiment evidence registry (PR 2)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from mmm.calibration.contracts import CalibrationUnit
from mmm.calibration.replay_estimand import REPLAY_TRANSFORM_MODE_FULL_PANEL, ReplayEstimandSpec
from mmm.calibration.replay_frames import build_calibration_unit_from_shift, panel_time_mask
from mmm.calibration.replay_lift import implied_lift_from_counterfactual
from mmm.config.schema import MMMConfig, RunEnvironment
from mmm.data.schema import PanelSchema
from mmm.evaluation.experiment_evidence_extension import load_evidence_from_path
from mmm.experiments.compatibility import (
    CompatibilityStatus,
    ExperimentCompatibilityResolver,
    ModelPanelContext,
    ReplayCompatibilityDecision,
    ReplayMode,
)
from mmm.experiments.evidence import ApprovalStatus, ExperimentEvidence, GeoGranularity
from mmm.experiments.evidence_quality import (
    CONSERVATIVE_DEFAULT_SE,
    EvidenceQualityContext,
    QualityTier,
    score_evidence_quality,
)
from mmm.experiments.shock_plan import AllocationQuality, CounterfactualShockPlanner


@dataclass
class WeightedReplayEntry:
    unit: CalibrationUnit
    evidence_weight: float
    experiment_id: str
    channel: str
    compatibility_status: str
    quality_tier: str
    replay_mode: str
    supports_subgeo_claims: bool = False
    allocation_method: str | None = None
    allocation_quality: str | None = None
    allocation_role: str = "computational_bridge_only"
    allocation_required: bool = False


@dataclass
class EvidenceReplayPrepareResult:
    used: list[WeightedReplayEntry] = field(default_factory=list)
    rejected: list[dict[str, Any]] = field(default_factory=list)
    diagnostic_only: list[str] = field(default_factory=list)
    n_loaded: int = 0
    warnings: list[str] = field(default_factory=list)
    compatibility_status_counts: dict[str, int] = field(default_factory=dict)
    shock_allocation_quality_counts: dict[str, int] = field(default_factory=dict)


def uses_weighted_evidence_replay(config: MMMConfig) -> bool:
    cal = config.calibration
    return bool(
        cal.use_replay_calibration
        and cal.replay_mode == "evidence_registry"
        and cal.evidence_weighting_enabled
    )


def uses_evidence_registry_replay(config: MMMConfig) -> bool:
    """True when prod/training replay path is evidence-registry (weighted or not)."""
    cal = config.calibration
    return bool(cal.use_replay_calibration and cal.replay_mode == "evidence_registry")


def uses_legacy_replay(config: MMMConfig) -> bool:
    if uses_weighted_evidence_replay(config):
        return False
    cal = config.calibration
    return bool(cal.use_replay_calibration and cal.replay_units_path)


def validate_evidence_registry_replay_config(config: MMMConfig) -> None:
    """Fail loudly when evidence-registry replay is misconfigured."""
    cal = config.calibration
    if cal.replay_mode != "evidence_registry":
        return
    if not cal.evidence_registry_path:
        raise ValueError(
            "calibration.replay_mode='evidence_registry' requires calibration.evidence_registry_path"
        )
    if not cal.compatibility_resolver_enabled:
        raise ValueError(
            "calibration.replay_mode='evidence_registry' requires calibration.compatibility_resolver_enabled=true"
        )
    if cal.evidence_weighting_enabled and not cal.use_replay_calibration:
        raise ValueError(
            "calibration.evidence_weighting_enabled requires calibration.use_replay_calibration=true"
        )


def _panel_context(panel: pd.DataFrame, schema: PanelSchema, config: MMMConfig) -> ModelPanelContext:
    geos = {str(x) for x in panel[schema.geo_column].unique()}
    wcol = schema.week_column
    wseries = panel[wcol]
    week_min_val: float | None = None
    week_max_val: float | None = None
    week_min_ts: pd.Timestamp | None = None
    week_max_ts: pd.Timestamp | None = None
    if pd.api.types.is_numeric_dtype(wseries):
        week_min_val = float(wseries.min())
        week_max_val = float(wseries.max())
    else:
        w = pd.to_datetime(wseries, errors="coerce")
        if w.notna().any():
            week_min_ts = w.min()
            week_max_ts = w.max()
    return ModelPanelContext(
        geo_column=schema.geo_column,
        channel_columns=schema.channel_columns,
        target_column=schema.target_column,
        panel_geos=geos,
        panel_week_min=week_min_ts,
        panel_week_max=week_max_ts,
        panel_week_min_value=week_min_val,
        panel_week_max_value=week_max_val,
        model_geo_granularity=GeoGranularity(config.calibration.model_geo_granularity),
        channel_mapping=dict(config.calibration.channel_mapping or {}),
    )


def _resolve_channel(evidence: ExperimentEvidence, ctx: ModelPanelContext) -> str | None:
    mapped = ctx.channel_mapping.get(evidence.channel, evidence.channel)
    if mapped in ctx.channel_columns:
        return mapped
    if evidence.channel in ctx.channel_columns:
        return evidence.channel
    return None


def _replay_estimand_for_evidence(
    evidence: ExperimentEvidence,
    compat: ReplayCompatibilityDecision,
    schema: PanelSchema,
    panel_geos: set[str],
) -> dict[str, Any]:
    scope_geos = [str(g) for g in evidence.geo_scope if str(g) in panel_geos]
    if compat.replay_mode in {
        ReplayMode.AGGREGATE_MODEL_TO_EXPERIMENT_SCOPE,
        ReplayMode.DIRECT_SAME_GRAIN,
    } and not scope_geos and evidence.geo_granularity in {GeoGranularity.NATIONAL, GeoGranularity.USER}:
        return {
            "geo_scope": "all",
            "geo_ids": [],
            "week_start": evidence.time_window.start,
            "week_end": evidence.time_window.end,
            "aggregation": "geo_mean_then_global_mean",
            "target_kpi_column": schema.target_column,
            "lift_scale": evidence.metadata.get("lift_scale", "mean_kpi_level_delta"),
            "notes": "aggregate_experiment_on_granular_panel",
        }
    if not scope_geos:
        scope_geos = sorted(panel_geos)
    agg = "mean"
    if compat.compatibility_status == CompatibilityStatus.AGGREGATE_ONLY:
        agg = "geo_mean_then_global_mean"
    return {
        "geo_scope": "listed" if scope_geos else "all",
        "geo_ids": scope_geos,
        "week_start": evidence.time_window.start,
        "week_end": evidence.time_window.end,
        "aggregation": agg,
        "target_kpi_column": schema.target_column,
        "lift_scale": evidence.metadata.get("lift_scale", "mean_kpi_level_delta"),
        "notes": f"evidence_registry:{evidence.experiment_id}",
    }


def _spend_multiplier(evidence: ExperimentEvidence, obs: pd.DataFrame, channel: str) -> float:
    meta = evidence.metadata
    if meta.get("spend_multiplier") is not None:
        return float(meta["spend_multiplier"])
    total = float(obs[channel].astype(float).sum())
    if total <= 0 or evidence.spend_delta is None:
        return 1.0
    delta = float(evidence.spend_delta)
    # Treat spend_delta as absolute spend removed from treated channel in window.
    withheld = abs(delta)
    return max(0.01, (total - withheld) / total)


def build_calibration_unit_from_evidence(
    evidence: ExperimentEvidence,
    panel: pd.DataFrame,
    schema: PanelSchema,
    *,
    channel: str,
    compat: ReplayCompatibilityDecision,
    spend_multiplier: float | None = None,
) -> CalibrationUnit | None:
    gcol, wcol = schema.geo_column, schema.week_column
    panel_geos = {str(x) for x in panel[gcol].unique()}
    re_dict = _replay_estimand_for_evidence(evidence, compat, schema, panel_geos)
    re_dict["replay_transform_mode"] = REPLAY_TRANSFORM_MODE_FULL_PANEL
    spec = ReplayEstimandSpec.from_dict(re_dict)
    window_mask = (
        panel[gcol].astype(str).isin(set(spec.geo_ids))
        if spec.geo_scope == "listed" and spec.geo_ids
        else pd.Series(True, index=panel.index)
    ) & panel_time_mask(panel[wcol], spec.week_start, spec.week_end)
    if not bool(window_mask.any()):
        return None
    window_df = panel.loc[window_mask]
    mult = spend_multiplier if spend_multiplier is not None else _spend_multiplier(evidence, window_df, channel)
    se = evidence.standard_error
    return build_calibration_unit_from_shift(
        panel,
        schema,
        unit_id=evidence.experiment_id,
        channel=channel,
        geo_ids=list(spec.geo_ids) if spec.geo_ids else sorted(panel_geos),
        week_start=spec.week_start,
        week_end=spec.week_end,
        spend_multiplier=mult,
        observed_lift=float(evidence.lift_estimate),
        lift_se=float(se) if se is not None and se > 0 else None,
        target_kpi=schema.target_column,
        estimand=evidence.estimand,
        lift_scale=str(re_dict.get("lift_scale", "")),
        replay_estimand=re_dict,
        experiment_id=evidence.experiment_id,
        calibration_readiness=evidence.approval_status.value,
    )


def prepare_evidence_replay(
    config: MMMConfig,
    panel: pd.DataFrame,
    schema: PanelSchema,
) -> EvidenceReplayPrepareResult:
    """Filter evidence, build replay units, attach weights (no replay execution)."""
    validate_evidence_registry_replay_config(config)
    path = config.calibration.evidence_registry_path
    assert path is not None
    evidence_list = load_evidence_from_path(path)
    ctx = _panel_context(panel, schema, config)
    resolver = ExperimentCompatibilityResolver()
    planner = CounterfactualShockPlanner()
    target_kpi = config.calibration.experiment_target_kpi or schema.target_column
    allow_missing_se = config.run_environment != RunEnvironment.PROD

    result = EvidenceReplayPrepareResult(n_loaded=len(evidence_list))
    status_counts: dict[str, int] = {}
    shock_counts: dict[str, int] = {}

    for ev in evidence_list:
        if ev.approval_status == ApprovalStatus.EXPIRED:
            result.rejected.append({"experiment_id": ev.experiment_id, "reason": "expired_approval"})
            continue

        ch = _resolve_channel(ev, ctx)
        compat = resolver.resolve(ev, ctx, target_kpi=target_kpi)
        status_counts[compat.compatibility_status.value] = (
            status_counts.get(compat.compatibility_status.value, 0) + 1
        )
        shock = planner.plan(ev, compat)
        shock_counts[shock.allocation_quality] = shock_counts.get(shock.allocation_quality, 0) + 1

        if compat.compatibility_status == CompatibilityStatus.REJECTED:
            result.rejected.append(
                {
                    "experiment_id": ev.experiment_id,
                    "reason": compat.rejection_reason or "rejected",
                }
            )
            continue

        if compat.compatibility_status == CompatibilityStatus.DIAGNOSTIC_ONLY:
            result.diagnostic_only.append(ev.experiment_id)
            result.rejected.append(
                {"experiment_id": ev.experiment_id, "reason": "diagnostic_only_compatibility"}
            )
            continue

        if shock.allocation_quality == AllocationQuality.REJECTED.value:
            result.rejected.append(
                {"experiment_id": ev.experiment_id, "reason": "shock_plan_rejected"}
            )
            continue

        if compat.allocation_required and shock.allocation_quality == AllocationQuality.LOW.value:
            result.rejected.append(
                {
                    "experiment_id": ev.experiment_id,
                    "reason": "allocation_required_but_low_quality_shock",
                }
            )
            continue

        if ev.standard_error is None or ev.standard_error <= 0:
            if not allow_missing_se:
                result.rejected.append(
                    {"experiment_id": ev.experiment_id, "reason": "missing_or_invalid_standard_error"}
                )
                continue
            result.warnings.append(
                f"conservative_default_se_for_{ev.experiment_id}: excluded_from_prod_paths"
            )

        qctx = EvidenceQualityContext(
            target_kpi=target_kpi,
            channel_match=ch is not None,
            compatibility=compat,
            allow_missing_se=allow_missing_se,
        )
        qscore = score_evidence_quality(ev, qctx)
        if qscore.quality_tier == QualityTier.REJECTED or qscore.evidence_weight <= 0:
            result.rejected.append(
                {
                    "experiment_id": ev.experiment_id,
                    "reason": "quality_rejected",
                    "quality_reasons": qscore.reasons,
                }
            )
            continue

        if ch is None:
            result.rejected.append({"experiment_id": ev.experiment_id, "reason": "channel_not_in_model"})
            continue

        unit = build_calibration_unit_from_evidence(
            ev, panel, schema, channel=ch, compat=compat
        )
        if unit is None:
            result.rejected.append(
                {"experiment_id": ev.experiment_id, "reason": "empty_replay_panel_slice"}
            )
            continue

        if compat.compatibility_status == CompatibilityStatus.AGGREGATE_ONLY:
            result.warnings.append(
                f"aggregate_evidence_{ev.experiment_id}_on_granular_model: no_subgeo_claims"
            )
        if compat.allocation_required:
            result.warnings.append(
                f"allocated_shock_{ev.experiment_id}: computational_bridge_only"
            )
        if qscore.expiration_status.startswith("stale"):
            result.warnings.append(f"stale_evidence_downweighted:{ev.experiment_id}")
        if qscore.quality_tier == QualityTier.LOW:
            result.warnings.append(f"low_quality_evidence_low_weight:{ev.experiment_id}")

        result.used.append(
            WeightedReplayEntry(
                unit=unit,
                evidence_weight=qscore.evidence_weight,
                experiment_id=ev.experiment_id,
                channel=ch,
                compatibility_status=compat.compatibility_status.value,
                quality_tier=qscore.quality_tier.value,
                replay_mode=compat.replay_mode.value,
                supports_subgeo_claims=compat.supports_subgeo_claims,
                allocation_method=shock.allocation_method,
                allocation_quality=shock.allocation_quality,
                allocation_role=shock.allocation_role,
                allocation_required=compat.allocation_required,
            )
        )

    result.compatibility_status_counts = status_counts
    result.shock_allocation_quality_counts = shock_counts
    return result


def aggregate_weighted_evidence_replay_loss(
    entries: list[WeightedReplayEntry],
    predict_fn: Any,
    *,
    schema: PanelSchema,
    target_col: str,
    config: MMMConfig | None = None,
) -> tuple[float, dict[str, Any]]:
    """
    Weighted replay loss: sum(w_i * z_i^2) / sum(w_i).

    ``z_i = (implied_lift - observed_lift) / se_i``
    """
    if target_col != schema.target_column:
        raise ValueError("target_col must match schema.target_column")
    weighted_errors: list[float] = []
    weights: list[float] = []
    unit_meta: list[dict[str, Any]] = []
    allow_missing_se = config is not None and config.run_environment != RunEnvironment.PROD

    for ent in entries:
        u = ent.unit
        if u.observed_spend_frame is None or u.counterfactual_spend_frame is None or u.observed_lift is None:
            continue
        if not u.replay_estimand:
            raise ValueError(f"replay unit {u.unit_id!r}: replay_estimand required")
        spec = ReplayEstimandSpec.from_dict(u.replay_estimand)
        r = implied_lift_from_counterfactual(
            panel_observed=u.observed_spend_frame,
            panel_counterfactual=u.counterfactual_spend_frame,
            predict_fn=predict_fn,
            schema=schema,
            estimand=spec,
        )
        implied = float(r["implied_mean_delta"])
        if u.lift_se is not None and u.lift_se > 0:
            se = float(u.lift_se)
        elif allow_missing_se:
            se = CONSERVATIVE_DEFAULT_SE
        else:
            continue
        w = float(ent.evidence_weight)
        z2 = w * float(((implied - float(u.observed_lift)) / se) ** 2)
        weighted_errors.append(z2)
        weights.append(w)
        unit_meta.append(
            {
                "experiment_id": ent.experiment_id,
                "unit_id": u.unit_id,
                "implied_delta": implied,
                "observed_lift": u.observed_lift,
                "se": se,
                "evidence_weight": w,
                "weighted_error": z2,
                "compatibility_status": ent.compatibility_status,
                "quality_tier": ent.quality_tier,
                "supports_subgeo_claims": ent.supports_subgeo_claims,
            }
        )

    if not weighted_errors or sum(weights) <= 0:
        return 0.0, {
            "n_units": 0,
            "weighted_replay_loss": 0.0,
            "units": unit_meta,
            "replay_mode_used": "evidence_registry",
        }

    loss = float(sum(weighted_errors) / sum(weights))
    return loss, {
        "n_units": len(weighted_errors),
        "weighted_replay_loss": loss,
        "sum_evidence_weight": float(sum(weights)),
        "units": unit_meta,
        "replay_mode_used": "evidence_registry",
        "replay_transform_mode": REPLAY_TRANSFORM_MODE_FULL_PANEL,
        "replay_uses_full_panel_transform": True,
        "evidence_weights": [float(x) for x in weights],
    }


def build_evidence_weighted_replay_summary(
    prepared: EvidenceReplayPrepareResult,
    loss_meta: dict[str, Any],
) -> dict[str, Any]:
    """Extension-report payload for evidence-weighted replay."""
    unit_governance = [
        {
            "experiment_id": ent.experiment_id,
            "channel": ent.channel,
            "quality_tier": ent.quality_tier,
            "compatibility_status": ent.compatibility_status,
            "supports_subgeo_claims": ent.supports_subgeo_claims,
            "allocation_role": ent.allocation_role,
            "allocation_method": ent.allocation_method,
            "allocation_required": ent.allocation_required,
            "lift_se": ent.unit.lift_se,
            "evidence_weight": ent.evidence_weight,
        }
        for ent in prepared.used
    ]
    return {
        "replay_mode_used": loss_meta.get("replay_mode_used", "evidence_registry"),
        "replay_transform_mode": loss_meta.get(
            "replay_transform_mode", REPLAY_TRANSFORM_MODE_FULL_PANEL
        ),
        "replay_uses_full_panel_transform": bool(
            loss_meta.get("replay_uses_full_panel_transform", True)
        ),
        "n_evidence_units_loaded": prepared.n_loaded,
        "n_evidence_units_used": len(prepared.used),
        "n_evidence_units_rejected": len(prepared.rejected),
        "weighted_replay_loss": loss_meta.get("weighted_replay_loss"),
        "rejected_evidence_reasons": list(prepared.rejected),
        "evidence_weights": loss_meta.get("evidence_weights", []),
        "compatibility_status_counts": dict(prepared.compatibility_status_counts),
        "shock_allocation_quality_counts": dict(prepared.shock_allocation_quality_counts),
        "diagnostic_only_experiments": list(prepared.diagnostic_only),
        "governance_warnings": list(prepared.warnings),
        "unit_governance": unit_governance,
        "units": loss_meta.get("units", []),
    }
