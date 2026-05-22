"""Experiment ↔ MMM replay compatibility resolver (Phase 1 — diagnostic only)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import pandas as pd

from mmm.experiments.evidence import ExperimentEvidence, GeoGranularity


class CompatibilityStatus(StrEnum):
    COMPATIBLE = "compatible"
    AGGREGATE_ONLY = "aggregate_only"
    ALLOCATION_REQUIRED = "allocation_required"
    DIAGNOSTIC_ONLY = "diagnostic_only"
    REJECTED = "rejected"


class ReplayMode(StrEnum):
    DIRECT_SAME_GRAIN = "direct_same_grain"
    AGGREGATE_MODEL_TO_EXPERIMENT_SCOPE = "aggregate_model_to_experiment_scope"
    ALLOCATE_EXPERIMENT_SHOCK_TO_MODEL_UNITS = "allocate_experiment_shock_to_model_units"
    REJECT_INCOMPATIBLE = "reject_incompatible"


# Coarse → fine ordering for granularity comparison.
_GEO_RANK: dict[GeoGranularity, int] = {
    GeoGranularity.NATIONAL: 0,
    GeoGranularity.REGION: 1,
    GeoGranularity.DMA: 2,
    GeoGranularity.GEO: 3,
    GeoGranularity.USER: 4,
}


@dataclass
class ModelPanelContext:
    """MMM panel / schema context for compatibility checks."""

    geo_column: str
    channel_columns: tuple[str, ...]
    target_column: str
    panel_geos: set[str]
    panel_week_min: pd.Timestamp | None = None
    panel_week_max: pd.Timestamp | None = None
    #: Set when ``week_column`` is numeric (dense week rank / index).
    panel_week_min_value: float | None = None
    panel_week_max_value: float | None = None
    model_geo_granularity: GeoGranularity = GeoGranularity.GEO
    channel_mapping: dict[str, str] = field(default_factory=dict)


@dataclass
class ReplayCompatibilityDecision:
    compatibility_status: CompatibilityStatus
    replay_mode: ReplayMode
    model_geo_granularity: str
    experiment_geo_granularity: str
    supports_model_level_calibration: bool
    supports_subgeo_claims: bool
    allocation_required: bool
    allowed_allocation_methods: list[str] = field(default_factory=list)
    rejection_reason: str | None = None
    warnings: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "compatibility_status": self.compatibility_status.value,
            "replay_mode": self.replay_mode.value,
            "model_geo_granularity": self.model_geo_granularity,
            "experiment_geo_granularity": self.experiment_geo_granularity,
            "supports_model_level_calibration": self.supports_model_level_calibration,
            "supports_subgeo_claims": self.supports_subgeo_claims,
            "allocation_required": self.allocation_required,
            "allowed_allocation_methods": list(self.allowed_allocation_methods),
            "rejection_reason": self.rejection_reason,
            "warnings": list(self.warnings),
        }


def _infer_model_granularity(
    panel_geos: set[str],
    declared: GeoGranularity,
) -> GeoGranularity:
    if declared != GeoGranularity.GEO:
        return declared
    n = len(panel_geos)
    if n <= 1:
        return GeoGranularity.NATIONAL
    if n <= 60:
        return GeoGranularity.DMA
    return GeoGranularity.GEO


def _time_overlaps(
    evidence: ExperimentEvidence,
    panel_min: pd.Timestamp | None,
    panel_max: pd.Timestamp | None,
    *,
    panel_min_value: float | None = None,
    panel_max_value: float | None = None,
) -> bool:
    if panel_min_value is not None and panel_max_value is not None:
        try:
            ex_start = float(evidence.time_window.start)
            ex_end = float(evidence.time_window.end)
        except (TypeError, ValueError):
            return False
        return not (ex_end < panel_min_value or ex_start > panel_max_value)
    if panel_min is None or panel_max is None:
        return True
    ex_start = pd.to_datetime(evidence.time_window.start, errors="coerce")
    ex_end = pd.to_datetime(evidence.time_window.end, errors="coerce")
    if pd.isna(ex_start) or pd.isna(ex_end):
        return False
    return not (ex_end < panel_min or ex_start > panel_max)


def _geo_overlap(evidence: ExperimentEvidence, panel_geos: set[str]) -> bool:
    if not evidence.geo_scope:
        return True
    scope = {str(g).upper() for g in evidence.geo_scope}
    panel_upper = {str(g).upper() for g in panel_geos}
    if scope.intersection(panel_upper):
        return True
    national_tokens = {"US", "USA", "NATIONAL", "ALL", "COUNTRY"}
    if (
        evidence.geo_granularity in {GeoGranularity.NATIONAL, GeoGranularity.USER}
        and scope.intersection(national_tokens)
    ):
        if panel_upper.intersection(national_tokens):
            return True
        if len(panel_geos) <= 1:
            return True
        market = str(evidence.metadata.get("market", "")).upper()
        if market and market in scope:
            return bool(panel_geos)
        if evidence.geo_granularity == GeoGranularity.USER and market in {"US", "USA"}:
            return bool(panel_geos)
    return False


def _map_channel(evidence: ExperimentEvidence, ctx: ModelPanelContext) -> str | None:
    mapped = ctx.channel_mapping.get(evidence.channel, evidence.channel)
    if mapped in ctx.channel_columns:
        return mapped
    if evidence.channel in ctx.channel_columns:
        return evidence.channel
    return None


class ExperimentCompatibilityResolver:
    """Resolve whether experiment evidence can replay against an MMM panel scope."""

    def resolve(
        self,
        evidence: ExperimentEvidence,
        ctx: ModelPanelContext,
        *,
        target_kpi: str | None = None,
    ) -> ReplayCompatibilityDecision:
        warnings: list[str] = []
        model_g = _infer_model_granularity(ctx.panel_geos, ctx.model_geo_granularity)
        exp_g = evidence.geo_granularity
        model_rank = _GEO_RANK[model_g]
        exp_rank = _GEO_RANK[exp_g]

        kpi_target = target_kpi or ctx.target_column
        if evidence.kpi != kpi_target and evidence.kpi not in {ctx.target_column, kpi_target}:
            return ReplayCompatibilityDecision(
                compatibility_status=CompatibilityStatus.REJECTED,
                replay_mode=ReplayMode.REJECT_INCOMPATIBLE,
                model_geo_granularity=model_g.value,
                experiment_geo_granularity=exp_g.value,
                supports_model_level_calibration=False,
                supports_subgeo_claims=False,
                allocation_required=False,
                rejection_reason="kpi_mismatch",
                warnings=warnings,
            )

        ch = _map_channel(evidence, ctx)
        if ch is None:
            return ReplayCompatibilityDecision(
                compatibility_status=CompatibilityStatus.REJECTED,
                replay_mode=ReplayMode.REJECT_INCOMPATIBLE,
                model_geo_granularity=model_g.value,
                experiment_geo_granularity=exp_g.value,
                supports_model_level_calibration=False,
                supports_subgeo_claims=False,
                allocation_required=False,
                rejection_reason="channel_not_in_model",
                warnings=warnings,
            )

        if not _time_overlaps(
            evidence,
            ctx.panel_week_min,
            ctx.panel_week_max,
            panel_min_value=ctx.panel_week_min_value,
            panel_max_value=ctx.panel_week_max_value,
        ):
            return ReplayCompatibilityDecision(
                compatibility_status=CompatibilityStatus.REJECTED,
                replay_mode=ReplayMode.REJECT_INCOMPATIBLE,
                model_geo_granularity=model_g.value,
                experiment_geo_granularity=exp_g.value,
                supports_model_level_calibration=False,
                supports_subgeo_claims=False,
                allocation_required=False,
                rejection_reason="time_window_no_overlap",
                warnings=warnings,
            )

        if not _geo_overlap(evidence, ctx.panel_geos):
            return ReplayCompatibilityDecision(
                compatibility_status=CompatibilityStatus.REJECTED,
                replay_mode=ReplayMode.REJECT_INCOMPATIBLE,
                model_geo_granularity=model_g.value,
                experiment_geo_granularity=exp_g.value,
                supports_model_level_calibration=False,
                supports_subgeo_claims=False,
                allocation_required=False,
                rejection_reason="geo_scope_no_overlap",
                warnings=warnings,
            )

        # User-level national experiment on DMA MMM → aggregate only, no DMA claims.
        if exp_g == GeoGranularity.USER and model_g == GeoGranularity.DMA:
            warnings.append(
                "user_level_national_experiment_on_dma_mmm: aggregate_replay_only; no_dma_subgeo_claims"
            )
            return ReplayCompatibilityDecision(
                compatibility_status=CompatibilityStatus.AGGREGATE_ONLY,
                replay_mode=ReplayMode.AGGREGATE_MODEL_TO_EXPERIMENT_SCOPE,
                model_geo_granularity=model_g.value,
                experiment_geo_granularity=exp_g.value,
                supports_model_level_calibration=True,
                supports_subgeo_claims=False,
                allocation_required=False,
                warnings=warnings,
            )

        # User-level national on national MMM → direct aggregate replay.
        if exp_g == GeoGranularity.USER and model_g == GeoGranularity.NATIONAL:
            return ReplayCompatibilityDecision(
                compatibility_status=CompatibilityStatus.COMPATIBLE,
                replay_mode=ReplayMode.DIRECT_SAME_GRAIN,
                model_geo_granularity=model_g.value,
                experiment_geo_granularity=exp_g.value,
                supports_model_level_calibration=True,
                supports_subgeo_claims=False,
                allocation_required=False,
                warnings=warnings,
            )

        if model_rank == exp_rank:
            return ReplayCompatibilityDecision(
                compatibility_status=CompatibilityStatus.COMPATIBLE,
                replay_mode=ReplayMode.DIRECT_SAME_GRAIN,
                model_geo_granularity=model_g.value,
                experiment_geo_granularity=exp_g.value,
                supports_model_level_calibration=True,
                supports_subgeo_claims=exp_g in {GeoGranularity.DMA, GeoGranularity.GEO},
                allocation_required=False,
                warnings=warnings,
            )

        if model_rank > exp_rank:
            # Model more granular than experiment → aggregate model to experiment scope.
            warnings.append("model_more_granular_than_experiment: aggregate_model_outputs_to_experiment_scope")
            return ReplayCompatibilityDecision(
                compatibility_status=CompatibilityStatus.AGGREGATE_ONLY,
                replay_mode=ReplayMode.AGGREGATE_MODEL_TO_EXPERIMENT_SCOPE,
                model_geo_granularity=model_g.value,
                experiment_geo_granularity=exp_g.value,
                supports_model_level_calibration=True,
                supports_subgeo_claims=False,
                allocation_required=False,
                warnings=warnings,
            )

        # Experiment more granular than model.
        if exp_rank > model_rank + 1:
            warnings.append(
                "experiment_much_more_granular_than_model: diagnostic_only unless design is representative"
            )
            return ReplayCompatibilityDecision(
                compatibility_status=CompatibilityStatus.DIAGNOSTIC_ONLY,
                replay_mode=ReplayMode.AGGREGATE_MODEL_TO_EXPERIMENT_SCOPE,
                model_geo_granularity=model_g.value,
                experiment_geo_granularity=exp_g.value,
                supports_model_level_calibration=False,
                supports_subgeo_claims=False,
                allocation_required=False,
                warnings=warnings,
            )

        # One level finer experiment → allocation may be required for replay bridge.
        methods = [
            "observed_withheld_spend",
            "observed_withheld_impressions",
            "eligible_user_weighted",
            "impression_or_opportunity_weighted",
            "observed_spend_weighted",
            "baseline_conversion_weighted",
            "uniform",
        ]
        warnings.append(
            "experiment_more_granular_than_model: allocation_is_computational_bridge_only; "
            "not_experimental_dma_truth"
        )
        return ReplayCompatibilityDecision(
            compatibility_status=CompatibilityStatus.ALLOCATION_REQUIRED,
            replay_mode=ReplayMode.ALLOCATE_EXPERIMENT_SHOCK_TO_MODEL_UNITS,
            model_geo_granularity=model_g.value,
            experiment_geo_granularity=exp_g.value,
            supports_model_level_calibration=False,
            supports_subgeo_claims=False,
            allocation_required=True,
            allowed_allocation_methods=methods,
            warnings=warnings,
        )
