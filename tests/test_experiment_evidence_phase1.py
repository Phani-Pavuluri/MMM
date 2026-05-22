"""Phase 1: experiment evidence contract, registry, compatibility, shock plan, quality."""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

from mmm.config.schema import CalibrationConfig, DataConfig, Framework, MMMConfig, ModelForm
from mmm.data.schema import PanelSchema
from mmm.evaluation.experiment_evidence_extension import build_experiment_evidence_reports
from mmm.experiments.compatibility import (
    CompatibilityStatus,
    ExperimentCompatibilityResolver,
    ModelPanelContext,
    ReplayCompatibilityDecision,
    ReplayMode,
)
from mmm.experiments.evidence import (
    ApprovalStatus,
    ExperimentEvidence,
    ExperimentType,
    GeoGranularity,
    TimeWindow,
)
from mmm.experiments.evidence_quality import EvidenceQualityContext, QualityTier, score_evidence_quality
from mmm.experiments.evidence_registry import ExperimentEvidenceRegistry
from mmm.experiments.shock_plan import CounterfactualShockPlanner


def _base_evidence(**kwargs: object) -> ExperimentEvidence:
    defaults = dict(
        experiment_id="exp-1",
        experiment_type=ExperimentType.GEOX,
        channel="tv",
        kpi="revenue",
        estimand="ATT_geo_time",
        lift_estimate=100.0,
        standard_error=25.0,
        spend_delta=5000.0,
        time_window=TimeWindow(start="2024-01-01", end="2024-06-01"),
        geo_scope=["US"],
        geo_granularity=GeoGranularity.NATIONAL,
        source_system="platform_x",
        freshness_date=date.today().isoformat(),
        approval_status=ApprovalStatus.ACCEPTED,
    )
    defaults.update(kwargs)
    return ExperimentEvidence(**defaults)  # type: ignore[arg-type]


def _panel_ctx(*, geos: set[str], granularity: GeoGranularity = GeoGranularity.DMA) -> ModelPanelContext:
    return ModelPanelContext(
        geo_column="geo_id",
        channel_columns=("tv", "search"),
        target_column="revenue",
        panel_geos=geos,
        panel_week_min=pd.Timestamp("2024-01-01"),
        panel_week_max=pd.Timestamp("2024-12-31"),
        model_geo_granularity=granularity,
    )


def test_registry_register_and_retrieve() -> None:
    reg = ExperimentEvidenceRegistry()
    ev = _base_evidence()
    reg.register(ev)
    reg.mark_accepted("exp-1")
    found = reg.retrieve(channel="tv", kpi="revenue", approval_only=True)
    assert len(found) == 1
    cov = reg.coverage()
    assert cov.n_accepted == 1


def test_registry_rejects_duplicate() -> None:
    reg = ExperimentEvidenceRegistry()
    reg.register(_base_evidence())
    with pytest.raises(ValueError):
        reg.register(_base_evidence())


def test_user_national_on_national_mmm_direct_replay() -> None:
    ev = _base_evidence(
        geo_granularity=GeoGranularity.USER,
        geo_scope=["US"],
        randomization_unit="user",
        metadata={"market": "US"},
    )
    ctx = _panel_ctx(geos={"US"}, granularity=GeoGranularity.NATIONAL)
    dec = ExperimentCompatibilityResolver().resolve(ev, ctx, target_kpi="revenue")
    assert dec.replay_mode == ReplayMode.DIRECT_SAME_GRAIN
    assert dec.supports_model_level_calibration is True


def test_user_national_on_dma_mmm_aggregate_only() -> None:
    ev = _base_evidence(
        geo_granularity=GeoGranularity.USER,
        geo_scope=["US"],
        randomization_unit="user",
        metadata={"market": "US"},
    )
    ctx = _panel_ctx(geos={f"dma_{i}" for i in range(10)}, granularity=GeoGranularity.DMA)
    dec = ExperimentCompatibilityResolver().resolve(ev, ctx, target_kpi="revenue")
    assert dec.replay_mode.value == "aggregate_model_to_experiment_scope"
    assert dec.supports_subgeo_claims is False


def test_dma_experiment_dma_mmm_same_grain() -> None:
    geos = {"501", "502"}
    ev = _base_evidence(geo_granularity=GeoGranularity.DMA, geo_scope=list(geos))
    ctx = _panel_ctx(geos=geos, granularity=GeoGranularity.DMA)
    dec = ExperimentCompatibilityResolver().resolve(ev, ctx, target_kpi="revenue")
    assert dec.replay_mode == ReplayMode.DIRECT_SAME_GRAIN
    assert dec.supports_subgeo_claims is True


def test_us_experiment_emea_panel_rejected() -> None:
    ev = _base_evidence(geo_scope=["US"])
    ctx = _panel_ctx(geos={"UK", "DE", "FR"}, granularity=GeoGranularity.GEO)
    dec = ExperimentCompatibilityResolver().resolve(ev, ctx, target_kpi="revenue")
    assert dec.rejection_reason == "geo_scope_no_overlap"


def test_missing_se_rejected_in_prod_context() -> None:
    ev = _base_evidence(standard_error=None)
    score = score_evidence_quality(ev, EvidenceQualityContext(allow_missing_se=False))
    assert score.quality_tier == QualityTier.REJECTED
    assert score.evidence_weight == 0.0


def test_missing_se_conservative_warning_when_allowed() -> None:
    ev = _base_evidence(standard_error=None)
    score = score_evidence_quality(ev, EvidenceQualityContext(allow_missing_se=True))
    assert score.evidence_weight > 0
    assert "conservative_default_se_used" in score.reasons


def test_stale_experiment_downweighted() -> None:
    old = (date.today() - timedelta(days=400)).isoformat()
    ev = _base_evidence(freshness_date=old)
    score = score_evidence_quality(ev, EvidenceQualityContext(stale_after_days=365))
    assert score.freshness_weight < 1.0
    assert "stale" in score.expiration_status or "stale" in " ".join(score.reasons)


def test_shock_plan_rejects_without_spend_or_exposure() -> None:
    ev = _base_evidence(spend_delta=None, exposure_delta=None)
    compat = ReplayCompatibilityDecision(
        compatibility_status=CompatibilityStatus.COMPATIBLE,
        replay_mode=ReplayMode.DIRECT_SAME_GRAIN,
        model_geo_granularity="dma",
        experiment_geo_granularity="dma",
        supports_model_level_calibration=True,
        supports_subgeo_claims=True,
        allocation_required=False,
    )
    plan = CounterfactualShockPlanner().plan(ev, compat)
    assert plan.allocation_quality == "rejected"


def test_weighted_replay_loss_term() -> None:
    from mmm.experiments.evidence_quality import aggregate_weighted_replay_loss, weighted_replay_loss_term

    t = weighted_replay_loss_term(110.0, 100.0, 10.0, 0.5)
    assert t > 0
    loss, meta = aggregate_weighted_replay_loss([(110.0, 100.0, 10.0, 1.0)])
    assert meta["n_terms"] == 1
    assert loss > 0


def test_evidence_extension_reports(tmp_path: Path) -> None:
    ev = _base_evidence()
    reg_path = tmp_path / "evidence.json"
    reg_path.write_text(
        json.dumps({"registry_version": "mmm_experiment_evidence_registry_v1", "experiments": {}}),
        encoding="utf-8",
    )
    data = {
        "registry_version": "mmm_experiment_evidence_registry_v1",
        "experiments": {ev.experiment_id: ev.to_registry_dict()},
    }
    reg_path.write_text(json.dumps(data), encoding="utf-8")

    rows = []
    for g in range(5):
        rows.append(
            {
                "geo_id": f"dma_{g}",
                "week_start_date": "2024-03-01",
                "revenue": 100.0,
                "tv": 10.0,
                "search": 5.0,
            }
        )
    panel = pd.DataFrame(rows)
    schema = PanelSchema("geo_id", "week_start_date", "revenue", ("tv", "search"))
    cfg = MMMConfig(
        framework=Framework.RIDGE_BO,
        model_form=ModelForm.SEMI_LOG,
        data=DataConfig(
            channel_columns=["tv", "search"],
            geo_column="geo_id",
            week_column="week_start_date",
            target_column="revenue",
        ),
        calibration=CalibrationConfig(
            evidence_registry_path=str(reg_path),
            compatibility_resolver_enabled=True,
            replay_mode="evidence_registry",
            model_geo_granularity="dma",
        ),
    )
    out = build_experiment_evidence_reports(cfg, panel=panel, schema=schema)
    assert "experiment_compatibility_report" in out
    assert "evidence_weighting_report" in out
    assert "counterfactual_shock_plan" in out
    assert out["experiment_compatibility_report"]["n_experiments"] == 1


def test_extension_skipped_when_not_enabled() -> None:
    cfg = MMMConfig(
        framework=Framework.RIDGE_BO,
        model_form=ModelForm.SEMI_LOG,
        data=DataConfig(channel_columns=["tv"]),
    )
    out = build_experiment_evidence_reports(cfg)
    assert out.get("skipped") is True
