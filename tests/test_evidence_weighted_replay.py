"""PR 2: Ridge weighted replay from experiment evidence registry."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from mmm.calibration.evidence_replay import (
    aggregate_weighted_evidence_replay_loss,
    build_calibration_unit_from_evidence,
    prepare_evidence_replay,
    uses_legacy_replay,
    uses_weighted_evidence_replay,
)
from mmm.calibration.replay_lift import aggregate_replay_calibration_loss
from mmm.config.schema import CalibrationConfig, DataConfig, Framework, MMMConfig, ModelForm, RunEnvironment
from mmm.data.schema import PanelSchema
from mmm.evaluation.calibration_extension import compute_replay_calibration_metrics
from mmm.experiments.compatibility import ExperimentCompatibilityResolver, ModelPanelContext
from mmm.experiments.evidence import (
    ApprovalStatus,
    ExperimentEvidence,
    ExperimentType,
    GeoGranularity,
    TimeWindow,
)
from mmm.experiments.evidence_quality import weighted_replay_loss_term
from mmm.models.ridge_bo.trainer import RidgeBOMMMTrainer


def _evidence(**kwargs: object) -> ExperimentEvidence:
    d = dict(
        experiment_id="e1",
        experiment_type=ExperimentType.GEOX,
        channel="tv",
        kpi="revenue",
        estimand="ATT",
        lift_estimate=10.0,
        standard_error=2.0,
        spend_delta=100.0,
        metadata={"spend_multiplier": 0.9, "market": "US"},
        time_window=TimeWindow(start="0", end="11"),
        geo_scope=["US"],
        geo_granularity=GeoGranularity.NATIONAL,
        source_system="x",
        freshness_date=date.today().isoformat(),
        approval_status=ApprovalStatus.ACCEPTED,
    )
    d.update(kwargs)
    return ExperimentEvidence(**d)  # type: ignore[arg-type]


def _panel(n_geos: int = 4) -> tuple[pd.DataFrame, PanelSchema]:
    rows = []
    for g in range(n_geos):
        for w in range(12):
            rows.append(
                {
                    "geo_id": f"dma_{g}",
                    "week_start_date": w,
                    "revenue": 100.0 + w,
                    "tv": 10.0 + g,
                }
            )
    schema = PanelSchema("geo_id", "week_start_date", "revenue", ("tv",))
    return pd.DataFrame(rows), schema


def _write_registry(tmp_path: Path, *evidence: ExperimentEvidence) -> Path:
    p = tmp_path / "evidence.json"
    data = {
        "registry_version": "mmm_experiment_evidence_registry_v1",
        "experiments": {e.experiment_id: e.to_registry_dict() for e in evidence},
    }
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def test_legacy_replay_unchanged_config() -> None:
    cfg = MMMConfig(
        framework=Framework.RIDGE_BO,
        data=DataConfig(channel_columns=["tv"]),
        calibration=CalibrationConfig(
            use_replay_calibration=True,
            replay_units_path="units.json",
            replay_mode="legacy",
        ),
    )
    assert uses_legacy_replay(cfg)
    assert not uses_weighted_evidence_replay(cfg)


def test_missing_registry_path_fails_at_config() -> None:
    with pytest.raises(ValueError, match="evidence_registry_path"):
        MMMConfig(
            framework=Framework.RIDGE_BO,
            data=DataConfig(channel_columns=["tv"]),
            calibration=CalibrationConfig(
                replay_mode="evidence_registry",
                compatibility_resolver_enabled=True,
            ),
        )


def test_compatibility_disabled_fails_at_config() -> None:
    with pytest.raises(ValueError, match="compatibility_resolver_enabled"):
        MMMConfig(
            framework=Framework.RIDGE_BO,
            data=DataConfig(channel_columns=["tv"]),
            calibration=CalibrationConfig(
                replay_mode="evidence_registry",
                evidence_registry_path="x.json",
            ),
        )


def test_validate_evidence_registry_replay_config_missing_path() -> None:
    with pytest.raises(ValueError, match="evidence_registry_path"):
        CalibrationConfig(
            replay_mode="evidence_registry",
            compatibility_resolver_enabled=True,
            evidence_weighting_enabled=True,
            use_replay_calibration=True,
        )


def test_weighted_replay_loss_formula() -> None:
    # z1=1, z2=4 with weights 1 and 3 -> (1*1 + 3*4)/(1+3) = 13/4
    t1 = weighted_replay_loss_term(11.0, 10.0, 1.0, 1.0)
    t2 = weighted_replay_loss_term(14.0, 10.0, 2.0, 3.0)
    assert abs(t1 - 1.0) < 1e-9
    assert abs(t2 - 3.0 * 4.0) < 1e-9
    loss = (t1 + t2) / (1.0 + 3.0)
    assert abs(loss - 3.25) < 1e-9


def test_rejected_evidence_excluded_from_prepare(tmp_path: Path) -> None:
    panel, schema = _panel()
    bad = _evidence(experiment_id="bad", channel="missing_channel", kpi="other_kpi")
    reg = _write_registry(tmp_path, bad)
    cfg = MMMConfig(
        framework=Framework.RIDGE_BO,
        data=DataConfig(
            channel_columns=["tv"],
            geo_column="geo_id",
            week_column="week_start_date",
            target_column="revenue",
        ),
        calibration=CalibrationConfig(
            use_replay_calibration=True,
            replay_mode="evidence_registry",
            evidence_weighting_enabled=True,
            compatibility_resolver_enabled=True,
            evidence_registry_path=str(reg),
            model_geo_granularity="dma",
        ),
    )
    prep = prepare_evidence_replay(cfg, panel, schema)
    assert prep.n_loaded == 1
    assert len(prep.used) == 0
    assert prep.rejected


def test_aggregate_only_user_national_dma_mmm(tmp_path: Path) -> None:
    panel, schema = _panel()
    ev = _evidence(
        experiment_id="nat_user",
        geo_granularity=GeoGranularity.USER,
        geo_scope=["US"],
        metadata={"market": "US", "spend_multiplier": 0.85},
    )
    reg = _write_registry(tmp_path, ev)
    cfg = MMMConfig(
        framework=Framework.RIDGE_BO,
        data=DataConfig(
            channel_columns=["tv"],
            geo_column="geo_id",
            week_column="week_start_date",
            target_column="revenue",
        ),
        calibration=CalibrationConfig(
            use_replay_calibration=True,
            replay_mode="evidence_registry",
            evidence_weighting_enabled=True,
            compatibility_resolver_enabled=True,
            evidence_registry_path=str(reg),
            model_geo_granularity="dma",
        ),
    )
    prep = prepare_evidence_replay(cfg, panel, schema)
    assert len(prep.used) == 1
    ent = prep.used[0]
    assert ent.supports_subgeo_claims is False
    assert ent.compatibility_status == "aggregate_only"
    assert any("no_subgeo" in w for w in prep.warnings)


def test_missing_se_conservative_in_research(tmp_path: Path) -> None:
    panel, schema = _panel()
    ev = _evidence(experiment_id="nose", standard_error=None)
    reg = _write_registry(tmp_path, ev)
    cfg = MMMConfig(
        framework=Framework.RIDGE_BO,
        run_environment=RunEnvironment.RESEARCH,
        data=DataConfig(
            channel_columns=["tv"],
            geo_column="geo_id",
            week_column="week_start_date",
            target_column="revenue",
        ),
        calibration=CalibrationConfig(
            use_replay_calibration=True,
            replay_mode="evidence_registry",
            evidence_weighting_enabled=True,
            compatibility_resolver_enabled=True,
            evidence_registry_path=str(reg),
            model_geo_granularity="dma",
        ),
    )
    prep = prepare_evidence_replay(cfg, panel, schema)
    assert len(prep.used) == 1
    assert any("conservative" in w for w in prep.warnings)


def test_missing_se_rejected_in_prod_env(tmp_path: Path) -> None:
    panel, schema = _panel()
    ev = _evidence(experiment_id="nose", standard_error=None)
    reg = _write_registry(tmp_path, ev)
    cfg = MMMConfig(
        framework=Framework.RIDGE_BO,
        data=DataConfig(
            channel_columns=["tv"],
            geo_column="geo_id",
            week_column="week_start_date",
            target_column="revenue",
        ),
        calibration=CalibrationConfig(
            use_replay_calibration=True,
            replay_mode="evidence_registry",
            evidence_weighting_enabled=True,
            compatibility_resolver_enabled=True,
            evidence_registry_path=str(reg),
            model_geo_granularity="dma",
        ),
    )
    cfg_prod = cfg.model_copy(update={"run_environment": RunEnvironment.PROD})
    prep = prepare_evidence_replay(cfg_prod, panel, schema)
    assert len(prep.used) == 0
    assert any(r.get("reason") == "missing_or_invalid_standard_error" for r in prep.rejected)


def test_expired_evidence_excluded(tmp_path: Path) -> None:
    panel, schema = _panel()
    ev = _evidence(experiment_id="exp", approval_status=ApprovalStatus.EXPIRED)
    reg = _write_registry(tmp_path, ev)
    cfg = MMMConfig(
        framework=Framework.RIDGE_BO,
        data=DataConfig(
            channel_columns=["tv"],
            geo_column="geo_id",
            week_column="week_start_date",
            target_column="revenue",
        ),
        calibration=CalibrationConfig(
            use_replay_calibration=True,
            replay_mode="evidence_registry",
            evidence_weighting_enabled=True,
            compatibility_resolver_enabled=True,
            evidence_registry_path=str(reg),
            model_geo_granularity="dma",
        ),
    )
    prep = prepare_evidence_replay(cfg, panel, schema)
    assert len(prep.used) == 0
    assert any(r.get("reason") == "expired_approval" for r in prep.rejected)


def test_weighted_vs_legacy_equal_weights(tmp_path: Path) -> None:
    """Single unit: legacy mean z^2 equals weighted sum/sum when weight=1."""
    panel, schema = _panel(n_geos=1)
    ev = _evidence(experiment_id="u1", geo_granularity=GeoGranularity.GEO, geo_scope=["dma_0"])
    ctx = ModelPanelContext(
        geo_column="geo_id",
        channel_columns=("tv",),
        target_column="revenue",
        panel_geos={"dma_0"},
        model_geo_granularity=GeoGranularity.GEO,
    )
    compat = ExperimentCompatibilityResolver().resolve(ev, ctx, target_kpi="revenue")
    unit = build_calibration_unit_from_evidence(ev, panel, schema, channel="tv", compat=compat)
    assert unit is not None

    def predict_fn(dfp: pd.DataFrame) -> np.ndarray:
        return np.full(len(dfp), 100.0)

    legacy_loss, _ = aggregate_replay_calibration_loss(
        [unit], predict_fn, schema=schema, target_col="revenue"
    )
    from mmm.calibration.evidence_replay import WeightedReplayEntry

    ent = WeightedReplayEntry(
        unit=unit,
        evidence_weight=1.0,
        experiment_id="u1",
        channel="tv",
        compatibility_status="compatible",
        quality_tier="high",
        replay_mode="direct_same_grain",
    )
    w_loss, w_meta = aggregate_weighted_evidence_replay_loss(
        [ent], predict_fn, schema=schema, target_col="revenue"
    )
    assert abs(legacy_loss - w_loss) < 1e-6
    assert w_meta["replay_mode_used"] == "evidence_registry"


def test_bo_objective_uses_weighted_replay(tmp_path: Path) -> None:
    panel, schema = _panel(n_geos=2)
    ev = _evidence(
        experiment_id="w1",
        geo_granularity=GeoGranularity.GEO,
        geo_scope=["dma_0", "dma_1"],
        metadata={"spend_multiplier": 0.8, "market": "US"},
    )
    reg = _write_registry(tmp_path, ev)
    cfg = MMMConfig(
        framework=Framework.RIDGE_BO,
        model_form=ModelForm.SEMI_LOG,
        data=DataConfig(
            channel_columns=["tv"],
            geo_column="geo_id",
            week_column="week_start_date",
            target_column="revenue",
        ),
        calibration=CalibrationConfig(
            use_replay_calibration=True,
            replay_mode="evidence_registry",
            evidence_weighting_enabled=True,
            compatibility_resolver_enabled=True,
            evidence_registry_path=str(reg),
            model_geo_granularity="geo",
        ),
        ridge_bo={"n_trials": 2},
        cv={"mode": "rolling", "n_splits": 2, "min_train_weeks": 4, "horizon_weeks": 2},
    )
    from mmm.calibration.contracts import CalibrationUnit
    from mmm.calibration.evidence_replay import EvidenceReplayPrepareResult, WeightedReplayEntry

    slice_df = panel.head(4).copy()
    cf = slice_df.copy()
    cf["tv"] = cf["tv"] * 0.8
    dummy_unit = CalibrationUnit(
        unit_id="w1",
        treated_channel_names=["tv"],
        observed_spend_frame=slice_df,
        counterfactual_spend_frame=cf,
        observed_lift=1.0,
        lift_se=1.0,
        replay_estimand={
            "geo_scope": "all",
            "geo_ids": [],
            "week_start": "0",
            "week_end": "11",
            "aggregation": "mean",
            "target_kpi_column": "revenue",
            "lift_scale": "mean_kpi_level_delta",
        },
    )
    with (
        patch(
            "mmm.models.ridge_bo.trainer.prepare_evidence_replay",
            return_value=EvidenceReplayPrepareResult(
                used=[
                    WeightedReplayEntry(
                        unit=dummy_unit,
                        evidence_weight=0.8,
                        experiment_id="w1",
                        channel="tv",
                        compatibility_status="compatible",
                        quality_tier="high",
                        replay_mode="direct_same_grain",
                    )
                ],
                n_loaded=1,
            ),
        ),
        patch(
            "mmm.models.ridge_bo.trainer.aggregate_weighted_evidence_replay_loss",
            return_value=(0.42, {"weighted_replay_loss": 0.42, "replay_mode_used": "evidence_registry"}),
        ) as mock_loss,
    ):
        tr = RidgeBOMMMTrainer(cfg, schema)
        tr.fit(panel)
        assert mock_loss.called


def test_extension_emits_evidence_weighted_replay_summary(tmp_path: Path) -> None:
    panel, schema = _panel()
    ev = _evidence(
        geo_granularity=GeoGranularity.GEO,
        geo_scope=[f"dma_{g}" for g in range(4)],
        metadata={"spend_multiplier": 0.9, "market": "US"},
    )
    reg = _write_registry(tmp_path, ev)
    cfg = MMMConfig(
        framework=Framework.RIDGE_BO,
        data=DataConfig(
            channel_columns=["tv"],
            geo_column="geo_id",
            week_column="week_start_date",
            target_column="revenue",
        ),
        calibration=CalibrationConfig(
            use_replay_calibration=True,
            replay_mode="evidence_registry",
            evidence_weighting_enabled=True,
            compatibility_resolver_enabled=True,
            evidence_registry_path=str(reg),
            model_geo_granularity="geo",
        ),
    )
    fit_out = {
        "artifacts": type(
            "A",
            (),
            {
                "best_params": {"decay": 0.5, "hill_half": 1.0, "hill_slope": 2.0},
                "coef": np.zeros(1),
                "intercept": np.zeros(1),
            },
        )()
    }
    loss, meta, is_replay = compute_replay_calibration_metrics(panel, schema, cfg, fit_out)
    assert is_replay
    assert loss is not None
    assert "evidence_weighted_replay_summary" in meta
    summary = meta["evidence_weighted_replay_summary"]
    assert summary["n_evidence_units_loaded"] == 1
    assert summary["replay_mode_used"] == "evidence_registry"
