"""BO replay holdout generalization disclosure (advisory; objective unchanged)."""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd

from mmm.calibration.contracts import CalibrationUnit
from mmm.calibration.evidence_replay import EvidenceReplayPrepareResult, WeightedReplayEntry
from mmm.calibration.replay_generalization import build_replay_calibration_metadata
from mmm.config.schema import CalibrationConfig, DataConfig, Framework, MMMConfig, ModelForm
from mmm.data.schema import PanelSchema
from mmm.governance.model_release import ModelReleaseState, infer_model_release_state
from mmm.models.ridge_bo.trainer import RidgeBOMMMTrainer


def _panel() -> tuple[pd.DataFrame, PanelSchema]:
    rows = [{"geo_id": "G0", "week_start_date": w, "revenue": 50.0, "tv": 2.0} for w in range(14)]
    schema = PanelSchema("geo_id", "week_start_date", "revenue", ("tv",))
    return pd.DataFrame(rows), schema


def test_no_holdout_warning_absent() -> None:
    meta = build_replay_calibration_metadata(
        train_loss=0.2,
        holdout_loss=None,
        n_units=1,
        replay_mode_used="legacy",
        replay_transform_mode="full_panel_transform_estimand_mask",
    )
    assert meta["replay_holdout_available"] is False
    assert meta["replay_overfit_warning"] == ""


def test_moderate_and_severe_gap_detected() -> None:
    mod = build_replay_calibration_metadata(
        train_loss=0.1,
        holdout_loss=0.2,
        n_units=2,
        replay_mode_used="legacy",
        replay_transform_mode="full_panel_transform_estimand_mask",
    )
    assert mod["replay_generalization_gap_severity"] == "moderate"
    assert mod["replay_overfit_warning"]
    sev = build_replay_calibration_metadata(
        train_loss=0.1,
        holdout_loss=0.4,
        n_units=2,
        replay_mode_used="legacy",
        replay_transform_mode="full_panel_transform_estimand_mask",
        gap_severe_threshold=0.25,
    )
    assert sev["replay_generalization_gap_severity"] == "severe"


def test_trainer_best_detail_contains_replay_fields(tmp_path) -> None:
    panel, schema = _panel()
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
            evidence_registry_path=str(tmp_path / "e.json"),
            compatibility_resolver_enabled=True,
        ),
        ridge_bo={"n_trials": 1},
        cv={"mode": "rolling", "n_splits": 2, "min_train_weeks": 4, "horizon_weeks": 2},
    )
    (tmp_path / "e.json").write_text("[]", encoding="utf-8")
    unit = CalibrationUnit(
        unit_id="u1",
        treated_channel_names=["tv"],
        observed_spend_frame=panel,
        counterfactual_spend_frame=panel,
        observed_lift=1.0,
        lift_se=1.0,
        replay_estimand={
            "geo_scope": "all",
            "geo_ids": [],
            "week_start": "0",
            "week_end": "13",
            "aggregation": "mean",
            "target_kpi_column": "revenue",
            "lift_scale": "mean_kpi_level_delta",
            "replay_transform_mode": "full_panel_transform_estimand_mask",
        },
    )
    prep = EvidenceReplayPrepareResult(
        used=[
            WeightedReplayEntry(
                unit=unit,
                evidence_weight=1.0,
                experiment_id="u1",
                channel="tv",
                compatibility_status="compatible",
                quality_tier="high",
                replay_mode="direct_same_grain",
            )
        ]
    )
    with patch("mmm.models.ridge_bo.trainer.prepare_evidence_replay", return_value=prep):
        out = RidgeBOMMMTrainer(cfg, schema).fit(panel)
    bd = out["best_detail"]
    assert bd.get("replay_holdout_available") is True
    assert "replay_train_loss" in bd
    assert "replay_generalization_gap" in bd


def test_severe_gap_optional_blocking() -> None:
    cfg = MMMConfig(framework=Framework.RIDGE_BO, data={"channel_columns": ["tv"]})
    release = infer_model_release_state(
        config=cfg,
        panel_qa_max_severity="info",
        governance_approved_for_optimization=True,
        governance_approved_for_reporting=True,
        ridge_fit_summary_present=True,
        invalidation_reasons=["severe_replay_generalization_gap"],
    )
    assert release["state"] == ModelReleaseState.INVALIDATED.value

    cfg2 = MMMConfig(
        framework=Framework.RIDGE_BO,
        data={"channel_columns": ["tv"]},
        calibration={"block_on_severe_replay_gap": False},
    )
    release2 = infer_model_release_state(
        config=cfg2,
        panel_qa_max_severity="info",
        governance_approved_for_optimization=True,
        governance_approved_for_reporting=True,
        ridge_fit_summary_present=True,
        invalidation_reasons=[],
    )
    assert release2["state"] == ModelReleaseState.PLANNING_ALLOWED.value
