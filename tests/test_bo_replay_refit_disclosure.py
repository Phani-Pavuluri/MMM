"""BO replay refit semantics disclosed in trial objective detail (audit fix P3)."""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd

from mmm.calibration.contracts import CalibrationUnit
from mmm.calibration.evidence_replay import EvidenceReplayPrepareResult, WeightedReplayEntry
from mmm.config.schema import CalibrationConfig, DataConfig, Framework, MMMConfig, ModelForm
from mmm.data.schema import PanelSchema
from mmm.models.ridge_bo.trainer import RidgeBOMMMTrainer


def test_fit_best_detail_includes_replay_refit_disclosure(tmp_path) -> None:
    rows = [{"geo_id": "G0", "week_start_date": w, "revenue": 10.0, "tv": 1.0} for w in range(12)]
    panel = pd.DataFrame(rows)
    schema = PanelSchema("geo_id", "week_start_date", "revenue", ("tv",))
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
            "week_end": "11",
            "aggregation": "mean",
            "target_kpi_column": "revenue",
            "lift_scale": "mean_kpi_level_delta",
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
    bd = out.get("best_detail") or {}
    assert bd.get("calibration_refit_mode") == "full_panel_same_hyperparameters"
    assert bd.get("replay_uses_full_panel_refit") is True
    assert isinstance(bd.get("replay_overfit_warning"), str)
    assert isinstance(bd.get("train_vs_holdout_replay_loss"), dict)
