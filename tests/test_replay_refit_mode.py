"""Fold-aligned replay refit mode and BO objective honesty."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from mmm.calibration.contracts import CalibrationUnit
from mmm.calibration.replay_estimand import REPLAY_TRANSFORM_MODE_FULL_PANEL
from mmm.calibration.replay_frames import build_calibration_unit_from_shift
from mmm.calibration.replay_refit_mode import (
    FULL_PANEL_REPLAY_OPTIMISM_WARNING,
    replay_refit_enters_objective,
    validate_replay_refit_mode,
)
from mmm.calibration.units_io import write_calibration_units_to_json
from mmm.config.schema import CalibrationConfig, CVConfig, Framework, MMMConfig, ModelForm
from mmm.data.schema import PanelSchema
from mmm.models.ridge_bo.trainer import RidgeBOMMMTrainer
from mmm.utils.synthetic import SyntheticGeoPanelSpec, generate_geo_panel


def test_replay_refit_mode_enum_validation() -> None:
    assert validate_replay_refit_mode("full_panel_refit") == "full_panel_refit"
    assert validate_replay_refit_mode("fold_aligned") == "fold_aligned"
    with pytest.raises(ValueError, match="replay_refit_mode"):
        validate_replay_refit_mode("invalid")


def test_default_backward_compatible_full_panel() -> None:
    cfg = MMMConfig(data={"channel_columns": ["c1"]})
    assert cfg.calibration.replay_refit_mode == "full_panel_refit"
    assert replay_refit_enters_objective("full_panel_refit", use_replay_calibration=True) is True
    assert replay_refit_enters_objective("holdout_only_diagnostic", use_replay_calibration=True) is False


def _replay_unit(panel: pd.DataFrame, schema: PanelSchema, mult: float = 1.2) -> CalibrationUnit:
    wcol = schema.week_column
    weeks = sorted(panel[wcol].unique())
    week_start, week_end = weeks[10], weeks[18]
    u = build_calibration_unit_from_shift(
        panel,
        schema,
        unit_id="u1",
        channel="c1",
        geo_ids=["G0"],
        week_start=week_start,
        week_end=week_end,
        spend_multiplier=mult,
        observed_lift=5.0,
        lift_se=1.0,
        replay_estimand={
            "geo_scope": "listed",
            "geo_ids": ["G0"],
            "week_start": week_start,
            "week_end": week_end,
            "aggregation": "mean",
            "target_kpi_column": "revenue",
            "lift_scale": "mean_kpi_level_delta",
            "replay_transform_mode": REPLAY_TRANSFORM_MODE_FULL_PANEL,
        },
    )
    assert u is not None
    return u


def test_fold_aligned_vs_full_panel_differ_on_leakage_sensitive_fixture(tmp_path: Path) -> None:
    spec = SyntheticGeoPanelSpec(n_geos=1, n_weeks=40, channels=("c1",), betas=(0.8,))
    panel, schema = generate_geo_panel(spec, seed=7)
    unit = _replay_unit(panel, schema)
    path = tmp_path / "units.json"
    write_calibration_units_to_json([unit], path)
    base_cv = CVConfig(mode="rolling", n_splits=3, min_train_weeks=15, horizon_weeks=4)
    base_data = {
        "geo_column": schema.geo_column,
        "week_column": schema.week_column,
        "target_column": schema.target_column,
        "channel_columns": list(schema.channel_columns),
    }
    cfg_full = MMMConfig(
        framework=Framework.RIDGE_BO,
        model_form=ModelForm.SEMI_LOG,
        data=base_data,
        cv=base_cv,
        ridge_bo={"n_trials": 2},
        calibration=CalibrationConfig(
            use_replay_calibration=True,
            replay_units_path=str(path),
            replay_refit_mode="full_panel_refit",
        ),
        random_seed=11,
    )
    cfg_fold = cfg_full.model_copy(
        update={"calibration": cfg_full.calibration.model_copy(update={"replay_refit_mode": "fold_aligned"})}
    )
    out_full = RidgeBOMMMTrainer(cfg_full, schema).fit(panel)
    out_fold = RidgeBOMMMTrainer(cfg_fold, schema).fit(panel)
    bd_full = out_full["best_detail"]
    bd_fold = out_fold["best_detail"]
    assert bd_full.get("replay_uses_full_panel_refit") is True
    assert bd_fold.get("replay_uses_full_panel_refit") is False
    assert bd_fold.get("calibration_refit_mode") == "fold_aligned_cv"
    assert FULL_PANEL_REPLAY_OPTIMISM_WARNING in str(bd_full.get("replay_overfit_warning", ""))
    assert bd_full.get("replay_train_loss") != bd_fold.get("replay_train_loss") or bd_fold.get("fold_replay_losses")


def test_holdout_only_diagnostic_zero_objective_replay(tmp_path: Path) -> None:
    spec = SyntheticGeoPanelSpec(n_geos=1, n_weeks=30, channels=("c1",), betas=(0.5,))
    panel, schema = generate_geo_panel(spec, seed=3)
    unit = _replay_unit(panel, schema)
    write_calibration_units_to_json([unit], tmp_path / "u.json")
    cfg = MMMConfig(
        framework=Framework.RIDGE_BO,
        data={
            "geo_column": schema.geo_column,
            "week_column": schema.week_column,
            "target_column": schema.target_column,
            "channel_columns": list(schema.channel_columns),
        },
        cv=CVConfig(mode="rolling", n_splits=2, min_train_weeks=12, horizon_weeks=3),
        ridge_bo={"n_trials": 1},
        calibration={
            "use_replay_calibration": True,
            "replay_units_path": str(tmp_path / "u.json"),
            "replay_refit_mode": "holdout_only_diagnostic",
        },
        random_seed=5,
    )
    out = RidgeBOMMMTrainer(cfg, schema).fit(panel)
    bd = out["best_detail"]
    assert bd.get("calibration_refit_mode") == "holdout_diagnostic_only"
    assert bd.get("calibration_score_source") == "predictive_only_replay_holdout_diagnostic"
    tvr = bd.get("train_vs_holdout_replay_loss") or {}
    assert float(tvr.get("replay_loss_in_objective", -1)) == 0.0
