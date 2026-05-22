"""Legacy and evidence-registry replay share full-panel transform semantics."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from mmm.calibration.evidence_replay import (
    WeightedReplayEntry,
    aggregate_weighted_evidence_replay_loss,
    build_calibration_unit_from_evidence,
)
from mmm.calibration.replay_estimand import REPLAY_TRANSFORM_MODE_FULL_PANEL
from mmm.calibration.replay_etl import SpendShiftSpec, build_replay_units_from_panel_shifts
from mmm.calibration.replay_frames import LEGACY_REPLAY_DEPRECATED_WARNING, normalize_replay_units_to_full_panel
from mmm.calibration.replay_generalization import replay_generalization_gap_severity
from mmm.calibration.replay_lift import aggregate_replay_calibration_loss
from mmm.config.schema import Framework, MMMConfig, ModelForm
from mmm.data.schema import PanelSchema
from mmm.experiments.compatibility import ExperimentCompatibilityResolver, ModelPanelContext
from mmm.experiments.evidence import (
    ApprovalStatus,
    ExperimentEvidence,
    ExperimentType,
    GeoGranularity,
    TimeWindow,
)
from mmm.features.design_matrix import build_design_matrix
from mmm.models.ridge_bo.ridge import fit_ridge, predict_ridge


def _panel() -> tuple[pd.DataFrame, PanelSchema]:
    rows = []
    for w in range(16):
        rows.append(
            {
                "geo_id": "G0",
                "week_start_date": w,
                "revenue": 120.0,
                "tv": 30.0 if w < 4 else 6.0,
            }
        )
    schema = PanelSchema("geo_id", "week_start_date", "revenue", ("tv",))
    return pd.DataFrame(rows), schema


def _predict(panel: pd.DataFrame, schema: PanelSchema, cfg: MMMConfig):
    bundle = build_design_matrix(panel, schema, cfg, decay=0.55, hill_half=1.0, hill_slope=2.0)
    coef, intercept = fit_ridge(bundle.X, bundle.y_modeling, 0.5)

    def predict_level(dfp: pd.DataFrame) -> np.ndarray:
        b = build_design_matrix(dfp, schema, cfg, decay=0.55, hill_half=1.0, hill_slope=2.0)
        return np.exp(predict_ridge(b.X, coef, intercept))

    return predict_level


def test_legacy_etl_matches_evidence_registry_implied_lift() -> None:
    panel, schema = _panel()
    cfg = MMMConfig(
        framework=Framework.RIDGE_BO,
        model_form=ModelForm.SEMI_LOG,
        data={
            "geo_column": "geo_id",
            "week_column": "week_start_date",
            "target_column": "revenue",
            "channel_columns": ["tv"],
        },
    )
    shift = SpendShiftSpec(
        unit_id="exp-a",
        channel="tv",
        spend_multiplier=0.6,
        geo_ids=["G0"],
        week_start=8,
        week_end=12,
        observed_lift=1.0,
        lift_se=0.5,
    )
    legacy_units = build_replay_units_from_panel_shifts(panel, schema, [shift], target_kpi="revenue")
    assert len(legacy_units) == 1
    assert legacy_units[0].replay_estimand["replay_transform_mode"] == REPLAY_TRANSFORM_MODE_FULL_PANEL
    assert len(legacy_units[0].observed_spend_frame) == len(panel)

    ev = ExperimentEvidence(
        experiment_id="exp-a",
        experiment_type=ExperimentType.GEOX,
        channel="tv",
        kpi="revenue",
        estimand="ATT",
        lift_estimate=1.0,
        standard_error=0.5,
        metadata={"spend_multiplier": 0.6},
        time_window=TimeWindow(start="8", end="12"),
        geo_scope=["G0"],
        geo_granularity=GeoGranularity.GEO,
        source_system="t",
        freshness_date=date.today().isoformat(),
        approval_status=ApprovalStatus.ACCEPTED,
    )
    ctx = ModelPanelContext(
        geo_column="geo_id",
        channel_columns=("tv",),
        target_column="revenue",
        panel_geos={"G0"},
        model_geo_granularity=GeoGranularity.GEO,
    )
    compat = ExperimentCompatibilityResolver().resolve(ev, ctx, target_kpi="revenue")
    ev_unit = build_calibration_unit_from_evidence(ev, panel, schema, channel="tv", compat=compat)
    assert ev_unit is not None

    predict_level = _predict(panel, schema, cfg)
    legacy_loss, legacy_meta = aggregate_replay_calibration_loss(
        legacy_units, predict_level, schema=schema, target_col="revenue"
    )
    ent = WeightedReplayEntry(
        unit=ev_unit,
        evidence_weight=1.0,
        experiment_id="exp-a",
        channel="tv",
        compatibility_status="compatible",
        quality_tier="high",
        replay_mode="direct_same_grain",
    )
    ev_loss, ev_meta = aggregate_weighted_evidence_replay_loss(
        [ent], predict_level, schema=schema, target_col="revenue"
    )
    assert abs(legacy_meta["units"][0]["implied_delta"] - ev_meta["units"][0]["implied_delta"]) < 1e-5
    assert abs(legacy_loss - ev_loss) < 1e-5


def test_normalize_upgrades_slice_units() -> None:
    panel, schema = _panel()
    slice_df = panel.loc[panel["week_start_date"].between(8, 12)].copy()
    unit_dict = {
        "unit_id": "u0",
        "treated_channel_names": ["tv"],
        "observed_spend_frame": slice_df.to_dict(orient="records"),
        "counterfactual_spend_frame": (slice_df.assign(tv=slice_df["tv"] * 0.6)).to_dict(orient="records"),
        "observed_lift": 1.0,
        "lift_se": 0.5,
        "replay_estimand": {
            "geo_scope": "listed",
            "geo_ids": ["G0"],
            "week_start": 8,
            "week_end": 12,
            "aggregation": "mean",
            "target_kpi_column": "revenue",
            "lift_scale": "mean_kpi_level_delta",
        },
    }
    import json
    import tempfile
    from pathlib import Path

    from mmm.calibration.units_io import load_calibration_units_from_json

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "units.json"
        p.write_text(json.dumps([unit_dict]), encoding="utf-8")
        loaded = load_calibration_units_from_json(p)
    upgraded, warnings = normalize_replay_units_to_full_panel(panel, schema, loaded)
    assert len(upgraded[0].observed_spend_frame) == len(panel)
    assert any("upgraded" in w for w in warnings)


def test_slice_without_estimand_emits_deprecated_warning() -> None:
    from mmm.calibration.contracts import CalibrationUnit

    panel, schema = _panel()
    slice_df = panel.head(3)
    unit = CalibrationUnit(
        unit_id="bad",
        treated_channel_names=["tv"],
        observed_spend_frame=slice_df,
        counterfactual_spend_frame=slice_df,
        observed_lift=1.0,
        lift_se=1.0,
        replay_estimand=None,
    )
    _, meta = aggregate_replay_calibration_loss(
        [unit],
        lambda dfp: np.full(len(dfp), 100.0),
        schema=schema,
        target_col="revenue",
    )
    assert any(LEGACY_REPLAY_DEPRECATED_WARNING in w for w in meta.get("legacy_replay_warnings", []))


def test_replay_gap_severity_thresholds() -> None:
    assert replay_generalization_gap_severity(0.05) == "none"
    assert replay_generalization_gap_severity(0.15) == "moderate"
    assert replay_generalization_gap_severity(0.3) == "severe"
    assert replay_generalization_gap_severity(None) == "none"
