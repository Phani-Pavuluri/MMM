"""Evidence replay must preserve full-panel adstock state (audit fix P1)."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from mmm.calibration.evidence_replay import build_calibration_unit_from_evidence
from mmm.calibration.replay_estimand import (
    REPLAY_TRANSFORM_MODE_FULL_PANEL,
    ReplayEstimandSpec,
    eval_mask_for_replay,
)
from mmm.calibration.replay_frames import build_full_panel_replay_frames
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
from mmm.transforms.stack import build_channel_features_from_params


def _panel_with_pre_window_impulse() -> tuple[pd.DataFrame, PanelSchema]:
    rows = []
    for w in range(16):
        tv = 50.0 if w < 4 else 5.0
        rows.append(
            {
                "geo_id": "G0",
                "week_start_date": w,
                "revenue": 200.0,
                "tv": tv,
            }
        )
    schema = PanelSchema("geo_id", "week_start_date", "revenue", ("tv",))
    return pd.DataFrame(rows), schema


def _evidence_window() -> ExperimentEvidence:
    return ExperimentEvidence(
        experiment_id="pre-adstock",
        experiment_type=ExperimentType.GEOX,
        channel="tv",
        kpi="revenue",
        estimand="ATT",
        lift_estimate=1.0,
        standard_error=0.5,
        spend_delta=10.0,
        metadata={"spend_multiplier": 0.5},
        time_window=TimeWindow(start="8", end="12"),
        geo_scope=["G0"],
        geo_granularity=GeoGranularity.GEO,
        source_system="test",
        freshness_date=date.today().isoformat(),
        approval_status=ApprovalStatus.ACCEPTED,
    )


def _legacy_slice_unit(
    panel: pd.DataFrame,
    schema: PanelSchema,
    channel: str,
    spec: ReplayEstimandSpec,
    mult: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    gcol, wcol = schema.geo_column, schema.week_column
    if spec.geo_scope == "all":
        geo_mask = pd.Series(True, index=panel.index)
    else:
        geo_mask = panel[gcol].astype(str).isin(set(spec.geo_ids))
    time_mask = (panel[wcol].astype(float) >= float(spec.week_start)) & (
        panel[wcol].astype(float) <= float(spec.week_end)
    )
    m = geo_mask & time_mask
    obs = panel.loc[m].copy()
    cf = obs.copy()
    cf[channel] = cf[channel].astype(float) * mult
    return obs.reset_index(drop=True), cf.reset_index(drop=True)


def test_calibration_unit_uses_full_panel_frames() -> None:
    panel, schema = _panel_with_pre_window_impulse()
    ev = _evidence_window()
    ctx = ModelPanelContext(
        geo_column="geo_id",
        channel_columns=("tv",),
        target_column="revenue",
        panel_geos={"G0"},
        model_geo_granularity=GeoGranularity.GEO,
    )
    compat = ExperimentCompatibilityResolver().resolve(ev, ctx, target_kpi="revenue")
    unit = build_calibration_unit_from_evidence(ev, panel, schema, channel="tv", compat=compat)
    assert unit is not None
    assert len(unit.observed_spend_frame) == len(panel)
    assert unit.replay_estimand is not None
    assert unit.replay_estimand.get("replay_transform_mode") == REPLAY_TRANSFORM_MODE_FULL_PANEL


def test_adstock_carryover_differs_full_panel_vs_window_slice() -> None:
    """Pre-window spend must change transformed media inside the experiment window."""
    panel, schema = _panel_with_pre_window_impulse()
    ev = _evidence_window()
    ctx = ModelPanelContext(
        geo_column="geo_id",
        channel_columns=("tv",),
        target_column="revenue",
        panel_geos={"G0"},
        model_geo_granularity=GeoGranularity.GEO,
    )
    compat = ExperimentCompatibilityResolver().resolve(ev, ctx, target_kpi="revenue")
    unit = build_calibration_unit_from_evidence(ev, panel, schema, channel="tv", compat=compat)
    assert unit is not None and unit.replay_estimand is not None
    spec = ReplayEstimandSpec.from_dict(unit.replay_estimand)
    slice_obs, _ = _legacy_slice_unit(panel, schema, "tv", spec, 0.5)
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
    x_full = build_channel_features_from_params(
        unit.observed_spend_frame, schema, cfg.transforms, decay=0.6, hill_half=1.0, hill_slope=2.0
    )
    x_slice = build_channel_features_from_params(
        slice_obs, schema, cfg.transforms, decay=0.6, hill_half=1.0, hill_slope=2.0
    )
    mask = eval_mask_for_replay(unit.observed_spend_frame, schema, spec)
    assert float(np.max(np.abs(x_full[mask, 0] - x_slice[:, 0]))) > 1e-4


def test_full_panel_frames_match_replay_helper() -> None:
    panel, schema = _panel_with_pre_window_impulse()
    spec = ReplayEstimandSpec.from_dict(
        {
            "geo_scope": "listed",
            "geo_ids": ["G0"],
            "week_start": "8",
            "week_end": "12",
            "aggregation": "mean",
            "target_kpi_column": "revenue",
            "lift_scale": "mean_kpi_level_delta",
            "replay_transform_mode": REPLAY_TRANSFORM_MODE_FULL_PANEL,
        }
    )
    frames = build_full_panel_replay_frames(panel, schema, spec, "tv", 0.5)
    assert frames is not None
    obs, cf = frames
    assert len(obs) == len(panel)
    assert (obs["tv"].iloc[:4] == cf["tv"].iloc[:4]).all()
    assert (cf["tv"].iloc[8:13] < obs["tv"].iloc[8:13]).any()
