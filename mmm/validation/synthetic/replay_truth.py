"""Authoritative replay lift for replay-recovery worlds (Phase 4B-4)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from mmm.calibration.replay_estimand import REPLAY_TRANSFORM_MODE_FULL_PANEL, ReplayEstimandSpec
from mmm.calibration.replay_frames import build_calibration_unit_from_shift, build_full_panel_replay_frames
from mmm.calibration.replay_lift import implied_lift_from_counterfactual
from mmm.config.schema import MMMConfig
from mmm.data.schema import PanelSchema
from mmm.features.design_matrix import build_design_matrix
from mmm.models.ridge_bo.ridge import predict_ridge
from mmm.transforms.stack import build_channel_features_from_params
from mmm.validation.synthetic.optimizer_truth import shared_ridge_transform_params

MIN_TRUE_REPLAY_LIFT = 0.005


def _truth_coef_vector(truth: dict[str, Any], channels: list[str]) -> np.ndarray:
    betas = truth["coefficient_truth"]["true_beta_by_channel"]
    return np.array([float(betas[c]) for c in channels], dtype=float)


def truth_predict_level(
    truth: dict[str, Any],
    schema: PanelSchema,
    config: MMMConfig,
) -> Any:
    """Level-KPI predict_fn using authoritative coefficients and transform hyperparameters."""

    def predict_fn(dfp: pd.DataFrame) -> np.ndarray:
        shared = shared_ridge_transform_params(truth)
        channels = list(truth["media_truth"]["channels"])
        coef = _truth_coef_vector(truth, channels)
        intercept = np.array([float(truth["coefficient_truth"]["intercept"])], dtype=float)
        bundle = build_design_matrix(
            dfp,
            schema,
            config,
            decay=shared["decay"],
            hill_half=shared["hill_half"],
            hill_slope=shared["hill_slope"],
        )
        ylog = predict_ridge(bundle.X, coef, intercept)
        return np.exp(ylog)

    return predict_fn


def replay_estimand_from_unit(unit: dict[str, Any], *, target_kpi: str) -> ReplayEstimandSpec:
    return ReplayEstimandSpec.from_dict(
        {
            "aggregation": str(unit.get("aggregation", "mean")),
            "geo_ids": list(unit["geos"]),
            "geo_scope": str(unit.get("geo_scope", "listed")),
            "lift_scale": str((unit.get("lift_definition") or {}).get("scale", "mean_kpi_level_delta")),
            "notes": "WORLD-010 replay recovery",
            "replay_transform_mode": str(
                unit.get("replay_transform_mode", REPLAY_TRANSFORM_MODE_FULL_PANEL)
            ),
            "target_kpi_column": target_kpi,
            "week_end": str(unit["week_end"]),
            "week_start": str(unit["week_start"]),
        }
    )


def build_calibration_unit_for_experiment(
    truth: dict[str, Any],
    panel: pd.DataFrame,
    schema: PanelSchema,
    unit: dict[str, Any],
) -> Any:
    """Full-panel replay unit from experiment_truth (production frame semantics)."""
    target_kpi = str(truth["outcome_truth"]["target_column"])
    lift = unit.get("lift_definition") or {}
    unc = unit.get("uncertainty") or {}
    spec = replay_estimand_from_unit(unit, target_kpi=target_kpi)
    mult = float(unit["counterfactual_spend_multiplier"])
    return build_calibration_unit_from_shift(
        panel,
        schema,
        unit_id=str(unit["unit_id"]),
        channel=str(unit["channel"]),
        geo_ids=list(unit["geos"]),
        week_start=unit["week_start"],
        week_end=unit["week_end"],
        spend_multiplier=mult,
        observed_lift=float(lift.get("value", 0.0)),
        lift_se=float(unc.get("se", 0.08)),
        target_kpi=target_kpi,
        estimand=str(unit.get("estimand", "geo_time_ATT")),
        lift_scale=str(lift.get("scale", "mean_kpi_level_delta")),
        replay_estimand=spec.to_json(),
        experiment_id=str(unit.get("experiment_id", unit["unit_id"])),
        calibration_readiness=str(unit.get("calibration_readiness", "approved")),
    )


def compute_true_replay_lift(
    truth: dict[str, Any],
    panel: pd.DataFrame,
    schema: PanelSchema,
    config: MMMConfig,
    unit: dict[str, Any],
) -> dict[str, Any]:
    """
    True experiment lift via ``implied_lift_from_counterfactual`` with truth coefficients.

    Observed spend path is the materialized panel; counterfactual applies
    ``counterfactual_spend_multiplier`` only inside the estimand mask on the full panel.
    """
    cal_unit = build_calibration_unit_for_experiment(truth, panel, schema, unit)
    if cal_unit is None or cal_unit.observed_spend_frame is None:
        raise ValueError(f"could not build replay frames for unit {unit.get('unit_id')}")
    spec = ReplayEstimandSpec.from_dict(cal_unit.replay_estimand or {})
    predict_fn = truth_predict_level(truth, schema, config)
    result = implied_lift_from_counterfactual(
        panel_observed=cal_unit.observed_spend_frame,
        panel_counterfactual=cal_unit.counterfactual_spend_frame,
        predict_fn=predict_fn,
        schema=schema,
        estimand=spec,
    )
    return {
        "true_experiment_lift": float(result["implied_mean_delta"]),
        "n_eval": int(result["n_eval"]),
        "replay_transform_mode": str(result["replay_transform_mode"]),
        "estimand_mask_used": spec.to_json(),
        "pre_window_adstock_preserved": _pre_window_adstock_preserved(
            truth, cal_unit.observed_spend_frame, schema, config, unit, spec
        ),
    }


def _pre_window_adstock_preserved(
    truth: dict[str, Any],
    panel_obs: pd.DataFrame,
    schema: PanelSchema,
    config: MMMConfig,
    unit: dict[str, Any],
    spec: ReplayEstimandSpec,
) -> bool:
    """Pre-window spend must change transformed media inside the experiment window (full-panel path)."""
    channel = str(unit["channel"])
    mult = float(unit["counterfactual_spend_multiplier"])
    shared = shared_ridge_transform_params(truth)
    frames = build_full_panel_replay_frames(panel_obs, schema, spec, channel, mult)
    if frames is None:
        return False
    obs, _ = frames
    decay = shared["decay"]
    hill_half = shared["hill_half"]
    hill_slope = shared["hill_slope"]
    x_full = build_channel_features_from_params(
        obs, schema, config.transforms, decay=decay, hill_half=hill_half, hill_slope=hill_slope
    )
    from mmm.calibration.replay_estimand import eval_mask_for_replay

    mask = eval_mask_for_replay(obs, schema, spec)
    # Window-slice baseline: features built only on experiment rows (adstock reset).
    gcol, wcol = schema.geo_column, schema.week_column
    geo_mask = obs[gcol].astype(str).isin(set(spec.geo_ids))
    time_mask = (obs[wcol] >= pd.to_datetime(spec.week_start)) & (
        obs[wcol] <= pd.to_datetime(spec.week_end)
    )
    slice_panel = obs.loc[geo_mask & time_mask].copy()
    x_slice = build_channel_features_from_params(
        slice_panel, schema, config.transforms, decay=decay, hill_half=hill_half, hill_slope=hill_slope
    )
    if not np.any(mask) or x_slice.size == 0:
        return False
    return float(np.max(np.abs(x_full[mask, 0] - x_slice[:, 0]))) > 1e-4


def validate_replay_lift_surface(lift: float, *, min_lift: float = MIN_TRUE_REPLAY_LIFT) -> None:
    if not np.isfinite(lift):
        raise ValueError("true replay lift must be finite")
    if abs(lift) < min_lift:
        raise ValueError(f"replay lift too small for recovery world: |lift|={abs(lift):.6f}")


def detect_window_slice_adstock_reset(
    truth: dict[str, Any],
    panel: pd.DataFrame,
    schema: PanelSchema,
    config: MMMConfig,
    unit: dict[str, Any],
) -> bool:
    """Return True when window-slice transform would reset adstock (unsupported)."""
    target_kpi = str(config.data.target_column) if hasattr(config, "data") else schema.target_column
    spec = replay_estimand_from_unit(unit, target_kpi=target_kpi)
    return not _pre_window_adstock_preserved(truth, panel, schema, config, unit, spec)


def populate_experiment_truth_from_panel(
    truth: dict[str, Any],
    panel: pd.DataFrame,
    schema: PanelSchema,
    config: MMMConfig,
) -> dict[str, Any]:
    """Fill lift_definition.value from authoritative replay computation."""
    units_in = list((truth.get("experiment_truth") or {}).get("units") or [])
    if not units_in:
        raise ValueError("experiment_truth.units required for replay recovery world")
    units_out: list[dict[str, Any]] = []
    for u in units_in:
        computed = compute_true_replay_lift(truth, panel, schema, config, u)
        validate_replay_lift_surface(float(computed["true_experiment_lift"]))
        out = dict(u)
        lift_def = dict(out.get("lift_definition") or {})
        lift_def["scale"] = str(lift_def.get("scale", "mean_kpi_level_delta"))
        lift_def["value"] = float(computed["true_experiment_lift"])
        out["lift_definition"] = lift_def
        out["replay_truth_method"] = "truth_coef_full_panel_transform_estimand_mask"
        out["pre_window_adstock_preserved"] = bool(computed["pre_window_adstock_preserved"])
        units_out.append(out)
    return {"units": units_out}


def build_world_010_truth(*, seed: int = 10010) -> dict[str, Any]:
    """
    Authoritative skeleton for WORLD-010.

    Call ``enrich_world_010_experiment_truth`` after the DGP panel exists to record true lift.
    """
    channel = "search"
    channels = [channel]
    n_periods = 16
    week_start = "2020-02-17"
    week_end = "2020-03-02"
    baseline = 10.0
    observed_mult = 1.5
    return {
        "artifact_truth": {
            "expected_certification_levels": {"synthetic": "exact"},
            "expected_failures": [],
            "expected_gates": [{"expected": "pass", "gate_id": "production_readiness_approved"}],
            "expected_warnings": [],
        },
        "coefficient_truth": {
            "controls": {},
            "intercept": 4.605170185988092,
            "interactions": [],
            "true_beta_by_channel": {channel: 0.45},
        },
        "decision_truth": {},
        "drift_truth": {
            "changepoints": [],
            "coefficient_drift": [],
            "policy_changes": [],
            "privacy_shifts": [],
        },
        "experiment_truth": {
            "units": [
                {
                    "aggregation": "mean",
                    "calibration_readiness": "approved",
                    "channel": channel,
                    "counterfactual_spend_multiplier": baseline / (baseline * observed_mult),
                    "estimand": "geo_time_ATT",
                    "experiment_id": "exp-search-boost-001",
                    "geo_scope": "listed",
                    "geos": ["G0", "G1"],
                    "lift_definition": {
                        "scale": "mean_kpi_level_delta",
                        "value": 0.0,
                    },
                    "replay_transform_mode": REPLAY_TRANSFORM_MODE_FULL_PANEL,
                    "spend_shock": {
                        "kind": "multiplier",
                        "observed_multiplier": observed_mult,
                    },
                    "uncertainty": {"se": 0.05},
                    "unit_id": "replay_search_g01",
                    "week_end": week_end,
                    "week_start": week_start,
                }
            ]
        },
        "geo_truth": {
            "geo_column_name": "geo_id",
            "geos": ["G0", "G1", "G2"],
            "hierarchy": None,
            "n_geos": 3,
            "weights": {"G0": 1.0 / 3.0, "G1": 1.0 / 3.0, "G2": 1.0 / 3.0},
        },
        "governance_truth": {
            "approved_for_optimization": True,
            "model_release_state": "planning_allowed",
            "replay_calibration_active": True,
            "require_production_certification": False,
            "require_promoted_model": False,
        },
        "media_truth": {
            "baseline_spend_by_channel": {channel: baseline},
            "channels": channels,
            "spend_process_spec": {
                "correlation_level": "low",
                "kind": "pre_impulse_constant",
                "impulse_periods": 4,
                "impulse_level": 20.0,
                "baseline_level": baseline,
            },
        },
        "metadata": {
            "archetype_id": "replay_recovery_world",
            "creation_timestamp": "2026-05-22T16:00:00Z",
            "description": "Replay calibration recovery — known experiment lift, full-panel estimand mask",
            "generation_seed": seed,
            "materialization_version": "dgp_materialize_v1.0.0",
            "negative_world": False,
            "scenario_tags": ["dgp:replay_recovery", "noise:none", "replay:known_lift"],
            "world_contract_version": "groundtruth_world_v1",
            "world_generator_version": "manual_replay_v1.0.0",
            "world_id": "WORLD-010-replay-recovery",
            "world_version": "1.0.0",
        },
        "outcome_truth": {
            "base_level_mean": 100.0,
            "model_form": "semi_log",
            "observation_noise_level": "low",
            "observation_noise_std": 0.0,
            "target_column": "revenue",
            "target_scale": "positive_level",
        },
        "time_truth": {
            "date_frequency": "weekly",
            "start_date": "2020-01-06",
            "end_date": "2020-04-27",
            "n_periods": n_periods,
            "seasonality_declared": False,
            "week_column_name": "week_start_date",
            "train_window": {"start_period_index": 0, "end_period_index": n_periods - 3},
            "eval_window": {
                "start_period_index": n_periods - 2,
                "end_period_index": n_periods - 1,
            },
        },
        "transform_truth": {
            "adstock_family": "geometric",
            "saturation_family": "hill",
            "adstock_decay_by_channel": {channel: 0.5},
            "hill_half_max_by_channel": {channel: 10.0},
            "hill_slope_by_channel": {channel: 2.0},
        },
    }


def enrich_world_010_experiment_truth(
    truth: dict[str, Any],
    panel: pd.DataFrame,
    schema: PanelSchema,
    config: MMMConfig,
) -> dict[str, Any]:
    """Fill experiment_truth lift from authoritative replay computation (in-memory only)."""
    out = dict(truth)
    out["experiment_truth"] = populate_experiment_truth_from_panel(truth, panel, schema, config)
    return out
