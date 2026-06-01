"""Authoritative drift / identifiability worlds for Phase 4B-5."""

from __future__ import annotations

from typing import Any

import numpy as np


def build_world_011_truth(*, seed: int = 11011) -> dict[str, Any]:
    """WORLD-011 — coefficient drift at known changepoint (single channel)."""
    channel = "media"
    n_periods = 16
    changepoint_index = 8
    pre_beta = 0.45
    post_beta = 0.12
    return {
        "artifact_truth": {
            "expected_certification_levels": {"synthetic": "exact"},
            "expected_failures": [],
            "expected_gates": [
                {"gate_id": "production_readiness_approved", "expected": "warn"},
            ],
            "expected_warnings": [
                {"warning_id": "coefficient_drift_detected", "severity": "high"},
            ],
        },
        "coefficient_truth": {
            "controls": {},
            "intercept": 4.605170185988092,
            "interactions": [],
            "true_beta_by_channel": {channel: pre_beta},
        },
        "decision_truth": {},
        "drift_truth": {
            "changepoints": [
                {
                    "period_index": changepoint_index,
                    "week_date": "2020-02-24",
                    "affected_domains": ["coefficient_truth"],
                }
            ],
            "coefficient_drift": [
                {
                    "channel": channel,
                    "start_period_index": changepoint_index,
                    "pre_beta": pre_beta,
                    "post_beta": post_beta,
                    "delta_beta": post_beta - pre_beta,
                }
            ],
            "policy_changes": [],
            "privacy_shifts": [],
            "expected_reliability": {
                "coef_recovery_across_regimes": False,
                "drift_warning_expected": True,
                "readiness_downgrade_expected": True,
                "post_period_fit_degradation_min_ratio": 1.15,
            },
        },
        "experiment_truth": {"units": []},
        "geo_truth": {
            "geo_column_name": "geo_id",
            "geos": ["G0", "G1"],
            "hierarchy": None,
            "n_geos": 2,
            "weights": {"G0": 0.5, "G1": 0.5},
        },
        "governance_truth": {
            "approved_for_optimization": False,
            "model_release_state": "review_required",
            "replay_calibration_active": False,
            "require_production_certification": False,
            "require_promoted_model": False,
        },
        "media_truth": {
            "baseline_spend_by_channel": {channel: 12.0},
            "channels": [channel],
            "spend_process_spec": {
                "correlation_level": "low",
                "kind": "constant",
                "level": 12.0,
            },
        },
        "metadata": {
            "archetype_id": "drift_recovery_world",
            "creation_timestamp": "2026-05-22T18:00:00Z",
            "description": "Drift recovery — known coefficient changepoint in KPI generation",
            "generation_seed": seed,
            "materialization_version": "dgp_materialize_v1.0.0",
            "negative_world": False,
            "scenario_tags": ["dgp:drift_recovery", "noise:none", "drift:known_changepoint"],
            "world_contract_version": "groundtruth_world_v1",
            "world_generator_version": "manual_drift_v1.0.0",
            "world_id": "WORLD-011-drift-recovery",
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


def build_world_012_truth(*, seed: int = 12012) -> dict[str, Any]:
    """WORLD-012 — severe collinearity with distinct true coefficients."""
    channels = ["search", "social"]
    return {
        "artifact_truth": {
            "expected_certification_levels": {"synthetic": "exact"},
            "expected_failures": [],
            "expected_gates": [
                {"gate_id": "production_readiness_approved", "expected": "warn"},
            ],
            "expected_warnings": [
                {"warning_id": "identifiability_collinearity", "severity": "high"},
            ],
        },
        "coefficient_truth": {
            "controls": {},
            "intercept": 4.605170185988092,
            "interactions": [],
            "true_beta_by_channel": {"search": 0.40, "social": 0.10},
        },
        "decision_truth": {},
        "drift_truth": {
            "changepoints": [],
            "coefficient_drift": [],
            "policy_changes": [],
            "privacy_shifts": [],
            "expected_reliability": {
                "coef_recovery_expected": False,
                "identifiability_warning_expected": True,
                "readiness_review_required": True,
                "min_channel_correlation": 0.95,
                "min_max_vif": 5.0,
            },
        },
        "experiment_truth": {"units": []},
        "geo_truth": {
            "geo_column_name": "geo_id",
            "geos": ["G0", "G1", "G2"],
            "hierarchy": None,
            "n_geos": 3,
            "weights": {"G0": 1.0 / 3.0, "G1": 1.0 / 3.0, "G2": 1.0 / 3.0},
        },
        "governance_truth": {
            "approved_for_optimization": False,
            "model_release_state": "review_required",
            "replay_calibration_active": False,
            "require_production_certification": False,
            "require_promoted_model": False,
        },
        "media_truth": {
            "baseline_spend_by_channel": {"search": 10.0, "social": 9.8},
            "channels": channels,
            "spend_process_spec": {
                "correlation_level": "severe",
                "kind": "collinear_block",
                "primary_channel": "search",
                "secondary_channel": "social",
                "scale": 0.98,
                "level": 10.0,
            },
        },
        "metadata": {
            "archetype_id": "identifiability_recovery_world",
            "creation_timestamp": "2026-05-22T18:00:00Z",
            "description": "Identifiability recovery — correlated spend, distinct true betas",
            "generation_seed": seed,
            "materialization_version": "dgp_materialize_v1.0.0",
            "negative_world": False,
            "scenario_tags": [
                "dgp:identifiability_recovery",
                "noise:none",
                "collinearity:severe",
            ],
            "world_contract_version": "groundtruth_world_v1",
            "world_generator_version": "manual_identifiability_v1.0.0",
            "world_id": "WORLD-012-identifiability-recovery",
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
            "n_periods": 16,
            "seasonality_declared": False,
            "week_column_name": "week_start_date",
            "train_window": {"start_period_index": 0, "end_period_index": 13},
            "eval_window": {"start_period_index": 14, "end_period_index": 15},
        },
        "transform_truth": {
            "adstock_family": "geometric",
            "saturation_family": "hill",
            "adstock_decay_by_channel": {c: 0.5 for c in channels},
            "hill_half_max_by_channel": {c: 10.0 for c in channels},
            "hill_slope_by_channel": {c: 2.0 for c in channels},
        },
    }


def panel_channel_correlation(panel: Any, schema: Any, channels: list[str]) -> float:
    """Max absolute Pearson correlation between channel spend columns."""
    import pandas as pd

    df = panel if isinstance(panel, pd.DataFrame) else pd.DataFrame(panel)
    if len(channels) < 2:
        return 0.0
    sub = df[list(channels)].astype(float)
    corr = sub.corr().to_numpy()
    if corr.size < 4:
        return 0.0
    off = corr[~np.eye(len(channels), dtype=bool)]
    return float(np.max(np.abs(off))) if off.size else 0.0
