"""Analytic/grid optimum for optimizer-recovery worlds (Phase 4B-3)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from mmm.config.schema import MMMConfig
from mmm.data.schema import PanelSchema
from mmm.models.ridge_bo.trainer import RidgeBOArtifacts
from mmm.planning.baseline import bau_baseline_from_panel
from mmm.planning.context import ridge_context_from_fit
from mmm.planning.decision_simulate import simulate


def _truth_coef_vector(truth: dict[str, Any], channels: list[str]) -> np.ndarray:
    betas = truth["coefficient_truth"]["true_beta_by_channel"]
    return np.array([float(betas[c]) for c in channels], dtype=float)


def shared_ridge_transform_params(truth: dict[str, Any]) -> dict[str, float]:
    transform = truth["transform_truth"]
    channels = list(truth["media_truth"]["channels"])
    decays = [float(transform["adstock_decay_by_channel"][c]) for c in channels]
    halves = [float(transform["hill_half_max_by_channel"][c]) for c in channels]
    slopes = [float(transform["hill_slope_by_channel"][c]) for c in channels]
    uniform = len(set(decays)) == 1 and len(set(halves)) == 1 and len(set(slopes)) == 1
    return {
        "decay": float(decays[0]) if uniform else float(np.mean(decays)),
        "hill_half": float(halves[0]) if uniform else float(np.mean(halves)),
        "hill_slope": float(slopes[0]) if uniform else float(np.mean(slopes)),
        "per_channel_uniform": uniform,
    }


# Minimum Δμ lift of optimum over equal-split BAU (provisional, TBD_v1_runtime).
MIN_OPTIMUM_LIFT_OVER_BAU = 0.02
# Minimum share gap: high_return must receive at least this fraction more than low_return.
MIN_HIGH_CHANNEL_SHARE_ADVANTAGE = 0.12


def truth_ridge_context(
    truth: dict[str, Any],
    panel: pd.DataFrame,
    schema: PanelSchema,
    config: MMMConfig,
) -> Any:
    """RidgeFitContext using authoritative coefficients (not fitted)."""
    channels = list(truth["media_truth"]["channels"])
    shared = shared_ridge_transform_params(truth)
    coef = _truth_coef_vector(truth, channels)
    intercept = np.array([float(truth["coefficient_truth"]["intercept"])], dtype=float)
    art = RidgeBOArtifacts(
        best_params={
            "decay": shared["decay"],
            "hill_half": shared["hill_half"],
            "hill_slope": shared["hill_slope"],
            "log_alpha": -6.0,
        },
        objective_history=[],
        coef=coef,
        intercept=intercept,
        leaderboard=[],
    )
    return ridge_context_from_fit(panel, schema, config, {"artifacts": art})


def grid_search_true_optimum(
    truth: dict[str, Any],
    panel: pd.DataFrame,
    schema: PanelSchema,
    config: MMMConfig,
    *,
    total_budget: float,
    channel_min: float = 0.0,
    channel_max: float | None = None,
    grid_steps: int = 161,
) -> dict[str, Any]:
    """
    Grid-search budget allocation maximizing Δμ vs BAU under true coefficients.

    Two-channel worlds only. Returns optimum spend, objective, regret vs equal split, and band metadata.
    """
    channels = list(truth["media_truth"]["channels"])
    if len(channels) != 2:
        raise ValueError("grid_search_true_optimum supports exactly two channels")
    high_ch, low_ch = channels[0], channels[1]
    betas = truth["coefficient_truth"]["true_beta_by_channel"]
    if float(betas[high_ch]) <= float(betas[low_ch]):
        raise ValueError("channels[0] must be the high-return channel (larger true_beta)")

    ctx = truth_ridge_context(truth, panel, schema, config)
    base = bau_baseline_from_panel(panel, schema)
    hi = float(channel_max if channel_max is not None else total_budget)
    lo = float(channel_min)

    best: dict[str, Any] | None = None
    grid = np.linspace(lo, hi, grid_steps)
    for x_hi in grid:
        x_lo = total_budget - x_hi
        if x_lo < lo - 1e-9 or x_lo > hi + 1e-9:
            continue
        plan = {high_ch: float(x_hi), low_ch: float(x_lo)}
        sim = simulate(plan, ctx, baseline_plan=base, uncertainty_mode="point")
        rec = {
            "spend": plan,
            "delta_mu": float(sim.delta_mu),
            "plan_mu": float(sim.plan_mu),
        }
        if best is None or rec["delta_mu"] > best["delta_mu"]:
            best = rec

    if best is None:
        raise ValueError("grid search found no feasible allocation")

    bau_sim = simulate(dict(base.spend_by_channel), ctx, baseline_plan=base, uncertainty_mode="point")
    bau_delta = float(bau_sim.delta_mu)
    lift = float(best["delta_mu"]) - bau_delta

    # Interior check: optimum not pinned to both bounds
    opt_hi = best["spend"][high_ch]
    interior = lo + 0.05 * total_budget < opt_hi < hi - 0.05 * total_budget

    share_hi = opt_hi / total_budget
    band = {
        "high_return_channel": high_ch,
        "low_return_channel": low_ch,
        "high_return_min_budget_share": max(0.52, share_hi - 0.08),
        "low_return_max_budget_share": min(0.48, (total_budget - opt_hi) / total_budget + 0.08),
        "high_return_must_exceed_low_return": True,
    }

    return {
        "true_optimal_budget": dict(best["spend"]),
        "true_optimal_delta_mu": float(best["delta_mu"]),
        "true_bau_delta_mu": bau_delta,
        "true_regret_at_bau": max(0.0, float(best["delta_mu"]) - bau_delta),
        "optimum_lift_over_bau": lift,
        "optimum_interior": interior,
        "expected_allocation_band": band,
        "grid_search_method": "truth_coef_full_panel_simulate",
        "grid_steps": grid_steps,
        "total_budget": float(total_budget),
    }


def validate_optimizer_surface(
    optimum: dict[str, Any],
    *,
    min_lift: float = MIN_OPTIMUM_LIFT_OVER_BAU,
    min_share_gap: float = MIN_HIGH_CHANNEL_SHARE_ADVANTAGE,
) -> None:
    """Reject flat or corner-dominated surfaces before certifying."""
    if not optimum.get("optimum_interior", False):
        raise ValueError("optimum pinned to bounds — surface too flat or corner-dominated")
    if float(optimum.get("optimum_lift_over_bau", 0.0)) < min_lift:
        raise ValueError(
            f"optimum lift over BAU {optimum['optimum_lift_over_bau']:.6f} < {min_lift} — surface too flat"
        )
    opt = optimum["true_optimal_budget"]
    channels = list(opt.keys())
    if len(channels) == 2:
        shares = [float(opt[c]) for c in channels]
        if abs(shares[0] - shares[1]) / max(sum(shares), 1e-9) < min_share_gap:
            raise ValueError("allocation nearly equal — ambiguous optimum")


def populate_optimizer_decision_truth(
    truth: dict[str, Any],
    panel: pd.DataFrame,
    schema: PanelSchema,
    config: MMMConfig,
    *,
    total_budget: float,
) -> dict[str, Any]:
    """Return decision_truth block with grid-search optimum (does not write world_truth.json)."""
    opt = grid_search_true_optimum(
        truth,
        panel,
        schema,
        config,
        total_budget=total_budget,
        channel_max=total_budget,
    )
    validate_optimizer_surface(opt)
    channels = list(truth["media_truth"]["channels"])
    base_spend = dict(truth["media_truth"].get("baseline_spend_by_channel") or {})
    if not base_spend:
        level = float((truth["media_truth"].get("spend_process_spec") or {}).get("level", 10.0))
        base_spend = {c: level for c in channels}
    equal = {c: total_budget / len(channels) for c in channels}
    return {
        "budget_constraints": [
            {
                "constraint_id": "total_budget_fixed",
                "total_budget": float(total_budget),
                "channel_min": {c: 0.0 for c in channels},
                "channel_max": {c: float(total_budget) for c in channels},
            }
        ],
        "scenarios": [
            {
                "scenario_id": "bau_equal_split",
                "baseline_spend_by_channel": {c: float(base_spend.get(c, equal[c])) for c in channels},
                "candidate_spend_by_channel": equal,
                "true_delta_mu": float(opt["true_bau_delta_mu"]),
            },
            {
                "scenario_id": "grid_optimum_plan",
                "baseline_spend_by_channel": {c: float(base_spend.get(c, equal[c])) for c in channels},
                "candidate_spend_by_channel": dict(opt["true_optimal_budget"]),
                "true_delta_mu": float(opt["true_optimal_delta_mu"]),
            },
        ],
        "true_optimal_budget": dict(opt["true_optimal_budget"]),
        "true_optimal_delta_mu": float(opt["true_optimal_delta_mu"]),
        "true_regret_at_bau": float(opt["true_regret_at_bau"]),
        "expected_allocation_band": dict(opt["expected_allocation_band"]),
        "optimizer_truth_method": opt["grid_search_method"],
        "optimum_lift_over_bau": float(opt["optimum_lift_over_bau"]),
    }


def build_world_009_truth(*, seed: int = 9009) -> dict[str, Any]:
    """
    Authoritative skeleton for WORLD-009; call ``enrich_world_009_decision_truth`` after panel exists.
    """
    channels = ["high_return", "low_return"]
    n_periods = 12
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
            "true_beta_by_channel": {"high_return": 0.52, "low_return": 0.06},
        },
        "decision_truth": {},
        "drift_truth": {
            "changepoints": [],
            "coefficient_drift": [],
            "policy_changes": [],
            "privacy_shifts": [],
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
            "approved_for_optimization": True,
            "model_release_state": "planning_allowed",
            "replay_calibration_active": False,
            "require_production_certification": False,
            "require_promoted_model": False,
        },
        "media_truth": {
            "baseline_spend_by_channel": {"high_return": 10.0, "low_return": 10.0},
            "channels": channels,
            "spend_process_spec": {
                "correlation_level": "low",
                "kind": "channel_modulated",
                "by_channel": {
                    "high_return": {"base": 10.0, "amplitude": 4.0},
                    "low_return": {"base": 10.0, "amplitude": 0.5},
                },
            },
        },
        "metadata": {
            "archetype_id": "optimizer_recovery_world",
            "creation_timestamp": "2026-05-22T12:00:00Z",
            "description": "Optimizer recovery world — high_return vs low_return, grid-recorded optimum",
            "generation_seed": seed,
            "materialization_version": "dgp_materialize_v1.0.0",
            "negative_world": False,
            "scenario_tags": ["dgp:optimizer_recovery", "noise:none", "optimizer:known_optimum"],
            "world_contract_version": "groundtruth_world_v1",
            "world_generator_version": "manual_optimizer_v1.0.0",
            "world_id": "WORLD-009-optimizer-recovery",
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
            "end_date": "2020-03-23",
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
            "adstock_decay_by_channel": {c: 0.5 for c in channels},
            "hill_half_max_by_channel": {c: 10.0 for c in channels},
            "hill_slope_by_channel": {c: 2.0 for c in channels},
        },
    }


def enrich_world_009_decision_truth(
    truth: dict[str, Any],
    panel: pd.DataFrame,
    schema: PanelSchema,
    config: MMMConfig,
    *,
    total_budget: float = 40.0,
) -> dict[str, Any]:
    """Fill decision_truth optimizer fields from grid search (in-memory only)."""
    out = dict(truth)
    out["decision_truth"] = populate_optimizer_decision_truth(
        truth, panel, schema, config, total_budget=total_budget
    )
    return out
