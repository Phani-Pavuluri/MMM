"""Deterministic GroundTruthWorld archetype generators (truth JSON only; no panel/DGP)."""

from __future__ import annotations

import random
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from mmm.calibration.replay_estimand import REPLAY_TRANSFORM_MODE_FULL_PANEL
from mmm.validation.synthetic._io import write_json
from mmm.validation.synthetic.materializer import MATERIALIZATION_VERSION, WORLD_CONTRACT_VERSION

GENERATOR_VERSION = "archetype_gen_v1.0.0"
BASELINE_ARCHETYPE_ID = "baseline_world"
REPLAY_ARCHETYPE_ID = "experiment_world"

_CALENDAR_START = date(2020, 1, 6)
_SIGNAL_TAGS = ("low", "medium", "high")
_NOISE_TAGS = ("low", "medium", "high")
_EXPERIMENT_QUALITY = ("low", "medium", "high")


def write_world_truth(bundle_dir: str | Path, truth: dict[str, Any]) -> Path:
    """Write ``world_truth.json`` only; derived artifacts come from ``materialize_world``."""
    bundle = Path(bundle_dir)
    bundle.mkdir(parents=True, exist_ok=True)
    world_id = str(truth["metadata"]["world_id"])
    if bundle.name != world_id:
        raise ValueError(f"bundle directory name {bundle.name!r} must match world_id {world_id!r}")
    out = bundle / "world_truth.json"
    write_json(out, truth)
    return out


def generate_baseline_world_truth(seed: int, world_id: str) -> dict[str, Any]:
    """Deterministic ``baseline_world`` archetype; no experiment units."""
    if seed < 0:
        raise ValueError("seed must be >= 0")
    rng = random.Random(seed)
    n_geos = 2 + (seed % 2)
    n_periods = 8 + (seed % 5) * 2
    channels = ["search", "social"]
    geos = [f"G{i}" for i in range(n_geos)]
    weights = _equal_weights(geos)
    spend_level = 8.0 + (seed % 7)
    beta_search = round(0.30 + rng.uniform(0.0, 0.20), 4)
    beta_social = round(0.08 + rng.uniform(0.0, 0.06), 4)
    base_kpi = 90.0 + (seed % 21)
    time_truth = _time_truth(n_periods=n_periods)
    train_end = max(3, n_periods - 3)
    time_truth["train_window"] = {"start_period_index": 0, "end_period_index": train_end}
    time_truth["eval_window"] = {
        "start_period_index": min(train_end + 1, n_periods - 1),
        "end_period_index": n_periods - 1,
    }
    scenario_id = f"baseline_shift_{seed % 1000:03d}"
    bump = round(1.0 + rng.uniform(0.1, 0.8), 3)

    return _assemble_truth(
        world_id=world_id,
        seed=seed,
        archetype_id=BASELINE_ARCHETYPE_ID,
        description=f"Generated baseline archetype (seed={seed})",
        scenario_tags=[
            f"signal:{_SIGNAL_TAGS[seed % 3]}",
            f"noise:{_NOISE_TAGS[(seed // 3) % 3]}",
        ],
        time_truth=time_truth,
        geo_truth=_geo_truth(geos, weights),
        media_truth=_media_truth(channels, spend_level=spend_level),
        outcome_truth=_outcome_truth(base_kpi),
        transform_truth=_transform_truth(channels),
        coefficient_truth=_coefficient_truth(
            channels,
            beta_by_channel={"search": beta_search, "social": beta_social},
        ),
        experiment_truth={"units": []},
        decision_truth={
            "budget_constraints": [],
            "scenarios": [
                {
                    "scenario_id": scenario_id,
                    "baseline_spend_by_channel": {c: spend_level for c in channels},
                    "candidate_spend_by_channel": {
                        "search": spend_level + bump,
                        "social": spend_level,
                    },
                    "true_delta_mu": round(bump * beta_search, 4),
                }
            ],
        },
        governance_truth=_governance_truth(replay_calibration_active=False),
    )


def generate_replay_world_truth(seed: int, world_id: str) -> dict[str, Any]:
    """Deterministic ``experiment_world`` archetype with one replay unit."""
    if seed < 0:
        raise ValueError("seed must be >= 0")
    rng = random.Random(seed)
    n_geos = 2 + (seed % 2)
    n_periods = 10 + (seed % 4) * 2
    channels = ["search"]
    geos = [f"G{i}" for i in range(n_geos)]
    weights = _equal_weights(geos)
    spend_level = 10.0 + (seed % 6)
    beta_search = round(0.28 + rng.uniform(0.0, 0.15), 4)
    base_kpi = 95.0 + (seed % 15)
    time_truth = _time_truth(n_periods=n_periods)
    train_end = max(5, n_periods - 3)
    time_truth["train_window"] = {"start_period_index": 0, "end_period_index": train_end}
    time_truth["eval_window"] = {
        "start_period_index": min(train_end + 1, n_periods - 1),
        "end_period_index": n_periods - 1,
    }

    week_start_idx = 2 + (seed % max(1, n_periods - 6))
    window_len = 2 + (seed % 3)
    week_end_idx = min(week_start_idx + window_len, n_periods - 1)
    week_start = _week_iso(week_start_idx)
    week_end = _week_iso(week_end_idx)
    lift_value = round(0.015 + rng.uniform(0.0, 0.02), 4)
    lift_se = round(0.05 + rng.uniform(0.0, 0.06), 4)
    unit_id = f"replay_u_{seed % 10000:04d}"

    return _assemble_truth(
        world_id=world_id,
        seed=seed,
        archetype_id=REPLAY_ARCHETYPE_ID,
        description=f"Generated experiment/replay archetype (seed={seed})",
        scenario_tags=[
            f"experiment_quality:{_EXPERIMENT_QUALITY[seed % 3]}",
            f"signal:{_SIGNAL_TAGS[(seed // 2) % 3]}",
        ],
        time_truth=time_truth,
        geo_truth=_geo_truth(geos, weights),
        media_truth=_media_truth(channels, spend_level=spend_level),
        outcome_truth=_outcome_truth(base_kpi),
        transform_truth=_transform_truth(channels),
        coefficient_truth=_coefficient_truth(channels, beta_by_channel={"search": beta_search}),
        experiment_truth={
            "units": [
                {
                    "unit_id": unit_id,
                    "experiment_id": f"exp-search-{seed % 1000:03d}",
                    "channel": "search",
                    "geos": geos,
                    "geo_scope": "listed",
                    "week_start": week_start,
                    "week_end": week_end,
                    "estimand": "geo_time_ATT",
                    "aggregation": "mean",
                    "replay_transform_mode": REPLAY_TRANSFORM_MODE_FULL_PANEL,
                    "calibration_readiness": "approved",
                    "lift_definition": {
                        "scale": "mean_kpi_level_delta",
                        "value": lift_value,
                    },
                    "uncertainty": {"se": lift_se},
                }
            ]
        },
        decision_truth={},
        governance_truth=_governance_truth(replay_calibration_active=True),
    )


def compose_archetype_truth(
    *,
    family: str,
    seed: int,
    world_id: str,
    n_geos: int,
    n_periods: int,
    channels: list[str],
    noise_level: str,
    correlation_level: str,
    seasonality: str,
    drift: bool,
    experiment_quality: str,
    privacy_loss: bool,
    missingness: str,
    scenario_tags: list[str] | None = None,
    description: str | None = None,
) -> dict[str, Any]:
    """Deterministic truth assembly from explicit parameters (used by ScenarioBuilder)."""
    if seed < 0:
        raise ValueError("seed must be >= 0")
    if n_geos < 1:
        raise ValueError("n_geos must be >= 1")
    if n_periods < 4:
        raise ValueError("n_periods must be >= 4")
    if not channels:
        raise ValueError("channels must be non-empty")

    rng = random.Random(seed)
    geos = [f"G{i}" for i in range(n_geos)]
    spend_level = 8.0 + (seed % 7)
    base_kpi = 90.0 + (seed % 21)
    if noise_level == "medium":
        base_kpi += 5.0
    elif noise_level == "high":
        base_kpi += 12.0

    betas = _betas_for_channels(rng, channels, correlation_level=correlation_level)
    time_truth = _time_truth(n_periods=n_periods)
    time_truth["seasonality_declared"] = seasonality in ("mild", "strong")
    train_end = max(3, n_periods - 3)
    time_truth["train_window"] = {"start_period_index": 0, "end_period_index": train_end}
    time_truth["eval_window"] = {
        "start_period_index": min(train_end + 1, n_periods - 1),
        "end_period_index": n_periods - 1,
    }

    media = _media_truth(channels, spend_level=spend_level)
    media["spend_process_spec"]["correlation_level"] = correlation_level

    outcome = _outcome_truth(base_kpi)
    outcome["observation_noise_level"] = noise_level

    tags = list(scenario_tags or [])
    tags.extend(
        [
            f"noise:{noise_level}",
            f"correlation:{correlation_level}",
            f"seasonality:{seasonality}",
            f"missingness:{missingness}",
        ]
    )
    if drift:
        tags.append("drift:on")
    if privacy_loss:
        tags.append("privacy_loss:on")

    is_replay = family == "replay"
    archetype = REPLAY_ARCHETYPE_ID if is_replay else BASELINE_ARCHETYPE_ID
    experiment_truth: dict[str, Any] = {"units": []}
    decision_truth: dict[str, Any] = {"budget_constraints": [], "scenarios": []}
    governance = _governance_truth(replay_calibration_active=False)

    if is_replay:
        governance = _governance_truth(replay_calibration_active=experiment_quality != "none")
        experiment_truth = {
            "units": _experiment_units(
                seed=seed,
                rng=rng,
                channels=channels,
                geos=geos,
                n_periods=n_periods,
                experiment_quality=experiment_quality,
            )
        }
    else:
        bump = round(1.0 + rng.uniform(0.1, 0.8), 3)
        primary = channels[0]
        decision_truth = {
            "budget_constraints": [],
            "scenarios": [
                {
                    "scenario_id": f"scenario_shift_{seed % 1000:03d}",
                    "baseline_spend_by_channel": {c: spend_level for c in channels},
                    "candidate_spend_by_channel": {
                        **{c: spend_level for c in channels},
                        primary: spend_level + bump,
                    },
                    "true_delta_mu": round(bump * betas[primary], 4),
                }
            ],
        }

    artifact = _artifact_truth()
    artifact["expected_warnings"] = _expected_warnings(
        correlation_level=correlation_level,
        missingness=missingness,
    )

    return _assemble_truth(
        world_id=world_id,
        seed=seed,
        archetype_id=archetype,
        description=description or f"Composed {family} scenario (seed={seed})",
        scenario_tags=tags,
        time_truth=time_truth,
        geo_truth=_geo_truth(geos, _equal_weights(geos)),
        media_truth=media,
        outcome_truth=outcome,
        transform_truth=_transform_truth(channels),
        coefficient_truth=_coefficient_truth(channels, beta_by_channel=betas),
        experiment_truth=experiment_truth,
        decision_truth=decision_truth,
        drift_truth=_drift_truth(
            drift=drift,
            privacy_loss=privacy_loss,
            n_periods=n_periods,
            channels=channels,
        ),
        governance_truth=governance,
        artifact_truth=artifact,
    )


def _betas_for_channels(
    rng: random.Random,
    channels: list[str],
    *,
    correlation_level: str,
) -> dict[str, float]:
    betas: dict[str, float] = {}
    base = 0.30 + rng.uniform(0.0, 0.15)
    for i, ch in enumerate(channels):
        if correlation_level == "severe" and i > 0:
            betas[ch] = round(base * (1.0 + 0.01 * i), 4)
        else:
            betas[ch] = round(base + rng.uniform(0.0, 0.08) * (i + 1), 4)
    return betas


def _experiment_units(
    *,
    seed: int,
    rng: random.Random,
    channels: list[str],
    geos: list[str],
    n_periods: int,
    experiment_quality: str,
) -> list[dict[str, Any]]:
    if experiment_quality == "none":
        return []
    channel = channels[0]
    week_start_idx = 2 + (seed % max(1, n_periods - 6))
    week_end_idx = min(week_start_idx + 2, n_periods - 1)
    lift_value = 0.02
    se_by_quality = {"weak": 0.12, "medium": 0.08, "high": 0.04}
    lift_se = se_by_quality.get(experiment_quality, 0.08)
    return [
        {
            "unit_id": f"replay_u_{seed % 10000:04d}",
            "experiment_id": f"exp-{channel}-{seed % 1000:03d}",
            "channel": channel,
            "geos": list(geos),
            "geo_scope": "listed",
            "week_start": _week_iso(week_start_idx),
            "week_end": _week_iso(week_end_idx),
            "estimand": "geo_time_ATT",
            "aggregation": "mean",
            "replay_transform_mode": REPLAY_TRANSFORM_MODE_FULL_PANEL,
            "calibration_readiness": "approved",
            "lift_definition": {"scale": "mean_kpi_level_delta", "value": lift_value},
            "uncertainty": {"se": lift_se},
        }
    ]


def _drift_truth(
    *,
    drift: bool,
    privacy_loss: bool,
    n_periods: int,
    channels: list[str],
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "changepoints": [],
        "coefficient_drift": [],
        "policy_changes": [],
        "privacy_shifts": [],
    }
    if drift:
        cp = max(1, n_periods // 2)
        out["changepoints"] = [
            {
                "period_index": cp,
                "affected_domains": ["coefficient_truth"],
            }
        ]
        out["coefficient_drift"] = [
            {
                "channel": channels[0],
                "start_period_index": cp,
                "delta_beta": -0.05,
            }
        ]
    if privacy_loss:
        out["privacy_shifts"] = [
            {
                "period_index": 0,
                "kind": "aggregation_noise",
                "severity": "mild",
            }
        ]
    return out


def _expected_warnings(
    *,
    correlation_level: str,
    missingness: str,
) -> list[dict[str, Any]]:
    warnings: list[dict[str, Any]] = []
    if correlation_level == "severe":
        warnings.append(
            {
                "warning_id": "identifiability_collinearity",
                "severity": "high",
            }
        )
    if missingness == "mild":
        warnings.append(
            {
                "warning_id": "panel_missingness",
                "severity": "moderate",
            }
        )
    return warnings


def _assemble_truth(
    *,
    world_id: str,
    seed: int,
    archetype_id: str,
    description: str,
    scenario_tags: list[str],
    time_truth: dict[str, Any],
    geo_truth: dict[str, Any],
    media_truth: dict[str, Any],
    outcome_truth: dict[str, Any],
    transform_truth: dict[str, Any],
    coefficient_truth: dict[str, Any],
    experiment_truth: dict[str, Any],
    decision_truth: dict[str, Any],
    governance_truth: dict[str, Any],
    drift_truth: dict[str, Any] | None = None,
    artifact_truth: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "metadata": {
            "world_id": world_id,
            "world_version": "1.0.0",
            "world_contract_version": WORLD_CONTRACT_VERSION,
            "world_generator_version": GENERATOR_VERSION,
            "materialization_version": MATERIALIZATION_VERSION,
            "generation_seed": int(seed),
            "scenario_tags": scenario_tags,
            "creation_timestamp": _creation_timestamp(seed),
            "archetype_id": archetype_id,
            "negative_world": False,
            "description": description,
        },
        "time_truth": time_truth,
        "geo_truth": geo_truth,
        "media_truth": media_truth,
        "outcome_truth": outcome_truth,
        "transform_truth": transform_truth,
        "coefficient_truth": coefficient_truth,
        "experiment_truth": experiment_truth,
        "decision_truth": decision_truth,
        "drift_truth": drift_truth if drift_truth is not None else {},
        "artifact_truth": artifact_truth if artifact_truth is not None else _artifact_truth(),
        "governance_truth": governance_truth,
    }


def _creation_timestamp(seed: int) -> str:
    base = datetime(2026, 1, 1, 0, 0, 0)
    dt = base + timedelta(seconds=int(seed) % 86400)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _time_truth(*, n_periods: int) -> dict[str, Any]:
    if n_periods < 4:
        raise ValueError("n_periods must be >= 4")
    end = _CALENDAR_START + timedelta(weeks=n_periods - 1)
    return {
        "date_frequency": "weekly",
        "start_date": _CALENDAR_START.isoformat(),
        "end_date": end.isoformat(),
        "n_periods": n_periods,
        "seasonality_declared": False,
        "week_column_name": "week_start_date",
        "train_window": {"start_period_index": 0, "end_period_index": n_periods - 2},
        "eval_window": {
            "start_period_index": n_periods - 1,
            "end_period_index": n_periods - 1,
        },
    }


def _week_iso(week_index: int) -> str:
    return (_CALENDAR_START + timedelta(weeks=week_index)).isoformat()


def _geo_truth(geos: list[str], weights: dict[str, float]) -> dict[str, Any]:
    return {
        "geos": geos,
        "n_geos": len(geos),
        "weights": weights,
        "hierarchy": None,
        "geo_column_name": "geo_id",
    }


def _equal_weights(geos: list[str]) -> dict[str, float]:
    w = 1.0 / len(geos)
    return {g: round(w, 10) for g in geos}


def _media_truth(channels: list[str], *, spend_level: float) -> dict[str, Any]:
    baseline = {c: spend_level for c in channels}
    return {
        "channels": channels,
        "baseline_spend_by_channel": baseline,
        "spend_process_spec": {
            "kind": "constant",
            "level": spend_level,
            "correlation_level": "low",
        },
    }


def _outcome_truth(base_kpi: float) -> dict[str, Any]:
    return {
        "target_column": "revenue",
        "target_scale": "positive_level",
        "model_form": "semi_log",
        "base_level_mean": base_kpi,
        "observation_noise_level": "low",
    }


def _transform_truth(channels: list[str]) -> dict[str, Any]:
    return {
        "adstock_family": "geometric",
        "saturation_family": "hill",
        "adstock_decay_by_channel": {c: 0.5 for c in channels},
        "hill_half_max_by_channel": {c: 10.0 for c in channels},
        "hill_slope_by_channel": {c: 2.0 for c in channels},
    }


def _coefficient_truth(
    channels: list[str],
    *,
    beta_by_channel: dict[str, float],
) -> dict[str, Any]:
    import math

    missing = set(channels) - set(beta_by_channel)
    if missing:
        raise ValueError(f"missing beta for channels: {sorted(missing)}")
    return {
        "intercept": math.log(100.0),
        "true_beta_by_channel": {c: float(beta_by_channel[c]) for c in channels},
        "controls": {},
        "interactions": [],
    }


def _governance_truth(*, replay_calibration_active: bool) -> dict[str, Any]:
    return {
        "approved_for_optimization": True,
        "model_release_state": "planning_allowed",
        "replay_calibration_active": replay_calibration_active,
        "require_production_certification": False,
        "require_promoted_model": False,
    }


def _artifact_truth() -> dict[str, Any]:
    return {
        "expected_certification_levels": {"synthetic": "exact"},
        "expected_gates": [{"gate_id": "production_readiness_approved", "expected": "pass"}],
        "expected_failures": [],
        "expected_warnings": [],
    }
