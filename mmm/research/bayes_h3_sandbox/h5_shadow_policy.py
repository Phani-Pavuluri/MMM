"""Frozen H5 shadow-run policy loading and validation (research only)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mmm.research.bayes_h3_sandbox.fencing import H5_MODEL_SPEC_VERSION
from mmm.research.bayes_h3_sandbox.h5_geometry_config import (
    HIERARCHY_FIXED_TAU,
    HIERARCHY_POOLED_CHANNEL,
    is_ablation_only_geometry,
    validate_geometry_config,
)
from mmm.research.bayes_h3_sandbox.h5_real_panel_preprocessing import (
    CHANNEL_POLICY_DROP_COLLINEAR,
    CHANNEL_POLICY_MODES,
    validate_collinearity_config,
)
from mmm.research.bayes_h3_sandbox.h5_shadow_protocol import validate_transform_config
from mmm.research.bayes_h3_sandbox.h5_shadow_runner import ShadowRunRequest
from mmm.research.bayes_h3_sandbox.h5_trust_diagnostics import research_production_flags

POLICY_TYPE_RESEARCH = "research_shadow_policy"

REQUIRED_EXCLUDED_MENTIONS: frozenset[str] = frozenset(
    {"optimizer", "DecisionSurface", "decision_surface", "recommendations"}
)

PRODUCTION_FLAG_KEYS: frozenset[str] = frozenset(
    {
        "hard_gate",
        "production_promotion",
        "approved_for_prod",
        "prod_decisioning_allowed",
    }
)


class H5ShadowPolicyError(ValueError):
    """Shadow policy validation failed — fail closed."""


def load_shadow_policy(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        raise H5ShadowPolicyError(f"shadow policy not found: {p}")
    loaded = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise H5ShadowPolicyError("shadow policy JSON must be an object")
    return loaded


def _geometry_from_policy(policy: dict[str, Any]) -> dict[str, Any]:
    geom = policy.get("h5_geometry_config") or policy.get("geometry_config")
    if not geom or not isinstance(geom, dict):
        raise H5ShadowPolicyError("h5_geometry_config is required")
    return dict(geom)


def _channel_lists(channel_policy: dict[str, Any]) -> tuple[list[str], list[str]]:
    dropped = channel_policy.get("dropped_channels") or channel_policy.get(
        "explicit_dropped_channels"
    )
    kept = channel_policy.get("kept_channels") or channel_policy.get("explicit_kept_channels")
    if not isinstance(dropped, list) or not dropped:
        raise H5ShadowPolicyError(
            "channel_policy requires non-empty dropped_channels (or explicit_dropped_channels)"
        )
    if not isinstance(kept, list) or not kept:
        raise H5ShadowPolicyError(
            "channel_policy requires non-empty kept_channels (or explicit_kept_channels)"
        )
    return [str(c) for c in dropped], [str(c) for c in kept]


def _validate_production_flags(policy: dict[str, Any]) -> None:
    flags = policy.get("production_flags")
    if not isinstance(flags, dict):
        raise H5ShadowPolicyError("production_flags object is required")
    for key in PRODUCTION_FLAG_KEYS:
        if flags.get(key) is True:
            raise H5ShadowPolicyError(f"production_flags.{key} must not be true for shadow policy")
    expected = research_production_flags()
    for key in PRODUCTION_FLAG_KEYS:
        if flags.get(key) is not expected.get(key):
            raise H5ShadowPolicyError(
                f"production_flags.{key} must be {expected.get(key)!r} (got {flags.get(key)!r})"
            )


def _validate_channel_policy_explicit(channel_policy: dict[str, Any]) -> None:
    if not isinstance(channel_policy, dict):
        raise H5ShadowPolicyError("channel_policy is required")
    mode = channel_policy.get("mode")
    if mode not in CHANNEL_POLICY_MODES:
        raise H5ShadowPolicyError(f"unsupported channel_policy.mode: {mode!r}")
    if mode == CHANNEL_POLICY_DROP_COLLINEAR:
        if channel_policy.get("no_silent_dropping") is not True:
            raise H5ShadowPolicyError(
                "drop_collinear_channels policy requires no_silent_dropping=true"
            )
        _channel_lists(channel_policy)


def _validate_geometry_not_ablation_promotable(geometry: dict[str, Any]) -> None:
    if is_ablation_only_geometry(geometry):
        raise H5ShadowPolicyError(
            "ablation-only geometry (pooled_channel_effects_ablation, fixed_tau_ablation) "
            "cannot be used in a promotable frozen shadow policy"
        )
    hier = geometry.get("hierarchy_policy")
    if hier in (HIERARCHY_POOLED_CHANNEL, HIERARCHY_FIXED_TAU):
        raise H5ShadowPolicyError(f"hierarchy_policy {hier!r} is ablation-only")


def validate_shadow_policy(policy: dict[str, Any]) -> None:
    """Fail closed on incomplete or production-unsafe shadow policies (no model run)."""
    if policy.get("policy_type") != POLICY_TYPE_RESEARCH:
        raise H5ShadowPolicyError(f"policy_type must be {POLICY_TYPE_RESEARCH!r}")
    if not str(policy.get("policy_id") or "").strip():
        raise H5ShadowPolicyError("policy_id is required")
    if not str(policy.get("dataset_snapshot_id") or "").strip():
        raise H5ShadowPolicyError("dataset_snapshot_id is required")
    if not str(policy.get("panel_id") or "").strip():
        raise H5ShadowPolicyError("panel_id is required")
    if not str(policy.get("panel_path") or "").strip():
        raise H5ShadowPolicyError("panel_path is required")
    if policy.get("model_spec_version") != H5_MODEL_SPEC_VERSION:
        raise H5ShadowPolicyError(f"model_spec_version must be {H5_MODEL_SPEC_VERSION!r}")
    if policy.get("enable_h5_sandbox") is not True:
        raise H5ShadowPolicyError("enable_h5_sandbox must be true")
    if policy.get("research_only") is not True:
        raise H5ShadowPolicyError("research_only must be true")

    panel_schema = policy.get("panel_schema")
    if not panel_schema or not isinstance(panel_schema, dict):
        raise H5ShadowPolicyError("panel_schema is required")

    transform_config = policy.get("transform_config")
    if not transform_config or not isinstance(transform_config, dict):
        raise H5ShadowPolicyError("transform_config is required")
    try:
        validate_transform_config(transform_config)
    except Exception as exc:
        raise H5ShadowPolicyError(str(exc)) from exc

    geometry = _geometry_from_policy(policy)
    try:
        validate_geometry_config(geometry)
        _validate_geometry_not_ablation_promotable(geometry)
    except H5ShadowPolicyError:
        raise
    except Exception as exc:
        raise H5ShadowPolicyError(str(exc)) from exc

    channel_policy = policy.get("channel_policy")
    if not channel_policy:
        raise H5ShadowPolicyError("channel_policy is required")
    try:
        validate_collinearity_config(channel_policy)
        _validate_channel_policy_explicit(channel_policy)
    except Exception as exc:
        raise H5ShadowPolicyError(str(exc)) from exc

    sampler = policy.get("sampler_profile")
    if not sampler or not isinstance(sampler, dict):
        raise H5ShadowPolicyError("sampler_profile is required")
    for key in ("draws", "tune", "chains", "target_accept"):
        if sampler.get(key) is None:
            raise H5ShadowPolicyError(f"sampler_profile.{key} is required")

    _validate_production_flags(policy)

    excluded = policy.get("excluded_fields")
    if not isinstance(excluded, list):
        raise H5ShadowPolicyError("excluded_fields must be a list")
    excluded_lower = {str(x).lower() for x in excluded}
    for forbidden in REQUIRED_EXCLUDED_MENTIONS:
        if forbidden.lower() not in excluded_lower and forbidden not in excluded:
            raise H5ShadowPolicyError(f"excluded_fields must mention {forbidden!r}")

    forbidden_claims = policy.get("forbidden_claims")
    if not isinstance(forbidden_claims, list) or not forbidden_claims:
        raise H5ShadowPolicyError("forbidden_claims must be a non-empty list")


def build_transform_config_from_policy(policy: dict[str, Any]) -> dict[str, Any]:
    """Merge transform_config, panel_schema, and channel_policy for the shadow runner."""
    transform = dict(policy["transform_config"])
    schema = dict(policy["panel_schema"])
    week_col = schema.get("week_column") or schema.get("date_column") or "week_start_date"
    target_col = schema.get("target_column") or schema.get("outcome_column") or "revenue"
    transform["panel_schema"] = {
        "geo_column": schema.get("geo_column", "geo_id"),
        "week_column": week_col,
        "target_column": target_col,
    }
    transform["control_columns"] = list(schema.get("control_columns") or [])
    media_cols = schema.get("media_columns")
    if media_cols:
        transforms = dict(transform.get("media_transforms_by_channel") or {})
        for ch in media_cols:
            transforms.setdefault(ch, "identity")
        transform["media_transforms_by_channel"] = transforms
    transform["channel_policy"] = dict(policy["channel_policy"])
    transform["frozen_policy_id"] = policy.get("policy_id")
    return transform


def assert_channel_policy_matches_explicit(
    policy_record: dict[str, Any],
    channel_policy: dict[str, Any],
) -> None:
    """Verify preprocessing dropped/kept exactly what the frozen policy declares."""
    if channel_policy.get("mode") != CHANNEL_POLICY_DROP_COLLINEAR:
        return
    expected_drop = set(_channel_lists(channel_policy)[0])
    expected_keep = set(_channel_lists(channel_policy)[1])
    actual_drop = {d.get("channel") for d in policy_record.get("dropped_channels") or []}
    actual_drop.discard(None)
    actual_keep = set(policy_record.get("kept_channels") or [])
    if actual_drop != expected_drop:
        raise H5ShadowPolicyError(
            f"implicit or unexpected channel drop: expected dropped {sorted(expected_drop)!r}, "
            f"got {sorted(actual_drop)!r}"
        )
    if actual_keep != expected_keep:
        raise H5ShadowPolicyError(
            f"channel keep mismatch: expected {sorted(expected_keep)!r}, got {sorted(actual_keep)!r}"
        )


def policy_to_shadow_runner_args(
    policy: dict[str, Any],
    *,
    policy_path: str | Path | None = None,
    output_path: str | Path | None = None,
    execute_fit: bool = True,
) -> dict[str, Any]:
    """Map a validated frozen policy to shadow-runner keyword arguments."""
    validate_shadow_policy(policy)
    geometry = _geometry_from_policy(policy)
    transform_config = build_transform_config_from_policy(policy)
    sampler = dict(policy["sampler_profile"])
    sampler_overrides = {
        "draws": int(sampler["draws"]),
        "tune": int(sampler["tune"]),
        "chains": int(sampler["chains"]),
        "target_accept": float(sampler["target_accept"]),
    }
    profile_name = str(sampler.get("profile") or "extended_mcmc")
    extended = profile_name.startswith("extended") or sampler_overrides.get("draws", 0) >= 600

    prescale = dict(policy.get("prescale") or {})
    sandbox_overrides: dict[str, Any] = {
        "h5_geometry_config": geometry,
        "h5_panel_context": "real_panel",
        "h5_real_panel": True,
    }
    if prescale.get("media_prescale"):
        sandbox_overrides["media_prescale"] = prescale["media_prescale"]
    if prescale.get("outcome_prescale"):
        sandbox_overrides["outcome_prescale"] = prescale["outcome_prescale"]

    return {
        "panel_path": policy.get("panel_path"),
        "panel_id": policy["panel_id"],
        "dataset_snapshot_id": policy["dataset_snapshot_id"],
        "transform_config": transform_config,
        "model_spec_version": policy["model_spec_version"],
        "enable_h5_sandbox": policy["enable_h5_sandbox"],
        "research_only": policy["research_only"],
        "output_path": output_path,
        "fast_mcmc": not extended,
        "extended_mcmc": extended,
        "execute_fit": execute_fit,
        "artifact_type": policy.get("artifact_type", "real_panel_shadow_artifact"),
        "policy_id": policy["policy_id"],
        "source_policy_path": str(policy_path) if policy_path else None,
        "frozen_policy": policy,
        "geometry_config": geometry,
        "sampler_profile_applied": {
            "profile": profile_name,
            **sampler_overrides,
        },
        "sandbox_model_overrides": sandbox_overrides,
        "channel_policy_declared": dict(policy["channel_policy"]),
        "forbidden_claims": list(policy.get("forbidden_claims") or []),
    }


def policy_to_shadow_request(
    policy: dict[str, Any],
    *,
    policy_path: str | Path | None = None,
    output_path: str | Path | None = None,
    execute_fit: bool = True,
) -> ShadowRunRequest:
    args = policy_to_shadow_runner_args(
        policy,
        policy_path=policy_path,
        output_path=output_path,
        execute_fit=execute_fit,
    )
    forbidden_claims = args.pop("forbidden_claims", None)
    frozen = args.pop("frozen_policy", None)
    geometry_config = args.pop("geometry_config", None)
    sampler_profile_applied = args.pop("sampler_profile_applied", None)
    sandbox_model_overrides = args.pop("sandbox_model_overrides", None)
    channel_policy_declared = args.pop("channel_policy_declared", None)
    policy_id = args.pop("policy_id", None)
    source_policy_path = args.pop("source_policy_path", None)
    _ = frozen, forbidden_claims

    return ShadowRunRequest(
        **args,
        policy_id=policy_id,
        source_policy_path=source_policy_path,
        geometry_config=geometry_config,
        sandbox_model_overrides=sandbox_model_overrides,
        sampler_profile_applied=sampler_profile_applied,
        channel_policy_declared=channel_policy_declared,
    )
