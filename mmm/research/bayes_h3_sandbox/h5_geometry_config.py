"""Explicit H5 sandbox geometry configuration (research only)."""

from __future__ import annotations

from typing import Any

PARAMETERIZATION_CENTERED = "centered"
PARAMETERIZATION_NON_CENTERED = "non_centered"

LIKELIHOOD_CURRENT_DEFAULT = "current_default"
LIKELIHOOD_PRESCALED_LOG_OUTCOME = "prescaled_log_outcome"
LIKELIHOOD_SIGMA_FLOOR = "sigma_floor"

HIERARCHY_FULL_GEO_CHANNEL = "full_geo_channel_hierarchy"
HIERARCHY_POOLED_CHANNEL = "pooled_channel_effects_ablation"
HIERARCHY_FIXED_TAU = "fixed_tau_ablation"

PARAMETERIZATION_MODES: frozenset[str] = frozenset(
    {PARAMETERIZATION_CENTERED, PARAMETERIZATION_NON_CENTERED}
)
LIKELIHOOD_SCALE_POLICIES: frozenset[str] = frozenset(
    {LIKELIHOOD_CURRENT_DEFAULT, LIKELIHOOD_PRESCALED_LOG_OUTCOME, LIKELIHOOD_SIGMA_FLOOR}
)
HIERARCHY_POLICIES: frozenset[str] = frozenset(
    {HIERARCHY_FULL_GEO_CHANNEL, HIERARCHY_POOLED_CHANNEL, HIERARCHY_FIXED_TAU}
)

DEFAULT_GEOMETRY_CONFIG: dict[str, Any] = {
    "parameterization": PARAMETERIZATION_NON_CENTERED,
    "likelihood_scale_policy": LIKELIHOOD_CURRENT_DEFAULT,
    "hierarchy_policy": HIERARCHY_FULL_GEO_CHANNEL,
}


class H5GeometryConfigError(ValueError):
    """H5 geometry config validation failed — fail closed."""


def resolve_geometry_config(overrides: dict[str, Any] | None) -> dict[str, Any]:
    """Merge explicit h5_geometry_config; when absent, preserve legacy H5 behavior."""
    raw = dict((overrides or {}).get("h5_geometry_config") or {})
    if not raw:
        return {
            **DEFAULT_GEOMETRY_CONFIG,
            "explicit": False,
            "legacy_default": True,
        }
    validate_geometry_config(raw)
    return {
        **raw,
        "explicit": True,
        "legacy_default": False,
    }


def validate_geometry_config(config: dict[str, Any] | None) -> None:
    if not config or not isinstance(config, dict):
        raise H5GeometryConfigError("h5_geometry_config must be a non-empty object when provided")
    param = config.get("parameterization")
    if param not in PARAMETERIZATION_MODES:
        raise H5GeometryConfigError(
            f"parameterization must be one of {sorted(PARAMETERIZATION_MODES)}"
        )
    like = config.get("likelihood_scale_policy")
    if like not in LIKELIHOOD_SCALE_POLICIES:
        raise H5GeometryConfigError(
            f"likelihood_scale_policy must be one of {sorted(LIKELIHOOD_SCALE_POLICIES)}"
        )
    hier = config.get("hierarchy_policy")
    if hier not in HIERARCHY_POLICIES:
        raise H5GeometryConfigError(f"hierarchy_policy must be one of {sorted(HIERARCHY_POLICIES)}")
    if hier == HIERARCHY_FIXED_TAU:
        ft = config.get("fixed_tau_value")
        if ft is None or float(ft) <= 0:
            raise H5GeometryConfigError("fixed_tau_ablation requires fixed_tau_value > 0")
    if like == LIKELIHOOD_SIGMA_FLOOR:
        sf = config.get("sigma_floor")
        if sf is None or float(sf) <= 0:
            raise H5GeometryConfigError("sigma_floor policy requires sigma_floor > 0")


def apply_likelihood_scale_policy(
    overrides: dict[str, Any],
    geometry: dict[str, Any],
) -> dict[str, Any]:
    """Map likelihood policy to sandbox model override flags (explicit only)."""
    out = dict(overrides)
    policy = geometry.get("likelihood_scale_policy", LIKELIHOOD_CURRENT_DEFAULT)
    if policy == LIKELIHOOD_PRESCALED_LOG_OUTCOME:
        out["media_prescale"] = "zscore_panel"
        out["outcome_prescale"] = "zscore_log"
    elif policy == LIKELIHOOD_SIGMA_FLOOR:
        out["sigma_floor"] = float(geometry.get("sigma_floor", 0.05))
    return out


def geometry_record_for_artifact(geometry: dict[str, Any]) -> dict[str, Any]:
    return {
        "parameterization": geometry.get("parameterization"),
        "likelihood_scale_policy": geometry.get("likelihood_scale_policy"),
        "hierarchy_policy": geometry.get("hierarchy_policy"),
        "explicit_config": geometry.get("explicit", False),
        "fixed_tau_value": geometry.get("fixed_tau_value"),
        "sigma_floor": geometry.get("sigma_floor"),
    }
