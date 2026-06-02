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

TAU_PARAM_CURRENT = "current"
TAU_PARAM_LOG_TAU = "log_tau"
TAU_PARAM_NONCENTERED_LOG_TAU = "noncentered_log_tau"

SIGMA_POLICY_CURRENT = "current_default"
SIGMA_POLICY_FLOOR = "sigma_floor"
SIGMA_POLICY_PRIOR_REGULARIZED = "sigma_prior_regularized"

BETA_PRIOR_CURRENT = "current_default"
BETA_PRIOR_STRONGER = "stronger_regularization"
BETA_PRIOR_CHANNEL_SCALED = "channel_scaled_regularization"

HIERARCHY_STRENGTH_LEARNED = "learned_tau"
HIERARCHY_STRENGTH_WEAK_REG = "weakly_regularized_tau"
HIERARCHY_STRENGTH_STRONG_REG = "strongly_regularized_tau"
HIERARCHY_STRENGTH_FIXED_TAU = "fixed_tau_ablation"

PARAMETERIZATION_MODES: frozenset[str] = frozenset(
    {PARAMETERIZATION_CENTERED, PARAMETERIZATION_NON_CENTERED}
)
LIKELIHOOD_SCALE_POLICIES: frozenset[str] = frozenset(
    {LIKELIHOOD_CURRENT_DEFAULT, LIKELIHOOD_PRESCALED_LOG_OUTCOME, LIKELIHOOD_SIGMA_FLOOR}
)
HIERARCHY_POLICIES: frozenset[str] = frozenset(
    {HIERARCHY_FULL_GEO_CHANNEL, HIERARCHY_POOLED_CHANNEL, HIERARCHY_FIXED_TAU}
)
TAU_PARAMETERIZATIONS: frozenset[str] = frozenset(
    {TAU_PARAM_CURRENT, TAU_PARAM_LOG_TAU, TAU_PARAM_NONCENTERED_LOG_TAU}
)
SIGMA_POLICIES: frozenset[str] = frozenset(
    {SIGMA_POLICY_CURRENT, SIGMA_POLICY_FLOOR, SIGMA_POLICY_PRIOR_REGULARIZED}
)
BETA_PRIOR_POLICIES: frozenset[str] = frozenset(
    {BETA_PRIOR_CURRENT, BETA_PRIOR_STRONGER, BETA_PRIOR_CHANNEL_SCALED}
)
HIERARCHY_STRENGTH_POLICIES: frozenset[str] = frozenset(
    {
        HIERARCHY_STRENGTH_LEARNED,
        HIERARCHY_STRENGTH_WEAK_REG,
        HIERARCHY_STRENGTH_STRONG_REG,
        HIERARCHY_STRENGTH_FIXED_TAU,
    }
)

DEFAULT_GEOMETRY_CONFIG: dict[str, Any] = {
    "parameterization": PARAMETERIZATION_NON_CENTERED,
    "likelihood_scale_policy": LIKELIHOOD_CURRENT_DEFAULT,
    "hierarchy_policy": HIERARCHY_FULL_GEO_CHANNEL,
    "tau_parameterization": TAU_PARAM_CURRENT,
    "sigma_policy": SIGMA_POLICY_CURRENT,
    "beta_prior_policy": BETA_PRIOR_CURRENT,
    "hierarchy_strength_policy": HIERARCHY_STRENGTH_LEARNED,
}

ABLATION_HIERARCHY_POLICIES: frozenset[str] = frozenset({HIERARCHY_POOLED_CHANNEL, HIERARCHY_FIXED_TAU})


class H5GeometryConfigError(ValueError):
    """H5 geometry config validation failed — fail closed."""


def is_ablation_only_geometry(geometry: dict[str, Any]) -> bool:
    hier = geometry.get("hierarchy_policy", HIERARCHY_FULL_GEO_CHANNEL)
    strength = geometry.get("hierarchy_strength_policy", HIERARCHY_STRENGTH_LEARNED)
    return hier in ABLATION_HIERARCHY_POLICIES or strength == HIERARCHY_STRENGTH_FIXED_TAU


def is_hierarchy_faithful_geometry(geometry: dict[str, Any]) -> bool:
    return not is_ablation_only_geometry(geometry)


def _merge_geometry_defaults(raw: dict[str, Any]) -> dict[str, Any]:
    merged = {**DEFAULT_GEOMETRY_CONFIG, **raw}
    if merged.get("hierarchy_strength_policy") == HIERARCHY_STRENGTH_FIXED_TAU:
        merged["hierarchy_policy"] = HIERARCHY_FIXED_TAU
    return merged


def resolve_geometry_config(overrides: dict[str, Any] | None) -> dict[str, Any]:
    """Merge explicit h5_geometry_config; when absent, preserve legacy H5 behavior."""
    raw = dict((overrides or {}).get("h5_geometry_config") or {})
    if not raw:
        return {
            **DEFAULT_GEOMETRY_CONFIG,
            "explicit": False,
            "legacy_default": True,
            "hierarchy_faithful": True,
            "ablation_only": False,
        }
    merged = _merge_geometry_defaults(raw)
    validate_geometry_config(merged)
    return {
        **merged,
        "explicit": True,
        "legacy_default": False,
        "hierarchy_faithful": is_hierarchy_faithful_geometry(merged),
        "ablation_only": is_ablation_only_geometry(merged),
    }


def validate_geometry_config(config: dict[str, Any] | None) -> None:
    if not config or not isinstance(config, dict):
        raise H5GeometryConfigError("h5_geometry_config must be a non-empty object when provided")
    cfg = _merge_geometry_defaults(config)
    param = cfg.get("parameterization")
    if param not in PARAMETERIZATION_MODES:
        raise H5GeometryConfigError(
            f"parameterization must be one of {sorted(PARAMETERIZATION_MODES)}"
        )
    like = cfg.get("likelihood_scale_policy")
    if like not in LIKELIHOOD_SCALE_POLICIES:
        raise H5GeometryConfigError(
            f"likelihood_scale_policy must be one of {sorted(LIKELIHOOD_SCALE_POLICIES)}"
        )
    hier = cfg.get("hierarchy_policy")
    if hier not in HIERARCHY_POLICIES:
        raise H5GeometryConfigError(f"hierarchy_policy must be one of {sorted(HIERARCHY_POLICIES)}")
    tau_param = cfg.get("tau_parameterization")
    if tau_param not in TAU_PARAMETERIZATIONS:
        raise H5GeometryConfigError(
            f"tau_parameterization must be one of {sorted(TAU_PARAMETERIZATIONS)}"
        )
    sigma_pol = cfg.get("sigma_policy")
    if sigma_pol not in SIGMA_POLICIES:
        raise H5GeometryConfigError(f"sigma_policy must be one of {sorted(SIGMA_POLICIES)}")
    beta_pol = cfg.get("beta_prior_policy")
    if beta_pol not in BETA_PRIOR_POLICIES:
        raise H5GeometryConfigError(
            f"beta_prior_policy must be one of {sorted(BETA_PRIOR_POLICIES)}"
        )
    strength = cfg.get("hierarchy_strength_policy")
    if strength not in HIERARCHY_STRENGTH_POLICIES:
        raise H5GeometryConfigError(
            f"hierarchy_strength_policy must be one of {sorted(HIERARCHY_STRENGTH_POLICIES)}"
        )
    if hier == HIERARCHY_FIXED_TAU or strength == HIERARCHY_STRENGTH_FIXED_TAU:
        ft = cfg.get("fixed_tau_value")
        if ft is None or float(ft) <= 0:
            raise H5GeometryConfigError("fixed_tau_ablation requires fixed_tau_value > 0")
    sigma_pol = cfg.get("sigma_policy")
    if sigma_pol == SIGMA_POLICY_FLOOR or cfg.get("likelihood_scale_policy") == LIKELIHOOD_SIGMA_FLOOR:
        sf = cfg.get("sigma_floor")
        if sf is None or float(sf) <= 0:
            raise H5GeometryConfigError("sigma_floor policy requires sigma_floor > 0")
    if strength == HIERARCHY_STRENGTH_FIXED_TAU and hier not in (
        HIERARCHY_FIXED_TAU,
        HIERARCHY_FULL_GEO_CHANNEL,
    ):
        raise H5GeometryConfigError(
            "fixed_tau_ablation strength requires hierarchy_policy fixed_tau_ablation or full hierarchy"
        )


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


def apply_geometry_priors(
    overrides: dict[str, Any],
    geometry: dict[str, Any],
    *,
    channel_scales: list[float] | None = None,
) -> dict[str, Any]:
    """Map H5l geometry policies to model prior / scaling overrides."""
    out = dict(overrides)
    strength = geometry.get("hierarchy_strength_policy", HIERARCHY_STRENGTH_LEARNED)
    if strength == HIERARCHY_STRENGTH_WEAK_REG:
        out["tau_channel_prior_sigma"] = 0.35
    elif strength == HIERARCHY_STRENGTH_STRONG_REG:
        out["tau_channel_prior_sigma"] = 0.15
    elif strength == HIERARCHY_STRENGTH_LEARNED:
        out.setdefault("tau_channel_prior_sigma", 0.5)

    beta_pol = geometry.get("beta_prior_policy", BETA_PRIOR_CURRENT)
    if beta_pol == BETA_PRIOR_STRONGER:
        out["mu_channel_prior_sigma"] = 0.25
        out["z_beta_prior_sigma"] = 0.5
    elif beta_pol == BETA_PRIOR_CHANNEL_SCALED:
        out["beta_channel_scales"] = list(channel_scales or [1.0] * 8)
    else:
        out.setdefault("mu_channel_prior_sigma", 0.5)
        out.setdefault("z_beta_prior_sigma", 1.0)

    sigma_pol = geometry.get("sigma_policy", SIGMA_POLICY_CURRENT)
    if sigma_pol == SIGMA_POLICY_FLOOR:
        out["sigma_floor"] = float(geometry.get("sigma_floor", 0.05))
    elif sigma_pol == SIGMA_POLICY_PRIOR_REGULARIZED:
        out["sigma_prior_sigma"] = 0.35

    if geometry.get("hierarchy_strength_policy") == HIERARCHY_STRENGTH_FIXED_TAU:
        out["hierarchy_policy_fixed"] = True

    return apply_likelihood_scale_policy(out, geometry)


def geometry_record_for_artifact(geometry: dict[str, Any]) -> dict[str, Any]:
    return {
        "parameterization": geometry.get("parameterization"),
        "likelihood_scale_policy": geometry.get("likelihood_scale_policy"),
        "hierarchy_policy": geometry.get("hierarchy_policy"),
        "tau_parameterization": geometry.get("tau_parameterization"),
        "sigma_policy": geometry.get("sigma_policy"),
        "beta_prior_policy": geometry.get("beta_prior_policy"),
        "hierarchy_strength_policy": geometry.get("hierarchy_strength_policy"),
        "hierarchy_faithful": geometry.get("hierarchy_faithful", is_hierarchy_faithful_geometry(geometry)),
        "ablation_only": geometry.get("ablation_only", is_ablation_only_geometry(geometry)),
        "explicit_config": geometry.get("explicit", False),
        "fixed_tau_value": geometry.get("fixed_tau_value"),
        "sigma_floor": geometry.get("sigma_floor"),
    }


def evidence_promotion_for_geometry(
    convergence_status: str,
    geometry: dict[str, Any],
) -> bool:
    """Evidence promotion requires converged diagnostics and hierarchy-faithful (non-ablation) geometry."""
    from mmm.research.bayes_h3_sandbox.h5_trust_diagnostics import evidence_promotion_allowed

    if is_ablation_only_geometry(geometry):
        return False
    return evidence_promotion_allowed(convergence_status)
