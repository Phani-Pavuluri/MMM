"""Fail-fast validation for config vs implemented code paths."""

from __future__ import annotations

from mmm.config.schema import (
    CVMode,
    CVSplitAxis,
    Framework,
    MMMConfig,
    ModelForm,
    NormalizationProfile,
    RunEnvironment,
)
from mmm.contracts.canonical_transforms import assert_canonical_media_stack_for_modeling
from mmm.governance.policy import PolicyError

# Prod Ridge+BO: explicit named objective profiles (weights still in YAML; name is the governance anchor).
PROD_RIDGE_BO_OBJECTIVE_NAMED_PROFILES: frozenset[str] = frozenset({"ridge_bo_standard_v1"})
PROD_RIDGE_BO_MODEL_FORM_CONTRACTS: dict[ModelForm, str] = {
    ModelForm.SEMI_LOG: "ridge_bo_semi_log_calendar_cv_v1",
    ModelForm.LOG_LOG: "ridge_bo_log_log_calendar_cv_v1",
}


def validate_transform_stack_for_framework(config: MMMConfig) -> None:
    """Single allowlist for modeling stacks (see ``mmm.contracts.canonical_transforms``)."""
    if config.framework not in (Framework.RIDGE_BO, Framework.BAYESIAN):
        return
    assert_canonical_media_stack_for_modeling(config)


# Backwards-compatible alias
def validate_implemented_transforms_for_framework(config: MMMConfig) -> None:
    validate_transform_stack_for_framework(config)


def validate_prod_cv_configuration(config: MMMConfig) -> None:
    """Prod: require calendar-based CV split axis (no geo_rank / geo_blocked holdouts)."""
    if config.run_environment != RunEnvironment.PROD:
        return
    cv = config.cv
    if cv.split_axis != CVSplitAxis.CALENDAR_WEEK:
        raise PolicyError(
            f"run_environment=prod requires cv.split_axis=calendar_week (got {cv.split_axis.value!r}); "
            "geo_rank and geo_blocked strategies are not allowed in prod."
        )


def validate_geo_budget_planning_consistency(config: MMMConfig) -> None:
    """
    Per-geo budget optimization must use geo-aware Δμ pooling; otherwise optimizer objectives
    disagree with training-time / reporting aggregation semantics.
    """
    if not config.budget.geo_budget_enabled:
        return
    agg = config.extensions.product.planning_delta_mu_aggregation
    if agg != "geo_mean_then_global_mean":
        raise PolicyError(
            "budget.geo_budget_enabled=True requires extensions.product.planning_delta_mu_aggregation="
            "'geo_mean_then_global_mean' "
            f"(got {agg!r}). Per-geo SLSQP must align with geo-then-global full-panel μ semantics."
        )


def validate_prod_model_form_contract(config: MMMConfig) -> None:
    """Prod Ridge+BO: require explicit YAML contract id aligned to ``model_form`` (no silent default link)."""
    if config.run_environment != RunEnvironment.PROD or config.framework != Framework.RIDGE_BO:
        return
    expected = PROD_RIDGE_BO_MODEL_FORM_CONTRACTS.get(config.model_form)
    if not expected:
        return
    got = (config.prod_canonical_modeling_contract_id or "").strip()
    if got != expected:
        raise PolicyError(
            "run_environment=prod with framework=ridge_bo requires prod_canonical_modeling_contract_id="
            f"{expected!r} for model_form={config.model_form.value} (got {got!r}). "
            "Add this key to YAML to acknowledge the supported semi-log vs log-log link and calendar-CV training contract."
        )


def validate_prod_explicit_modeling_policy(config: MMMConfig) -> None:
    """
    Prod: forbid implicit CV strategy (AUTO), forbid silent research normalization, and for Ridge+BO
    require an explicit named objective profile.
    """
    if config.run_environment != RunEnvironment.PROD:
        return
    if config.cv.mode == CVMode.AUTO:
        raise PolicyError(
            "run_environment=prod forbids cv.mode=auto: calendar CV must use an explicit strategy "
            "(cv.mode=rolling or cv.mode=expanding). Auto mode silently chooses rolling vs expanding "
            "from panel geometry and is research-only."
        )
    if config.objective.normalization_profile != NormalizationProfile.STRICT_PROD:
        raise PolicyError(
            "run_environment=prod requires objective.normalization_profile=strict_prod in YAML "
            f"(got {config.objective.normalization_profile.value!r}). "
            "Implicit research normalization is not allowed in prod."
        )
    if config.framework != Framework.RIDGE_BO:
        return
    name = (config.objective.named_profile or "").strip()
    if name not in PROD_RIDGE_BO_OBJECTIVE_NAMED_PROFILES:
        raise PolicyError(
            "run_environment=prod with framework=ridge_bo requires objective.named_profile to be one of "
            f"{sorted(PROD_RIDGE_BO_OBJECTIVE_NAMED_PROFILES)!r} (got {config.objective.named_profile!r}). "
            "Set objective.named_profile: ridge_bo_standard_v1 after reviewing objective.weights."
        )


def apply_environment_objective_profile_inplace(config: MMMConfig) -> None:
    """
    Map run_environment → objective normalization profile **in place** (Pydantic v2-safe).

    Prod: ``strict_prod`` only — must be set explicitly in YAML (no silent coercion from ``research``).
    """
    prof = config.objective.normalization_profile
    if config.run_environment == RunEnvironment.PROD:
        if prof == NormalizationProfile.RESEARCH:
            raise PolicyError(
                "run_environment=prod requires objective.normalization_profile=strict_prod to be set "
                "explicitly in config (got research). Silent coercion from the factory default is forbidden."
            )
        if prof != NormalizationProfile.STRICT_PROD:
            raise ValueError(
                f"run_environment=prod requires objective.normalization_profile=strict_prod "
                f"(got {prof.value!r}); research/debug normalization is forbidden in prod."
            )
        return
    if prof != NormalizationProfile.RESEARCH:
        return
    if config.run_environment == RunEnvironment.DEV:
        config.objective = config.objective.model_copy(update={"normalization_profile": NormalizationProfile.DEBUG})


def apply_environment_objective_profile(config: MMMConfig) -> MMMConfig:
    """Backward-compatible wrapper (mutates and returns the same instance)."""
    apply_environment_objective_profile_inplace(config)
    return config
