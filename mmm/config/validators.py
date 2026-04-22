"""Fail-fast validation for config vs implemented code paths."""

from __future__ import annotations

from mmm.config.schema import Framework, MMMConfig


def validate_implemented_transforms_for_framework(config: MMMConfig) -> None:
    """Ridge+BO fast path only implements geometric adstock + Hill saturation."""
    if config.framework != Framework.RIDGE_BO:
        return
    if config.transforms.adstock != "geometric":
        raise ValueError(
            f"framework=ridge_bo only supports transforms.adstock='geometric' (got {config.transforms.adstock!r}). "
            "Weibull adstock is not implemented in the Ridge+BO trainer."
        )
    if config.transforms.saturation != "hill":
        raise ValueError(
            f"framework=ridge_bo only supports transforms.saturation='hill' (got {config.transforms.saturation!r})."
        )


def apply_environment_objective_profile(config: MMMConfig) -> MMMConfig:
    """Map run_environment → default normalization profile when still at factory default."""
    from mmm.config.schema import NormalizationProfile, RunEnvironment

    if config.objective.normalization_profile != NormalizationProfile.RESEARCH:
        return config
    if config.run_environment == RunEnvironment.PROD:
        obj = config.objective.model_copy(
            update={"normalization_profile": NormalizationProfile.STRICT_PROD}
        )
        return config.model_copy(update={"objective": obj})
    if config.run_environment == RunEnvironment.DEV:
        obj = config.objective.model_copy(update={"normalization_profile": NormalizationProfile.DEBUG})
        return config.model_copy(update={"objective": obj})
    return config
