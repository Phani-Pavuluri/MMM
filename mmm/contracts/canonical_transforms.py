"""
Single source of truth: which (framework, model_form, adstock, saturation) stacks the codebase can execute.

All canonical Ridge+BO / Bayesian training, design-matrix construction, and decision preflights must
agree with this module — not with YAML literals alone.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mmm.config.schema import MMMConfig

# Tuples: (framework, model_form, adstock, saturation) — only implemented paths.
CANONICAL_MEDIA_STACKS: frozenset[tuple[str, str, str, str]] = frozenset(
    {
        ("ridge_bo", "semi_log", "geometric", "hill"),
        ("ridge_bo", "log_log", "geometric", "hill"),
        ("bayesian", "semi_log", "geometric", "hill"),
        ("bayesian", "log_log", "geometric", "hill"),
    }
)


def canonical_media_stack_key(cfg: MMMConfig) -> tuple[str, str, str, str]:
    return (
        cfg.framework.value,
        cfg.model_form.value,
        cfg.transforms.adstock,
        cfg.transforms.saturation,
    )


def assert_canonical_media_stack_for_modeling(cfg: MMMConfig) -> None:
    """
    Fail fast if the config requests a stack the modeling / feature builders cannot honor.

    Used by config validation and by runtime services (defense in depth).
    """
    key = canonical_media_stack_key(cfg)
    if key not in CANONICAL_MEDIA_STACKS:
        supported = ", ".join(f"{t[0]}+{t[1]}+{t[2]}+{t[3]}" for t in sorted(CANONICAL_MEDIA_STACKS))
        raise ValueError(
            f"Unsupported canonical media stack (framework, model_form, adstock, saturation)={key!r}. "
            f"Implemented stacks: {supported}"
        )
