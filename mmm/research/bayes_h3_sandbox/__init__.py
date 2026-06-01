"""Bayes-H3 research sandbox — guardrails and entrypoint (not production decisioning)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mmm.research.bayes_h3_sandbox.diagnostic_trust import build_diagnostic_trust_stub
from mmm.research.bayes_h3_sandbox.fencing import (
    BayesSandboxGuardError,
    assert_not_production_decision_surface,
    assert_optimizer_input_not_bayes_sandbox,
    reject_if_prod_decisioning_flags,
)
from mmm.research.bayes_h3_sandbox.labels import (
    RESEARCH_ONLY_LABEL,
    apply_research_only_envelope,
    validate_research_only_artifact,
)

if TYPE_CHECKING:
    from mmm.research.bayes_h3_sandbox.entrypoint import (
        SANDBOX_ENTRYPOINT,
        run_sandbox_fit,
        wrap_sandbox_artifact,
    )

_LAZY_ENTRYPOINT_EXPORTS = frozenset({"SANDBOX_ENTRYPOINT", "run_sandbox_fit", "wrap_sandbox_artifact"})


def __getattr__(name: str) -> Any:
    if name in _LAZY_ENTRYPOINT_EXPORTS:
        from mmm.research.bayes_h3_sandbox import entrypoint as _entrypoint

        return getattr(_entrypoint, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "SANDBOX_ENTRYPOINT",
    "RESEARCH_ONLY_LABEL",
    "BayesSandboxGuardError",
    "apply_research_only_envelope",
    "assert_not_production_decision_surface",
    "assert_optimizer_input_not_bayes_sandbox",
    "build_diagnostic_trust_stub",
    "reject_if_prod_decisioning_flags",
    "run_sandbox_fit",
    "validate_research_only_artifact",
    "wrap_sandbox_artifact",
]
