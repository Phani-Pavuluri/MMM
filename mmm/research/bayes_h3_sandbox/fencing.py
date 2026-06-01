"""Guards preventing Bayes sandbox artifacts from entering production decision paths."""

from __future__ import annotations

from typing import Any

from mmm.config.schema import Framework, MMMConfig, RunEnvironment
from mmm.research.bayes_h3_sandbox.labels import (
    validate_research_only_artifact,
)


class BayesSandboxGuardError(ValueError):
    """Bayes-H3 sandbox guard violated — production decisioning blocked."""


H5_MODEL_SPEC_VERSION = "bayes_h5_sandbox_spec_v1"


def assert_h5_sandbox_gates(
    *,
    model_spec_version: str | None,
    enable_h5_sandbox: bool,
    research_only: bool = True,
) -> None:
    """Fail closed unless H5 spec is explicitly requested inside the research sandbox."""
    if model_spec_version == H5_MODEL_SPEC_VERSION:
        if not enable_h5_sandbox:
            raise BayesSandboxGuardError(
                f"{H5_MODEL_SPEC_VERSION} requires enable_h5_sandbox=True (research-only gated path)"
            )
        if not research_only:
            raise BayesSandboxGuardError("H5 sandbox path requires research_only=True")
        return
    if enable_h5_sandbox and model_spec_version != H5_MODEL_SPEC_VERSION:
        raise BayesSandboxGuardError(
            "enable_h5_sandbox=True requires model_spec_version="
            f"{H5_MODEL_SPEC_VERSION!r}"
        )


def reject_if_prod_decisioning_flags(artifact: dict[str, Any]) -> None:
    """Reject artifacts that claim production approval or decisioning."""
    if artifact.get("approved_for_prod") is True:
        raise BayesSandboxGuardError("approved_for_prod must not be true for Bayes sandbox")
    if artifact.get("prod_decisioning_allowed") is True:
        raise BayesSandboxGuardError("prod_decisioning_allowed must not be true for Bayes sandbox")
    if artifact.get("decision_grade") is True:
        raise BayesSandboxGuardError("decision_grade must not be true for Bayes sandbox")


def assert_not_production_decision_surface(artifact: dict[str, Any]) -> None:
    """Block packaging sandbox output as a production DecisionSurface."""
    validate_research_only_artifact(artifact)
    reject_if_prod_decisioning_flags(artifact)
    if (
        artifact.get("decision_surface") is not None
        and artifact.get("production_decision_surface") is not False
        and artifact.get("decision_surface_production") is True
    ):
        raise BayesSandboxGuardError("cannot attach production DecisionSurface to sandbox artifact")
    if artifact.get("production_decision_surface") is True:
        raise BayesSandboxGuardError("production DecisionSurface emission blocked for Bayes sandbox")


def assert_optimizer_input_not_bayes_sandbox(
    payload: Any,
    *,
    config: MMMConfig | None = None,
    context: str = "optimizer",
) -> None:
    """Reject optimizer inputs that are Bayes sandbox posterior/coef artifacts."""
    if isinstance(payload, dict) and (
        payload.get("bayes_h3_sandbox") is True
        or (payload.get("research_only") is True and ("idata" in payload or "linear_coef_draws" in payload))
    ):
        raise BayesSandboxGuardError(
            f"{context}: cannot use Bayes sandbox posterior/coef output as production optimizer input"
        )

    if (
        config is not None
        and config.run_environment == RunEnvironment.PROD
        and config.framework == Framework.BAYESIAN
        and isinstance(payload, dict)
        and payload.get("bayes_h3_sandbox") is True
    ):
        raise BayesSandboxGuardError("production optimize_budget_via_simulation cannot consume Bayes-H3 sandbox output")


def assert_no_production_recommendation(artifact: dict[str, Any]) -> None:
    if artifact.get("production_recommendation") is True:
        raise BayesSandboxGuardError("production recommendation artifacts blocked for Bayes sandbox")
    if artifact.get("recommendation") is not None and artifact.get("decision_grade") is True:
        raise BayesSandboxGuardError("decision-grade recommendation blocked for Bayes sandbox")


def wrap_legacy_trainer_warning() -> dict[str, str]:
    return {
        "usage": "sandbox_only",
        "note": (
            "BayesianMMMTrainer (pymc_trainer) outputs are research/diagnostic only. "
            "Use mmm.research.bayes_h3_sandbox.run_sandbox_fit for Bayes-H3 sandbox work."
        ),
    }
