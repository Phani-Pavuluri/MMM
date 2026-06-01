"""Single Bayes-H3 research sandbox entrypoint — diagnostic fit only."""

from __future__ import annotations

from typing import Any

import pandas as pd

from mmm.config.schema import MMMConfig
from mmm.data.schema import PanelSchema
from mmm.research.bayes_h3_sandbox.fencing import (
    H5_MODEL_SPEC_VERSION,
    assert_h5_sandbox_gates,
    assert_no_production_recommendation,
    assert_not_production_decision_surface,
    wrap_legacy_trainer_warning,
)
from mmm.research.bayes_h3_sandbox.labels import apply_research_only_envelope, validate_research_only_artifact
from mmm.research.bayes_h3_sandbox.model import (
    build_diagnostic_trust_from_fit,
    fit_h3_sandbox_hierarchical,
    fit_h5_sandbox_hierarchical,
)

SANDBOX_ENTRYPOINT = "mmm.research.bayes_h3_sandbox.run_sandbox_fit"


def wrap_sandbox_artifact(
    fit_out: dict[str, Any],
    *,
    diagnostic_trust: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Wrap a raw fit dict as a governed Bayes-H3 sandbox artifact."""
    artifact = apply_research_only_envelope(dict(fit_out))
    artifact["sandbox_entrypoint"] = SANDBOX_ENTRYPOINT
    artifact["outputs_are_diagnostic_only"] = True
    artifact["production_decision_surface"] = False
    artifact["production_recommendation"] = False
    artifact["legacy_trainer_policy"] = wrap_legacy_trainer_warning()
    artifact["diagnostic_trust_report"] = diagnostic_trust or build_diagnostic_trust_from_fit(fit_out)
    validate_research_only_artifact(artifact)
    assert_not_production_decision_surface(artifact)
    assert_no_production_recommendation(artifact)
    return artifact


def run_sandbox_fit(
    config: MMMConfig,
    schema: PanelSchema,
    df: pd.DataFrame,
    *,
    diagnostic_trust: dict[str, Any] | None = None,
    geo_hierarchy_mapping: dict[str, Any] | None = None,
    calibration_signals_stub: list[dict[str, Any]] | None = None,
    sandbox_model_overrides: dict[str, Any] | None = None,
    model_spec_version: str | None = None,
    enable_h5_sandbox: bool = False,
    research_only: bool = True,
) -> dict[str, Any]:
    """
    Run the Bayes-H3 hierarchical sandbox fit (research / diagnostic only).

    - Sandbox only — not production decisioning
    - Diagnostic posterior / coefficients only
    - No production DecisionSurface, optimizer, or recommendations
    - H5 (`bayes_h5_sandbox_spec_v1`) requires explicit ``enable_h5_sandbox=True``
    """
    assert_h5_sandbox_gates(
        model_spec_version=model_spec_version,
        enable_h5_sandbox=enable_h5_sandbox,
        research_only=research_only,
    )
    if model_spec_version == H5_MODEL_SPEC_VERSION:
        raw = fit_h5_sandbox_hierarchical(
            config,
            schema,
            df,
            geo_hierarchy_mapping=geo_hierarchy_mapping,
            calibration_signals_stub=calibration_signals_stub,
            sandbox_model_overrides=sandbox_model_overrides,
        )
    else:
        raw = fit_h3_sandbox_hierarchical(
            config,
            schema,
            df,
            geo_hierarchy_mapping=geo_hierarchy_mapping,
            calibration_signals_stub=calibration_signals_stub,
            sandbox_model_overrides=sandbox_model_overrides,
        )
    trust = diagnostic_trust or build_diagnostic_trust_from_fit(raw)
    artifact = wrap_sandbox_artifact(raw, diagnostic_trust=trust)
    if model_spec_version == H5_MODEL_SPEC_VERSION:
        artifact["model_spec_version"] = H5_MODEL_SPEC_VERSION
        artifact["enable_h5_sandbox"] = True
        artifact["hard_gate"] = False
        artifact["production_promotion"] = False
    return artifact
