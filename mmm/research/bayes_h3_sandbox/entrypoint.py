"""Single Bayes-H3 research sandbox entrypoint — diagnostic fit only."""

from __future__ import annotations

from typing import Any

import pandas as pd

from mmm.config.schema import MMMConfig
from mmm.data.schema import PanelSchema
from mmm.research.bayes_h3_sandbox.diagnostic_trust import build_diagnostic_trust_stub
from mmm.research.bayes_h3_sandbox.fencing import (
    assert_no_production_recommendation,
    assert_not_production_decision_surface,
    wrap_legacy_trainer_warning,
)
from mmm.research.bayes_h3_sandbox.labels import apply_research_only_envelope, validate_research_only_artifact

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
    artifact["diagnostic_trust_report"] = diagnostic_trust or build_diagnostic_trust_stub()
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
) -> dict[str, Any]:
    """
    Run PyMC hierarchical fit through the Bayes-H3 sandbox boundary.

    - Sandbox only — not production decisioning
    - Diagnostic posterior / coefficients only
    - No production DecisionSurface, optimizer, or recommendations
    """
    from mmm.models.bayesian.pymc_trainer import BayesianMMMTrainer

    trainer = BayesianMMMTrainer(config, schema)
    raw = trainer.fit(df)
    trust = diagnostic_trust
    if trust is None and isinstance(raw.get("ppc"), dict):
        di = raw["ppc"].get("decision_inference") or {}
        trust = build_diagnostic_trust_stub(
            posterior_summary={"ppc_keys": sorted(str(k) for k in raw["ppc"])},
            convergence_diagnostics=di if isinstance(di, dict) else {},
            hierarchy_evidence=raw.get("bayesian_hierarchy_report") or {},
            pooling_diagnostics=(
                raw.get("bayesian_hierarchy_report", {}).get("pooling", {})
                if isinstance(raw.get("bayesian_hierarchy_report"), dict)
                else {}
            ),
        )
    return wrap_sandbox_artifact(raw, diagnostic_trust=trust)
