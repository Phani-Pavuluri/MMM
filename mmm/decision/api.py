"""Stable Python entrypoints that share ``mmm.decision.service`` with the CLI (no duplicated policy logic)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mmm.config.load import load_config
from mmm.config.schema import RunEnvironment
from mmm.contracts.runtime_validation import SemanticContractError
from mmm.decision.legacy_budget_optimization import run_legacy_diagnostic_optimize_budget
from mmm.decision.service import load_scenario_yaml, optimize_budget_decision, simulate_decision
from mmm.governance.decision_safety import MSG_OPTIMIZATION_BLOCKED
from mmm.governance.policy import PolicyError


def run_decision_simulation(
    *,
    config: Path,
    scenario: Path,
    extension_report: Path,
    out: Path,
) -> dict[str, Any]:
    """Run full-panel simulate via ``simulate_decision`` (same policy gates as ``mmm decide simulate``)."""
    cfg = load_config(config)
    er = json.loads(extension_report.read_text(encoding="utf-8"))
    try:
        raw = load_scenario_yaml(scenario)
        payload = simulate_decision(cfg=cfg, scenario=raw, extension_report=er, out=out)
    except PolicyError as e:
        raise SemanticContractError(str(e)) from e
    except ValueError as e:
        raise SemanticContractError(str(e)) from e
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return payload


def run_decision_optimization(
    *,
    config: Path,
    extension_report: Path | None,
    out: Path,
    curve_bundle: Path | None = None,
    allow_unsafe_decision_apis: bool = False,
    legacy_diagnostic_curve_optimizer: bool = False,
) -> dict[str, Any]:
    """
    Full-panel optimize via ``optimize_budget_decision`` (same runtime policy as CLI).

    **Non-canonical:** If full-panel inputs are unavailable *and* ``run_environment`` is not prod,
    ``legacy_diagnostic_curve_optimizer`` may run ``run_legacy_diagnostic_optimize_budget`` (curve /
    placeholder path). That path is not decision-grade.
    """
    cfg = load_config(config)
    if extension_report is None or not extension_report.exists():
        if cfg.run_environment == RunEnvironment.PROD:
            raise SemanticContractError(
                "Production optimize-budget requires an existing extension_report JSON from training."
            )
        raise SemanticContractError("extension_report path is required and must exist.")

    er = json.loads(extension_report.read_text(encoding="utf-8"))
    ridge = er.get("ridge_fit_summary")
    full_model_ready = bool(cfg.data.path and isinstance(ridge, dict) and ridge.get("coef"))

    if full_model_ready:
        try:
            payload = optimize_budget_decision(cfg=cfg, extension_report=er, out=out)
        except PolicyError as e:
            raise SemanticContractError(str(e)) from e
        except (ValueError, KeyError) as e:
            raise SemanticContractError(str(e)) from e
    elif cfg.run_environment == RunEnvironment.PROD:
        raise SemanticContractError(
            "Production optimize-budget requires config.data.path and extension_report "
            "with ridge_fit_summary.coef (full-panel Δμ objective via optimize_budget_decision)."
        )
    elif not cfg.allow_unsafe_decision_apis or not allow_unsafe_decision_apis:
        raise SemanticContractError(MSG_OPTIMIZATION_BLOCKED)
    elif not legacy_diagnostic_curve_optimizer:
        raise SemanticContractError(
            "Full-panel inputs unavailable. Pass legacy_diagnostic_curve_optimizer=True (non-prod only) "
            "for the legacy diagnostic curve path, or provide data.path + ridge_fit_summary."
        )
    else:
        import os

        if os.environ.get("MMM_UNSAFE_LEGACY_DIAGNOSTIC_BUDGET", "").strip() != "1":
            raise SemanticContractError(
                "Legacy diagnostic path requires MMM_UNSAFE_LEGACY_DIAGNOSTIC_BUDGET=1 "
                "(explicit unsafe acknowledgement)."
            )
        try:
            payload = run_legacy_diagnostic_optimize_budget(
                cfg=cfg,
                er_data=er,
                curve_bundle=curve_bundle,
                allow_unsafe_decision_apis=allow_unsafe_decision_apis,
            )
        except PolicyError as e:
            raise SemanticContractError(str(e)) from e
        except ValueError as e:
            raise SemanticContractError(str(e)) from e

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return payload
