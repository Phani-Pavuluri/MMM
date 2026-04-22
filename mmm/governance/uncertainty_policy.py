"""Uncertainty presentation rules (Ridge vs Bayesian, production vs research)."""

from __future__ import annotations

from typing import Any

from mmm.config.schema import Framework, MMMConfig, RunEnvironment


def ridge_forbids_precise_monetary_ci(config: MMMConfig) -> bool:
    """Ridge + prod: no precise monetary confidence intervals in decision-facing surfaces."""
    return config.framework == Framework.RIDGE_BO and config.run_environment == RunEnvironment.PROD


def bayesian_intervals_allowed(fit_meta: dict[str, Any] | None) -> bool:
    """Credibility-style bands only when inference and PPC gates pass."""
    if not fit_meta:
        return False
    return bool(fit_meta.get("posterior_diagnostics_ok")) and bool(fit_meta.get("posterior_predictive_ok"))


def assert_no_forbidden_ridge_money_ci_keys(payload: dict[str, Any], *, config: MMMConfig) -> None:
    """
    Fail-closed guard for API/report dicts: forbid known money-CI section keys in Ridge prod.

    Callers attach ``decision_policy`` / ``economics_contract`` instead of emitting calibrated $ CI.
    """
    if not ridge_forbids_precise_monetary_ci(config):
        return
    forbidden = (
        "credible_interval_revenue_usd",
        "ci95_revenue_usd",
        "posterior_interval_money",
    )
    keys = set(payload.keys())
    bad = [k for k in forbidden if k in keys]
    if bad:
        raise ValueError(f"Ridge production forbids monetary CI keys: {bad}")
