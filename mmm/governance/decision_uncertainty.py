"""Decision-facing uncertainty disclosure (no fabricated intervals)."""

from __future__ import annotations

from typing import Any

from mmm.config.schema import Framework, MMMConfig, RunEnvironment
from mmm.governance.uncertainty_policy import bayesian_intervals_allowed, ridge_forbids_precise_monetary_ci

RIDGE_DISCLOSURE = (
    "Point estimates only; calibrated monetary confidence intervals are not currently supported."
)


_METHODS_INVESTIGATED = {
    "bootstrap_intervals": "not_implemented",
    "conformal_intervals": "not_implemented",
    "calibration_coverage_checks": "disclosure_only",
    "note": (
        "Bootstrap and conformal monetary intervals were evaluated for Ridge production "
        "and deferred — point estimates with explicit disclosure remain the contract."
    ),
}


def build_decision_uncertainty(
    config: MMMConfig,
    *,
    fit_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Artifact block ``decision_uncertainty`` for bundles, CLI, and reports."""
    if config.framework == Framework.RIDGE_BO:
        return {
            "uncertainty_available": False,
            "uncertainty_unavailable": True,
            "uncertainty_method": "point_estimate",
            "confidence_supported": False,
            "disclosure_text": RIDGE_DISCLOSURE,
            "ridge_production_forbids_precise_monetary_ci": ridge_forbids_precise_monetary_ci(config),
            "methods_investigated": _METHODS_INVESTIGATED,
        }
    if config.framework == Framework.BAYESIAN:
        intervals_ok = bayesian_intervals_allowed(fit_meta)
        prod_blocked = config.run_environment == RunEnvironment.PROD
        avail = bool(intervals_ok and not prod_blocked)
        return {
            "uncertainty_available": avail,
            "uncertainty_unavailable": not avail,
            "uncertainty_method": "posterior_draws" if intervals_ok else "diagnostic_only",
            "confidence_supported": avail,
            "methods_investigated": _METHODS_INVESTIGATED,
            "disclosure_text": (
                "Bayesian posterior intervals are research/diagnostic only; "
                "production decision surfaces require Ridge full-panel Δμ."
                if prod_blocked
                else "Posterior intervals available when PPC gates pass (research surfaces)."
            ),
            "prod_decisioning_allowed": False,
        }
    return {
        "uncertainty_available": False,
        "uncertainty_unavailable": True,
        "uncertainty_method": "unknown_framework",
        "confidence_supported": False,
        "disclosure_text": RIDGE_DISCLOSURE,
        "methods_investigated": _METHODS_INVESTIGATED,
    }
