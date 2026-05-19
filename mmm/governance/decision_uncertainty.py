"""Decision-facing uncertainty disclosure (no fabricated intervals)."""

from __future__ import annotations

from typing import Any

from mmm.config.schema import Framework, MMMConfig, RunEnvironment
from mmm.governance.uncertainty_policy import bayesian_intervals_allowed, ridge_forbids_precise_monetary_ci

RIDGE_DISCLOSURE = (
    "Point estimates only; calibrated monetary confidence intervals are not currently supported."
)


def build_decision_uncertainty(
    config: MMMConfig,
    *,
    fit_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Artifact block ``decision_uncertainty`` for bundles, CLI, and reports."""
    if config.framework == Framework.RIDGE_BO:
        return {
            "uncertainty_available": False,
            "uncertainty_method": "point_estimate",
            "confidence_supported": False,
            "disclosure_text": RIDGE_DISCLOSURE,
            "ridge_production_forbids_precise_monetary_ci": ridge_forbids_precise_monetary_ci(config),
        }
    if config.framework == Framework.BAYESIAN:
        intervals_ok = bayesian_intervals_allowed(fit_meta)
        prod_blocked = config.run_environment == RunEnvironment.PROD
        return {
            "uncertainty_available": bool(intervals_ok and not prod_blocked),
            "uncertainty_method": "posterior_draws" if intervals_ok else "diagnostic_only",
            "confidence_supported": bool(intervals_ok and not prod_blocked),
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
        "uncertainty_method": "unknown_framework",
        "confidence_supported": False,
        "disclosure_text": RIDGE_DISCLOSURE,
    }
