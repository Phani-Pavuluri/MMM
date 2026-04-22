"""E7: structured uncertainty buckets for reporting (not full inference)."""

from __future__ import annotations

from typing import Any


class UncertaintyDecomposer:
    """Separate uncertainty sources when partial inputs exist."""

    @staticmethod
    def build_report(
        *,
        posterior_width: dict[str, float] | None = None,
        bootstrap_width: dict[str, float] | None = None,
        cv_mae_std: float | None = None,
        experiment_se_scale: float | None = None,
        optimization_robustness: float | None = None,
    ) -> dict[str, Any]:
        return {
            "parameter_uncertainty": posterior_width or {},
            "model_specification_uncertainty": {"note": "compare across specs / not quantified unless ensemble"},
            "cv_uncertainty": {"mae_std_across_splits": cv_mae_std},
            "experiment_uncertainty": {"inverse_se_scale": experiment_se_scale},
            "optimization_uncertainty": {"spread_proxy": optimization_robustness},
            "interpretation": (
                "Intervals are not automatically calibrated unless Bayesian or bootstrap enabled."
            ),
            "decision_safe_intervals": False,
            "ridge_path_note": "Ridge+BO: intervals are approximate unless bootstrap or Bayesian draws are wired.",
        }
