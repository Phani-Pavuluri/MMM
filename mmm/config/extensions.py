"""Optional diagnostics, governance, gates, and feature-engine settings (additive to MMMConfig)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from mmm.causal.estimand import EstimandConfig


class ProductScopeConfig(BaseModel):
    """
    Product / economics scope.

    ``full_model``: budget decisions are scored through full-panel μ simulation (default).
    ``curve_local``: legacy curve interpolation path for diagnostics / warm starts only — not final truth.
    """

    planner_mode: Literal["full_model", "curve_local"] = "full_model"
    ridge_uncertainty_stance: Literal["exploratory", "bootstrap_ready"] = "exploratory"
    bayesian_maturity: Literal["experimental", "supported"] = "experimental"
    #: How row-level μ is collapsed for Δμ when multiple geos exist (Ridge full-panel path).
    planning_delta_mu_aggregation: Literal["global_row_mean", "geo_mean_then_global_mean"] = (
        "global_row_mean"
    )
    #: ``draws`` enables posterior draw–based P10/P50/P90 and risk-aware optimizers (with diagnostics + coef draws).
    #: ``disclosure_only`` adds artifact notes without requiring draw payloads.
    posterior_planning_mode: Literal["off", "disclosure_only", "draws"] = "off"


class IdentifiabilityRunConfig(BaseModel):
    enabled: bool = True
    bootstrap_rounds: int = 15
    bootstrap_frac: float = 0.85
    vif_threshold: float = 10.0
    condition_threshold: float = 1e4


class GovernanceConfig(BaseModel):
    """Scorecard thresholds; approvals are advisory unless CLI enforces."""

    max_mae_ratio_vs_baseline: float = 1.15
    max_identifiability_risk: float = 0.65
    max_high_vif_channels: int = 2
    require_falsification_pass: bool = False
    #: Mean per-unit standardized squared error from **replay** calibration (chi²-like scale).
    max_replay_calibration_chi2: float = 12.0
    #: Upper bound for legacy (non-replay) calibration loss after SE normalization in loss term.
    max_legacy_calibration_loss: float = 25.0
    require_posterior_predictive_pass: bool = False
    #: ArviZ ``summary`` thresholds for **decision-grade** Bayesian claims (see ``bayesian_inference_report``).
    bayesian_max_rhat: float = 1.01
    bayesian_min_ess_bulk: float = 200.0
    bayesian_max_divergences: int = 0
    #: When set, ``posterior_predictive_ok`` requires ``mean_abs_gap`` on the modeling scale to be <= this.
    bayesian_max_mean_abs_ppc_gap: float | None = None


class OptimizationGateConfig(BaseModel):
    """E11: block unsafe optimization."""

    enabled: bool = False
    require_governance_optimization_flag: bool = True
    require_response_curve_safe: bool = True
    max_identifiability_risk: float = 0.7
    allow_missing_extension_report: bool = False


class FeatureEngineConfig(BaseModel):
    """E3: optional extra controls (trainer still uses data.control_columns; these feed diagnostics)."""

    trend_spline_knots: int = 0
    fourier_yearly_harmonics: int = 0
    holiday_country: str | None = None


class FalsificationConfig(BaseModel):
    enabled: bool = True
    placebo_draws: int = 5
    loo_geo: bool = False


class CurveResponseConfig(BaseModel):
    """Steady-state response curve construction (Sprint 6)."""

    steady_state_horizon_weeks: int = Field(default=52, ge=2, le=520)
    require_curve_stress_for_optimization: bool = True


class ExtensionSuiteConfig(BaseModel):
    identifiability: IdentifiabilityRunConfig = Field(default_factory=IdentifiabilityRunConfig)
    governance: GovernanceConfig = Field(default_factory=GovernanceConfig)
    optimization_gates: OptimizationGateConfig = Field(default_factory=OptimizationGateConfig)
    estimand: EstimandConfig = Field(default_factory=EstimandConfig)
    product: ProductScopeConfig = Field(default_factory=ProductScopeConfig)
    features: FeatureEngineConfig = Field(default_factory=FeatureEngineConfig)
    falsification: FalsificationConfig = Field(default_factory=FalsificationConfig)
    curves: CurveResponseConfig = Field(default_factory=CurveResponseConfig)

    model_config = {"extra": "forbid"}
