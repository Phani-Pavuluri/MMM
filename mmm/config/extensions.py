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
    #: How transforms enter Bayesian decision runs vs Ridge+BO (cross-framework comparability).
    bayesian_decision_transform_stance: Literal[
        "fixed_yaml_features_labeled",
        "research_only_experimental_harmonize",
    ] = "fixed_yaml_features_labeled"
    #: Full-panel budget SLSQP: number of feasible randomized starts (deterministic seed from ridge_bo.sampler_seed).
    simulation_optimizer_n_starts: int = Field(default=15, ge=10, le=20)
    #: After the best start, re-solve from perturbed feasible points; used for allocation
    #: stability / decision_safe gating.
    simulation_optimizer_stability_checks: int = Field(default=3, ge=0, le=32)
    #: Max L1 distance between normalized allocations across stability re-solves; above => not decision-safe.
    simulation_optimizer_stability_max_l1: float = Field(default=0.22, ge=0.0, le=1.0)


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
    #: When not ``None``, optimization is blocked if falsification ``flags`` count exceeds this cap
    #: even when ``require_falsification_pass`` is ``False`` (graded / prod-tightening gate).
    falsification_max_allowed_flags_for_optimization: int | None = None
    #: Fraction of ``max_identifiability_risk`` above which we label identifiability as limiting decision safety.
    identifiability_decision_safety_margin: float = Field(default=0.85, ge=0.5, le=1.0)
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
    #: When set (0–1), ``posterior_predictive_ok`` additionally requires ``empirical_coverage_p90`` >= this.
    bayesian_min_ppc_empirical_coverage: float | None = Field(default=None, ge=0.0, le=1.0)
    #: Max share of total inverse-SE mass one replay unit may hold in prod (mitigate SE-underreport dominance).
    replay_max_unit_inverse_se_influence_share: float = Field(default=0.65, ge=0.25, le=1.0)
    #: When True, prod decision paths may proceed past identifiability caps only with a validated waiver artifact.
    allow_identifiability_waiver: bool = False
    #: Post-fit gate: block if ``abs(residual_lag1_autocorr)`` exceeds this (Ridge extension path).
    post_fit_residual_autocorr_block_abs: float = Field(default=0.55, ge=0.0, le=0.99)
    post_fit_oot_mae_ratio_warn: float = Field(default=1.45, ge=1.0, le=20.0)
    post_fit_oot_mae_ratio_block: float = Field(default=2.35, ge=1.0, le=20.0)
    post_fit_objective_trial_cv_warn: float = Field(default=0.12, ge=0.0, le=1.0)
    post_fit_kpi_log_mean_ratio_warn_abs: float = Field(default=0.22, ge=0.0, le=2.0)
    #: When set in **prod**, ``operational_health`` blocks if falsification reports fewer distinct placebo families.
    falsification_prod_min_reported_placebo_families: int | None = Field(default=None, ge=1, le=32)


class OptimizationGateConfig(BaseModel):
    """E11: block unsafe optimization."""

    enabled: bool = True
    require_governance_optimization_flag: bool = True
    require_response_curve_safe: bool = True
    max_identifiability_risk: float = 0.7
    allow_missing_extension_report: bool = False
    #: In ``PROD``, deny optimization when extension ``panel_qa.max_severity`` is ``block``.
    prod_block_on_panel_qa_block: bool = True
    #: In ``PROD``, also deny when ``panel_qa.max_severity`` is ``warn`` (strict data discipline).
    prod_block_on_panel_qa_warn: bool = False


class FeatureEngineConfig(BaseModel):
    """E3: optional extra controls (trainer still uses data.control_columns; these feed diagnostics)."""

    trend_spline_knots: int = 0
    fourier_yearly_harmonics: int = 0
    holiday_country: str | None = None


class FalsificationConfig(BaseModel):
    enabled: bool = True
    placebo_draws: int = 5
    #: ``fixed_scale_floor`` uses analytic scale floors; ``empirical_y_permutation`` builds a null from permuted ``y``.
    null_calibration_method: Literal["fixed_scale_floor", "empirical_y_permutation"] = "fixed_scale_floor"
    empirical_null_n_permutations: int = Field(default=24, ge=4, le=128)
    loo_geo: bool = False
    #: Second independent Gaussian noise column (same ridge ``alpha`` as media path).
    dual_noise_placebo: bool = True
    #: Randomly permute media columns before fitting — fungible channels absorb similar mass under collinearity.
    media_column_permutation_placebo: bool = True
    #: Circular time-shift of the media design matrix vs ``y`` (breaks timing alignment).
    time_shifted_media_placebo: bool = True
    #: Pool random channel subsets into two aggregate columns and refit (collinearity / fungibility stress).
    grouped_channel_placebo: bool = True
    #: Shuffle rows' media vectors **within each geo** (requires ``geo_ids`` passed into falsification run).
    within_geo_media_row_shuffle_placebo: bool = False
    #: Held-out geo stress: fit on a random half of geos' rows, score coef transfer (requires ``geo_ids``).
    geo_split_coef_transfer_placebo: bool = False


class CurveResponseConfig(BaseModel):
    """Steady-state response curve construction (Sprint 6)."""

    steady_state_horizon_weeks: int = Field(default=52, ge=2, le=520)
    require_curve_stress_for_optimization: bool = True


class PanelQAConfig(BaseModel):
    """Data QA above ``validate_panel`` — extension artifacts + optional PROD training blocks."""

    enabled: bool = True
    #: In ``PROD``, fail training when ``max_severity == "block"`` (e.g. duplicate geo-week keys).
    prod_block_severity: Literal["off", "block"] = "off"
    #: Warn when missing (geo, week) cells vs a full rectangular grid exceed this fraction.
    missing_week_warn_fraction: float = Field(default=0.12, ge=0.0, le=1.0)
    spend_spike_abs_z: float = Field(default=8.0, ge=3.0, le=30.0)
    #: Warn when this fraction of rows have all channel spends <= 0.
    all_channel_zero_warn_fraction: float = Field(default=0.2, ge=0.0, le=1.0)


class ExtensionSuiteConfig(BaseModel):
    identifiability: IdentifiabilityRunConfig = Field(default_factory=IdentifiabilityRunConfig)
    governance: GovernanceConfig = Field(default_factory=GovernanceConfig)
    optimization_gates: OptimizationGateConfig = Field(default_factory=OptimizationGateConfig)
    estimand: EstimandConfig = Field(default_factory=EstimandConfig)
    product: ProductScopeConfig = Field(default_factory=ProductScopeConfig)
    features: FeatureEngineConfig = Field(default_factory=FeatureEngineConfig)
    falsification: FalsificationConfig = Field(default_factory=FalsificationConfig)
    curves: CurveResponseConfig = Field(default_factory=CurveResponseConfig)
    panel_qa: PanelQAConfig = Field(default_factory=PanelQAConfig)

    model_config = {"extra": "forbid"}
