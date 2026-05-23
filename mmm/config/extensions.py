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
    #: When False, ``approved_for_reporting`` / optimization scorecard paths ignore baseline beat checks.
    require_beats_baselines_for_approval: bool = True
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


class PlanningPolicyConfig(BaseModel):
    """Guardrails for non-media control assumptions on decision paths."""

    promo_columns: list[str] = Field(default_factory=list)
    pricing_columns: list[str] = Field(default_factory=list)
    macro_columns: list[str] = Field(default_factory=list)
    seasonality_columns: list[str] = Field(default_factory=list)
    #: When True in prod, decision paths with sensitive controls + observed assumptions fail closed.
    strict_prod_requires_explicit_control_scenario: bool = False
    #: Optional name-based warnings for columns in data.control_columns (never blocks alone).
    name_heuristic_warnings: bool = True
    #: When True, decision artifacts include full overlay row lists; otherwise SHA-256 hashes only.
    store_full_control_overlays_in_artifacts: bool = False


class ExperimentSchedulerConfig(BaseModel):
    """Prioritize experimentation effort from post-fit diagnostics (no execution)."""

    enabled: bool = True
    high_priority_threshold: float = Field(default=0.62, ge=0.0, le=1.0)
    low_priority_threshold: float = Field(default=0.38, ge=0.0, le=1.0)
    #: Score when calibration evidence is absent (higher → more stale / needs experiment).
    staleness_absent_calibration: float = Field(default=1.0, ge=0.0, le=1.0)
    staleness_partial_calibration: float = Field(default=0.55, ge=0.0, le=1.0)
    staleness_strong_calibration: float = Field(default=0.12, ge=0.0, le=1.0)


class FeatureSeparabilityConfig(BaseModel):
    """Diagnostic separability guidance for split media variables (no automatic merges)."""

    enabled: bool = True
    #: Explicit groups override auto prefix detection when non-empty.
    feature_groups: dict[str, list[str]] = Field(default_factory=dict)
    auto_group_prefix: bool = True
    correlation_moderate: float = Field(default=0.5, ge=0.0, le=1.0)
    correlation_high: float = Field(default=0.8, ge=0.0, le=1.0)
    sign_flip_rate_unstable: float = Field(default=0.2, ge=0.0, le=1.0)
    coef_cv_unstable: float = Field(default=0.5, ge=0.0, le=10.0)
    vif_healthy: float = Field(default=5.0, ge=1.0, le=100.0)
    vif_warning: float = Field(default=10.0, ge=1.0, le=200.0)
    contribution_share_variance_unstable: float = Field(default=0.08, ge=0.0, le=1.0)
    business_importance_high_spend_share: float = Field(default=0.08, ge=0.0, le=1.0)
    business_importance_high_contribution_share: float = Field(default=0.10, ge=0.0, le=1.0)
    #: Minimum group spend share of panel media before ``experiment_recommended`` (tiny splits → caution only).
    experiment_min_group_spend_share: float = Field(default=0.03, ge=0.0, le=1.0)
    reuse_identifiability_bootstrap: bool = True
    bootstrap_rounds: int = Field(default=12, ge=0, le=64)


class ContinuousValidationConfig(BaseModel):
    """Diagnostic comparison of prior model predictions vs new experiment evidence."""

    enabled: bool = False
    registry_dir: str | None = None
    lookback_days: int = Field(default=365, ge=1)
    require_experiment_se: bool = False
    experiment_registry_path: str | None = None


class DecisionValidationConfig(BaseModel):
    """Diagnostic comparison of prior decisions vs subsequent experiment evidence."""

    enabled: bool = False
    decision_registry_dir: str | None = None
    experiment_registry_path: str | None = None
    lookback_days: int = Field(default=180, ge=1)


class RobustOptimizationResearchConfig(BaseModel):
    """PR 5B: research-only robust optimization diagnostics (not prod optimize-budget)."""

    enabled: bool = False
    risk_lambda: float = Field(default=1.0, ge=0.0)
    lcb_z_score: float = Field(default=1.0, ge=0.0)
    n_candidates: int = Field(default=6, ge=2, le=32)
    n_stability_scenarios: int = Field(default=6, ge=2, le=32)
    budget_perturbation_pct: float = Field(default=0.05, ge=0.0, le=0.25)
    frontier_lambda_grid: list[float] = Field(default_factory=lambda: [0.0, 0.5, 1.0, 2.0, 5.0])


class UncertaintyPropagationConfig(BaseModel):
    """PR 5A: research-only uncertainty propagation reports (no optimizer / prod monetary CIs)."""

    enabled: bool = False
    ridge_summarize_bootstrap: bool = True
    #: Reserved; conformal summarization not implemented in this package version.
    ridge_summarize_conformal: bool = False


class RidgeUncertaintyResearchConfig(BaseModel):
    """Optional Ridge interval research (never enables production decision CIs)."""

    enabled: bool = False
    bootstrap_rounds: int = Field(default=8, ge=4, le=32)


class PerformanceAuditConfig(BaseModel):
    """Extension runtime telemetry (diagnostic only)."""

    enabled: bool = True


class ReproducibilityCertificationConfig(BaseModel):
    """Post-fit reproducibility snapshot and self-certification (diagnostic only)."""

    enabled: bool = False


class PerformanceCertificationConfig(BaseModel):
    """Synthetic scaling performance certification (diagnostic only; can be slow)."""

    enabled: bool = False
    include_medium_scenario: bool = False
    include_large_scenario: bool = False
    n_trials_per_scenario: int = Field(default=1, ge=1, le=5)
    seed: int = 42


class DriftHistoricalConfig(BaseModel):
    """Historical drift context (diagnostic only — no auto-retrain)."""

    prior_run_dir: str | None = None
    #: Directory holding ``accepted_runs.jsonl`` for cross-run comparisons.
    registry_dir: str | None = None
    #: When True, compare against latest registry entry (excluding current run).
    use_registry: bool = False


class PanelQAConfig(BaseModel):
    """Data QA above ``validate_panel`` — extension artifacts + optional PROD training blocks."""

    enabled: bool = True
    #: In ``PROD``, fail training when ``max_severity == "block"`` (e.g. duplicate geo-week keys).
    prod_block_severity: Literal["off", "block"] = "off"
    #: When ``True`` in prod, allows ``prod_block_severity: off`` (explicit waiver of training blocks).
    prod_block_waiver: bool = False
    #: Warn when missing (geo, week) cells vs a full rectangular grid exceed this fraction.
    missing_week_warn_fraction: float = Field(default=0.12, ge=0.0, le=1.0)
    spend_spike_abs_z: float = Field(default=8.0, ge=3.0, le=30.0)
    #: Warn when this fraction of rows have all channel spends <= 0.
    all_channel_zero_warn_fraction: float = Field(default=0.2, ge=0.0, le=1.0)


class ExtensionSuiteConfig(BaseModel):
    continuous_validation: ContinuousValidationConfig = Field(default_factory=ContinuousValidationConfig)
    decision_validation: DecisionValidationConfig = Field(default_factory=DecisionValidationConfig)
    robust_optimization_research: RobustOptimizationResearchConfig = Field(
        default_factory=RobustOptimizationResearchConfig
    )
    uncertainty_propagation: UncertaintyPropagationConfig = Field(
        default_factory=UncertaintyPropagationConfig
    )
    identifiability: IdentifiabilityRunConfig = Field(default_factory=IdentifiabilityRunConfig)
    feature_separability: FeatureSeparabilityConfig = Field(default_factory=FeatureSeparabilityConfig)
    experiment_scheduler: ExperimentSchedulerConfig = Field(default_factory=ExperimentSchedulerConfig)
    planning_policy: PlanningPolicyConfig = Field(default_factory=PlanningPolicyConfig)
    governance: GovernanceConfig = Field(default_factory=GovernanceConfig)
    optimization_gates: OptimizationGateConfig = Field(default_factory=OptimizationGateConfig)
    estimand: EstimandConfig = Field(default_factory=EstimandConfig)
    product: ProductScopeConfig = Field(default_factory=ProductScopeConfig)
    features: FeatureEngineConfig = Field(default_factory=FeatureEngineConfig)
    falsification: FalsificationConfig = Field(default_factory=FalsificationConfig)
    curves: CurveResponseConfig = Field(default_factory=CurveResponseConfig)
    panel_qa: PanelQAConfig = Field(default_factory=PanelQAConfig)
    drift_historical: DriftHistoricalConfig = Field(default_factory=DriftHistoricalConfig)
    ridge_uncertainty_research: RidgeUncertaintyResearchConfig = Field(
        default_factory=RidgeUncertaintyResearchConfig
    )
    performance_audit: PerformanceAuditConfig = Field(default_factory=PerformanceAuditConfig)
    reproducibility_certification: ReproducibilityCertificationConfig = Field(
        default_factory=ReproducibilityCertificationConfig
    )
    performance_certification: PerformanceCertificationConfig = Field(
        default_factory=PerformanceCertificationConfig
    )

    model_config = {"extra": "forbid"}
