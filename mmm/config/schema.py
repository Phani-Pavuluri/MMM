"""Canonical configuration models (YAML + API aligned)."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from mmm.config.extensions import ExtensionSuiteConfig


class ModelForm(str, Enum):
    SEMI_LOG = "semi_log"
    LOG_LOG = "log_log"


class PoolingMode(str, Enum):
    NONE = "none"
    FULL = "full"
    PARTIAL = "partial"


class Framework(str, Enum):
    BAYESIAN = "bayesian"
    RIDGE_BO = "ridge_bo"


class CVMode(str, Enum):
    ROLLING = "rolling"
    EXPANDING = "expanding"
    AUTO = "auto"


class CVSplitAxis(str, Enum):
    """``geo_rank`` = legacy within-geo dense week rank; ``calendar_week`` = global calendar index."""

    GEO_RANK = "geo_rank"
    CALENDAR_WEEK = "calendar_week"
    GEO_BLOCKED = "geo_blocked"


class FitMetric(str, Enum):
    RMSE = "rmse"
    MAE = "mae"
    MAPE = "mape"
    WMAPE = "wmape"


class NormalizationProfile(str, Enum):
    """How composite objective terms are scaled before applying weights."""

    STRICT_PROD = "strict_prod"
    RESEARCH = "research"
    DEBUG = "debug"


class RunEnvironment(str, Enum):
    """Deployment context for governance defaults."""

    DEV = "dev"
    RESEARCH = "research"
    STAGING = "staging"
    PROD = "prod"


class BayesianBackend(str, Enum):
    PYMC = "pymc"
    STAN = "stan"


class ArtifactBackend(str, Enum):
    LOCAL = "local"
    MLFLOW = "mlflow"


class DataConfig(BaseModel):
    """Input contract."""

    path: str | None = None
    geo_column: str = "geo_id"
    week_column: str = "week_start_date"
    target_column: str = "revenue"
    channel_columns: list[str] = Field(default_factory=list)
    control_columns: list[str] = Field(default_factory=list)
    wide_format: bool = True
    #: Optional durable dataset version (lakehouse snapshot id, etc.) for decision bundle lineage.
    data_version_id: str | None = None


class TransformConfig(BaseModel):
    adstock: Literal["geometric", "weibull"] = "geometric"
    saturation: Literal["hill", "log", "logistic"] = "hill"
    adstock_params: dict[str, float] = Field(default_factory=dict)
    saturation_params: dict[str, float] = Field(default_factory=dict)
    custom_plugin: str | None = None


class CVConfig(BaseModel):
    mode: CVMode = CVMode.AUTO
    n_splits: int = 4
    min_train_weeks: int = 52
    horizon_weeks: int = 4
    gap_weeks: int = 0
    split_axis: CVSplitAxis = Field(
        default=CVSplitAxis.CALENDAR_WEEK,
        description="Use calendar_week for weekly panels; geo_rank is legacy within-geo dense rank.",
    )
    #: ``None`` inherits from top-level ``random_seed`` at resolve time.
    geo_blocked_seed: int | None = None


class RidgeBOConfig(BaseModel):
    n_trials: int = 32
    timeout_sec: float | None = None
    #: ``None`` inherits from top-level ``random_seed`` at resolve time.
    sampler_seed: int | None = None
    alpha_ridge_bounds: tuple[float, float] = (1e-4, 1e3)
    use_pruner: bool = True


class BayesianConfig(BaseModel):
    backend: BayesianBackend = BayesianBackend.PYMC
    draws: int = 1000
    tune: int = 1000
    chains: int = 4
    target_accept: float = 0.9
    #: ``None`` inherits from top-level ``random_seed`` at resolve time.
    nuts_seed: int | None = None
    media_coef_sigma: float = 0.8
    control_coef_sigma: float = 2.0
    #: ``half_normal_nonneg`` encodes a positivity prior on **media** channels (default). ``normal_symmetric``
    #: allows negative media elasticities when KPI/channel semantics require it (research; document in artifacts).
    media_channel_prior: Literal["half_normal_nonneg", "normal_symmetric"] = "half_normal_nonneg"
    prior_predictive_draws: int = 0
    posterior_predictive_draws: int = 0
    #: Research-only experiment lift likelihood (PR 3); does not enable prod decisioning.
    use_experiment_likelihood: bool = False
    experiment_registry_path: str | None = None
    experiment_likelihood_weight: float = 1.0
    min_experiment_quality_tier: Literal["high", "medium", "low"] = "medium"
    allow_aggregate_only_evidence: bool = True
    allow_allocated_shocks: bool = True
    allow_conservative_missing_se: bool = False
    allow_level_lift_mismatch_research: bool = False
    exp_likelihood_research_only: bool = True
    #: Research-only hierarchical media priors: child ~ Normal(parent, hier_sigma_group).
    use_hierarchy: bool = False
    hierarchy_research_only: bool = True
    #: Prior scale for ``hier_sigma_group`` HalfNormal.
    hierarchy_group_sigma_prior: float = 0.5

    @model_validator(mode="after")
    def _validate_experiment_likelihood(self) -> BayesianConfig:
        if self.use_experiment_likelihood:
            if not self.experiment_registry_path:
                raise ValueError(
                    "bayesian.use_experiment_likelihood requires bayesian.experiment_registry_path"
                )
            if not self.exp_likelihood_research_only:
                raise ValueError(
                    "bayesian.exp_likelihood_research_only must remain true "
                    "(experiment likelihood cannot enable prod decisioning)"
                )
            if float(self.experiment_likelihood_weight) <= 0:
                raise ValueError("bayesian.experiment_likelihood_weight must be positive")
        if self.use_hierarchy:
            if not self.hierarchy_research_only:
                raise ValueError(
                    "bayesian.hierarchy_research_only must remain true "
                    "(Bayesian hierarchy cannot enable prod decisioning)"
                )
            if self.hierarchy_group_sigma_prior <= 0:
                raise ValueError("bayesian.hierarchy_group_sigma_prior must be positive")
        return self


class ObjectiveWeights(BaseModel):
    predictive: float = 1.0
    calibration: float = 1.0
    stability: float = 0.5
    plausibility: float = 0.25
    complexity: float = 0.1

    @model_validator(mode="after")
    def _nonnegative_and_positive_sum(self) -> ObjectiveWeights:
        d = self.model_dump()
        for k, v in d.items():
            if float(v) < 0:
                raise ValueError(f"objective.weights.{k} must be >= 0 (got {v})")
        if sum(float(x) for x in d.values()) <= 0:
            raise ValueError("objective.weights must sum to a positive value")
        return self


class CompositeObjectiveConfig(BaseModel):
    """Ridge+BO composite objective — components are normalized before weighting."""

    primary_metric: FitMetric = FitMetric.WMAPE
    weights: ObjectiveWeights = Field(default_factory=ObjectiveWeights)
    multi_objective: bool = False
    pareto_store_trials: bool = True
    normalization_profile: NormalizationProfile = NormalizationProfile.RESEARCH
    #: Prod Ridge+BO: required explicit objective contract name (no silent default-objective behavior).
    named_profile: str | None = Field(
        default=None,
        description="Named Ridge+BO objective profile for prod governance (e.g. ridge_bo_standard_v1).",
    )


class HierarchyConfig(BaseModel):
    """Explicit hierarchical borrowing for Ridge BO (opt-in; never inferred from data)."""

    enabled: bool = False
    hierarchy_definition_path: str | None = None
    hierarchy_type: Literal["geography", "channel", "campaign"] = "geography"
    regularization_strength: float = 0.1
    min_children_per_parent: int = 2
    allow_cross_branch_pooling: bool = False

    @model_validator(mode="after")
    def _validate_hierarchy_config(self) -> HierarchyConfig:
        if self.enabled and not self.hierarchy_definition_path:
            raise ValueError(
                "hierarchy.enabled=true requires hierarchy.hierarchy_definition_path "
                "(explicit HierarchyDefinition JSON)"
            )
        if self.regularization_strength < 0:
            raise ValueError("hierarchy.regularization_strength must be >= 0")
        if self.min_children_per_parent < 1:
            raise ValueError("hierarchy.min_children_per_parent must be >= 1")
        return self


class CalibrationConfig(BaseModel):
    enabled: bool = False
    experiments_path: str | None = None
    #: JSON list of :class:`mmm.calibration.contracts.CalibrationUnit`-compatible dicts with
    #: ``observed_spend_frame`` / ``counterfactual_spend_frame`` for replay (decision-safe path).
    replay_units_path: str | None = None
    #: Optional explicit train/holdout JSON lists; when both set, auto-split from ``replay_units_path`` is skipped.
    train_replay_units_path: str | None = None
    holdout_replay_units_path: str | None = None
    use_replay_calibration: bool = False
    #: When true, BO replay objective uses train units only; holdout units are diagnostic (extension artifact).
    use_replay_holdout_split: bool = False
    replay_holdout_fraction: float = Field(default=0.25, gt=0.0, lt=1.0)
    replay_holdout_min_train_units: int = Field(default=1, ge=1)
    replay_holdout_min_holdout_units: int = Field(default=1, ge=1)
    lift_column: str = "lift"
    lift_se_column: str | None = "lift_se"
    match_levels: list[str] = Field(
        default_factory=lambda: ["geo", "time_window", "channel", "device", "product"]
    )
    experiment_target_kpi: str | None = None
    use_quality_weights: bool = True
    #: Optional JSON registry (see ``mmm.experiments.durable_registry``) listing approved experiment_ids.
    experiment_registry_path: str | None = None
    #: When ``True`` in PROD, every replay unit must carry a non-empty ``experiment_id`` that is
    #: ``approved`` in the registry.
    require_approved_experiment_registry: bool = False
    #: Phase 1+ experiment evidence registry (JSON list or ``mmm_experiment_evidence_registry_v1``).
    evidence_registry_path: str | None = None
    #: Opt-in weighted replay loss (Ridge BO); default legacy unweighted path.
    evidence_weighting_enabled: bool = False
    #: Emit experiment_compatibility_report and related diagnostics.
    compatibility_resolver_enabled: bool = False
    #: ``legacy`` keeps existing replay calibration; ``evidence_registry`` is diagnostic-first.
    replay_mode: Literal["legacy", "evidence_registry"] = "legacy"
    #: Ridge hierarchical penalty between parent/child geo or channel groups (research).
    hierarchical_regularization_enabled: bool = False
    #: Declared model geo granularity for compatibility resolver.
    model_geo_granularity: Literal["national", "region", "dma", "geo", "user"] = "geo"
    #: Map experiment platform channel names to panel channel_columns.
    channel_mapping: dict[str, str] = Field(default_factory=dict)
    #: Prod evidence-registry replay: allow units without SE (discouraged; default fail-closed).
    allow_missing_se_in_prod_evidence_replay: bool = False
    #: Advisory threshold for ``replay_generalization_gap`` severity (holdout − train replay loss).
    replay_generalization_gap_threshold: float = Field(default=0.25, ge=0.0)
    #: When ``True``, severe replay gap may invalidate model release (default advisory warning only).
    block_on_severe_replay_gap: bool = False
    #: How replay calibration coefficients are fit for the BO objective (default backward compatible).
    replay_refit_mode: Literal["full_panel_refit", "fold_aligned", "holdout_only_diagnostic"] = (
        "full_panel_refit"
    )

    @model_validator(mode="after")
    def _validate_calibration_match_levels(self) -> CalibrationConfig:
        from mmm.calibration.replay_refit_mode import validate_replay_refit_mode

        validate_replay_refit_mode(self.replay_refit_mode)
        from mmm.calibration.matching import validate_calibration_match_levels

        validate_calibration_match_levels(self.match_levels)
        if self.enabled and self.experiments_path and "channel" not in self.match_levels:
            raise ValueError(
                "calibration.enabled with experiments_path requires 'channel' in calibration.match_levels "
                "(experiment rows are always filtered by channel against panel channel_columns)."
            )
        if self.replay_mode == "evidence_registry":
            if not self.evidence_registry_path:
                raise ValueError(
                    "calibration.replay_mode='evidence_registry' requires calibration.evidence_registry_path"
                )
            if not self.compatibility_resolver_enabled:
                raise ValueError(
                    "calibration.replay_mode='evidence_registry' requires "
                    "calibration.compatibility_resolver_enabled=true"
                )
        if self.evidence_weighting_enabled and self.replay_mode != "evidence_registry":
            raise ValueError(
                "calibration.evidence_weighting_enabled requires calibration.replay_mode='evidence_registry'"
            )
        if self.evidence_weighting_enabled and not self.use_replay_calibration:
            raise ValueError(
                "calibration.evidence_weighting_enabled requires calibration.use_replay_calibration=true"
            )
        return self


class GovernanceWorkflowConfig(BaseModel):
    """Explicit promotion workflow for prod decision surfaces (no auto-promotion)."""

    require_promoted_model_for_prod_decision: bool = False
    promotion_registry_path: str | None = None
    #: Warn when experiment evidence is older than this many days (continuous validation / registry).
    calibration_max_age_days: int = Field(default=180, ge=1)
    #: Relative max coef shift vs accepted/promoted reference before review action.
    coefficient_shift_threshold: float = Field(default=0.30, gt=0.0, le=5.0)
    #: Fraction of evaluated replay comparisons classified as miss before review action.
    replay_miss_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    #: When true, severe drift downgrades model_release to invalidated (no auto-retrain).
    require_review_on_drift: bool = False

    model_config = {"extra": "forbid"}


class BudgetConfig(BaseModel):
    enabled: bool = False
    total_budget: float | None = None
    channel_min: dict[str, float] = Field(default_factory=dict)
    channel_max: dict[str, float] = Field(default_factory=dict)
    objective: Literal["revenue", "conversions", "roi", "profit"] = "revenue"
    risk_metric: Literal["p50", "p10"] = "p50"
    #: When ``True``, ``optimize_budget_via_simulation`` optimizes per-(geo, channel) spends.
    geo_budget_enabled: bool = False
    #: Per-geo per-channel lower bounds (geo id string → channel → min spend).
    geo_channel_min: dict[str, dict[str, float]] = Field(default_factory=dict)
    #: Per-geo per-channel upper bounds.
    geo_channel_max: dict[str, dict[str, float]] = Field(default_factory=dict)
    #: Minimum total media spend (sum of channels) per geo.
    geo_floor_total: dict[str, float] = Field(default_factory=dict)
    #: Maximum total media spend per geo.
    geo_cap_total: dict[str, float] = Field(default_factory=dict)
    #: Named group → list of geo ids; used with ``geo_group_max_total``.
    geo_groups: dict[str, list[str]] = Field(default_factory=dict)
    #: Named group → max sum of **all** channel spends for geos in that group (single budget pool per group).
    geo_group_max_total: dict[str, float] = Field(default_factory=dict)


class ArtifactConfig(BaseModel):
    #: ``mlflow`` is **experimental** in this package (remote tracking not contract-tested); prefer ``local`` for CI.
    backend: ArtifactBackend = ArtifactBackend.LOCAL
    run_dir: str = "./mmm_runs"
    mlflow_experiment: str | None = None
    write_data_fingerprint: bool = True


class MMMConfig(BaseModel):
    """Top-level run configuration (canonical YAML shape)."""

    run_id: str | None = None
    run_environment: RunEnvironment = RunEnvironment.RESEARCH
    override_unsafe: bool = False
    #: Required in prod when ``override_unsafe=True`` — JSON waiver artifact (see ``unsafe_override_waiver``).
    override_unsafe_waiver_path: str | None = None
    framework: Framework = Framework.RIDGE_BO
    model_form: ModelForm = ModelForm.SEMI_LOG
    pooling: PoolingMode = PoolingMode.PARTIAL
    transforms: TransformConfig = Field(default_factory=TransformConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    cv: CVConfig = Field(default_factory=CVConfig)
    ridge_bo: RidgeBOConfig = Field(default_factory=RidgeBOConfig)
    bayesian: BayesianConfig = Field(default_factory=BayesianConfig)
    objective: CompositeObjectiveConfig = Field(default_factory=CompositeObjectiveConfig)
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    governance: GovernanceWorkflowConfig = Field(default_factory=GovernanceWorkflowConfig)
    hierarchy: HierarchyConfig = Field(default_factory=HierarchyConfig)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    artifacts: ArtifactConfig = Field(default_factory=ArtifactConfig)
    extensions: ExtensionSuiteConfig = Field(default_factory=ExtensionSuiteConfig)
    random_seed: int = 42
    #: Child seeds; ``None`` inherits per ``mmm.contracts.seed_resolution`` (see ``seed_resolution`` artifact).
    bootstrap_seed: int | None = None
    extension_seed: int | None = None
    experiment_scheduler_seed: int | None = None
    simulation_seed: int | None = None
    strict_schema: bool = True
    #: When ``False`` (default): Ridge BO does not use coefficient-vs-experiment calibration in the
    #: objective; governance never approves optimization; CLI ``optimize-budget`` is blocked unless
    #: YAML sets this ``True`` **and** ``--allow-unsafe-decision-apis`` is passed.
    allow_unsafe_decision_apis: bool = False
    #: Prod **Ridge+BO** only: explicit acknowledgement that ``model_form`` / link matches a supported contract.
    #: Must be ``ridge_bo_semi_log_calendar_cv_v1`` when ``model_form=semi_log`` in prod.
    #: ``log_log`` is research-only and cannot be used with ``run_environment=prod``.
    prod_canonical_modeling_contract_id: str | None = Field(default=None)

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _validate_channels_and_prod_safety(self) -> MMMConfig:
        import os

        from mmm.config.validators import (
            apply_environment_objective_profile_inplace,
            validate_geo_budget_planning_consistency,
            validate_prod_cv_configuration,
            validate_prod_explicit_modeling_policy,
            validate_prod_model_form_contract,
            validate_transform_stack_for_framework,
        )

        apply_environment_objective_profile_inplace(self)
        if not self.data.channel_columns:
            raise ValueError("data.channel_columns must be non-empty")
        validate_transform_stack_for_framework(self)
        validate_geo_budget_planning_consistency(self)
        if self.calibration.hierarchical_regularization_enabled and not self.hierarchy.enabled:
            raise ValueError(
                "calibration.hierarchical_regularization_enabled is deprecated; set hierarchy.enabled=true "
                "and hierarchy.hierarchy_definition_path instead"
            )
        if self.hierarchy.enabled and self.framework != Framework.RIDGE_BO:
            raise ValueError("hierarchy.enabled is supported only for framework=ridge_bo")
        from mmm.governance.model_form_policy import assert_log_log_hierarchy_blocked

        assert_log_log_hierarchy_blocked(self)
        from mmm.hierarchy.bayesian_hierarchy import validate_bayesian_hierarchy_config

        validate_bayesian_hierarchy_config(self)
        if self.run_environment == RunEnvironment.PROD:
            validate_prod_cv_configuration(self)
            validate_prod_explicit_modeling_policy(self)
            validate_prod_model_form_contract(self)
            dvid = str(self.data.data_version_id or os.environ.get("MMM_DATA_VERSION_ID") or "").strip()
            if not dvid:
                raise ValueError(
                    "run_environment=prod requires data.data_version_id (dataset snapshot / durable version id). "
                    "Tests may set MMM_DATA_VERSION_ID only as a non-production escape hatch."
                )
            if self.allow_unsafe_decision_apis:
                raise ValueError("run_environment=prod requires allow_unsafe_decision_apis=False")
            if self.override_unsafe:
                wpath = str(self.override_unsafe_waiver_path or "").strip()
                if not wpath:
                    raise ValueError(
                        "run_environment=prod forbids override_unsafe=True without "
                        "override_unsafe_waiver_path (signed waiver JSON)"
                    )
                from mmm.governance.unsafe_override_waiver import load_unsafe_override_waiver

                load_unsafe_override_waiver(wpath)
            if not self.extensions.optimization_gates.enabled:
                raise ValueError("run_environment=prod requires extensions.optimization_gates.enabled=True")
            if self.framework == Framework.BAYESIAN:
                if int(self.bayesian.posterior_predictive_draws or 0) <= 0:
                    raise ValueError(
                        "run_environment=prod with framework=bayesian requires bayesian.posterior_predictive_draws>0"
                    )
                gv = self.extensions.governance
                if gv.bayesian_max_mean_abs_ppc_gap is None:
                    raise ValueError(
                        "run_environment=prod with framework=bayesian requires "
                        "extensions.governance.bayesian_max_mean_abs_ppc_gap (finite PPC mean-abs gap gate)"
                    )
            fe = self.extensions.features
            if fe.trend_spline_knots > 0 or fe.fourier_yearly_harmonics > 0 or (fe.holiday_country or "").strip():
                raise ValueError(
                    "run_environment=prod forbids implicit FeatureEngine-only controls "
                    "(trend_spline_knots / fourier_yearly_harmonics / holiday_country); "
                    "model them via explicit data.control_columns instead."
                )
        return self

    def model_dump_resolved(self) -> dict[str, Any]:
        return self.model_dump(mode="json")
