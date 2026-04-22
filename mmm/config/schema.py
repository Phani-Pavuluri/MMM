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
    geo_blocked_seed: int = 42


class RidgeBOConfig(BaseModel):
    n_trials: int = 32
    timeout_sec: float | None = None
    sampler_seed: int = 42
    alpha_ridge_bounds: tuple[float, float] = (1e-4, 1e3)
    use_pruner: bool = True


class BayesianConfig(BaseModel):
    backend: BayesianBackend = BayesianBackend.PYMC
    draws: int = 1000
    tune: int = 1000
    chains: int = 4
    target_accept: float = 0.9
    nuts_seed: int = 42
    media_coef_sigma: float = 0.8
    control_coef_sigma: float = 2.0
    prior_predictive_draws: int = 0
    posterior_predictive_draws: int = 0


class ObjectiveWeights(BaseModel):
    predictive: float = 1.0
    calibration: float = 1.0
    stability: float = 0.5
    plausibility: float = 0.25
    complexity: float = 0.1


class CompositeObjectiveConfig(BaseModel):
    """Ridge+BO composite objective — components are normalized before weighting."""

    primary_metric: FitMetric = FitMetric.WMAPE
    weights: ObjectiveWeights = Field(default_factory=ObjectiveWeights)
    multi_objective: bool = False
    pareto_store_trials: bool = True
    normalization_profile: NormalizationProfile = NormalizationProfile.RESEARCH


class CalibrationConfig(BaseModel):
    enabled: bool = False
    experiments_path: str | None = None
    #: JSON list of :class:`mmm.calibration.contracts.CalibrationUnit`-compatible dicts with
    #: ``observed_spend_frame`` / ``counterfactual_spend_frame`` for replay (decision-safe path).
    replay_units_path: str | None = None
    use_replay_calibration: bool = False
    lift_column: str = "lift"
    lift_se_column: str | None = "lift_se"
    match_levels: list[str] = Field(
        default_factory=lambda: ["geo", "time_window", "channel", "device", "product"]
    )
    experiment_target_kpi: str | None = None
    use_quality_weights: bool = True
    #: Optional JSON registry (see ``mmm.experiments.durable_registry``) listing approved experiment_ids.
    experiment_registry_path: str | None = None
    #: When ``True`` in PROD, every replay unit must carry a non-empty ``experiment_id`` that is ``approved`` in the registry.
    require_approved_experiment_registry: bool = False


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
    backend: ArtifactBackend = ArtifactBackend.LOCAL
    run_dir: str = "./mmm_runs"
    mlflow_experiment: str | None = None
    write_data_fingerprint: bool = True


class MMMConfig(BaseModel):
    """Top-level run configuration (canonical YAML shape)."""

    run_id: str | None = None
    run_environment: RunEnvironment = RunEnvironment.RESEARCH
    override_unsafe: bool = False
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
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    artifacts: ArtifactConfig = Field(default_factory=ArtifactConfig)
    extensions: ExtensionSuiteConfig = Field(default_factory=ExtensionSuiteConfig)
    random_seed: int = 42
    strict_schema: bool = True
    #: When ``False`` (default): Ridge BO does not use coefficient-vs-experiment calibration in the
    #: objective; governance never approves optimization; CLI ``optimize-budget`` is blocked unless
    #: YAML sets this ``True`` **and** ``--allow-unsafe-decision-apis`` is passed.
    allow_unsafe_decision_apis: bool = False

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _validate_channels_and_prod_safety(self) -> MMMConfig:
        if not self.data.channel_columns:
            raise ValueError("data.channel_columns must be non-empty")
        from mmm.config.validators import validate_implemented_transforms_for_framework

        validate_implemented_transforms_for_framework(self)
        if self.run_environment == RunEnvironment.PROD:
            if self.allow_unsafe_decision_apis:
                raise ValueError("run_environment=prod requires allow_unsafe_decision_apis=False")
            if not self.extensions.optimization_gates.enabled:
                raise ValueError("run_environment=prod requires extensions.optimization_gates.enabled=True")
            if self.framework == Framework.BAYESIAN and int(self.bayesian.posterior_predictive_draws or 0) <= 0:
                raise ValueError(
                    "run_environment=prod with framework=bayesian requires bayesian.posterior_predictive_draws>0"
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
