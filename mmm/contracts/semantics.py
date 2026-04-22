"""
Canonical semantics for the MMM package.

All modules that speak about targets, calibration, contributions, or optimization
should align with these definitions (import types here; avoid redefining ad hoc strings).

This module is documentation-as-code: it does not enforce behavior at runtime alone;
callers should validate configs against these specs where appropriate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

# ---------------------------------------------------------------------------
# Modeling target
# ---------------------------------------------------------------------------


class ModelFormKind(str, Enum):
    """Outcome ↔ spend link in the mean structure (before stochastic layer)."""

    SEMI_LOG = "semi_log"
    """log(Y) ~ f(transformed_spend) + controls. Default weekly geo path."""

    LOG_LOG = "log_log"
    """log(Y) ~ log(transformed_spend) + controls. Stronger compression of media."""


class OutcomeScale(str, Enum):
    """Scale on which the likelihood / loss is defined."""

    LEVEL = "level"
    LOG = "log"


@dataclass(frozen=True)
class ModelingTargetSpec:
    """
    Canonical modeling target: what the inner estimator fits and predicts.

    - `column`: column name in the panel (e.g. revenue).
    - `form`: SEMI_LOG vs LOG_LOG (mean link).
    - `outcome_scale`: where Gaussian / robust loss applies (typically LOG for SEMI_LOG).
    - `floor`: small positive floor for log if needed (document in artifacts).
    """

    column: str
    form: ModelFormKind
    outcome_scale: OutcomeScale = OutcomeScale.LOG
    floor: float = 1e-12

    def summary(self) -> str:
        return (
            f"target={self.column}, form={self.form.value}, "
            f"likelihood_scale={self.outcome_scale.value}, log_floor={self.floor}"
        )


# ---------------------------------------------------------------------------
# Calibration estimand
# ---------------------------------------------------------------------------


class CalibrationKPI(str, Enum):
    """What the experiment measured (must align with modeling target or explicit mapping)."""

    SAME_AS_MODEL_TARGET = "same_as_model_target"
    REVENUE = "revenue"
    CONVERSIONS = "conversions"
    CUSTOM = "custom"


class LiftDefinition(str, Enum):
    """How experimental lift is defined (must match replay construction)."""

    RELATIVE_INCREMENTAL = "relative_incremental"
    """(Y_post - Y_counterfactual) / baseline or similar; define in replay."""

    ABSOLUTE_INCREMENTAL = "absolute_incremental"
    """Level difference in KPI."""

    LOG_RELATIVE = "log_relative"
    """Difference on log scale; only comparable to model if model is on same scale."""


class EffectHorizon(str, Enum):
    """What duration the experiment identifies."""

    SHORT_RUN = "short_run"
    LONG_RUN = "long_run"


@dataclass(frozen=True)
class CalibrationEstimandSpec:
    """
    Canonical calibration estimand: what “lift” means for replay and loss.

    - `kpi`: measured outcome type; `SAME_AS_MODEL_TARGET` ties to `ModelingTargetSpec.column`.
    - `lift_definition`: how observed lift in data is defined.
    - `horizon`: short- vs long-run effect (drives windows and replay length).
    - `aggregation`: e.g. geo_week, national_week (must match experiment aggregation).
    - `requires_counterfactual_path`: if True, replay must use observed + counterfactual spend paths.
    """

    kpi: CalibrationKPI
    lift_definition: LiftDefinition
    horizon: EffectHorizon
    aggregation: str = "geo_week"
    requires_counterfactual_path: bool = True
    notes: tuple[str, ...] = ()

    def summary(self) -> str:
        return (
            f"kpi={self.kpi.value}, lift={self.lift_definition.value}, "
            f"horizon={self.horizon.value}, agg={self.aggregation}, "
            f"counterfactual_path={self.requires_counterfactual_path}"
        )


# ---------------------------------------------------------------------------
# Contribution interpretation
# ---------------------------------------------------------------------------


class ContributionScale(str, Enum):
    """Where decomposition components live."""

    MODELING_LINEAR_PREDICTOR = "modeling_linear_predictor"
    """Additive pieces in the same space as X @ beta before inverse link (e.g. log scale for semi-log)."""

    LEVEL_APPROX = "level_approx"
    """Approximate mapping to level; not exact additive dollars unless documented method."""

    SHARE_DECOMPOSITION = "share_decomposition"
    """Non-additive shares that sum to 1; not incremental dollars."""


@dataclass(frozen=True)
class ContributionInterpretation:
    """
    Canonical contribution semantics: what “channel contribution” means in outputs.

    - `scale`: where `contrib__*` columns live.
    - `is_exact_additive_in_target`: if True, row-wise sum of components equals target definition exactly.
    - `safe_for_budgeting`: if False, contributions must not drive spend allocation without curves/PPC.
    - `caveats`: human-readable warnings for reports.
    """

    scale: ContributionScale
    is_exact_additive_in_target: bool
    safe_for_budgeting: bool
    caveats: tuple[str, ...] = field(default_factory=tuple)

    @staticmethod
    def ridge_log_surrogate_default() -> ContributionInterpretation:
        return ContributionInterpretation(
            scale=ContributionScale.MODELING_LINEAR_PREDICTOR,
            is_exact_additive_in_target=False,
            safe_for_budgeting=False,
            caveats=(
                "Components are on the modeling linear predictor (e.g. log scale); "
                "they are not literal incremental dollars without an explicit inverse map.",
            ),
        )


# ---------------------------------------------------------------------------
# Optimization-safe outputs
# ---------------------------------------------------------------------------


class CurveNumericalStatus(str, Enum):
    """Stress-test status for response surfaces."""

    UNKNOWN = "unknown"
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"


@dataclass(frozen=True)
class OptimizationSafetySpec:
    """
    Canonical definition of when optimizer inputs are safe.

    - `curves_monotone`: required monotone media response on spend grid (per policy).
    - `curve_stress_passed`: finite-difference / bootstrap stability checks.
    - `governance_approved_for_optimization`: scorecard flag from governance layer.
    - `identifiability_status`: ok | skipped | failed — skipped must not imply low risk in prod.
    """

    curves_monotone: bool
    curve_stress_passed: bool
    governance_approved_for_optimization: bool
    identifiability_status: Literal["ok", "skipped", "failed"]

    def is_safe_to_optimize(self, *, strict: bool = True) -> bool:
        if not self.curves_monotone or not self.curve_stress_passed:
            return False
        if not self.governance_approved_for_optimization:
            return False
        if self.identifiability_status == "failed":
            return False
        return not (strict and self.identifiability_status == "skipped")

    def block_reasons(self, *, strict: bool = True) -> list[str]:
        reasons: list[str] = []
        if not self.curves_monotone:
            reasons.append("curves_not_monotone")
        if not self.curve_stress_passed:
            reasons.append("curve_stress_failed")
        if not self.governance_approved_for_optimization:
            reasons.append("governance_not_approved_for_optimization")
        if self.identifiability_status == "failed":
            reasons.append("identifiability_failed")
        if strict and self.identifiability_status == "skipped":
            reasons.append("identifiability_skipped_in_strict_mode")
        return reasons
