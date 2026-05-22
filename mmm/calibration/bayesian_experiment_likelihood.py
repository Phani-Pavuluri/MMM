"""Bayesian experiment likelihood (research-only) using experiment evidence layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from mmm.calibration.evidence_replay import (
    _panel_context,
    _resolve_channel,
    build_calibration_unit_from_evidence,
)
from mmm.calibration.replay_estimand import ReplayEstimandSpec
from mmm.config.schema import Framework, MMMConfig, ModelForm, PoolingMode
from mmm.data.schema import PanelSchema
from mmm.evaluation.experiment_evidence_extension import load_evidence_from_path
from mmm.experiments.compatibility import (
    CompatibilityStatus,
    ExperimentCompatibilityResolver,
)
from mmm.experiments.evidence import ApprovalStatus
from mmm.experiments.evidence_quality import (
    CONSERVATIVE_DEFAULT_SE,
    EvidenceQualityContext,
    QualityTier,
    score_evidence_quality,
)
from mmm.experiments.shock_plan import AllocationQuality, CounterfactualShockPlanner
from mmm.hierarchy.pooling import partial_pooling_indices
from mmm.transforms.stack import build_channel_features_from_params
from mmm.utils.math import safe_log

LOG_MODELING_LIFT_SCALES = frozenset({"log_mean_kpi_delta", "mean_log_kpi_delta", "log_relative"})
LEVEL_LIFT_SCALES = frozenset({"mean_kpi_level_delta", "absolute_incremental"})
_BRIDGE_ROLE = "computational_bridge_only"

_TIER_RANK = {
    QualityTier.HIGH.value: 3,
    QualityTier.MEDIUM.value: 2,
    QualityTier.LOW.value: 1,
    QualityTier.REJECTED.value: 0,
}


class LiftScaleMismatchError(ValueError):
    """Experiment lift scale incompatible with Bayesian model outcome scale."""


@dataclass
class BayesianExperimentLikelihoodTerm:
    experiment_id: str
    channel: str
    observed_lift: float
    adjusted_se: float
    evidence_weight: float
    reported_se: float
    lift_scale: str
    compatibility_status: str
    quality_tier: str
    aggregation: str
    supports_subgeo_claims: bool
    allocation_role: str
    allocation_required: bool
    X_obs: np.ndarray
    X_cf: np.ndarray
    geo_idx: np.ndarray


@dataclass
class BayesianExperimentPrepareResult:
    used: list[BayesianExperimentLikelihoodTerm] = field(default_factory=list)
    rejected: list[dict[str, Any]] = field(default_factory=list)
    n_loaded: int = 0
    warnings: list[str] = field(default_factory=list)
    compatibility_status_counts: dict[str, int] = field(default_factory=dict)


def uses_bayesian_experiment_likelihood(config: MMMConfig) -> bool:
    return bool(config.bayesian.use_experiment_likelihood)


def validate_bayesian_experiment_likelihood_config(config: MMMConfig) -> None:
    b = config.bayesian
    if not b.use_experiment_likelihood:
        return
    if not b.experiment_registry_path:
        raise ValueError(
            "bayesian.use_experiment_likelihood requires bayesian.experiment_registry_path"
        )
    if not b.exp_likelihood_research_only:
        raise ValueError(
            "bayesian.exp_likelihood_research_only must remain true (experiment likelihood is research-only)"
        )
    if config.framework != Framework.BAYESIAN:
        raise ValueError("bayesian.use_experiment_likelihood requires framework=bayesian")


def validate_experiment_lift_scale(
    lift_scale: str,
    model_form: ModelForm,
    *,
    allow_level_on_log_model: bool = False,
) -> None:
    """Reject level-reported experiment lift on log-outcome models unless explicitly allowed."""
    ls = str(lift_scale or "").strip()
    if not ls:
        raise LiftScaleMismatchError("experiment lift_scale is required for Bayesian experiment likelihood")
    if model_form == ModelForm.SEMI_LOG and ls in LEVEL_LIFT_SCALES and not allow_level_on_log_model:
        raise LiftScaleMismatchError(
            f"lift_scale {ls!r} is KPI-level while model_form=semi_log fits log({{target}}); "
            "use log_mean_kpi_delta or enable explicit level-lift override (research only)"
        )
    if model_form == ModelForm.LOG_LOG and ls in LEVEL_LIFT_SCALES and not allow_level_on_log_model:
        raise LiftScaleMismatchError(
            f"lift_scale {ls!r} is KPI-level while model_form=log_log; use a log-scale lift definition"
        )
    if model_form in {ModelForm.SEMI_LOG, ModelForm.LOG_LOG} and ls not in LOG_MODELING_LIFT_SCALES | LEVEL_LIFT_SCALES:
        raise LiftScaleMismatchError(
            f"unsupported lift_scale {ls!r}; expected one of {sorted(LOG_MODELING_LIFT_SCALES | LEVEL_LIFT_SCALES)}"
        )


def compute_adjusted_standard_error(
    *,
    reported_se: float | None,
    evidence_weight: float,
    compatibility_status: str,
    allocation_required: bool,
    allocation_quality: str | None,
    expiration_status: str,
    allow_missing_se: bool,
) -> tuple[float, list[str]]:
    """Inflate SE when evidence is weaker (lower weight → higher sigma)."""
    reasons: list[str] = []
    if reported_se is None or reported_se <= 0:
        if not allow_missing_se:
            raise ValueError("missing_or_invalid_standard_error")
        reasons.append("conservative_default_se_used")
        base = CONSERVATIVE_DEFAULT_SE
    else:
        base = float(reported_se)

    w = max(float(evidence_weight), 1e-6)
    adj = base / np.sqrt(w)
    if compatibility_status == CompatibilityStatus.AGGREGATE_ONLY.value:
        adj *= 1.15
        reasons.append("aggregate_only_inflation")
    if allocation_required:
        adj *= 1.25
        reasons.append("allocation_uncertainty_inflation")
    if allocation_quality == AllocationQuality.LOW.value:
        adj *= 1.1
        reasons.append("low_allocation_quality_inflation")
    if str(expiration_status).startswith("stale"):
        adj *= 1.2
        reasons.append("stale_evidence_inflation")
    return float(max(adj, 1e-6)), reasons


def _min_tier_rank(min_tier: str) -> int:
    return _TIER_RANK.get(str(min_tier).lower(), 2)


def _build_features_for_frames(
    df: Any,
    schema: PanelSchema,
    config: MMMConfig,
    *,
    decay: float,
    hill_half: float,
    hill_slope: float,
) -> tuple[np.ndarray, np.ndarray]:
    import pandas as pd

    panel = pd.DataFrame(df)
    X = build_channel_features_from_params(
        panel,
        schema,
        config.transforms,
        decay=decay,
        hill_half=hill_half,
        hill_slope=hill_slope,
        modeling_config=config,
    )
    if config.model_form == ModelForm.LOG_LOG:
        X = safe_log(np.maximum(X, 1e-9))
    if schema.control_columns:
        X = np.column_stack([X] + [panel[c].to_numpy(dtype=float) for c in schema.control_columns])
    geo_idx = partial_pooling_indices(panel, schema)
    return np.asarray(X, dtype=float), np.asarray(geo_idx, dtype=int)


def prepare_bayesian_experiment_likelihood_terms(
    config: MMMConfig,
    panel: Any,
    schema: PanelSchema,
) -> BayesianExperimentPrepareResult:
    """Prepare experiment units for PyMC likelihood (research-only)."""
    import pandas as pd

    validate_bayesian_experiment_likelihood_config(config)
    path = config.bayesian.experiment_registry_path
    assert path is not None
    panel_df = pd.DataFrame(panel)
    evidence_list = load_evidence_from_path(path)
    ctx = _panel_context(panel_df, schema, config)
    resolver = ExperimentCompatibilityResolver()
    planner = CounterfactualShockPlanner()
    target_kpi = config.calibration.experiment_target_kpi or schema.target_column
    allow_missing_se = bool(config.bayesian.allow_conservative_missing_se)
    min_rank = _min_tier_rank(config.bayesian.min_experiment_quality_tier)

    decay = float(config.transforms.adstock_params.get("decay", 0.5))
    hill_half = float(config.transforms.saturation_params.get("half_max", 1.0))
    hill_slope = float(config.transforms.saturation_params.get("slope", 2.0))

    result = BayesianExperimentPrepareResult(n_loaded=len(evidence_list))
    status_counts: dict[str, int] = {}

    for ev in evidence_list:
        if ev.approval_status in {ApprovalStatus.REJECTED, ApprovalStatus.EXPIRED}:
            result.rejected.append(
                {"experiment_id": ev.experiment_id, "reason": f"approval_{ev.approval_status.value}"}
            )
            continue

        compat = resolver.resolve(ev, ctx, target_kpi=target_kpi)
        status_counts[compat.compatibility_status.value] = (
            status_counts.get(compat.compatibility_status.value, 0) + 1
        )
        shock = planner.plan(ev, compat)

        if compat.compatibility_status == CompatibilityStatus.REJECTED:
            result.rejected.append(
                {"experiment_id": ev.experiment_id, "reason": compat.rejection_reason or "rejected"}
            )
            continue
        if compat.compatibility_status == CompatibilityStatus.DIAGNOSTIC_ONLY:
            result.rejected.append(
                {"experiment_id": ev.experiment_id, "reason": "diagnostic_only_compatibility"}
            )
            continue
        if compat.compatibility_status == CompatibilityStatus.AGGREGATE_ONLY and not (
            config.bayesian.allow_aggregate_only_evidence
        ):
            result.rejected.append(
                {"experiment_id": ev.experiment_id, "reason": "aggregate_only_not_allowed"}
            )
            continue
        if compat.allocation_required and not config.bayesian.allow_allocated_shocks:
            result.rejected.append(
                {"experiment_id": ev.experiment_id, "reason": "allocated_shock_not_allowed"}
            )
            continue
        if shock.allocation_quality == AllocationQuality.REJECTED.value:
            result.rejected.append({"experiment_id": ev.experiment_id, "reason": "shock_plan_rejected"})
            continue

        ch = _resolve_channel(ev, ctx)
        if ch is None:
            result.rejected.append({"experiment_id": ev.experiment_id, "reason": "channel_not_in_model"})
            continue

        qctx = EvidenceQualityContext(
            target_kpi=target_kpi,
            channel_match=True,
            compatibility=compat,
            allow_missing_se=allow_missing_se,
        )
        qscore = score_evidence_quality(ev, qctx)
        if _TIER_RANK.get(qscore.quality_tier.value, 0) < min_rank:
            result.rejected.append(
                {
                    "experiment_id": ev.experiment_id,
                    "reason": "quality_tier_below_minimum",
                    "quality_tier": qscore.quality_tier.value,
                }
            )
            continue
        if qscore.quality_tier == QualityTier.REJECTED or qscore.evidence_weight <= 0:
            result.rejected.append(
                {"experiment_id": ev.experiment_id, "reason": "quality_rejected", "reasons": qscore.reasons}
            )
            continue

        unit = build_calibration_unit_from_evidence(ev, panel_df, schema, channel=ch, compat=compat)
        if unit is None or unit.observed_spend_frame is None or unit.counterfactual_spend_frame is None:
            result.rejected.append(
                {"experiment_id": ev.experiment_id, "reason": "missing_replay_frames"}
            )
            continue

        spec = ReplayEstimandSpec.from_dict(unit.replay_estimand or {})
        lift_scale = str(spec.lift_scale or ev.metadata.get("lift_scale", "log_mean_kpi_delta"))
        try:
            validate_experiment_lift_scale(
                lift_scale,
                config.model_form,
                allow_level_on_log_model=config.bayesian.allow_level_lift_mismatch_research,
            )
        except LiftScaleMismatchError as e:
            result.rejected.append(
                {"experiment_id": ev.experiment_id, "reason": "lift_scale_mismatch", "detail": str(e)}
            )
            continue

        try:
            adj_se, se_reasons = compute_adjusted_standard_error(
                reported_se=ev.standard_error,
                evidence_weight=qscore.evidence_weight,
                compatibility_status=compat.compatibility_status.value,
                allocation_required=compat.allocation_required,
                allocation_quality=shock.allocation_quality,
                expiration_status=qscore.expiration_status,
                allow_missing_se=allow_missing_se,
            )
        except ValueError:
            result.rejected.append(
                {"experiment_id": ev.experiment_id, "reason": "missing_or_invalid_standard_error"}
            )
            continue

        X_obs, g_obs = _build_features_for_frames(
            unit.observed_spend_frame,
            schema,
            config,
            decay=decay,
            hill_half=hill_half,
            hill_slope=hill_slope,
        )
        X_cf, g_cf = _build_features_for_frames(
            unit.counterfactual_spend_frame,
            schema,
            config,
            decay=decay,
            hill_half=hill_half,
            hill_slope=hill_slope,
        )
        if len(X_obs) != len(X_cf) or not np.array_equal(g_obs, g_cf):
            result.rejected.append(
                {"experiment_id": ev.experiment_id, "reason": "obs_cf_row_misalignment"}
            )
            continue

        if compat.compatibility_status == CompatibilityStatus.AGGREGATE_ONLY:
            result.warnings.append(f"aggregate_only_{ev.experiment_id}: no_subgeo_claims")
        if compat.allocation_required:
            result.warnings.append(f"allocated_shock_{ev.experiment_id}: computational_bridge_only")

        result.used.append(
            BayesianExperimentLikelihoodTerm(
                experiment_id=ev.experiment_id,
                channel=ch,
                observed_lift=float(ev.lift_estimate),
                adjusted_se=adj_se,
                evidence_weight=float(qscore.evidence_weight),
                reported_se=float(ev.standard_error or adj_se),
                lift_scale=lift_scale,
                compatibility_status=compat.compatibility_status.value,
                quality_tier=qscore.quality_tier.value,
                aggregation=spec.aggregation,
                supports_subgeo_claims=compat.supports_subgeo_claims,
                allocation_role=shock.allocation_role,
                allocation_required=compat.allocation_required,
                X_obs=X_obs,
                X_cf=X_cf,
                geo_idx=g_obs,
            )
        )
        _ = se_reasons

    result.compatibility_status_counts = status_counts
    return result


def _symbolic_implied_lift(
    pm: Any,
    *,
    alpha_geo: Any,
    beta: Any,
    pooling: PoolingMode,
    term: BayesianExperimentLikelihoodTerm,
) -> Any:
    """MMM implied lift on the Bayesian modeling (log) scale."""
    Xo = pm.Data(f"x_obs_{term.experiment_id}", term.X_obs)
    Xc = pm.Data(f"x_cf_{term.experiment_id}", term.X_cf)
    gi = pm.Data(f"geo_{term.experiment_id}", term.geo_idx.astype(int))

    if pooling == PoolingMode.FULL:
        mu_obs = alpha_geo[gi] + pm.math.dot(Xo, beta)
        mu_cf = alpha_geo[gi] + pm.math.dot(Xc, beta)
    else:
        mu_obs = alpha_geo[gi] + (Xo * beta[gi]).sum(axis=-1)
        mu_cf = alpha_geo[gi] + (Xc * beta[gi]).sum(axis=-1)

    delta = mu_obs - mu_cf
    if term.aggregation == "geo_mean_then_global_mean":
        geos = np.unique(term.geo_idx)
        per_geo = []
        for g in geos:
            mask = term.geo_idx == g
            per_geo.append(pm.math.mean(delta[mask]))
        return pm.math.mean(pm.math.stack(per_geo))
    return pm.math.mean(delta)


def register_pymc_experiment_likelihoods(
    pm: Any,
    *,
    pooling: PoolingMode,
    alpha_geo: Any,
    beta: Any,
    terms: list[BayesianExperimentLikelihoodTerm],
    likelihood_weight: float = 1.0,
) -> list[str]:
    """Add Normal experiment-lift likelihoods inside an open PyMC model context."""
    if not terms:
        return []
    var_names: list[str] = []
    lw = max(float(likelihood_weight), 1e-6)
    for term in terms:
        implied = _symbolic_implied_lift(
            pm,
            alpha_geo=alpha_geo,
            beta=beta,
            pooling=pooling,
            term=term,
        )
        implied_name = f"mmm_implied_lift_{term.experiment_id}"
        pm.Deterministic(implied_name, implied)
        sigma_eff = term.adjusted_se / np.sqrt(lw * term.evidence_weight)
        obs_name = f"exp_lift_obs_{term.experiment_id}"
        pm.Normal(
            obs_name,
            mu=implied,
            sigma=float(sigma_eff),
            observed=term.observed_lift,
        )
        var_names.extend([implied_name, obs_name])
    return var_names


def build_bayesian_experiment_likelihood_report(
    config: MMMConfig,
    idata: Any,
    prepared: BayesianExperimentPrepareResult,
    *,
    pymc_var_names: list[str] | None = None,
) -> dict[str, Any]:
    """Post-fit diagnostic report (research-only; prod decisioning remains blocked)."""
    b = config.bayesian
    enabled = uses_bayesian_experiment_likelihood(config)
    governance_warnings = [
        "Bayesian experiment likelihood is research-only.",
        "Does not enable Bayesian prod budget or planning decisioning.",
        "Experiment evidence constrains the posterior but does not prove causal validity beyond experiment scope.",
        "Aggregate experiments do not support subgeo claims.",
        "Allocated spend shocks are computational bridges only, not experimental subgeo truth.",
    ]
    governance_warnings.extend(prepared.warnings)

    report: dict[str, Any] = {
        "enabled": enabled,
        "research_only": bool(b.exp_likelihood_research_only),
        "prod_decisioning_allowed": False,
        "n_evidence_units_loaded": prepared.n_loaded,
        "n_evidence_units_used": len(prepared.used),
        "n_evidence_units_rejected": len(prepared.rejected),
        "rejected_reasons": list(prepared.rejected),
        "compatibility_status_counts": dict(prepared.compatibility_status_counts),
        "likelihood_terms": [],
        "adjusted_standard_errors": [],
        "evidence_weights": [],
        "warnings": list(prepared.warnings),
        "governance_warnings": governance_warnings,
        "posterior_experiment_fit": {},
        "posterior_predictive_experiment_checks": {},
    }

    if not enabled or not prepared.used:
        return report

    likelihood_terms: list[dict[str, Any]] = []
    posterior_fit: dict[str, Any] = {}
    for term in prepared.used:
        likelihood_terms.append(
            {
                "experiment_id": term.experiment_id,
                "channel": term.channel,
                "observed_lift": term.observed_lift,
                "adjusted_se": term.adjusted_se,
                "reported_se": term.reported_se,
                "evidence_weight": term.evidence_weight,
                "lift_scale": term.lift_scale,
                "compatibility_status": term.compatibility_status,
                "quality_tier": term.quality_tier,
                "supports_subgeo_claims": term.supports_subgeo_claims,
                "allocation_role": term.allocation_role,
            }
        )
        report["adjusted_standard_errors"].append(term.adjusted_se)
        report["evidence_weights"].append(term.evidence_weight)

        imp_name = f"mmm_implied_lift_{term.experiment_id}"
        if idata is not None and hasattr(idata, "posterior") and imp_name in idata.posterior:
            imp_s = np.asarray(idata.posterior[imp_name].stack(sample=("chain", "draw"))).ravel()
            posterior_fit[term.experiment_id] = {
                "implied_lift_mean": float(np.mean(imp_s)),
                "implied_lift_p10": float(np.quantile(imp_s, 0.1)),
                "implied_lift_p90": float(np.quantile(imp_s, 0.9)),
                "observed_lift": term.observed_lift,
                "abs_error_mean": float(np.mean(np.abs(imp_s - term.observed_lift))),
            }

    report["likelihood_terms"] = likelihood_terms
    report["posterior_experiment_fit"] = posterior_fit
    if posterior_fit:
        report["posterior_predictive_experiment_checks"] = {
            "note": "Posterior implied lift vs observed experiment lift on modeling scale.",
            "n_terms_with_posterior": len(posterior_fit),
        }
    if pymc_var_names:
        report["pymc_likelihood_var_names"] = list(pymc_var_names)
    return report
