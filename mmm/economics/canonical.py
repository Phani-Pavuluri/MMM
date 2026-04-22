"""
Single canonical economics contract for decomposition, curves, simulator, optimizer, reporting.

All dollar-interpretable surfaces should carry the same ``economics_contract`` dict (version + scope
+ definitions) so operators cannot drift definitions across modules.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from mmm.config.schema import MMMConfig, ModelForm

ECONOMICS_CONTRACT_VERSION = "mmm_economics_contract_v1"

# Minimum keys expected on ``economics_output_metadata`` for business / decision JSON.
BUSINESS_ECONOMICS_METADATA_KEYS: frozenset[str] = frozenset(
    {
        "economics_contract_version",
        "economics_version",
        "target_kpi_column",
        "surface",
        "uncertainty_mode",
        "computation_mode",
        "baseline_type",
        "kpi_level_values_exact",
        "kpi_level_values_labeled_approximate",
        "planner_mode",
        "decision_safe",
    }
)

EconomicsOutputSurface = Literal[
    "full_model_simulation",
    "curve_diagnostic",
    "decomposition",
    "replay_calibration",
    "other",
]

# Replay ``lift_scale`` values that align with level-KPI deltas from predict_fn (exp of modeling scale).
REPLAY_LIFT_SCALES_KPI_LEVEL: frozenset[str] = frozenset(
    {
        "mean_kpi_level_delta",
        "sum_kpi_level_delta",
    }
)

PlannerMode = Literal["full_model", "curve_local"]
RidgeUncertaintyStance = Literal["exploratory", "bootstrap_ready"]
BayesianMaturity = Literal["experimental", "supported"]


def build_economics_contract(config: MMMConfig) -> dict[str, Any]:
    """Build the one JSON-serializable contract for this resolved config."""
    prod = config.extensions.product
    planner_mode: PlannerMode = prod.planner_mode  # type: ignore[assignment]
    ridge_uq: RidgeUncertaintyStance = prod.ridge_uncertainty_stance  # type: ignore[assignment]
    bayes_mat: BayesianMaturity = prod.bayesian_maturity  # type: ignore[assignment]

    mf = config.model_form.value if isinstance(config.model_form, ModelForm) else str(config.model_form)
    target = config.data.target_column
    defs = _definition_texts(model_form=mf, target_kpi=target, planner_mode=str(planner_mode))
    return {
        "contract_version": ECONOMICS_CONTRACT_VERSION,
        "planner_mode": planner_mode,
        "ridge_uncertainty_stance": ridge_uq,
        "bayesian_maturity": bayes_mat,
        "framework": config.framework.value,
        "model_form": mf,
        "target_kpi_column": target,
        "canonical_economic_quantity": (
            "Δμ = mean(μ̂(candidate spend plan)) − mean(μ̂(reference baseline)) on the modeling scale "
            f"for KPI column {target!r}; level-KPI summaries use the same μ path as training."
        ),
        "decision_source_of_truth": "full_panel_simulation",
        "default_baseline_policy": "bau_last_calendar_week_mean_across_geos",
        "ridge_production_uncertainty_rule": (
            "point estimates only for production decisions; no precise monetary credible intervals "
            "on the Ridge path in prod."
        ),
        "allowed_optimizer_objective_keys": [
            "response_on_modeling_scale",
            "kpi_level_implied_by_partial_curve",
            "delta_mu_full_model_simulation",
        ],
        **defs,
    }


def economics_output_metadata(
    config: MMMConfig,
    *,
    uncertainty_mode: str = "point",
    surface: EconomicsOutputSurface = "other",
    kpi_level_values_exact: bool | None = None,
    baseline_type: str | None = None,
    decision_safe: bool | None = None,
) -> dict[str, Any]:
    """
    Attach to any business-facing JSON so consumers know KPI, baseline policy, and exactness stance.

    ``kpi_level_values_exact`` defaults from framework + environment + surface when omitted.

    ``baseline_type`` should be the **actual** simulation baseline enum value (e.g. ``bau``); when
    omitted, the literal ``unspecified`` is stored so validators can detect incomplete payloads.

    ``decision_safe`` is explicit planner gating (may be ``None`` only for non-decision diagnostics).
    """
    from mmm.config.schema import Framework, RunEnvironment

    exact = kpi_level_values_exact
    if exact is None:
        if surface == "full_model_simulation" and config.framework == Framework.RIDGE_BO:
            exact = uncertainty_mode == "point"
        elif surface == "replay_calibration" and config.framework == Framework.RIDGE_BO:
            exact = True
        elif surface in ("curve_diagnostic", "decomposition"):
            exact = False
        else:
            exact = None
    approx = (
        config.run_environment == RunEnvironment.PROD
        and config.framework == Framework.RIDGE_BO
        and uncertainty_mode != "point"
    )
    if approx or exact is False:
        computation_mode = "approximate"
    elif exact is True:
        computation_mode = "exact"
    else:
        computation_mode = "unknown"

    baseline_resolved = baseline_type if baseline_type is not None else "unspecified"
    contract = build_economics_contract(config)
    return {
        "economics_contract_version": ECONOMICS_CONTRACT_VERSION,
        "economics_version": ECONOMICS_CONTRACT_VERSION,
        "target_kpi_column": config.data.target_column,
        "kpi_unit_semantics": "same_units_as_training_target_column",
        "default_baseline_for_planning": "bau_last_calendar_week_mean_across_geos",
        "canonical_delta_quantity": contract["canonical_economic_quantity"],
        "surface": surface,
        "uncertainty_mode": uncertainty_mode,
        "computation_mode": computation_mode,
        "baseline_type": baseline_resolved,
        "decision_safe": decision_safe,
        "kpi_level_values_exact": exact,
        "kpi_level_values_labeled_approximate": bool(approx),
        "planner_mode": config.extensions.product.planner_mode,
    }


def validate_business_economics_metadata(
    meta: dict[str, Any],
    *,
    require_specific_baseline: bool = True,
    require_decision_safe_bool: bool = False,
) -> None:
    """
    Fail closed if ``meta`` is missing required economics fields (post-``economics_output_metadata``).

    Use ``require_specific_baseline=False`` for fit-only / curve diagnostic payloads where baseline is N/A.
    """
    missing = sorted(BUSINESS_ECONOMICS_METADATA_KEYS - set(meta.keys()))
    if missing:
        raise ValueError(f"economics_output_metadata missing keys: {missing}")
    if meta.get("baseline_type") in (None, "unspecified") and require_specific_baseline:
        raise ValueError("economics_output_metadata.baseline_type must be set to a specific baseline for this surface")
    if require_decision_safe_bool and not isinstance(meta.get("decision_safe"), bool):
        raise ValueError("economics_output_metadata.decision_safe must be a bool for this surface")


def _definition_texts(*, model_form: str, target_kpi: str, planner_mode: str) -> dict[str, Any]:
    return {
        "incremental_revenue_definition": (
            f"Δ expected {target_kpi} vs a stated baseline path under the fitted μ structure; "
            "full_model uses full-panel μ with all channels, controls, and recursive adstock; "
            "curve_local uses partial steady-state r(S) for diagnostics only."
        ),
        "roi_definition": f"Δ {target_kpi} / Δ spend when both are consistently defined for the same counterfactual.",
        "mroas_definition": f"d(Δ {target_kpi})/d(spend) on the stated channel partial curve.",
        "mu_channel_partial_definition": (
            "Partial curve r(S)=β·x(S) after transforms; full μ uses the same design matrix as training."
        ),
        "semi_log_note": (
            "Y≈exp(μ); anchored level curves calibrate exp(μ_rest+r(S)) to a KPI anchor on the grid."
        ),
        "log_log_note": "Media enters log domain where configured; marginal w.r.t. spend follows curve gradient.",
        "disclosure": (
            f"planner_mode={planner_mode}: final budget scoring uses full-panel simulation when configured; "
            "serialized curves are diagnostic / explanatory only unless explicitly labeled otherwise."
        ),
    }


def attach_contract_to_curve_artifact(artifact: dict[str, Any], contract: dict[str, Any]) -> dict[str, Any]:
    """Attach / overwrite ``economics_contract``; keep legacy ``economics`` defs for readability."""
    from mmm.decomposition.economics import economics_definitions_block

    out = {**artifact, "economics_contract": contract}
    mf = str(out.get("model_form", contract.get("model_form", "semi_log")))
    rb = out.get("roi_bridge") if isinstance(out.get("roi_bridge"), dict) else {}
    tk = str(rb.get("target_kpi_column") or contract.get("target_kpi_column") or "revenue")
    out["economics"] = economics_definitions_block(model_form=mf, target_kpi_column=tk)
    out["curve_role"] = "diagnostic"
    out["decision_source_of_truth"] = "simulation"
    out["curve_truth_disclosure"] = (
        "Curve bundles are not the final planning truth; optimize-budget uses full-panel Δμ when "
        "ridge_fit_summary + training panel are available."
    )
    hz = out.get("horizon_weeks")
    out["curve_generation_assumptions"] = {
        "steady_state_horizon_weeks": hz,
        "baseline_anchor_context": "mean KPI level in panel used for optional level ROI bridge",
    }
    return out


def validate_optimizer_objective_key(objective_value_key: str, contract: dict[str, Any]) -> None:
    allowed = contract.get("allowed_optimizer_objective_keys") or []
    if objective_value_key not in allowed:
        raise ValueError(
            f"objective_value_key {objective_value_key!r} not allowed by economics_contract; "
            f"allowed={allowed}"
        )


def assert_planner_scope_supported(contract: dict[str, Any]) -> None:
    pm = contract.get("planner_mode")
    if pm is not None and pm not in ("full_model", "curve_local"):
        raise ValueError(f"Unsupported planner_mode in economics_contract: {pm!r}")


def economics_contract_for_curve_bundles(
    bundles: list[dict[str, Any]],
    *,
    strict: bool = False,
) -> dict[str, Any] | None:
    """
    Resolve ``economics_contract`` attached to serialized curve bundles.

    Non-strict: first bundle with a dict ``economics_contract`` containing ``contract_version``.

    Strict (production): **every** bundle must carry the same contract (JSON-equal with sorted keys);
    used by CLI and other fail-closed paths so operators cannot mix curve vintages.
    """
    if not bundles:
        return None

    def _norm(ec: dict[str, Any]) -> str:
        return json.dumps(ec, sort_keys=True, default=str)

    if strict:
        ref_s: str | None = None
        ref_dict: dict[str, Any] | None = None
        for b in bundles:
            ch = str(b.get("channel", "?"))
            ec = b.get("economics_contract")
            if not isinstance(ec, dict) or not ec.get("contract_version"):
                raise ValueError(
                    "economics_contract with contract_version is required on every curve bundle in prod; "
                    f"missing or invalid on channel {ch!r}"
                )
            s = _norm(ec)
            if ref_s is None:
                ref_s = s
                ref_dict = ec
            elif s != ref_s:
                raise ValueError(
                    "economics_contract differs across channel bundles; re-export curves from a single training run."
                )
        return ref_dict

    for b in bundles:
        ec = b.get("economics_contract")
        if isinstance(ec, dict) and ec.get("contract_version"):
            return ec
    return None
