"""
Explicit allowlist for ``simulation_at_recommendation`` payloads during optimize enrichment.

Replaces implicit DFS heuristics: only documented keys survive normalization; forbidden keys
(synthetic extension payloads, posterior exploration blocks, etc.) are rejected outright.
"""

from __future__ import annotations

from typing import Any

from mmm.config.schema import MMMConfig, RunEnvironment
from mmm.governance.policy import PolicyError

OPTIMIZE_SIMULATION_AT_RECOMMENDATION_ALLOWLIST_POLICY = "optimize_sim_at_rec_allowlist_v1"

# Keys emitted by ``SimulationResult.to_json`` plus documented ``extra={...}`` from
# ``mmm.planning.decision_simulate.simulate`` (keep in sync when that contract grows).
SIMULATION_AT_RECOMMENDATION_ALLOWED_TOP_LEVEL_KEYS: frozenset[str] = frozenset(
    {
        "baseline_mu",
        "plan_mu",
        "delta_mu",
        "delta_spend",
        "roi",
        "mroas",
        "baseline_type",
        "baseline_definition",
        "uncertainty_mode",
        "decision_safe",
        "economics_version",
        "planner_mode",
        "canonical_quantity",
        "mean_kpi_level_baseline",
        "mean_kpi_level_plan",
        "delta_kpi_level",
        "disclosure",
        "p10",
        "p50",
        "p90",
        "horizon_weeks",
        "candidate_plan_type",
        "counterfactual_construction_method",
        "spend_path_assumption",
        "aggregation_semantics",
        "kpi_column",
        "baseline_plan_source",
        "baseline_suitable_for_decisioning",
        "controls_path_semantics",
        "scenario_overlay_summary",
        "seasonality_path",
        "promo_path",
        "geo_time_aggregation",
        "posterior_planning_mode",
        "posterior_planning_gate",
        "plan_spend_for_economics",
        "spend_economics_mode",
        "baseline_has_per_geo_spend",
        "recommended_spend_plan_by_geo",
        "n_piecewise_segments",
        "economics_output_metadata",
        "economics_metadata",
        "decision_geography_contract",
    }
)

# Keys added by ``enrich_decision_simulation_json`` (must remain allowlisted post-enrich pass).
SIMULATION_AT_RECOMMENDATION_ENRICHMENT_KEYS: frozenset[str] = frozenset(
    {
        "artifact_tier",
        "decision_safe",
        "approximate",
        "not_for_budgeting",
        "economics_contract_version",
        "kpi_column",
        "kpi_unit_semantics",
        "baseline_type",
        "unsupported_questions",
    }
)

SIMULATION_AT_RECOMMENDATION_ALLOWED_TOP_LEVEL_KEYS_FULL: frozenset[str] = (
    SIMULATION_AT_RECOMMENDATION_ALLOWED_TOP_LEVEL_KEYS | SIMULATION_AT_RECOMMENDATION_ENRICHMENT_KEYS
)

# Never allow nested extension / approximate exploration blobs to masquerade as simulation JSON.
SIMULATION_AT_RECOMMENDATION_FORBIDDEN_TOP_LEVEL_KEYS: frozenset[str] = frozenset(
    {
        "posterior_exploration_quantity",
        "extension_report",
        "curve_bundles",
        "curve_bundle",
        "identifiability_waiver",
        "governance",
        "ridge_fit_summary",
    }
)


def apply_simulation_at_recommendation_allowlist(
    sim_js: dict[str, Any],
    *,
    cfg: MMMConfig,
    context: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Return ``(filtered_sim_js, audit)``.

    - Forbidden keys always raise ``PolicyError`` (all environments).
    - Unknown keys are stripped with audit metadata (non-prod: permissive strip).
    - PROD: unknown keys raise ``PolicyError`` to avoid silent loss of semantic payloads.
    """
    audit: dict[str, Any] = {
        "policy_name": OPTIMIZE_SIMULATION_AT_RECOMMENDATION_ALLOWLIST_POLICY,
        "context": context,
        "removed_keys": [],
        "forbidden_keys_seen": sorted(k for k in sim_js if k in SIMULATION_AT_RECOMMENDATION_FORBIDDEN_TOP_LEVEL_KEYS),
    }
    bad = [k for k in sim_js if k in SIMULATION_AT_RECOMMENDATION_FORBIDDEN_TOP_LEVEL_KEYS]
    if bad:
        raise PolicyError(
            f"{context}: simulation_at_recommendation contains forbidden top-level keys {bad!r} "
            "(nested extension/posterior payloads must not be embedded here; see optimize_enrichment policy)."
        )
    allowed = SIMULATION_AT_RECOMMENDATION_ALLOWED_TOP_LEVEL_KEYS
    removed = sorted(k for k in sim_js if k not in allowed)
    if removed:
        audit["removed_keys"] = removed
        if cfg.run_environment == RunEnvironment.PROD:
            raise PolicyError(
                f"{context}: simulation_at_recommendation has unknown keys {removed!r} under "
                f"{OPTIMIZE_SIMULATION_AT_RECOMMENDATION_ALLOWLIST_POLICY}; refusing silent strip in prod."
            )
    filtered = {k: sim_js[k] for k in sim_js if k in allowed}
    return filtered, audit


def apply_simulation_at_recommendation_allowlist_post_enrich(sim_js: dict[str, Any], *, context: str) -> None:
    """Validate keys after business-surface enrichment (fail closed in prod on drift)."""
    bad = sorted(k for k in sim_js if k not in SIMULATION_AT_RECOMMENDATION_ALLOWED_TOP_LEVEL_KEYS_FULL)
    if bad:
        raise PolicyError(
            f"{context}: post-enrich simulation_at_recommendation has unknown keys {bad!r} "
            f"(allowed policy={OPTIMIZE_SIMULATION_AT_RECOMMENDATION_ALLOWLIST_POLICY})."
        )


__all__ = [
    "OPTIMIZE_SIMULATION_AT_RECOMMENDATION_ALLOWLIST_POLICY",
    "SIMULATION_AT_RECOMMENDATION_ALLOWED_TOP_LEVEL_KEYS",
    "SIMULATION_AT_RECOMMENDATION_ALLOWED_TOP_LEVEL_KEYS_FULL",
    "SIMULATION_AT_RECOMMENDATION_FORBIDDEN_TOP_LEVEL_KEYS",
    "apply_simulation_at_recommendation_allowlist",
    "apply_simulation_at_recommendation_allowlist_post_enrich",
]
