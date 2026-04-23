"""Single shared decision path for simulate + optimize (CLI + Python API)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

from mmm.artifacts.decision_bundle import build_decision_bundle, compute_unsupported_questions
from mmm.artifacts.schema import SimulationDecisionResult
from mmm.config.schema import Framework, MMMConfig, RunEnvironment
from mmm.contracts.business_surface import (
    BusinessSurfaceMetadataError,
    enrich_decision_simulation_json,
    optimization_response_business_metadata,
    validate_business_facing_payload,
)
from mmm.contracts.canonical_transforms import assert_canonical_media_stack_for_modeling
from mmm.contracts.quantity_models import reject_approximate_quantity_subtrees_in_payload
from mmm.contracts.runtime_validation import SemanticContractError
from mmm.data.fingerprint import fingerprint_panel
from mmm.data.loader import DatasetBuilder
from mmm.data.panel_order import sort_panel_for_modeling
from mmm.data.schema import validate_panel
from mmm.decision.core import finalize_and_validate_cli_decision_bundle
from mmm.decision.extension_gate import optimization_gate_result
from mmm.decision.gates import allow_decision_pipeline
from mmm.decision.optimize_enrichment import (
    OPTIMIZE_SIMULATION_AT_RECOMMENDATION_ALLOWLIST_POLICY,
    apply_simulation_at_recommendation_allowlist,
    apply_simulation_at_recommendation_allowlist_post_enrich,
)
from mmm.governance.policy import (
    PolicyError,
    cv_mode_key_from_config,
    require_allow_unsafe,
    require_bayesian_block,
    require_decision_safe_result,
    require_identifiability_for_prod_decision,
    require_panel_qa_pass,
    require_planning_allowed,
    require_replay_calibration,
    require_safe_cv,
    runtime_policy_from_config,
)
from mmm.optimization.budget.simulation_optimizer import optimize_budget_via_simulation
from mmm.optimization.safety_gate import OptimizationSafetyGate
from mmm.planning.context import ridge_context_from_summary


def _prod_extension_gate(cfg: MMMConfig, er: dict[str, Any]) -> Any:
    gov = er.get("governance", {}) if isinstance(er, dict) else {}
    resp = er.get("response_diagnostics") if isinstance(er, dict) else None
    ident = float(er.get("identifiability", {}).get("identifiability_score", 0.0)) if isinstance(er, dict) else 0.0
    pq = er.get("panel_qa") if isinstance(er, dict) else None
    gate = OptimizationSafetyGate(cfg.extensions.optimization_gates)
    return gate.check(
        governance=gov if isinstance(gov, dict) else {},
        response_diag=resp,
        identifiability_score=ident,
        run_environment=cfg.run_environment,
        extension_report_present=True,
        panel_qa=pq if isinstance(pq, dict) else None,
    )


def _apply_runtime_policy_prechecks(cfg: MMMConfig, er: dict[str, Any], policy: Any) -> None:
    if cfg.framework in (Framework.RIDGE_BO, Framework.BAYESIAN):
        assert_canonical_media_stack_for_modeling(cfg)
    require_allow_unsafe(policy)
    require_bayesian_block(cfg.framework, policy)
    require_safe_cv(cv_mode_key_from_config(cfg), policy)
    mr = er.get("model_release") if isinstance(er, dict) else None
    st = mr.get("state") if isinstance(mr, dict) else None
    require_planning_allowed(st, policy)
    require_panel_qa_pass(er.get("panel_qa") if isinstance(er, dict) else None, policy)
    require_replay_calibration(
        er.get("calibration_summary") if isinstance(er, dict) else None,
        er.get("experiment_matching") if isinstance(er, dict) else None,
        policy,
    )
    require_identifiability_for_prod_decision(cfg, er, policy)


def _scenario_simulate(
    cfg: MMMConfig,
    er: dict[str, Any],
    raw: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    reject_approximate_quantity_subtrees_in_payload(raw, context="scenario_yaml")

    from mmm.planning.baseline import (
        channel_means_from_geo_plan,
        locked_geo_plan_baseline,
        locked_plan_baseline,
    )
    from mmm.planning.control_overlay import ControlOverlaySpec
    from mmm.planning.decision_simulate import simulate as decision_simulate
    from mmm.planning.spend_path import (
        PiecewiseSpendPath,
        counterfactual_piecewise_spend_panel,
        time_mean_spend_by_channel,
    )

    rs = er.get("ridge_fit_summary")
    if not isinstance(rs, dict) or not rs.get("coef"):
        raise PolicyError("extension_report must contain ridge_fit_summary.coef")
    builder = DatasetBuilder(cfg.data)
    schema = builder.schema()
    panel = sort_panel_for_modeling(validate_panel(builder.build(), schema), schema)
    ctx = ridge_context_from_summary(panel, schema, cfg, rs)
    geos_cli = sorted({str(x) for x in panel[schema.geo_column].unique()})
    spend_path = None
    if isinstance(raw.get("candidate_spend_path"), dict):
        spend_path = PiecewiseSpendPath.from_dict(raw["candidate_spend_path"])
    has_geo_cand = isinstance(raw.get("candidate_spend_by_geo"), dict)
    has_scalar_cand = isinstance(raw.get("candidate_spend"), dict)
    if spend_path is not None and has_geo_cand:
        raise ValueError("scenario YAML: candidate_spend_path cannot be combined with candidate_spend_by_geo")
    if has_geo_cand and has_scalar_cand:
        raise ValueError("scenario YAML: use either candidate_spend or candidate_spend_by_geo, not both")
    if spend_path is None and not has_scalar_cand and not has_geo_cand:
        raise ValueError(
            "scenario YAML must include candidate_spend, candidate_spend_by_geo, and/or candidate_spend_path."
        )
    spend_plan_geo = None
    if spend_path is not None:
        if has_scalar_cand and isinstance(raw.get("candidate_spend"), dict):
            cand = {str(k): float(v) for k, v in raw["candidate_spend"].items()}
        else:
            tmp_df = counterfactual_piecewise_spend_panel(panel, schema, spend_path)
            cand = time_mean_spend_by_channel(tmp_df, schema)
    elif has_geo_cand:
        raw_geo = raw["candidate_spend_by_geo"]
        spend_plan_geo = {
            str(g): {str(c): float(v) for c, v in row.items()} for g, row in raw_geo.items() if isinstance(row, dict)
        }
        cand = channel_means_from_geo_plan(spend_plan_geo, schema, geos_cli)
    else:
        cs = raw.get("candidate_spend")
        if not isinstance(cs, dict):
            raise ValueError("scenario YAML must include candidate_spend when candidate_spend_path is absent.")
        cand = {str(k): float(v) for k, v in cs.items()}
    base_plan = None
    if isinstance(raw.get("baseline_spend"), dict):
        base_plan = locked_plan_baseline(
            {str(k): float(v) for k, v in raw["baseline_spend"].items()},
            source="scenario_yaml:baseline_spend",
            notes="baseline_spend from scenario YAML (non-BAU; labeled locked_plan).",
        )
    elif isinstance(raw.get("baseline_spend_by_geo"), dict):
        raw_bg = raw["baseline_spend_by_geo"]
        by_geo = {
            str(g): {str(c): float(v) for c, v in row.items()} for g, row in raw_bg.items() if isinstance(row, dict)
        }
        base_plan = locked_geo_plan_baseline(
            by_geo,
            source="scenario_yaml:baseline_spend_by_geo",
            notes="baseline_spend_by_geo from scenario YAML (non-BAU; labeled locked_plan).",
        )
    co_b = ControlOverlaySpec.from_dict(raw["control_overlay_baseline"]) if isinstance(
        raw.get("control_overlay_baseline"), dict
    ) else None
    co_p = ControlOverlaySpec.from_dict(raw["control_overlay_plan"]) if isinstance(
        raw.get("control_overlay_plan"), dict
    ) else None
    co_single = (
        ControlOverlaySpec.from_dict(raw["control_overlay"]) if isinstance(raw.get("control_overlay"), dict) else None
    )
    ctrls_yaml = raw.get("controls_plan")
    sim = decision_simulate(
        cand,
        ctx,
        baseline_plan=base_plan,
        uncertainty_mode="point",
        spend_path_plan=spend_path,
        spend_plan_geo=spend_plan_geo,
        controls_plan=ctrls_yaml if co_p is None and co_single is None else None,
        control_overlay_baseline=co_b,
        control_overlay_plan=co_p,
        control_overlay=co_single if co_p is None else None,
    )
    return sim, sim.to_json()


def simulate_decision(
    *,
    cfg: MMMConfig,
    scenario: dict[str, Any],
    extension_report: dict[str, Any],
    out: Path | None,
) -> dict[str, Any]:
    """
    Full-panel Δμ simulate with centralized runtime policy.

    Raises ``PolicyError`` on policy violations; ``ValueError`` on invalid scenario shape.
    """
    policy = runtime_policy_from_config(cfg)
    if cfg.run_environment == RunEnvironment.PROD:
        if not cfg.data.path:
            raise PolicyError("Production simulate requires config.data.path")
        if out is None:
            raise PolicyError("Production simulate requires --out PATH to persist the JSON decision artifact")
    if not cfg.data.path:
        raise ValueError("simulate requires config.data.path to the training panel")
    _apply_runtime_policy_prechecks(cfg, extension_report, policy)

    gr_prod = None
    if cfg.run_environment == RunEnvironment.PROD:
        gr_prod = _prod_extension_gate(cfg, extension_report)
        if not gr_prod.allowed:
            raise PolicyError("optimization_gate_blocked: " + "; ".join(gr_prod.reasons))

    sim, sim_js = _scenario_simulate(cfg, extension_report, scenario)
    er = extension_report
    sim_js_pre, _sim_audit = apply_simulation_at_recommendation_allowlist(
        dict(sim_js), cfg=cfg, context="simulate_decision.simulation_json.pre_enrich"
    )
    _uq_sim = compute_unsupported_questions(cfg, er if isinstance(er, dict) else None)
    _gate_sim = bool(gr_prod.allowed) if cfg.run_environment == RunEnvironment.PROD and gr_prod is not None else True
    sim_js = enrich_decision_simulation_json(
        sim_js_pre, cfg=cfg, unsupported_questions=_uq_sim, governance_gate_allowed=_gate_sim
    )
    apply_simulation_at_recommendation_allowlist_post_enrich(
        sim_js, context="simulate_decision.simulation_json.post_enrich"
    )
    try:
        validate_business_facing_payload(
            sim_js,
            require_decision_tier=(cfg.run_environment == RunEnvironment.PROD),
            require_unsupported_questions=True,
        )
    except BusinessSurfaceMetadataError as e:
        raise PolicyError(str(e)) from e

    if cfg.run_environment != RunEnvironment.PROD:
        return {"simulation": sim_js}

    assert gr_prod is not None
    gov = er.get("governance") if isinstance(er, dict) else {}
    pq = er.get("panel_qa") if isinstance(er, dict) else {}
    gr = gr_prod
    builder = DatasetBuilder(cfg.data)
    schema = builder.schema()
    panel = sort_panel_for_modeling(validate_panel(builder.build(), schema), schema)
    fp = fingerprint_panel(panel, schema)
    em = er.get("experiment_matching") if isinstance(er, dict) else None
    mr = er.get("model_release") if isinstance(er, dict) else None
    policy_hash = policy.policy_fingerprint()
    mr_id = None
    if isinstance(mr, dict):
        mr_id = str(hash(tuple(sorted((str(k), str(v)) for k, v in mr.items()))))

    bundle = build_decision_bundle(
        config=cfg,
        schema=schema,
        governance=gov if isinstance(gov, dict) else {},
        optimization_gate=gr.to_json(),
        calibration_summary=er.get("calibration_summary") if isinstance(er, dict) else None,
        simulation_contract={
            "source": "decision_service_simulate",
            "decision_simulate": "mmm.planning.decision_simulate.simulate",
            "curve_bundles_role": "diagnostic_only",
            "objective": "delta_mu",
        },
        data_fingerprint=fp,
        uncertainty_mode="point",
        decision_safe=bool(gr.allowed),
        governance_passed=bool(gr.allowed),
        optimizer_success=None,
        baseline_type=str(sim_js.get("baseline_definition") or "bau"),
        economics_surface="full_model_simulation",
        panel_qa=pq if isinstance(pq, dict) else None,
        experiment_matching=em if isinstance(em, dict) else None,
        model_release=mr if isinstance(mr, dict) else None,
        simulation_json=sim_js,
        extension_report=er if isinstance(er, dict) else None,
        runtime_policy_hash=policy_hash,
        model_release_id=mr_id,
    )
    try:
        finalize_and_validate_cli_decision_bundle(bundle, cfg, simulation_json=sim_js)
    except SemanticContractError as e:
        raise PolicyError(str(e)) from e

    canon = SimulationDecisionResult.from_simulation_json(
        sim_js,
        governance_refs={"optimization_gate": gr.to_json(), "governance": gov},
        lineage_refs={"bundle_keys": list(bundle.keys())},
    )
    require_decision_safe_result(canon.as_result_dict(), policy)
    return {"simulation": sim_js, "decision_bundle": bundle, "decision_result": canon.model_dump(mode="json")}


def optimize_budget_decision(
    *,
    cfg: MMMConfig,
    extension_report: dict[str, Any],
    out: Path | None,
) -> dict[str, Any]:
    """Full-panel optimize with centralized runtime policy."""
    policy = runtime_policy_from_config(cfg)
    if cfg.run_environment == RunEnvironment.PROD and out is None:
        raise PolicyError("Production optimize-budget requires --out PATH to persist the decision artifact")
    _apply_runtime_policy_prechecks(cfg, extension_report, policy)

    er_data = extension_report
    gov = er_data.get("governance", {}) if isinstance(er_data, dict) else {}
    gr = optimization_gate_result(cfg, er_data, extension_report_present=True)
    if not gr.allowed:
        raise PolicyError("optimization_gate_blocked: " + "; ".join(gr.reasons))

    ridge_summary = er_data.get("ridge_fit_summary") if isinstance(er_data, dict) else None
    full_model_ready = bool(cfg.data.path and isinstance(ridge_summary, dict) and ridge_summary.get("coef"))
    if cfg.run_environment == RunEnvironment.PROD and not full_model_ready:
        raise PolicyError("Production optimize requires data.path + ridge_fit_summary in extension_report")

    if not full_model_ready:
        raise PolicyError("full_model_ready inputs required for optimize_budget_decision service path")

    builder = DatasetBuilder(cfg.data)
    schema = builder.schema()
    panel = sort_panel_for_modeling(validate_panel(builder.build(), schema), schema)
    ctx = ridge_context_from_summary(panel, schema, cfg, ridge_summary)  # type: ignore[arg-type]
    names = list(cfg.data.channel_columns)
    n = len(names)
    total_budget = float(cfg.budget.total_budget or n * 1e5)
    channel_min = np.array([float(cfg.budget.channel_min.get(c, 0.0)) for c in names], dtype=float)
    channel_max = np.array([float(cfg.budget.channel_max.get(c, 1e6)) for c in names], dtype=float)
    current = np.ones(n, dtype=float) * 1e5
    with allow_decision_pipeline():
        res = optimize_budget_via_simulation(
            ctx,
            current_spend=current,
            total_budget=total_budget,
            channel_min=channel_min,
            channel_max=channel_max,
        )
    optimizer_success = bool(res.get("optimizer_success", res.get("success")))
    governance_passed = bool(gr.allowed)
    allocation_stable = bool(res.get("allocation_stable", True))
    optimizer_internal_safe = bool(res.get("decision_safe", False))
    stability_score = float(res.get("stability_score", 0.0))
    extrapolation_ok = True
    sim_at: dict[str, Any] = dict(res.get("simulation_at_recommendation") or {})
    if sim_at:
        sim_at_pre, allow_audit = apply_simulation_at_recommendation_allowlist(
            sim_at, cfg=cfg, context="optimize_budget_decision.simulation_at_recommendation.pre_enrich"
        )
        _uq_opt = compute_unsupported_questions(cfg, er_data if isinstance(er_data, dict) else None)
        sim_at = enrich_decision_simulation_json(
            sim_at_pre,
            cfg=cfg,
            unsupported_questions=_uq_opt,
            governance_gate_allowed=bool(gr.allowed),
        )
        apply_simulation_at_recommendation_allowlist_post_enrich(
            sim_at, context="optimize_budget_decision.simulation_at_recommendation.post_enrich"
        )
        res = {
            **res,
            "simulation_at_recommendation": sim_at,
            "simulation_at_recommendation_allowlist_audit": {
                "policy": OPTIMIZE_SIMULATION_AT_RECOMMENDATION_ALLOWLIST_POLICY,
                "pre_enrich": allow_audit,
            },
        }
        reject_approximate_quantity_subtrees_in_payload(sim_at, context="optimizer.simulation_at_recommendation")
    em = sim_at.get("economics_metadata") if sim_at else None
    if isinstance(em, dict):
        extrapolation_ok = not bool(em.get("extrapolation_flag", False))
    gates_enabled = bool(cfg.extensions.optimization_gates.enabled)
    decision_safe = bool(
        governance_passed
        and optimizer_success
        and allocation_stable
        and optimizer_internal_safe
        and extrapolation_ok
        and gates_enabled
    )
    pq_b = er_data.get("panel_qa") if isinstance(er_data, dict) else None
    em_b = er_data.get("experiment_matching") if isinstance(er_data, dict) else None
    mr_b = er_data.get("model_release") if isinstance(er_data, dict) else None
    policy_hash = policy.policy_fingerprint()
    mr_id = None
    if isinstance(mr_b, dict):
        mr_id = str(hash(tuple(sorted((str(k), str(v)) for k, v in mr_b.items()))))
    bundle = build_decision_bundle(
        config=cfg,
        schema=schema,
        governance=gov,
        optimization_gate=gr.to_json(),
        simulation_contract={"source": "full_model_simulation_slsqp", "objective": "delta_mu"},
        data_fingerprint=fingerprint_panel(panel, schema),
        uncertainty_mode="point",
        decision_safe=decision_safe,
        governance_passed=governance_passed,
        optimizer_success=optimizer_success,
        model_summary={
            "objective_delta_mu": res.get("objective_delta_mu"),
            "optimizer_success": optimizer_success,
            "allocation_stable": allocation_stable,
            "optimizer_decision_safe": optimizer_internal_safe,
            "stability_score": stability_score,
            "multistart": res.get("multistart"),
            "stability": res.get("stability"),
            "decision_safe_components": {
                "governance_passed": governance_passed,
                "optimizer_success": optimizer_success,
                "allocation_stable": allocation_stable,
                "optimizer_internal_safe": optimizer_internal_safe,
                "no_extrapolation": extrapolation_ok,
                "gates_enabled": gates_enabled,
            },
        },
        economics_surface="full_model_simulation",
        panel_qa=pq_b if isinstance(pq_b, dict) else None,
        experiment_matching=em_b if isinstance(em_b, dict) else None,
        model_release=mr_b if isinstance(mr_b, dict) else None,
        simulation_json=sim_at if sim_at else None,
        extension_report=er_data if isinstance(er_data, dict) else None,
        runtime_policy_hash=policy_hash,
        model_release_id=mr_id,
    )
    _sim_for_val = sim_at if sim_at else None
    try:
        finalize_and_validate_cli_decision_bundle(bundle, cfg, simulation_json=_sim_for_val)
    except SemanticContractError as e:
        raise PolicyError(str(e)) from e
    bs = optimization_response_business_metadata(cfg=cfg, bundle=bundle, governance_gate_allowed=gr.allowed)
    if cfg.run_environment == RunEnvironment.PROD:
        try:
            validate_business_facing_payload(
                bs,
                require_decision_tier=True,
                require_unsupported_questions=True,
            )
        except BusinessSurfaceMetadataError as e:
            raise PolicyError(str(e)) from e
    if sim_at:
        canon = SimulationDecisionResult.from_simulation_json(
            sim_at,
            governance_refs={"optimization_gate": gr.to_json()},
            lineage_refs={"bundle_keys": list(bundle.keys())},
        )
        require_decision_safe_result(canon.as_result_dict(), policy)
    return {"optimization": res, "decision_bundle": bundle, "business_surface": bs}


def load_scenario_yaml(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("scenario YAML must be a mapping")
    return raw
