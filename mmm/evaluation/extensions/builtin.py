"""Built-in post-fit extension run functions (registration in sibling plugin modules)."""

from __future__ import annotations

from typing import Any

from mmm.artifacts.decision_bundle import build_decision_bundle, validate_prod_decision_bundle
from mmm.config.schema import Framework, RunEnvironment
from mmm.config.transform_policy import build_transform_policy_manifest
from mmm.contracts.quantity_models import PosteriorExplorationQuantityResult, UncertaintyBucketsQuantityResult
from mmm.contracts.run_manifest import build_run_manifest
from mmm.data.fingerprint import fingerprint_panel
from mmm.data.panel_order import sort_panel_for_modeling
from mmm.data.panel_qa import run_panel_qa
from mmm.economics.canonical import (
    build_economics_contract,
    economics_output_metadata,
    validate_business_economics_metadata,
)
from mmm.evaluation.baselines import media_shuffled_within_geo, run_baselines
from mmm.evaluation.calibration_extension import compute_replay_calibration_metrics
from mmm.evaluation.curve_decision_alignment import evaluate_curve_decision_alignment
from mmm.evaluation.drift_monitor import build_drift_report
from mmm.evaluation.experiment_scheduler import compute_experiment_scheduler_report
from mmm.evaluation.extensions.context import ExtensionContext
from mmm.evaluation.feature_pipeline import build_extension_design_bundle
from mmm.evaluation.feature_separability import compute_feature_separability_report
from mmm.evaluation.post_fit_validation import compute_post_fit_validation_bundle
from mmm.features.builder import build_extra_control_matrix
from mmm.governance.decision_safety import decision_safety_artifact
from mmm.governance.decision_uncertainty import build_decision_uncertainty
from mmm.governance.model_release import infer_model_release_state
from mmm.governance.operational_health import compute_operational_health
from mmm.governance.split_channel_policy import apply_split_channel_governance
from mmm.governance.uncertainty_policy import ridge_forbids_precise_monetary_ci
from mmm.guidance.recommend import recommend_configuration
from mmm.optimization.safety_gate import OptimizationSafetyGate
from mmm.planning.context import ridge_fit_summary_from_artifacts
from mmm.reporting.model_card import generate_model_card
from mmm.reporting.roi_sections import curve_bundles_to_roi_summary
from mmm.services.calibration_service import run_calibration_extensions
from mmm.services.curve_service import build_curve_diagnostics_bundle
from mmm.services.diagnostics_service import run_core_diagnostics
from mmm.services.governance_service import build_governance_bundle
from mmm.uncertainty.decomposition import UncertaintyDecomposer


def _ridge_log_alpha(ctx: ExtensionContext) -> float | None:
    if ctx.config.framework == Framework.RIDGE_BO and ctx.fit_out.get("artifacts") is not None:
        return float(ctx.fit_out["artifacts"].best_params.get("log_alpha", 0.0))
    return None


def _run_bootstrap(ctx: ExtensionContext) -> None:
    ctx.out["economics_contract"] = build_economics_contract(ctx.config)
    ctx.out["transform_policy"] = build_transform_policy_manifest(ctx.config)
    ctx.out["decision_uncertainty"] = build_decision_uncertainty(ctx.config)


def _run_panel_qa(ctx: ExtensionContext) -> None:
    ctx.out["panel_qa"] = run_panel_qa(ctx.panel_s, ctx.schema, ctx.ext.panel_qa)


def _run_design_matrix(ctx: ExtensionContext) -> None:
    _, ctx.X_media = build_extension_design_bundle(ctx.panel_s, ctx.schema, ctx.config, ctx.fit_out)


def _run_core_diagnostics(ctx: ExtensionContext) -> None:
    assert ctx.X_media is not None
    ctx.out.update(
        run_core_diagnostics(
            panel=ctx.panel_s,
            schema=ctx.schema,
            config=ctx.config,
            X_media=ctx.X_media,
            rng=ctx.rng,
            ridge_log_alpha=_ridge_log_alpha(ctx),
        )
    )


def _run_post_fit_validation(ctx: ExtensionContext) -> None:
    ctx.out["post_fit_validation"] = compute_post_fit_validation_bundle(
        panel=ctx.panel_s,
        schema=ctx.schema,
        config=ctx.config,
        fit_out=ctx.fit_out,
        yhat=ctx.yhat,
    )


def _run_feature_engine_preview(ctx: ExtensionContext) -> None:
    extra = build_extra_control_matrix(ctx.panel_s, ctx.schema, ctx.ext.features)
    if extra.size:
        ctx.out["feature_engine_preview"] = {"extra_control_columns": int(extra.shape[1])}


def _run_baselines(ctx: ExtensionContext) -> None:
    assert ctx.X_media is not None
    df_shuf = media_shuffled_within_geo(ctx.panel_s, ctx.schema, rng=ctx.rng)
    _, ctx.x_shuf = build_extension_design_bundle(
        sort_panel_for_modeling(df_shuf, ctx.schema), ctx.schema, ctx.config, ctx.fit_out
    )
    ctx.baselines_result = run_baselines(
        ctx.panel_s,
        ctx.schema,
        ctx.yhat,
        ctx.X_media,
        rng=ctx.rng,
        X_media_shuffled_same_transform=ctx.x_shuf,
    )
    ctx.out["baselines"] = ctx.baselines_result.to_json()


def _run_curves(ctx: ExtensionContext) -> None:
    ctx.curve_bundle = build_curve_diagnostics_bundle(
        panel=ctx.panel_s, schema=ctx.schema, config=ctx.config, fit_out=ctx.fit_out
    )
    cb = ctx.curve_bundle
    ctx.out["response_diagnostics"] = cb["response_diagnostics"]
    ctx.out["curve_stress"] = cb["curve_stress"]
    ctx.out["curve_safe_for_optimization"] = cb["safe_for_optimization"]
    if cb.get("curve_bundle"):
        ctx.out["curve_bundle"] = cb["curve_bundle"]
    if cb.get("curve_bundles"):
        ctx.out["curve_bundles"] = cb["curve_bundles"]
        ctx.out["roi_summary"] = curve_bundles_to_roi_summary(cb["curve_bundles"])


def _run_calibration_extensions(ctx: ExtensionContext) -> None:
    ctx.out.update(run_calibration_extensions(ctx.config, panel=ctx.panel_s, schema=ctx.schema))


def _run_replay_calibration(ctx: ExtensionContext) -> None:
    replay_loss, replay_meta, is_replay = compute_replay_calibration_metrics(
        ctx.panel_s, ctx.schema, ctx.config, ctx.fit_out
    )
    ctx.is_replay = is_replay
    ctx.replay_meta = replay_meta
    ctx.cal_loss = float(replay_loss) if replay_loss is not None else None
    ctx.out["calibration_summary"] = {
        "replay_calibration_active": is_replay,
        "replay_loss": ctx.cal_loss,
        "replay_meta": replay_meta if is_replay else None,
    }


def _run_governance(ctx: ExtensionContext) -> None:
    assert ctx.baselines_result is not None
    id_json = ctx.out.get("identifiability", {})
    bayesian_di: dict[str, Any] | None = None
    if ctx.config.framework == Framework.BAYESIAN:
        ppc_b = ctx.fit_out.get("ppc")
        if isinstance(ppc_b, dict):
            bayesian_di = ppc_b.get("decision_inference")
    ctx.out["governance"] = build_governance_bundle(
        config=ctx.config,
        panel=ctx.panel_s,
        schema=ctx.schema,
        yhat=ctx.yhat,
        baselines=ctx.baselines_result,
        identifiability_json=id_json,
        falsification_flags=list(ctx.out.get("falsification", {}).get("flags", [])),
        calibration_loss=ctx.cal_loss,
        calibration_is_replay=ctx.is_replay,
        calibration_raw=ctx.replay_meta if ctx.is_replay else None,
        bayesian_decision_inference=bayesian_di,
    )


def _run_split_channel_governance(ctx: ExtensionContext) -> None:
    apply_split_channel_governance(ctx.out)


def _run_feature_separability(ctx: ExtensionContext) -> None:
    assert ctx.X_media is not None
    id_json = ctx.out.get("identifiability", {})
    gov_js = ctx.out.get("governance") if isinstance(ctx.out.get("governance"), dict) else {}
    ctx.out["feature_separability_report"] = compute_feature_separability_report(
        panel=ctx.panel_s,
        schema=ctx.schema,
        config=ctx.config,
        fit_out=ctx.fit_out,
        X_media=ctx.X_media,
        identifiability_json=id_json if isinstance(id_json, dict) else None,
        experiment_matching_json=ctx.out.get("experiment_matching")
        if isinstance(ctx.out.get("experiment_matching"), dict)
        else None,
        rng=ctx.rng,
        governance_approved_for_optimization=bool(gov_js.get("approved_for_optimization")),
    )


def _run_bayesian_posterior_quantity(ctx: ExtensionContext) -> None:
    if ctx.config.framework != Framework.BAYESIAN:
        return
    bayesian_di: dict[str, Any] | None = None
    ppc_b = ctx.fit_out.get("ppc")
    if isinstance(ppc_b, dict):
        bayesian_di = ppc_b.get("decision_inference")
    di_summary: dict[str, Any] = {"surface": "extension_train_diagnostic"}
    if isinstance(bayesian_di, dict):
        di_summary["decision_inference_keys"] = sorted(str(k) for k in bayesian_di)
    ctx.out["posterior_exploration_quantity"] = PosteriorExplorationQuantityResult(
        draw_summary=di_summary,
        validity_diagnostics={"framework": "bayesian", "prod_decisioning_allowed": False},
    ).section_dict()


def _run_uncertainty_decomposition(ctx: ExtensionContext) -> None:
    id_json = ctx.out.get("identifiability", {})
    ud_report = UncertaintyDecomposer.build_report(
        bootstrap_width={"coef_dispersion_proxy": float(id_json.get("instability_score", 0.0))},
        experiment_se_scale=1.0,
    )
    ctx.out["uncertainty_decomposition"] = UncertaintyBucketsQuantityResult(
        validity_diagnostics={"uncertainty_buckets": ud_report},
    ).section_dict()


def _run_guidance(ctx: ExtensionContext) -> None:
    id_json = ctx.out.get("identifiability", {})
    id_score = float(id_json.get("identifiability_score", 0.0))
    ctx.out["guidance"] = recommend_configuration(
        len(ctx.panel), len(ctx.schema.channel_columns), id_score
    )


def _run_ridge_fit_summary(ctx: ExtensionContext) -> None:
    if ctx.config.framework == Framework.RIDGE_BO and ctx.fit_out.get("artifacts"):
        ctx.out["ridge_fit_summary"] = ridge_fit_summary_from_artifacts(ctx.fit_out["artifacts"])


def _run_model_release(ctx: ExtensionContext) -> None:
    gov_js = ctx.out.get("governance") or {}
    pq_js = ctx.out.get("panel_qa") or {}
    post_fit_validation = ctx.out.get("post_fit_validation")
    post_fit_validation_js = post_fit_validation if isinstance(post_fit_validation, dict) else None
    operational_health = ctx.out.get("operational_health")
    operational_health_js = operational_health if isinstance(operational_health, dict) else None
    ctx.out["model_release"] = infer_model_release_state(
        config=ctx.config,
        panel_qa_max_severity=str(pq_js.get("max_severity", "info")),
        governance_approved_for_optimization=bool(gov_js.get("approved_for_optimization")),
        governance_approved_for_reporting=bool(gov_js.get("approved_for_reporting")),
        ridge_fit_summary_present=bool(ctx.out.get("ridge_fit_summary")),
        post_fit_validation=post_fit_validation_js,
        operational_health=operational_health_js,
    )


def _run_decision_policy(ctx: ExtensionContext) -> None:
    ctx.out["decision_policy"] = {
        "canonical_economic_quantity": "delta_mu_mean_row_mu_modeling_scale",
        "default_baseline_type": "bau",
        "planner_mode": ctx.config.extensions.product.planner_mode,
        "uncertainty_mode_default": "point",
        "decision_safe_default": True,
        "ridge_production_forbids_precise_monetary_ci": ridge_forbids_precise_monetary_ci(ctx.config),
        "curve_bundles_are_diagnostic_only": True,
        "decision_source_of_truth": "full_panel_simulation_when_available",
        "economics_contract_version": ctx.out["economics_contract"].get("contract_version"),
    }


def _run_economics_output_metadata(ctx: ExtensionContext) -> None:
    ctx.out["economics_output_metadata"] = economics_output_metadata(
        ctx.config,
        uncertainty_mode="point",
        surface="curve_diagnostic",
        baseline_type="extension_train_reference",
        decision_safe=False,
    )
    validate_business_economics_metadata(
        ctx.out["economics_output_metadata"],
        require_specific_baseline=True,
        require_decision_safe_bool=True,
    )


def _run_optimization_gate(ctx: ExtensionContext) -> None:
    id_json = ctx.out.get("identifiability", {})
    gate = OptimizationSafetyGate(ctx.ext.optimization_gates)
    gov = ctx.out.get("governance") or {}
    ctx.gate_result = gate.check(
        governance=gov,
        response_diag=ctx.out.get("response_diagnostics"),
        identifiability_score=float(id_json.get("identifiability_score", 0.0)),
        run_environment=ctx.config.run_environment,
        extension_report_present=True,
        panel_qa=ctx.out.get("panel_qa") if isinstance(ctx.out.get("panel_qa"), dict) else None,
    )


def _run_operational_health(ctx: ExtensionContext) -> None:
    assert ctx.gate_result is not None
    ctx.out["operational_health"] = compute_operational_health(
        config=ctx.config,
        extension_report=ctx.out,
        optimization_gate_allowed=bool(ctx.gate_result.allowed),
    )


def _run_experiment_scheduler(ctx: ExtensionContext) -> None:
    assert ctx.gate_result is not None
    ch_cols = list(ctx.schema.channel_columns)
    panel_media_total = (
        sum(
            float(ctx.panel_s[c].astype(float).clip(lower=0.0).sum())
            for c in ch_cols
            if c in ctx.panel_s.columns
        )
        + 1e-12
    )
    spend_share_panel = {
        c: float(ctx.panel_s[c].astype(float).clip(lower=0.0).sum()) / panel_media_total
        for c in ch_cols
        if c in ctx.panel_s.columns
    }
    ctx.out["experiment_scheduler_report"] = compute_experiment_scheduler_report(
        config=ctx.config,
        extension_report=ctx.out,
        channel_columns=ch_cols,
        channel_spend_share_of_panel=spend_share_panel,
        target_column=ctx.schema.target_column,
        optimization_gate_allowed=bool(ctx.gate_result.allowed),
    )


def _run_fingerprint_and_drift(ctx: ExtensionContext) -> None:
    if not (
        ctx.config.artifacts.write_data_fingerprint
        or ctx.config.run_environment == RunEnvironment.PROD
    ):
        return
    ctx.fingerprint = fingerprint_panel(
        ctx.panel_s,
        ctx.schema,
        config=ctx.config,
        seed_resolution=ctx.seed_resolution,
    )
    ctx.out["data_fingerprint"] = ctx.fingerprint
    ctx.out["drift_report"] = build_drift_report(
        panel=ctx.panel_s,
        schema=ctx.schema,
        config=ctx.config,
        reference_fingerprint=ctx.fingerprint,
        reference_panel=ctx.panel_s,
        calibration_summary=ctx.out.get("calibration_summary"),
        seed_resolution=ctx.seed_resolution,
    )


def _run_curve_decision_alignment(ctx: ExtensionContext) -> None:
    if not ctx.curve_bundle.get("curve_bundles"):
        return
    ctx.out["curve_decision_alignment"] = evaluate_curve_decision_alignment(
        panel=ctx.panel_s,
        schema=ctx.schema,
        config=ctx.config,
        fit_out=ctx.fit_out,
        curve_bundles=ctx.curve_bundle["curve_bundles"],
    )


def _run_decision_bundle(ctx: ExtensionContext) -> None:
    assert ctx.gate_result is not None
    gov = ctx.out.get("governance") or {}
    cal_summary = dict(ctx.out["calibration_summary"])
    if ctx.is_replay:
        cal_summary["economics_output_metadata_replay"] = economics_output_metadata(
            ctx.config,
            uncertainty_mode="point",
            surface="replay_calibration",
            baseline_type="replay_calibration_reference",
            decision_safe=False,
        )
        validate_business_economics_metadata(
            cal_summary["economics_output_metadata_replay"],
            require_specific_baseline=True,
            require_decision_safe_bool=True,
        )
    econ_surface = (
        "replay_calibration"
        if ctx.is_replay
        else ("full_model_simulation" if ctx.out.get("ridge_fit_summary") else "other")
    )
    bundle_baseline_type = "replay_calibration" if ctx.is_replay else "extension_train_reference"
    gov_ok = bool(ctx.gate_result.allowed)
    opt_ok = bool(gov.get("approved_for_optimization"))
    model_summary = {
        "framework": ctx.config.framework.value,
        "n_panel_rows": int(len(ctx.panel_s)),
        "n_channels": len(ctx.schema.channel_columns),
        "ridge_bo_n_trials": ctx.config.ridge_bo.n_trials,
    }
    decision_flags = {
        "allow_unsafe_decision_apis": ctx.config.allow_unsafe_decision_apis,
        "governance_approved_for_optimization": bool(gov.get("approved_for_optimization")),
        "optimization_gate_allowed": ctx.gate_result.allowed,
        "decision_safety": decision_safety_artifact(
            allow_unsafe_decision_apis=ctx.config.allow_unsafe_decision_apis
        ),
    }
    ctx.out["artifact_tier_disclosure"] = {
        "extension_report_role": "training_diagnostics",
        "nested_decision_bundle_artifact_tier": "research",
        "cli_decision_bundle_artifact_tier": "decision",
        "note": (
            "Training extension_report and nested decision_bundle are research/diagnostic tier. "
            "Production budgeting requires a fresh mmm decide simulate|optimize-budget JSON bundle "
            "with artifact_tier=decision (see docs/planning_artifact_schema.md and prod safety checklist)."
        ),
    }
    ctx.out["decision_bundle"] = build_decision_bundle(
        config=ctx.config,
        schema=ctx.schema,
        governance=gov,
        optimization_gate=ctx.gate_result.to_json(),
        calibration_summary=cal_summary,
        simulation_contract={
            "source": "extension_train",
            "decision_simulate": "mmm.planning.decision_simulate.simulate",
            "curve_bundles_role": "diagnostic_only",
            "curve_policy_note": "Curves explain; full-panel simulation decides.",
        },
        data_fingerprint=ctx.fingerprint,
        decision_uncertainty=ctx.out["decision_uncertainty"],
        uncertainty_mode="point",
        decision_safe=bool(gov_ok and opt_ok),
        governance_passed=gov_ok,
        optimizer_success=None,
        baseline_type=bundle_baseline_type,
        model_summary=model_summary,
        decision_safety_flags=decision_flags,
        economics_surface=econ_surface,
        panel_qa=ctx.out.get("panel_qa") if isinstance(ctx.out.get("panel_qa"), dict) else None,
        experiment_matching=ctx.out.get("experiment_matching")
        if isinstance(ctx.out.get("experiment_matching"), dict)
        else None,
        model_release=ctx.out.get("model_release") if isinstance(ctx.out.get("model_release"), dict) else None,
        artifact_tier="research",
        extension_report=ctx.out if isinstance(ctx.out, dict) else None,
    )
    if ctx.config.run_environment == RunEnvironment.PROD:
        miss = validate_prod_decision_bundle(
            ctx.out["decision_bundle"],
            run_environment=ctx.config.run_environment,
            decision_cli_surface=False,
        )
        if miss:
            raise RuntimeError("PROD extension decision_bundle failed completeness audit: " + "; ".join(miss))


def _run_run_manifest(ctx: ExtensionContext) -> None:
    ctx.out["run_manifest"] = build_run_manifest(ctx.out, run_id=ctx.config.run_id)


def _run_model_card(ctx: ExtensionContext) -> None:
    ctx.out["model_card_md"] = generate_model_card(
        extension_report=ctx.out,
        decision_bundle=ctx.out.get("decision_bundle"),
    )
    if ctx.store:
        (ctx.store.run_path / "model_card.md").write_text(ctx.out["model_card_md"], encoding="utf-8")
