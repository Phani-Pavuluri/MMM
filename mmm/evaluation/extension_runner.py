"""Post-fit diagnostics extensions — thin orchestration over services."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from mmm.artifacts.base import ArtifactStoreBase
from mmm.artifacts.decision_bundle import build_decision_bundle
from mmm.config.schema import Framework, MMMConfig
from mmm.config.transform_policy import build_transform_policy_manifest
from mmm.data.fingerprint import fingerprint_panel
from mmm.data.panel_qa import run_panel_qa
from mmm.economics.canonical import (
    build_economics_contract,
    economics_output_metadata,
    validate_business_economics_metadata,
)
from mmm.governance.decision_safety import decision_safety_artifact
from mmm.governance.uncertainty_policy import ridge_forbids_precise_monetary_ci
from mmm.planning.context import ridge_fit_summary_from_artifacts
from mmm.data.schema import PanelSchema
from mmm.guidance.recommend import recommend_configuration
from mmm.services.calibration_service import run_calibration_extensions
from mmm.services.curve_service import build_curve_diagnostics_bundle
from mmm.services.diagnostics_service import run_core_diagnostics
from mmm.reporting.roi_sections import curve_bundles_to_roi_summary
from mmm.optimization.safety_gate import OptimizationSafetyGate
from mmm.services.governance_service import build_governance_bundle
from mmm.governance.model_release import infer_model_release_state
from mmm.uncertainty.decomposition import UncertaintyDecomposer
from mmm.data.panel_order import sort_panel_for_modeling
from mmm.evaluation.baselines import media_shuffled_within_geo, run_baselines
from mmm.evaluation.calibration_extension import compute_replay_calibration_metrics
from mmm.evaluation.feature_pipeline import build_extension_design_bundle
from mmm.features.builder import build_extra_control_matrix


def run_post_fit_extensions(
    *,
    panel: pd.DataFrame,
    schema: PanelSchema,
    config: MMMConfig,
    fit_out: dict[str, Any],
    yhat: np.ndarray,
    store: ArtifactStoreBase | None,
) -> dict[str, Any]:
    ext = config.extensions
    out: dict[str, Any] = {
        "economics_contract": build_economics_contract(config),
        "transform_policy": build_transform_policy_manifest(config),
    }
    rng = np.random.default_rng(config.random_seed)

    panel_s = sort_panel_for_modeling(panel, schema)
    out["panel_qa"] = run_panel_qa(panel_s, schema, ext.panel_qa)
    _bundle, X_media = build_extension_design_bundle(panel_s, schema, config, fit_out)

    ridge_la: float | None = None
    if config.framework == Framework.RIDGE_BO and fit_out.get("artifacts") is not None:
        ridge_la = float(fit_out["artifacts"].best_params.get("log_alpha", 0.0))
    out.update(
        run_core_diagnostics(
            panel=panel_s,
            schema=schema,
            config=config,
            X_media=X_media,
            rng=rng,
            ridge_log_alpha=ridge_la,
        )
    )

    extra = build_extra_control_matrix(panel_s, schema, ext.features)
    if extra.size:
        out["feature_engine_preview"] = {"extra_control_columns": int(extra.shape[1])}

    x_shuf: np.ndarray | None = None
    if rng is not None:
        df_shuf = media_shuffled_within_geo(panel_s, schema, rng=rng)
        _, x_shuf = build_extension_design_bundle(
            sort_panel_for_modeling(df_shuf, schema), schema, config, fit_out
        )
    bl = run_baselines(
        panel_s,
        schema,
        yhat,
        X_media,
        rng=rng,
        X_media_shuffled_same_transform=x_shuf,
    )
    out["baselines"] = bl.to_json()

    curve_bundle = build_curve_diagnostics_bundle(panel=panel_s, schema=schema, config=config, fit_out=fit_out)
    out["response_diagnostics"] = curve_bundle["response_diagnostics"]
    out["curve_stress"] = curve_bundle["curve_stress"]
    out["curve_safe_for_optimization"] = curve_bundle["safe_for_optimization"]
    if curve_bundle.get("curve_bundle"):
        out["curve_bundle"] = curve_bundle["curve_bundle"]
    if curve_bundle.get("curve_bundles"):
        out["curve_bundles"] = curve_bundle["curve_bundles"]
        out["roi_summary"] = curve_bundles_to_roi_summary(curve_bundle["curve_bundles"])

    out.update(run_calibration_extensions(config))

    replay_loss, replay_meta, is_replay = compute_replay_calibration_metrics(
        panel_s, schema, config, fit_out
    )
    cal_loss = float(replay_loss) if replay_loss is not None else None
    id_json = out.get("identifiability", {})
    bayesian_di: dict[str, Any] | None = None
    if config.framework == Framework.BAYESIAN:
        ppc_b = fit_out.get("ppc")
        if isinstance(ppc_b, dict):
            bayesian_di = ppc_b.get("decision_inference")
    out["governance"] = build_governance_bundle(
        config=config,
        panel=panel_s,
        schema=schema,
        yhat=yhat,
        baselines=bl,
        identifiability_json=id_json,
        falsification_flags=list(out.get("falsification", {}).get("flags", [])),
        calibration_loss=cal_loss,
        calibration_is_replay=is_replay,
        calibration_raw=replay_meta if is_replay else None,
        bayesian_decision_inference=bayesian_di,
    )

    out["uncertainty_decomposition"] = UncertaintyDecomposer.build_report(
        bootstrap_width={"coef_dispersion_proxy": float(id_json.get("instability_score", 0.0))},
        experiment_se_scale=1.0,
    )
    id_score = float(id_json.get("identifiability_score", 0.0))
    out["guidance"] = recommend_configuration(len(panel), len(schema.channel_columns), id_score)

    if config.framework == Framework.RIDGE_BO and fit_out.get("artifacts"):
        out["ridge_fit_summary"] = ridge_fit_summary_from_artifacts(fit_out["artifacts"])
    gov_js = out.get("governance") or {}
    pq_js = out.get("panel_qa") or {}
    out["model_release"] = infer_model_release_state(
        config=config,
        panel_qa_max_severity=str(pq_js.get("max_severity", "info")),
        governance_approved_for_optimization=bool(gov_js.get("approved_for_optimization")),
        governance_approved_for_reporting=bool(gov_js.get("approved_for_reporting")),
        ridge_fit_summary_present=bool(out.get("ridge_fit_summary")),
    )
    out["decision_policy"] = {
        "canonical_economic_quantity": "delta_mu_mean_row_mu_modeling_scale",
        "default_baseline_type": "bau",
        "planner_mode": config.extensions.product.planner_mode,
        "uncertainty_mode_default": "point",
        "decision_safe_default": True,
        "ridge_production_forbids_precise_monetary_ci": ridge_forbids_precise_monetary_ci(config),
        "curve_bundles_are_diagnostic_only": True,
        "decision_source_of_truth": "full_panel_simulation_when_available",
        "economics_contract_version": out["economics_contract"].get("contract_version"),
    }

    out["economics_output_metadata"] = economics_output_metadata(
        config,
        uncertainty_mode="point",
        surface="curve_diagnostic",
        baseline_type="extension_train_reference",
        decision_safe=False,
    )
    validate_business_economics_metadata(
        out["economics_output_metadata"],
        require_specific_baseline=True,
        require_decision_safe_bool=True,
    )

    gate = OptimizationSafetyGate(ext.optimization_gates)
    gov = out.get("governance") or {}
    gr = gate.check(
        governance=gov,
        response_diag=out.get("response_diagnostics"),
        identifiability_score=float(id_json.get("identifiability_score", 0.0)),
        run_environment=config.run_environment,
        extension_report_present=True,
    )
    cal_summary = {
        "replay_calibration_active": is_replay,
        "replay_loss": cal_loss,
        "replay_meta": replay_meta if is_replay else None,
    }
    if is_replay:
        cal_summary["economics_output_metadata_replay"] = economics_output_metadata(
            config,
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
    model_summary = {
        "framework": config.framework.value,
        "n_panel_rows": int(len(panel_s)),
        "n_channels": len(schema.channel_columns),
        "ridge_bo_n_trials": config.ridge_bo.n_trials,
    }
    decision_flags = {
        "allow_unsafe_decision_apis": config.allow_unsafe_decision_apis,
        "governance_approved_for_optimization": bool(gov.get("approved_for_optimization")),
        "optimization_gate_allowed": gr.allowed,
        "decision_safety": decision_safety_artifact(allow_unsafe_decision_apis=config.allow_unsafe_decision_apis),
    }
    fp = fingerprint_panel(panel_s, schema) if config.artifacts.write_data_fingerprint else None
    econ_surface = "replay_calibration" if is_replay else (
        "full_model_simulation" if out.get("ridge_fit_summary") else "other"
    )
    bundle_baseline_type = "replay_calibration" if is_replay else "extension_train_reference"
    gov_ok = bool(gr.allowed)
    opt_ok = bool(gov.get("approved_for_optimization"))
    out["decision_bundle"] = build_decision_bundle(
        config=config,
        schema=schema,
        governance=gov,
        optimization_gate=gr.to_json(),
        calibration_summary=cal_summary,
        simulation_contract={
            "source": "extension_train",
            "decision_simulate": "mmm.planning.decision_simulate.simulate",
            "curve_bundles_role": "diagnostic_only",
        },
        data_fingerprint=fp,
        uncertainty_mode="point",
        decision_safe=bool(gov_ok and opt_ok),
        governance_passed=gov_ok,
        optimizer_success=None,
        baseline_type=bundle_baseline_type,
        model_summary=model_summary,
        decision_safety_flags=decision_flags,
        economics_surface=econ_surface,
    )

    if store:
        store.log_dict("extension_report", out)

    return out
