"""Unified machine-readable bundle for decision-facing CLI/API outputs."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import mmm.version as mmm_version
from mmm.artifacts.schema import ARTIFACT_BUNDLE_SCHEMA_VERSION
from mmm.config.schema import Framework, MMMConfig, RunEnvironment
from mmm.contracts.artifact_tier import DECISION_TIER_VALUE
from mmm.contracts.lineage import build_decision_tier_lineage_payload
from mmm.contracts.quantity_models import (
    CurveQuantityResult,
    DecompositionQuantityResult,
    ROIApproxQuantityResult,
)
from mmm.governance.semantics import DecisionSemantics, SafetyFlags
from mmm.data.schema import PanelSchema
from mmm.economics.canonical import (
    EconomicsOutputSurface,
    build_economics_contract,
    economics_output_metadata,
    validate_business_economics_metadata,
)


def decision_resolved_config_snapshot(config: MMMConfig) -> dict[str, Any]:
    """Non-secret fields from resolved config for reproducibility inside decision bundles."""
    d = config.model_dump(mode="json")
    data = d.get("data") or {}
    cal = d.get("calibration") or {}
    ext = d.get("extensions") or {}
    return {
        "run_environment": d.get("run_environment"),
        "framework": d.get("framework"),
        "model_form": d.get("model_form"),
        "allow_unsafe_decision_apis": d.get("allow_unsafe_decision_apis"),
        "data": {
            "path": data.get("path"),
            "geo_column": data.get("geo_column"),
            "week_column": data.get("week_column"),
            "target_column": data.get("target_column"),
            "channel_columns": data.get("channel_columns"),
            "control_columns": data.get("control_columns"),
            "data_version_id": data.get("data_version_id"),
        },
        "calibration": {
            "enabled": cal.get("enabled"),
            "use_replay_calibration": cal.get("use_replay_calibration"),
            "replay_units_path": cal.get("replay_units_path"),
            "experiment_target_kpi": cal.get("experiment_target_kpi"),
        },
        "extensions": {
            "planner_mode": (ext.get("product") or {}).get("planner_mode"),
            "optimization_gates_enabled": (ext.get("optimization_gates") or {}).get("enabled"),
        },
    }


PROD_EXTENSION_BUNDLE_REQUIRED_TOP_LEVEL = (
    "bundle_version",
    "economics_contract",
    "economics_contract_version",
    "economics_output_metadata",
    "resolved_config_snapshot",
    "config_fingerprint_sha256",
    "config_sha",
    "data_fingerprint",
    "governance",
    "simulation_contract",
    "package_version",
    "artifact_tier",
    "semantic_contract",
    "unsupported_questions",
    "baseline_type",
)

PROD_DECISION_CLI_BUNDLE_REQUIRED_TOP_LEVEL = PROD_EXTENSION_BUNDLE_REQUIRED_TOP_LEVEL + (
    "panel_fingerprint",
    "git_sha",
    "dependency_digest",
    "dependency_lock_digest",
    "config_hash",
    "surface",
    "primary_semantics",
    "artifact_schema_version",
    "dataset_snapshot_id",
    "model_release_id",
    "runtime_policy_hash",
    "python_version",
    "created_at",
    "artifact_sections",
)


def compute_unsupported_questions(cfg: MMMConfig, extension_report: dict[str, Any] | None) -> list[str]:
    """List business questions this run cannot answer decision-grade (transparency)."""
    qs: list[str] = []
    if cfg.framework == Framework.BAYESIAN:
        qs.append("Decision-grade Bayesian uncertainty for prod budget optimization (blocked in prod).")
        qs.append("Bayesian posterior draws as the sole basis for audited dollar ROI (not certified here).")
    if cfg.framework == Framework.RIDGE_BO and cfg.run_environment == RunEnvironment.PROD:
        qs.append("Precise monetary credible intervals on Ridge in prod (point decisions only per contract).")
    qs.append("Exact channel-level dollar attribution from decomposition columns (approximate / not additive dollars).")
    qs.append("Curve-based ROI as the budget truth (curves are diagnostic-only vs full-panel Δμ).")
    er = extension_report or {}
    if not isinstance(er.get("ridge_fit_summary"), dict) or not (er.get("ridge_fit_summary") or {}).get("coef"):
        qs.append("Full-panel Δμ planning without ridge_fit_summary in extension_report.")
    if isinstance(er.get("panel_qa"), dict):
        sev = str(er["panel_qa"].get("max_severity", "")).lower()
        if sev == "warn":
            qs.append("Strict planning under panel_qa warn severity (prod may downgrade release state).")
    return qs


def validate_prod_decision_bundle(
    bundle: dict[str, Any],
    *,
    run_environment: RunEnvironment,
    decision_cli_surface: bool = False,
) -> list[str]:
    """Return human-readable issues; empty means OK for strict prod completeness checks.

    ``decision_cli_surface=True`` applies full lineage + ``artifact_tier=decision`` rules for
    ``mmm decide …`` outputs. Training-time extension bundles use ``decision_cli_surface=False``.
    """
    if run_environment != RunEnvironment.PROD:
        return []
    missing: list[str] = []
    tier = bundle.get("artifact_tier")
    keys = (
        PROD_DECISION_CLI_BUNDLE_REQUIRED_TOP_LEVEL
        if decision_cli_surface
        else PROD_EXTENSION_BUNDLE_REQUIRED_TOP_LEVEL
    )
    for k in keys:
        if k not in bundle or bundle.get(k) in (None, "", {}):
            missing.append(f"missing_or_empty:{k}")
    gov = bundle.get("governance")
    if not isinstance(gov, dict) or not gov:
        missing.append("governance_must_be_non_empty_dict")
    fp = bundle.get("data_fingerprint")
    if not isinstance(fp, dict) or not fp:
        missing.append("data_fingerprint_must_be_non_empty_dict")
    if decision_cli_surface:
        pf = bundle.get("panel_fingerprint")
        if not isinstance(pf, dict) or not pf:
            missing.append("panel_fingerprint_must_be_non_empty_dict")
        if bundle.get("data_fingerprint") != bundle.get("panel_fingerprint"):
            missing.append("panel_fingerprint_must_equal_data_fingerprint")
        if tier != DECISION_TIER_VALUE:
            missing.append("artifact_tier_must_be_decision")
        gh = bundle.get("git_sha")
        if not gh or not isinstance(gh, str):
            missing.append("git_sha_required_for_prod_bundle_set_MMM_GIT_SHA_if_needed")
        dd = bundle.get("dependency_digest")
        if not dd or not isinstance(dd, str):
            missing.append("dependency_digest_required")
    econ = bundle.get("economics_output_metadata")
    if not isinstance(econ, dict):
        missing.append("economics_output_metadata_must_be_dict")
    elif econ.get("computation_mode") is None and econ.get("kpi_level_values_exact") is None:
        missing.append("economics_output_metadata_missing_exactness_fields")
    sem = bundle.get("semantic_contract")
    if not isinstance(sem, dict) or not sem:
        missing.append("semantic_contract_required")
    if not isinstance(bundle.get("unsupported_questions"), list):
        missing.append("unsupported_questions_must_be_list")
    if bundle.get("config_sha") != bundle.get("config_fingerprint_sha256"):
        missing.append("config_sha_must_match_config_fingerprint_sha256")
    if decision_cli_surface and bundle.get("artifact_tier") != DECISION_TIER_VALUE:
        missing.append("artifact_tier_must_be_decision_for_cli")
    return missing


def build_semantic_contract(
    *,
    config: MMMConfig,
    baseline_type: str,
    aggregation_semantics: str | None,
) -> dict[str, Any]:
    return {
        "estimand": "delta_mu_full_panel",
        "aggregation": aggregation_semantics or "mean_mu_over_all_panel_rows_equal_weight",
        "scale": f"model_form={config.model_form.value};target={config.data.target_column}",
        "baseline_definition": baseline_type,
    }


def build_decision_bundle(
    *,
    config: MMMConfig,
    schema: PanelSchema | None = None,
    governance: dict[str, Any] | None = None,
    optimization_gate: dict[str, Any] | None = None,
    calibration_summary: dict[str, Any] | None = None,
    simulation_contract: dict[str, Any] | None = None,
    data_fingerprint: dict[str, Any] | None = None,
    config_fingerprint_sha256: str | None = None,
    uncertainty_mode: str = "point",
    decision_safe: bool | None = None,
    governance_passed: bool | None = None,
    optimizer_success: bool | None = None,
    baseline_type: str | None = "bau",
    model_summary: dict[str, Any] | None = None,
    decision_safety_flags: dict[str, Any] | None = None,
    economics_surface: EconomicsOutputSurface = "other",
    panel_qa: dict[str, Any] | None = None,
    experiment_matching: dict[str, Any] | None = None,
    model_release: dict[str, Any] | None = None,
    package_version: str | None = None,
    simulation_json: dict[str, Any] | None = None,
    extension_report: dict[str, Any] | None = None,
    artifact_tier: str | None = None,
    approximate: bool | None = None,
    not_for_budgeting: bool | None = None,
    runtime_policy_hash: str | None = None,
    model_release_id: str | None = None,
) -> dict[str, Any]:
    """Assemble required metadata for optimize-budget / canonical simulate / decision reports."""
    econ = build_economics_contract(config)
    cfg_blob = json.dumps(config.model_dump_resolved(), sort_keys=True, default=str)
    cf_sha = config_fingerprint_sha256 or hashlib.sha256(cfg_blob.encode()).hexdigest()
    bt = baseline_type if baseline_type not in (None, "") else "bau"
    econ_meta = economics_output_metadata(
        config,
        uncertainty_mode=uncertainty_mode,
        surface=economics_surface,
        baseline_type=bt,
        decision_safe=decision_safe,
    )
    validate_business_economics_metadata(
        econ_meta,
        require_specific_baseline=economics_surface in ("full_model_simulation", "replay_calibration"),
        require_decision_safe_bool=economics_surface == "full_model_simulation",
    )
    agg_sem = None
    if simulation_json:
        agg_sem = simulation_json.get("aggregation_semantics")
    sem = build_semantic_contract(config=config, baseline_type=str(bt), aggregation_semantics=agg_sem)

    tier = artifact_tier
    if tier is None:
        if economics_surface == "full_model_simulation":
            tier = DECISION_TIER_VALUE
        elif economics_surface in ("curve_diagnostic", "decomposition"):
            tier = "diagnostic"
        else:
            tier = "research"

    strict_lineage = str(tier) == DECISION_TIER_VALUE
    rph = runtime_policy_hash
    if rph is None:
        from mmm.governance.policy import runtime_policy_from_config

        rph = runtime_policy_from_config(config).policy_fingerprint()
    mr_id_resolved = model_release_id
    if mr_id_resolved is None and isinstance(model_release, dict):
        rid = str(model_release.get("release_id") or model_release.get("model_release_id") or "").strip()
        mr_id_resolved = rid or None
    if mr_id_resolved is None and isinstance(model_release, dict) and model_release:
        mr_id_resolved = str(abs(hash(tuple(sorted((str(k), str(v)) for k, v in model_release.items())))))
    if strict_lineage and not str(mr_id_resolved or "").strip():
        mr_id_resolved = f"model_release_fallback_from_config_hash:{cf_sha[:32]}"
    panel_fp: dict[str, Any] = data_fingerprint if isinstance(data_fingerprint, dict) else {}
    lineage_payload = build_decision_tier_lineage_payload(
        config=config,
        config_fingerprint_sha256=cf_sha,
        panel_fingerprint=panel_fp,
        runtime_policy_hash=rph,
        model_release_id=mr_id_resolved,
        artifact_schema_version=ARTIFACT_BUNDLE_SCHEMA_VERSION,
        package_version=package_version,
        strict=strict_lineage,
    )

    approx = approximate
    if approx is None:
        approx = bool(
            econ_meta.get("computation_mode") == "approximate"
            or econ_meta.get("kpi_level_values_exact") is False
        )

    nfb = not_for_budgeting
    if nfb is None:
        nfb = tier != DECISION_TIER_VALUE

    uq = compute_unsupported_questions(config, extension_report)

    primary_sem = (
        DecisionSemantics.FULL_PANEL_DELTA_MU.value
        if economics_surface == "full_model_simulation"
        else DecisionSemantics.APPROX_CURVE.value
    )
    surf = "decision" if tier == DECISION_TIER_VALUE else ("diagnostic" if tier == "diagnostic" else "research")
    _sum_safety = SafetyFlags(
        decision_safe=bool(decision_safe),
        prod_safe=bool(decision_safe and config.run_environment == RunEnvironment.PROD),
        approximate=bool(approx),
        unsupported_for=[],
    )
    roi_q = ROIApproxQuantityResult()
    decomp_q = DecompositionQuantityResult()
    curve_q = CurveQuantityResult()
    artifact_sections: dict[str, Any] = {
        "decision_summary": {
            "semantics": DecisionSemantics.FULL_PANEL_DELTA_MU.value,
            "safety": _sum_safety.model_dump(),
        },
        "roi_bridge": roi_q.section_dict(),
        "decomposition": decomp_q.section_dict(),
        "curve_diagnostic": curve_q.section_dict(),
    }

    out: dict[str, Any] = {
        "bundle_version": "mmm_decision_bundle_v2",
        "economics_contract": econ,
        "economics_version": econ.get("contract_version"),
        "economics_contract_version": str(econ.get("contract_version") or ""),
        "economics_output_metadata": econ_meta,
        "resolved_config_snapshot": decision_resolved_config_snapshot(config),
        "planner_mode": config.extensions.product.planner_mode,
        "run_environment": config.run_environment.value,
        "framework": config.framework.value,
        "uncertainty_mode": uncertainty_mode,
        "decision_safe": decision_safe,
        "governance_passed": governance_passed,
        "optimizer_success": optimizer_success,
        "config_fingerprint_sha256": cf_sha,
        "config_sha": cf_sha,
        "config_hash": cf_sha,
        "package_version": package_version or str(mmm_version.__version__),
        "baseline_policy_default": "bau",
        "baseline_type": bt,
        "canonical_economic_quantity": econ.get("canonical_economic_quantity"),
        "semantic_contract": sem,
        "artifact_tier": tier,
        "tier": tier,
        "surface": surf,
        "primary_semantics": primary_sem,
        "artifact_schema_version": ARTIFACT_BUNDLE_SCHEMA_VERSION,
        "approximate": approx,
        "not_for_budgeting": nfb,
        "unsupported_questions": uq,
        "artifact_sections": artifact_sections,
        "kpi_column": schema.target_column if schema is not None else config.data.target_column,
        "kpi_unit_semantics": str(econ_meta.get("kpi_unit_semantics") or "same_units_as_training_target_column"),
    }
    out.update(lineage_payload)
    out["config_fingerprint_sha256"] = cf_sha
    out["config_sha"] = cf_sha
    out["config_hash"] = cf_sha
    if config.framework == Framework.BAYESIAN:
        out["bayesian_prod_policy"] = {
            "prod_decisioning_allowed": False,
            "allowed_surfaces": ["research", "diagnostic"],
            "reason_blocked": "current policy disallows Bayesian decisioning on prod decision surfaces",
        }
    if isinstance(extension_report, dict) and extension_report.get("_identifiability_waiver_applied"):
        wj = extension_report["_identifiability_waiver_applied"]
        out["identifiability_waiver_applied"] = wj
        if isinstance(wj, dict) and wj.get("waiver_id"):
            out["identifiability_waiver_id"] = wj["waiver_id"]
    if schema is not None:
        out["schema_channels"] = list(schema.channel_columns)
        out["target_kpi"] = schema.target_column
    if governance is not None:
        out["governance"] = governance
    if optimization_gate is not None:
        out["optimization_gate"] = optimization_gate
    if calibration_summary is not None:
        out["calibration_summary"] = calibration_summary
    if simulation_contract is not None:
        out["simulation_contract"] = simulation_contract
    if data_fingerprint is not None:
        out["data_fingerprint"] = data_fingerprint
        out["panel_fingerprint"] = data_fingerprint
    if model_summary is not None:
        out["model_summary"] = model_summary
    if decision_safety_flags is not None:
        out["decision_safety_flags"] = decision_safety_flags
    if panel_qa is not None:
        out["panel_qa"] = panel_qa
    if experiment_matching is not None:
        out["experiment_matching"] = experiment_matching
    if model_release is not None:
        out["model_release"] = model_release
    return out
