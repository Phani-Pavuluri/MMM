"""Unified machine-readable bundle for decision-facing CLI/API outputs."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from mmm.config.schema import MMMConfig
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
) -> dict[str, Any]:
    """Assemble required metadata for optimize-budget / canonical simulate / decision reports."""
    econ = build_economics_contract(config)
    cfg_blob = json.dumps(config.model_dump_resolved(), sort_keys=True, default=str)
    cf_sha = config_fingerprint_sha256 or hashlib.sha256(cfg_blob.encode()).hexdigest()
    econ_meta = economics_output_metadata(
        config,
        uncertainty_mode=uncertainty_mode,
        surface=economics_surface,
        baseline_type=baseline_type,
        decision_safe=decision_safe,
    )
    validate_business_economics_metadata(
        econ_meta,
        require_specific_baseline=economics_surface in ("full_model_simulation", "replay_calibration"),
        require_decision_safe_bool=economics_surface == "full_model_simulation",
    )
    out: dict[str, Any] = {
        "bundle_version": "mmm_decision_bundle_v1",
        "economics_contract": econ,
        "economics_version": econ.get("contract_version"),
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
        "baseline_policy_default": "bau",
        "canonical_economic_quantity": econ.get("canonical_economic_quantity"),
    }
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
    if model_summary is not None:
        out["model_summary"] = model_summary
    if decision_safety_flags is not None:
        out["decision_safety_flags"] = decision_safety_flags
    return out
