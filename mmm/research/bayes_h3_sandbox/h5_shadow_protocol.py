"""Bayes-H5e real-panel shadow-run protocol helpers (research only — no production execution)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mmm.research.bayes_h3_sandbox.fencing import H5_MODEL_SPEC_VERSION
from mmm.research.bayes_h3_sandbox.h5_trust_diagnostics import (
    FIELDS_EXCLUDED_FROM_PRODUCTION,
    research_production_flags,
)

PROTOCOL_ID = "INV-H5E_REAL_PANEL_SHADOW_RUN_PROTOCOL"
SCHEMA_ID = "BAYES_H5E_SHADOW_RUN_SCHEMA_20260601"
SCHEMA_VERSION = "bayes_h5e_shadow_run_schema_v1"
DEFAULT_SCHEMA_PATH = Path("docs/05_validation/archives/BAYES_H5E_SHADOW_RUN_SCHEMA_20260601.json")

REQUIRED_LINEAGE_FIELDS: frozenset[str] = frozenset(
    {
        "run_id",
        "dataset_snapshot_id",
        "panel_id",
        "data_snapshot_hash",
        "mmm_config_hash",
        "run_environment",
        "sandbox_entrypoint",
        "model_spec_version",
        "enable_h5_sandbox",
        "research_only",
    }
)

REQUIRED_RECORD_FIELDS: frozenset[str] = frozenset(
    {
        *REQUIRED_LINEAGE_FIELDS,
        "transform_config",
        "calibration_signal_summary",
        "posterior_diagnostics",
        "trust_report_candidate_diagnostics",
        "ridge_comparison",
        "production_flags",
        "label",
    }
)

OPTIONAL_COMPARISON_FIELDS: frozenset[str] = frozenset(
    {
        "recovery_style_diagnostics",
        "geox_cls_comparison",
    }
)

FORBIDDEN_OUTPUT_FIELDS: frozenset[str] = frozenset(
    {
        *FIELDS_EXCLUDED_FROM_PRODUCTION,
        "optimizer_input",
        "production_trust_report",
        "budget_recommendation_payload",
    }
)

TRANSFORM_CONFIG_REQUIRED_KEYS: frozenset[str] = frozenset(
    {
        "media_transforms_by_channel",
        "transform_registry_id",
        "transform_mismatch_mode",
    }
)


class H5ShadowProtocolError(ValueError):
    """H5 shadow-run protocol validation failed — fail closed."""


def load_shadow_run_schema(path: str | Path | None = None) -> dict[str, Any]:
    p = Path(path or DEFAULT_SCHEMA_PATH)
    if not p.is_file():
        raise H5ShadowProtocolError(f"shadow-run schema not found: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def validate_transform_config(transform_config: dict[str, Any] | None) -> None:
    if not transform_config or not isinstance(transform_config, dict):
        raise H5ShadowProtocolError("transform_config is required and must be a non-empty object")
    missing = TRANSFORM_CONFIG_REQUIRED_KEYS - set(transform_config.keys())
    if missing:
        raise H5ShadowProtocolError(f"transform_config missing keys: {sorted(missing)}")


def validate_shadow_run_record(record: dict[str, Any]) -> None:
    """Validate one shadow-run output record (research only)."""
    missing = REQUIRED_RECORD_FIELDS - set(record.keys())
    if missing:
        raise H5ShadowProtocolError(f"shadow-run record missing required fields: {sorted(missing)}")

    if not str(record.get("dataset_snapshot_id", "")).strip():
        raise H5ShadowProtocolError("dataset_snapshot_id must be non-empty")

    validate_transform_config(record.get("transform_config"))

    if record.get("model_spec_version") != H5_MODEL_SPEC_VERSION:
        raise H5ShadowProtocolError(
            f"model_spec_version must be {H5_MODEL_SPEC_VERSION!r} for H5 shadow runs"
        )
    if record.get("enable_h5_sandbox") is not True:
        raise H5ShadowProtocolError("enable_h5_sandbox must be true")
    if record.get("research_only") is not True:
        raise H5ShadowProtocolError("research_only must be true")

    flags = record.get("production_flags") or {}
    expected = research_production_flags()
    for key, val in expected.items():
        if flags.get(key) is not val:
            raise H5ShadowProtocolError(f"production_flags.{key} must be {val!r}")

    for forbidden in FORBIDDEN_OUTPUT_FIELDS:
        if forbidden in record and record.get(forbidden) is not None:
            raise H5ShadowProtocolError(f"forbidden production field present: {forbidden!r}")

    ridge = record.get("ridge_comparison") or {}
    if ridge.get("decision_grade") is True:
        raise H5ShadowProtocolError("ridge_comparison must remain diagnostic-only in shadow runs")
    if ridge.get("used_for_optimizer") is True:
        raise H5ShadowProtocolError("ridge_comparison.used_for_optimizer must not be true")

    trust = record.get("trust_report_candidate_diagnostics") or {}
    if trust.get("production_trust_report") is not None:
        raise H5ShadowProtocolError("trust_report_candidate_diagnostics is research-only, not prod TrustReport")

    cal = record.get("calibration_signal_summary") or {}
    if cal.get("likelihood_integrated") is True:
        raise H5ShadowProtocolError(
            "calibration_signal_summary.likelihood_integrated must be false for H5e shadow protocol v1"
        )


def validate_shadow_run_schema_document(schema: dict[str, Any]) -> None:
    """Validate the published shadow-run schema JSON artifact."""
    required = {
        "schema_id",
        "schema_version",
        "model_spec_version",
        "required_lineage_fields",
        "required_record_fields",
        "excluded_fields",
        "production_flags",
        "example_shadow_run_record",
    }
    missing = required - set(schema.keys())
    if missing:
        raise H5ShadowProtocolError(f"schema document missing keys: {sorted(missing)}")

    if schema.get("schema_id") != SCHEMA_ID:
        raise H5ShadowProtocolError(f"schema_id must be {SCHEMA_ID!r}")

    flags = schema.get("production_flags") or {}
    for key, val in research_production_flags().items():
        if flags.get(key) is not val:
            raise H5ShadowProtocolError(f"schema production_flags.{key} must be {val!r}")

    validate_shadow_run_record(schema["example_shadow_run_record"])


def build_dry_run_shadow_run_record(
    *,
    run_id: str = "BAYES-H5E-SHADOW-DRY-RUN-0001",
    dataset_snapshot_id: str = "dry-run-snapshot-v1",
    panel_id: str = "dry-run-panel-v1",
) -> dict[str, Any]:
    """Construct a minimal valid shadow-run record for schema/tests (no MCMC execution)."""
    record: dict[str, Any] = {
        "run_id": run_id,
        "dataset_snapshot_id": dataset_snapshot_id,
        "panel_id": panel_id,
        "data_snapshot_hash": "sha256:dry-run-panel-placeholder",
        "mmm_config_hash": "sha256:dry-run-config-placeholder",
        "run_environment": "research",
        "sandbox_entrypoint": "mmm.research.bayes_h3_sandbox.run_sandbox_fit",
        "model_spec_version": H5_MODEL_SPEC_VERSION,
        "enable_h5_sandbox": True,
        "research_only": True,
        "label": "RESEARCH ONLY — NOT DECISION GRADE",
        "transform_config": {
            "transform_registry_id": "bayes_h5_media_transform_registry_v1",
            "media_transforms_by_channel": {"tv": "geometric_adstock", "search": "identity"},
            "transform_params_by_channel": {"tv": {"decay": 0.7}},
            "transform_mismatch_mode": "aligned",
            "policy_note": "Declared per client media spec; shadow run does not auto-infer prod FE transforms",
        },
        "calibration_signal_summary": {
            "signals_present": [],
            "likelihood_integrated": False,
            "stub_slots_only": True,
            "note": "CalibrationSignal ingress per platform contract; no prod likelihood in H5e v1",
        },
        "posterior_diagnostics": {
            "outputs_are_diagnostic_only": True,
            "convergence_diagnostics": {"status": "not_executed_dry_run"},
            "pooling_diagnostics": {"status": "not_executed_dry_run"},
        },
        "recovery_style_diagnostics": {
            "mode": "report_only",
            "known_truth_available": False,
            "note": "Real panels have no synthetic truth; compare Ridge/GeoX only",
        },
        "trust_report_candidate_diagnostics": {
            "mapping_version": "bayes_h5d_trust_diagnostic_mapping_v1",
            "warning_codes": ["h5:production:block"],
            "trust_report_candidate_fields": {"transform_alignment_status": "aligned"},
            "production_trust_report": None,
        },
        "ridge_comparison": {
            "ridge_run_id": None,
            "ridge_model_form": "semi_log",
            "comparison_mode": "diagnostic_only",
            "used_for_optimizer": False,
            "decision_grade": False,
            "metrics": {
                "note": "Compare coefficient stability / residual diagnostics vs H5 posterior summaries",
            },
        },
        "geox_cls_comparison": {
            "available": False,
            "experiment_evidence_refs": [],
            "note": "Optional cross-check vs historical GeoX/CLS CalibrationSignal or replay evidence when documented",
        },
        "excluded_fields": list(FORBIDDEN_OUTPUT_FIELDS),
        "production_flags": research_production_flags(),
        "outputs_are_diagnostic_only": True,
    }
    validate_shadow_run_record(record)
    return record


def build_shadow_run_schema_document() -> dict[str, Any]:
    """Build the canonical H5e shadow-run schema JSON (includes dry-run example)."""
    example = build_dry_run_shadow_run_record()
    doc: dict[str, Any] = {
        "schema_id": SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "protocol_investigation": PROTOCOL_ID,
        "model_spec_version": H5_MODEL_SPEC_VERSION,
        "label": "RESEARCH ONLY — NOT DECISION GRADE",
        "research_only": True,
        "purpose": (
            "Schema for Bayes-H5 real-panel shadow runs. Outputs are diagnostic only and must not "
            "feed optimizer, DecisionSurface, recommendations, or production TrustReport."
        ),
        "required_lineage_fields": sorted(REQUIRED_LINEAGE_FIELDS),
        "required_record_fields": sorted(REQUIRED_RECORD_FIELDS),
        "optional_comparison_fields": sorted(OPTIONAL_COMPARISON_FIELDS),
        "excluded_fields": sorted(FORBIDDEN_OUTPUT_FIELDS),
        "production_flags": research_production_flags(),
        "transform_config_policy": {
            "registry_id": "bayes_h5_media_transform_registry_v1",
            "must_be_declared_explicitly": True,
            "must_not_use_production_feature_engineering_pipeline": True,
            "aligned_vs_mismatch_documented": True,
        },
        "calibration_signal_policy": {
            "ingress": "CalibrationSignal stubs only in H5e v1",
            "likelihood_integrated": False,
            "no_prod_decisioning": True,
        },
        "example_shadow_run_record": example,
        "note": "Example record is dry-run only (no MCMC). Authorized execution is a separate step.",
    }
    validate_shadow_run_schema_document(doc)
    return doc


def write_shadow_run_schema_artifact(path: str | Path | None = None) -> dict[str, Any]:
    out = Path(path or DEFAULT_SCHEMA_PATH)
    doc = build_shadow_run_schema_document()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    doc["artifact_path"] = str(out)
    return doc
