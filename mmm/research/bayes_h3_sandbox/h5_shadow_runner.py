"""Bayes-H5 real-panel shadow-run execution harness (research only)."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from mmm.config.schema import BayesianBackend, Framework, MMMConfig, ModelForm, PoolingMode, RunEnvironment
from mmm.data.schema import PanelSchema, validate_panel
from mmm.research.bayes_h3_sandbox.entrypoint import SANDBOX_ENTRYPOINT, run_sandbox_fit
from mmm.research.bayes_h3_sandbox.fencing import H5_MODEL_SPEC_VERSION
from mmm.research.bayes_h3_sandbox.fixtures import (
    TOY_CALIBRATION_SIGNAL_STUB,
    TOY_GEO_HIERARCHY,
    TOY_SEED,
    toy_sandbox_bundle,
)
from mmm.research.bayes_h3_sandbox.h5_shadow_protocol import (
    FORBIDDEN_OUTPUT_FIELDS,
    H5ShadowProtocolError,
    research_production_flags,
    validate_shadow_run_record,
    validate_transform_config,
)
from mmm.research.bayes_h3_sandbox.h5_trust_diagnostics import (
    MAPPING_VERSION as H5D_MAPPING_VERSION,
    PANEL_CONTEXT_REAL,
    PANEL_CONTEXT_SYNTHETIC_FIXTURE,
    build_sampler_diagnostics,
    build_shadow_trust_diagnostics,
)
from mmm.research.bayes_h3_sandbox.h5_real_panel_preprocessing import (
    H5RealPanelPreprocessingError,
    apply_channel_policy,
    validate_collinearity_config,
)
from mmm.research.bayes_h3_sandbox.labels import RESEARCH_ONLY_LABEL
from mmm.research.bayes_h3_sandbox.recovery_worlds import SAMPLER_EXTENDED, SAMPLER_FAST

HARNESS_VERSION = "bayes_h5m_shadow_runner_v1"
DEFAULT_DRY_RUN_ARTIFACT = Path("docs/05_validation/archives/BAYES_H5F_SHADOW_RUN_DRY_RUN_20260601.json")
ARCHIVES_DIR = Path("docs/05_validation/archives")

FIXTURE_PANEL_ID = "synthetic_h5_shadow_fixture"
FIXTURE_DATASET_SNAPSHOT_ID = "synthetic_fixture_only"

DEFAULT_FIXTURE_TRANSFORM_CONFIG: dict[str, Any] = {
    "transform_registry_id": "bayes_h5_media_transform_registry_v1",
    "media_transforms_by_channel": {"tv": "identity", "search": "identity"},
    "transform_params_by_channel": {},
    "transform_mismatch_mode": "aligned",
    "policy_note": "Synthetic fixture — not real-panel evidence",
}


class H5ShadowRunnerError(ValueError):
    """H5 shadow-run harness failed — fail closed."""


@dataclass(frozen=True)
class ShadowRunRequest:
    panel_id: str
    dataset_snapshot_id: str
    transform_config: dict[str, Any]
    model_spec_version: str = H5_MODEL_SPEC_VERSION
    enable_h5_sandbox: bool = True
    research_only: bool = True
    panel_path: str | Path | None = None
    panel_df: pd.DataFrame | None = None
    output_path: str | Path | None = None
    fast_mcmc: bool = True
    extended_mcmc: bool = False
    execute_fit: bool = True
    artifact_type: str = "real_panel_shadow_artifact"
    calibration_signals_stub: list[dict[str, Any]] | None = None
    geo_hierarchy_mapping: dict[str, Any] | None = None
    ridge_comparison: dict[str, Any] | None = None
    geox_cls_comparison: dict[str, Any] | None = None
    run_id: str | None = None
    requested_production_flags: dict[str, bool] | None = None
    policy_id: str | None = None
    source_policy_path: str | None = None
    geometry_config: dict[str, Any] | None = None
    sandbox_model_overrides: dict[str, Any] | None = None
    sampler_profile_applied: dict[str, Any] | None = None
    channel_policy_declared: dict[str, Any] | None = None
    channel_policy_applied: dict[str, Any] | None = None


def _sha256_bytes(payload: bytes) -> str:
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _sha256_panel(df: pd.DataFrame) -> str:
    return _sha256_bytes(df.to_csv(index=False).encode("utf-8"))


def _sha256_config(config: MMMConfig) -> str:
    payload = json.dumps(config.model_dump(), sort_keys=True, default=str).encode("utf-8")
    return _sha256_bytes(payload)


def load_transform_config(value: str | Path | dict[str, Any]) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    p = Path(value)
    if not p.is_file():
        raise H5ShadowRunnerError(f"transform_config path not found: {p}")
    loaded = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise H5ShadowRunnerError("transform_config JSON must be an object")
    return loaded


def load_panel_from_path(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.is_file():
        raise H5ShadowRunnerError(f"panel path not found: {p}")
    if p.suffix.lower() == ".parquet":
        return pd.read_parquet(p)
    if p.suffix.lower() in (".csv", ".tsv"):
        return pd.read_csv(p)
    raise H5ShadowRunnerError(f"unsupported panel format: {p.suffix!r} (use .csv or .parquet)")


def reject_requested_production_flags(flags: dict[str, bool] | None) -> None:
    if not flags:
        return
    for key in ("hard_gate", "production_promotion", "approved_for_prod", "prod_decisioning_allowed"):
        if flags.get(key) is True:
            raise H5ShadowRunnerError(f"requested production flag {key!r}=true is not allowed for H5 shadow runs")


def validate_shadow_run_request(request: ShadowRunRequest) -> None:
    if not str(request.panel_id or "").strip():
        raise H5ShadowRunnerError("panel_id is required")
    if not str(request.dataset_snapshot_id or "").strip():
        raise H5ShadowRunnerError("dataset_snapshot_id is required")
    try:
        validate_transform_config(request.transform_config)
        validate_collinearity_config(request.transform_config.get("channel_policy"))
    except (H5ShadowProtocolError, H5ShadowRunnerError, H5RealPanelPreprocessingError) as exc:
        raise H5ShadowRunnerError(str(exc)) from exc
    if request.model_spec_version != H5_MODEL_SPEC_VERSION:
        raise H5ShadowRunnerError(f"model_spec_version must be {H5_MODEL_SPEC_VERSION!r}")
    if request.enable_h5_sandbox is not True:
        raise H5ShadowRunnerError("enable_h5_sandbox must be true")
    if request.research_only is not True:
        raise H5ShadowRunnerError("research_only must be true")
    if request.panel_df is None and request.panel_path is None and request.artifact_type != "dry_run_shadow_artifact":
        raise H5ShadowRunnerError("panel_path or panel_df is required")
    reject_requested_production_flags(request.requested_production_flags)


def default_output_path(panel_id: str, *, artifact_type: str, run_date: date | None = None) -> Path:
    if artifact_type == "dry_run_shadow_artifact":
        return DEFAULT_DRY_RUN_ARTIFACT
    d = (run_date or date.today()).strftime("%Y%m%d")
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", panel_id).strip("_") or "panel"
    return ARCHIVES_DIR / f"BAYES_H5F_SHADOW_RUN_{safe}_{d}.json"


def resolve_sampler_profile(*, fast_mcmc: bool, extended_mcmc: bool) -> tuple[str, dict[str, Any]]:
    if extended_mcmc:
        return "extended", dict(SAMPLER_EXTENDED)
    if fast_mcmc:
        return "fast", dict(SAMPLER_FAST)
    return "default", {}


def _config_from_panel(
    df: pd.DataFrame,
    schema: PanelSchema,
    *,
    fast_mcmc: bool,
    extended_mcmc: bool = False,
    nuts_seed: int = TOY_SEED,
    sampler_overrides: dict[str, Any] | None = None,
) -> tuple[MMMConfig, str]:
    bayesian: dict[str, Any] = {
        "backend": BayesianBackend.PYMC,
        "nuts_seed": nuts_seed,
        "prior_predictive_draws": 0,
        "posterior_predictive_draws": 0,
    }
    if sampler_overrides:
        profile_name = "policy"
        bayesian.update(sampler_overrides)
    else:
        profile_name, sampler = resolve_sampler_profile(fast_mcmc=fast_mcmc, extended_mcmc=extended_mcmc)
        bayesian.update(sampler)
    cfg = MMMConfig(
        framework=Framework.BAYESIAN,
        run_environment=RunEnvironment.RESEARCH,
        model_form=ModelForm.SEMI_LOG,
        pooling=PoolingMode.PARTIAL,
        data={
            "path": None,
            "geo_column": schema.geo_column,
            "week_column": schema.week_column,
            "target_column": schema.target_column,
            "channel_columns": list(schema.channel_columns),
            "control_columns": list(schema.control_columns),
        },
        bayesian=bayesian,
    )
    return cfg, profile_name


def _resolve_panel_schema(transform_config: dict[str, Any]) -> tuple[str, str, str]:
    """Resolve geo/week/target columns from transform_config.panel_schema or defaults."""
    raw = transform_config.get("panel_schema") or {}
    if raw and not isinstance(raw, dict):
        raise H5ShadowRunnerError("transform_config.panel_schema must be an object when present")
    geo = str(raw.get("geo_column") or "geo_id")
    week = str(raw.get("week_column") or "week")
    target = str(raw.get("target_column") or "y")
    return geo, week, target


def _infer_schema(df: pd.DataFrame, transform_config: dict[str, Any]) -> PanelSchema:
    channels = tuple(transform_config.get("media_transforms_by_channel", {}).keys())
    if not channels:
        raise H5ShadowRunnerError("transform_config.media_transforms_by_channel must list channel columns")
    geo_col, week_col, target_col = _resolve_panel_schema(transform_config)
    for col in (geo_col, week_col, target_col):
        if col not in df.columns:
            raise H5ShadowRunnerError(f"panel missing required column {col!r}")
    for ch in channels:
        if ch not in df.columns:
            raise H5ShadowRunnerError(f"panel missing channel column {ch!r}")
    controls = tuple(transform_config.get("control_columns") or ())
    for ctrl in controls:
        if ctrl not in df.columns:
            raise H5ShadowRunnerError(f"panel missing control column {ctrl!r}")
    return PanelSchema(geo_col, week_col, target_col, channels, controls)


def _resolve_real_panel_inputs(
    df: pd.DataFrame,
    transform_config: dict[str, Any],
    *,
    channel_policy_declared: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, PanelSchema, dict[str, Any], dict[str, Any] | None]:
    schema = _infer_schema(df, transform_config)
    policy_record: dict[str, Any] | None = None
    if transform_config.get("channel_policy"):
        df, schema, transform_config, policy_record = apply_channel_policy(df, schema, transform_config)
        if channel_policy_declared:
            from mmm.research.bayes_h3_sandbox.h5_shadow_policy import assert_channel_policy_matches_explicit

            assert_channel_policy_matches_explicit(policy_record, channel_policy_declared)
    return df, schema, transform_config, policy_record


def _panel_context_for_request(request: ShadowRunRequest) -> str:
    if request.artifact_type == "dry_run_shadow_artifact":
        return PANEL_CONTEXT_SYNTHETIC_FIXTURE
    return PANEL_CONTEXT_REAL


def build_shadow_run_record(
    request: ShadowRunRequest,
    *,
    fit_artifact: dict[str, Any] | None,
    schema: PanelSchema,
    config: MMMConfig,
    panel_hash: str,
    config_hash: str,
    sampler_profile: str = "fast",
) -> dict[str, Any]:
    panel_context = _panel_context_for_request(request)
    run_id = request.run_id or f"BAYES-H5H-SHADOW-{request.panel_id}-{date.today().strftime('%Y%m%d')}"
    cal_stub = request.calibration_signals_stub if request.calibration_signals_stub is not None else []

    if fit_artifact is not None:
        posterior = {
            "outputs_are_diagnostic_only": True,
            "posterior_summary": fit_artifact.get("posterior_summary"),
            "convergence_diagnostics": fit_artifact.get("convergence_diagnostics"),
            "pooling_diagnostics": fit_artifact.get("pooling_diagnostics"),
            "h5_transform_diagnostics": fit_artifact.get("h5_transform_diagnostics"),
            "diagnostic_trust_report_kind": (fit_artifact.get("diagnostic_trust_report") or {}).get(
                "trust_report_kind"
            ),
        }
        sampler_diag = build_sampler_diagnostics(fit_artifact, config, sampler_profile=sampler_profile)
        trust_diag = build_shadow_trust_diagnostics(
            fit_artifact,
            request.transform_config,
            panel_context=panel_context,
            convergence_status=sampler_diag["convergence_status"],
        )
    else:
        posterior = {
            "outputs_are_diagnostic_only": True,
            "convergence_diagnostics": {"status": "not_executed"},
            "pooling_diagnostics": {"status": "not_executed"},
        }
        sampler_diag = {"convergence_status": "not_executed", "evidence_promotion_allowed": False}
        trust_diag = {
            "mapping_version": H5D_MAPPING_VERSION,
            "warning_codes": ["h5:production:block", "h5:evidence:blocked"],
            "trust_report_candidate_fields": {
                "transform_alignment_status": "aligned",
                "panel_context": panel_context,
                "convergence_status": "not_executed",
                "evidence_promotion_allowed": False,
            },
            "production_trust_report": None,
        }

    record: dict[str, Any] = {
        "run_id": run_id,
        "dataset_snapshot_id": request.dataset_snapshot_id,
        "panel_id": request.panel_id,
        "data_snapshot_hash": panel_hash,
        "mmm_config_hash": config_hash,
        "run_environment": RunEnvironment.RESEARCH.value,
        "sandbox_entrypoint": SANDBOX_ENTRYPOINT,
        "model_spec_version": request.model_spec_version,
        "enable_h5_sandbox": True,
        "research_only": True,
        "label": RESEARCH_ONLY_LABEL,
        "transform_config": dict(request.transform_config),
        "calibration_signal_summary": {
            "signals_present": list(cal_stub),
            "likelihood_integrated": False,
            "stub_slots_only": True,
        },
        "posterior_diagnostics": posterior,
        "recovery_style_diagnostics": {
            "mode": "report_only",
            "known_truth_available": False,
            "note": "Real panels have no synthetic truth; use Ridge/GeoX comparison only",
        },
        "trust_report_candidate_diagnostics": trust_diag,
        "ridge_comparison": request.ridge_comparison
        or {
            "ridge_run_id": None,
            "comparison_mode": "diagnostic_only",
            "used_for_optimizer": False,
            "decision_grade": False,
            "metrics": {"note": "Optional Ridge fit on same dataset_snapshot_id"},
        },
        "geox_cls_comparison": request.geox_cls_comparison
        or {
            "available": False,
            "experiment_evidence_refs": [],
            "note": "Optional historical GeoX/CLS evidence cross-check",
        },
        "excluded_fields": sorted(FORBIDDEN_OUTPUT_FIELDS),
        "production_flags": research_production_flags(),
        "outputs_are_diagnostic_only": True,
    }
    if panel_context == PANEL_CONTEXT_REAL:
        record["real_panel_diagnostics"] = {
            "panel_context": panel_context,
            "sampler_diagnostics": sampler_diag,
            "convergence_status": sampler_diag.get("convergence_status"),
            "evidence_promotion_allowed": bool(sampler_diag.get("evidence_promotion_allowed")),
            "generative_transform_truth": "unknown",
            "note": "Real-panel shadow diagnostics — not synthetic generative truth",
        }
    validate_shadow_run_record(record)
    return record


def build_shadow_run_artifact(
    request: ShadowRunRequest,
    *,
    record: dict[str, Any] | None = None,
) -> dict[str, Any]:
    validate_shadow_run_request(request)
    sampler_profile = "fast"

    if request.artifact_type == "dry_run_shadow_artifact":
        cfg, schema, df = toy_sandbox_bundle(fast_mcmc=request.fast_mcmc and not request.extended_mcmc)
        sampler_profile = resolve_sampler_profile(
            fast_mcmc=request.fast_mcmc and not request.extended_mcmc,
            extended_mcmc=request.extended_mcmc,
        )[0]
        if request.transform_config == DEFAULT_FIXTURE_TRANSFORM_CONFIG:
            transform_config = DEFAULT_FIXTURE_TRANSFORM_CONFIG
        else:
            transform_config = request.transform_config
        panel_id = request.panel_id or FIXTURE_PANEL_ID
        snapshot_id = request.dataset_snapshot_id or FIXTURE_DATASET_SNAPSHOT_ID
        request = ShadowRunRequest(
            panel_id=panel_id,
            dataset_snapshot_id=snapshot_id,
            transform_config=transform_config,
            model_spec_version=request.model_spec_version,
            enable_h5_sandbox=request.enable_h5_sandbox,
            research_only=request.research_only,
            panel_df=df,
            output_path=request.output_path,
            fast_mcmc=request.fast_mcmc,
            extended_mcmc=request.extended_mcmc,
            execute_fit=request.execute_fit,
            artifact_type="dry_run_shadow_artifact",
            calibration_signals_stub=TOY_CALIBRATION_SIGNAL_STUB,
            geo_hierarchy_mapping=TOY_GEO_HIERARCHY,
            ridge_comparison=request.ridge_comparison,
            geox_cls_comparison=request.geox_cls_comparison,
            run_id=request.run_id,
            requested_production_flags=request.requested_production_flags,
        )
    else:
        if request.panel_df is not None:
            df = request.panel_df.copy()
        else:
            df = load_panel_from_path(request.panel_path)  # type: ignore[arg-type]
        transform_config = dict(request.transform_config)
        df, schema, transform_config, policy_record = _resolve_real_panel_inputs(
            df,
            transform_config,
            channel_policy_declared=request.channel_policy_declared,
        )
        sampler_ov = None
        if request.sampler_profile_applied:
            sampler_ov = {
                k: request.sampler_profile_applied[k]
                for k in ("draws", "tune", "chains", "target_accept")
                if k in request.sampler_profile_applied
            }
        request = ShadowRunRequest(
            panel_id=request.panel_id,
            dataset_snapshot_id=request.dataset_snapshot_id,
            transform_config=transform_config,
            model_spec_version=request.model_spec_version,
            enable_h5_sandbox=request.enable_h5_sandbox,
            research_only=request.research_only,
            panel_path=request.panel_path,
            panel_df=df,
            output_path=request.output_path,
            fast_mcmc=request.fast_mcmc,
            extended_mcmc=request.extended_mcmc,
            execute_fit=request.execute_fit,
            artifact_type=request.artifact_type,
            calibration_signals_stub=request.calibration_signals_stub,
            geo_hierarchy_mapping=request.geo_hierarchy_mapping,
            ridge_comparison=request.ridge_comparison,
            geox_cls_comparison=request.geox_cls_comparison,
            run_id=request.run_id,
            requested_production_flags=request.requested_production_flags,
            policy_id=request.policy_id,
            source_policy_path=request.source_policy_path,
            geometry_config=request.geometry_config,
            sandbox_model_overrides=request.sandbox_model_overrides,
            sampler_profile_applied=request.sampler_profile_applied,
            channel_policy_declared=request.channel_policy_declared,
            channel_policy_applied=policy_record,
        )
        cfg, sampler_profile = _config_from_panel(
            df,
            schema,
            fast_mcmc=request.fast_mcmc,
            extended_mcmc=request.extended_mcmc,
            sampler_overrides=sampler_ov,
        )
        if request.sampler_profile_applied:
            sampler_profile = str(request.sampler_profile_applied.get("profile") or sampler_profile)

    df = validate_panel(df, schema, integrity_qa=False, calendar_strict=False)
    panel_hash = _sha256_panel(df)
    config_hash = _sha256_config(cfg)

    fit_artifact: dict[str, Any] | None = None
    if request.execute_fit:
        panel_context = _panel_context_for_request(request)
        if panel_context == PANEL_CONTEXT_REAL:
            generative_kind = "real_panel"
        else:
            generative_kind = "linear"
        overrides = {
            "media_transforms_by_channel": dict(
                request.transform_config.get("media_transforms_by_channel") or {}
            ),
            "transform_params_by_channel": dict(
                request.transform_config.get("transform_params_by_channel") or {}
            ),
            "h5_generative_transform": generative_kind,
            "h5_panel_context": panel_context,
            "h5_real_panel": panel_context == PANEL_CONTEXT_REAL,
            "h5_transform_mismatch_mode": str(request.transform_config.get("transform_mismatch_mode", "aligned")),
        }
        if request.sandbox_model_overrides:
            overrides.update(request.sandbox_model_overrides)
        fit_artifact = run_sandbox_fit(
            cfg,
            schema,
            df,
            geo_hierarchy_mapping=request.geo_hierarchy_mapping,
            calibration_signals_stub=request.calibration_signals_stub or [],
            sandbox_model_overrides=overrides,
            model_spec_version=H5_MODEL_SPEC_VERSION,
            enable_h5_sandbox=True,
            research_only=True,
        )

    shadow_record = record or build_shadow_run_record(
        request,
        fit_artifact=fit_artifact,
        schema=schema,
        config=cfg,
        panel_hash=panel_hash,
        config_hash=config_hash,
        sampler_profile=sampler_profile,
    )

    prod_flags = research_production_flags()
    conv = (fit_artifact or {}).get("convergence_diagnostics") or {}
    from mmm.research.bayes_h3_sandbox.h5_trust_diagnostics import (
        classify_convergence_status,
        evidence_promotion_allowed,
    )

    convergence_status = classify_convergence_status(
        rhat_max=conv.get("rhat_max"),
        divergence_count=conv.get("divergence_count"),
    )
    promo = evidence_promotion_allowed(convergence_status)

    artifact: dict[str, Any] = {
        "harness_version": HARNESS_VERSION,
        "artifact_type": request.artifact_type,
        "schema_reference": "BAYES_H5E_SHADOW_RUN_SCHEMA_20260601",
        "label": RESEARCH_ONLY_LABEL,
        "research_only": True,
        "shadow_run": shadow_record,
        "sampler_profile": sampler_profile,
        "fast_mcmc_profile": request.fast_mcmc and not request.extended_mcmc,
        "extended_mcmc_profile": request.extended_mcmc,
        "execute_fit": request.execute_fit,
        "note": (
            "H5 shadow-run harness output — research only. "
            "Not production TrustReport, optimizer, or DecisionSurface."
        ),
        "production_flags": prod_flags,
        "outputs_are_diagnostic_only": True,
        "convergence_status": convergence_status,
        "rhat_max": conv.get("rhat_max"),
        "divergence_count": conv.get("divergence_count"),
        "evidence_promotion_allowed": promo,
        "excluded_fields": sorted(FORBIDDEN_OUTPUT_FIELDS),
    }
    if request.policy_id:
        from mmm.research.bayes_h3_sandbox.h5_geometry_config import geometry_record_for_artifact

        artifact["policy_id"] = request.policy_id
        artifact["source_policy_path"] = request.source_policy_path
        artifact["channel_policy_applied"] = request.channel_policy_applied
        artifact["channel_policy_declared"] = request.channel_policy_declared
        geom = request.geometry_config or {}
        artifact["geometry_config_applied"] = geometry_record_for_artifact(geom)
        artifact["sampler_profile_applied"] = request.sampler_profile_applied
        artifact["trust_report_candidate_diagnostics"] = shadow_record.get(
            "trust_report_candidate_diagnostics"
        )
    if request.artifact_type == "dry_run_shadow_artifact":
        artifact["note"] = (
            "Dry-run shadow artifact on synthetic fixture only — does not constitute real-panel evidence."
        )
    return artifact


def write_shadow_run_artifact(request: ShadowRunRequest) -> dict[str, Any]:
    artifact = build_shadow_run_artifact(request)
    out = Path(
        request.output_path
        or default_output_path(request.panel_id, artifact_type=request.artifact_type)
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2, default=str) + "\n", encoding="utf-8")
    artifact["artifact_path"] = str(out)
    return artifact


def run_fixture_dry_run_shadow(
    *,
    execute_fit: bool = True,
    fast_mcmc: bool = True,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run synthetic fixture shadow path (labeled dry_run — not real-panel evidence)."""
    return write_shadow_run_artifact(
        ShadowRunRequest(
            panel_id=FIXTURE_PANEL_ID,
            dataset_snapshot_id=FIXTURE_DATASET_SNAPSHOT_ID,
            transform_config=dict(DEFAULT_FIXTURE_TRANSFORM_CONFIG),
            artifact_type="dry_run_shadow_artifact",
            execute_fit=execute_fit,
            fast_mcmc=fast_mcmc,
            output_path=output_path or DEFAULT_DRY_RUN_ARTIFACT,
            calibration_signals_stub=TOY_CALIBRATION_SIGNAL_STUB,
            geo_hierarchy_mapping=TOY_GEO_HIERARCHY,
        )
    )


def validate_shadow_run_artifact_file(artifact: dict[str, Any]) -> None:
    if artifact.get("artifact_type") not in ("dry_run_shadow_artifact", "real_panel_shadow_artifact"):
        raise H5ShadowRunnerError("invalid artifact_type")
    flags = artifact.get("production_flags") or {}
    if not flags:
        flags = {k: artifact[k] for k in research_production_flags() if k in artifact}
    for key, val in research_production_flags().items():
        if flags.get(key) is not val:
            raise H5ShadowRunnerError(f"artifact production_flags.{key} must be {val!r}")
    shadow = artifact.get("shadow_run")
    if not isinstance(shadow, dict):
        raise H5ShadowRunnerError("shadow_run object required")
    validate_shadow_run_record(shadow)
    prod_flag_keys = set(research_production_flags())
    for forbidden in FORBIDDEN_OUTPUT_FIELDS:
        if forbidden in prod_flag_keys:
            if artifact.get(forbidden) is True:
                raise H5ShadowRunnerError(f"forbidden production flag on envelope: {forbidden!r}=true")
            continue
        if forbidden in artifact and artifact.get(forbidden) is not None:
            raise H5ShadowRunnerError(f"forbidden field on artifact envelope: {forbidden!r}")


def _build_cli_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Bayes-H5 real-panel shadow-run harness (research only)")
    p.add_argument("--panel-path", type=str, help="Path to panel CSV or Parquet")
    p.add_argument("--panel-id", type=str, default=None)
    p.add_argument("--dataset-snapshot-id", type=str, default=None)
    p.add_argument(
        "--transform-config",
        type=str,
        default=None,
        help="JSON object or path to JSON file with media_transforms_by_channel",
    )
    p.add_argument("--output-path", type=str, default=None)
    p.add_argument("--fast-mcmc", action="store_true", help="Use fast MCMC profile (200/200/2)")
    p.add_argument(
        "--extended-mcmc",
        action="store_true",
        help="Use extended MCMC profile (600/600/4 chains, target_accept=0.95)",
    )
    p.add_argument(
        "--fixture-dry-run",
        action="store_true",
        help="Run synthetic fixture dry-run (not real-panel evidence)",
    )
    p.add_argument("--no-fit", action="store_true", help="Build artifact without executing MCMC")
    p.add_argument(
        "--policy-path",
        type=str,
        default=None,
        help="Frozen shadow policy JSON — fully specifies the run (overrides other panel args)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_cli_parser().parse_args(argv)
    if args.fixture_dry_run:
        artifact = run_fixture_dry_run_shadow(
            execute_fit=not args.no_fit,
            fast_mcmc=(args.fast_mcmc or True) and not args.extended_mcmc,
            output_path=args.output_path,
        )
    elif args.policy_path:
        from mmm.research.bayes_h3_sandbox.h5_shadow_policy import (
            load_shadow_policy,
            policy_to_shadow_request,
        )

        policy = load_shadow_policy(args.policy_path)
        request = policy_to_shadow_request(
            policy,
            policy_path=args.policy_path,
            output_path=args.output_path,
            execute_fit=not args.no_fit,
        )
        artifact = write_shadow_run_artifact(request)
    else:
        missing = [
            name
            for name, val in (
                ("--panel-path", args.panel_path),
                ("--panel-id", args.panel_id),
                ("--dataset-snapshot-id", args.dataset_snapshot_id),
                ("--transform-config", args.transform_config),
            )
            if not val
        ]
        if missing:
            raise SystemExit(
                f"required arguments missing: {', '.join(missing)} "
                "(or provide --policy-path)"
            )
        artifact = write_shadow_run_artifact(
            ShadowRunRequest(
                panel_path=args.panel_path,
                panel_id=args.panel_id,
                dataset_snapshot_id=args.dataset_snapshot_id,
                transform_config=load_transform_config(args.transform_config),
                output_path=args.output_path,
                fast_mcmc=(args.fast_mcmc or True) and not args.extended_mcmc,
                extended_mcmc=args.extended_mcmc,
                execute_fit=not args.no_fit,
                artifact_type="real_panel_shadow_artifact",
            )
        )
    validate_shadow_run_artifact_file(artifact)
    print(artifact["artifact_path"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
