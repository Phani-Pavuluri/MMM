"""Bayes-H5j collinearity-aware geometry ablation runner (sample panel only)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from mmm.data.schema import PanelSchema, validate_panel
from mmm.research.bayes_h3_sandbox.entrypoint import run_sandbox_fit
from mmm.research.bayes_h3_sandbox.fencing import H5_MODEL_SPEC_VERSION
from mmm.research.bayes_h3_sandbox.h5_convergence_diagnostics import (
    DATASET_SNAPSHOT_ID,
    DEFAULT_PANEL_PATH,
    DEFAULT_TRANSFORM_CONFIG,
    PANEL_ID,
)
from mmm.research.bayes_h3_sandbox.h5_real_panel_preprocessing import (
    CHANNEL_POLICY_COMPOSITE,
    CHANNEL_POLICY_DROP_COLLINEAR,
    CHANNEL_POLICY_KEEP_ALL,
    CHANNEL_POLICY_POOLED,
    CHANNEL_POLICY_SINGLE,
    apply_channel_policy,
    apply_preprocessing_config,
    compute_media_correlation_matrix,
    detect_collinear_channel_groups,
)
from mmm.research.bayes_h3_sandbox.h5_shadow_runner import (
    _config_from_panel,
    _infer_schema,
    load_panel_from_path,
    load_transform_config,
)
from mmm.research.bayes_h3_sandbox.h5_trust_diagnostics import (
    classify_convergence_status,
    derive_real_panel_transform_warning_codes,
    evidence_promotion_allowed,
    research_production_flags,
)
from mmm.research.bayes_h3_sandbox.recovery_worlds import SAMPLER_EXTENDED, SAMPLER_FAST

INVESTIGATION_ID = "INV-H5J_COLLINEARITY_GEOMETRY_ABLATIONS"
ARTIFACT_ID = "BAYES_H5J_COLLINEARITY_GEOMETRY_ABLATIONS_20260601"
DEFAULT_H5I_DIAGNOSTICS = Path(
    "docs/05_validation/archives/BAYES_H5I_CONVERGENCE_DIAGNOSTICS_EXAMPLES_MMM_SAMPLE_PANEL_V1_20260601.json"
)
DEFAULT_OUTPUT = Path(f"docs/05_validation/archives/{ARTIFACT_ID}.json")

BASELINE_H5I_REF = "H5I-BASELINE-FAST-REPLAY"
PRESAMPLE_PREPROCESS = {"media_prescale": "zscore_panel"}
PRESCALE_MODEL_OVERRIDES = {
    "h5_panel_context": "real_panel",
    "h5_real_panel": True,
    "media_prescale": "zscore_panel",
    "outcome_prescale": "zscore_log",
}


@dataclass(frozen=True)
class GeometryAblationSpec:
    variant_id: str
    channel_policy: dict[str, Any]
    preprocessing: dict[str, Any] | None
    sampler_overrides: dict[str, Any]
    sampler_profile: str
    model_overrides: dict[str, Any]
    notes: str
    skip_fit: bool = False
    reference_only: bool = False


def default_ablation_specs() -> list[GeometryAblationSpec]:
    return [
        GeometryAblationSpec(
            variant_id="H5J-A-BASELINE-REPLAY",
            channel_policy={"mode": CHANNEL_POLICY_KEEP_ALL},
            preprocessing=None,
            sampler_overrides=dict(SAMPLER_FAST),
            sampler_profile="fast",
            model_overrides={"h5_panel_context": "real_panel", "h5_real_panel": True},
            notes="All channels, fast MCMC — aligns with H5i baseline replay",
        ),
        GeometryAblationSpec(
            variant_id="H5J-B-PRESCALE-EXTENDED",
            channel_policy={"mode": CHANNEL_POLICY_KEEP_ALL},
            preprocessing=PRESAMPLE_PREPROCESS,
            sampler_overrides=dict(SAMPLER_EXTENDED),
            sampler_profile="extended",
            model_overrides=dict(PRESCALE_MODEL_OVERRIDES),
            notes="Prescale media + zscore log outcome, extended MCMC",
        ),
        GeometryAblationSpec(
            variant_id="H5J-C-SINGLE-SEARCH-PRESCALE-EXTENDED",
            channel_policy={"mode": CHANNEL_POLICY_SINGLE, "channel": "search"},
            preprocessing=PRESAMPLE_PREPROCESS,
            sampler_overrides=dict(SAMPLER_EXTENDED),
            sampler_profile="extended",
            model_overrides=dict(PRESCALE_MODEL_OVERRIDES),
            notes="Single channel search only",
        ),
        GeometryAblationSpec(
            variant_id="H5J-D-DROP-COLLINEAR-PRESCALE-EXTENDED",
            channel_policy={
                "mode": CHANNEL_POLICY_DROP_COLLINEAR,
                "max_abs_corr_threshold": 0.95,
            },
            preprocessing=PRESAMPLE_PREPROCESS,
            sampler_overrides=dict(SAMPLER_EXTENDED),
            sampler_profile="extended",
            model_overrides=dict(PRESCALE_MODEL_OVERRIDES),
            notes="Drop redundant channel per collinear group (expected: social or tv)",
        ),
        GeometryAblationSpec(
            variant_id="H5J-E-COMPOSITE-SOCIAL-TV-PRESCALE-EXTENDED",
            channel_policy={
                "mode": CHANNEL_POLICY_COMPOSITE,
                "source_channels": ["social", "tv"],
                "method": "first_principal_component",
                "output_channel": "social_tv_pc1",
                "remaining_channels": ["search"],
            },
            preprocessing=PRESAMPLE_PREPROCESS,
            sampler_overrides=dict(SAMPLER_EXTENDED),
            sampler_profile="extended",
            model_overrides=dict(PRESCALE_MODEL_OVERRIDES),
            notes="Composite social+tv via PC1, keep search",
        ),
        GeometryAblationSpec(
            variant_id="H5J-F-POOLED-CHANNEL-EFFECTS",
            channel_policy={"mode": CHANNEL_POLICY_POOLED},
            preprocessing=PRESAMPLE_PREPROCESS,
            sampler_overrides=dict(SAMPLER_EXTENDED),
            sampler_profile="extended",
            model_overrides=dict(PRESCALE_MODEL_OVERRIDES),
            notes="Pooled coefficients ablation — not implemented",
            skip_fit=True,
        ),
    ]


def _load_baseline_h5i() -> dict[str, Any]:
    p = DEFAULT_H5I_DIAGNOSTICS
    if p.is_file():
        return json.loads(p.read_text(encoding="utf-8"))
    return {}


def run_geometry_ablation(
    spec: GeometryAblationSpec,
    *,
    base_transform_config: dict[str, Any],
    panel_path: str | Path = DEFAULT_PANEL_PATH,
    execute_fit: bool = True,
) -> dict[str, Any]:
    if spec.reference_only or spec.skip_fit:
        return {
            "variant_id": spec.variant_id,
            "channel_config": spec.channel_policy,
            "preprocessing_config": spec.preprocessing,
            "sampler_profile": spec.sampler_profile,
            "sampler_settings": spec.sampler_overrides,
            "model_overrides": spec.model_overrides,
            "rhat_max": None,
            "ess_min": None,
            "divergence_count": None,
            "convergence_status": "not_executed",
            "evidence_promotion_allowed": False,
            "warnings": ["h5:production:block", "h5:evidence:blocked"],
            "notes": spec.notes,
            "recommendation": "not_implemented" if spec.skip_fit else "reference_only",
            **research_production_flags(),
        }

    transform_config = dict(base_transform_config)
    transform_config["channel_policy"] = dict(spec.channel_policy)

    df = load_panel_from_path(panel_path)
    schema = _infer_schema(df, transform_config)
    df = validate_panel(df, schema, integrity_qa=False, calendar_strict=False)

    policy_record: dict[str, Any] = {}
    try:
        df, schema, transform_config, policy_record = apply_channel_policy(df, schema, transform_config)
    except Exception as exc:
        return {
            "variant_id": spec.variant_id,
            "channel_config": spec.channel_policy,
            "preprocessing_config": spec.preprocessing,
            "sampler_profile": spec.sampler_profile,
            "convergence_status": "not_executed",
            "evidence_promotion_allowed": False,
            "warnings": ["h5:production:block", "h5:evidence:blocked"],
            "error": str(exc),
            "notes": spec.notes,
            "recommendation": "failed_preprocessing",
            **research_production_flags(),
        }

    cfg, _ = _config_from_panel(
        df,
        schema,
        fast_mcmc=spec.sampler_profile == "fast",
        extended_mcmc=spec.sampler_profile == "extended",
    )
    cfg = cfg.model_copy(update={"bayesian": cfg.bayesian.model_copy(update=spec.sampler_overrides)})

    fit_artifact: dict[str, Any] | None = None
    if execute_fit:
        overrides = {
            "media_transforms_by_channel": dict(transform_config["media_transforms_by_channel"]),
            "transform_params_by_channel": dict(transform_config.get("transform_params_by_channel") or {}),
            "h5_transform_mismatch_mode": str(transform_config.get("transform_mismatch_mode", "aligned")),
            **spec.model_overrides,
        }
        if spec.preprocessing:
            overrides.setdefault("media_prescale", spec.preprocessing.get("media_prescale"))
            overrides.setdefault("outcome_prescale", "zscore_log" if spec.model_overrides.get("outcome_prescale") else None)
        fit_artifact = run_sandbox_fit(
            cfg,
            schema,
            df,
            sandbox_model_overrides=overrides,
            model_spec_version=H5_MODEL_SPEC_VERSION,
            enable_h5_sandbox=True,
            research_only=True,
        )

    conv = (fit_artifact or {}).get("convergence_diagnostics") or {}
    status = classify_convergence_status(
        rhat_max=conv.get("rhat_max"),
        divergence_count=conv.get("divergence_count"),
    )
    warnings = derive_real_panel_transform_warning_codes(transform_config)
    if status != "converged_diagnostic_only":
        warnings = sorted(set(warnings + ["h5:evidence:blocked"]))

    return {
        "variant_id": spec.variant_id,
        "channel_config": spec.channel_policy,
        "channel_policy_record": policy_record,
        "preprocessing_config": spec.preprocessing,
        "sampler_profile": spec.sampler_profile,
        "sampler_settings": spec.sampler_overrides,
        "model_overrides": spec.model_overrides,
        "active_channels": list(schema.channel_columns),
        "rhat_max": conv.get("rhat_max"),
        "ess_min": conv.get("ess_bulk_min"),
        "divergence_count": conv.get("divergence_count"),
        "convergence_status": status,
        "evidence_promotion_allowed": evidence_promotion_allowed(status),
        "worst_rhat_parameters": conv.get("worst_rhat_parameters"),
        "warnings": warnings,
        "notes": spec.notes,
        "recommendation": _recommendation(status),
        **research_production_flags(),
    }


def _recommendation(status: str) -> str:
    if status == "converged_diagnostic_only":
        return "report_only_pass_eligibility_candidate"
    if status == "weak_convergence":
        return "report_only_weak"
    return "report_only_fail"


def _recommended_next_action(ablations: list[dict[str, Any]]) -> str:
    eligible = [a for a in ablations if a.get("evidence_promotion_allowed")]
    if eligible:
        best = min(eligible, key=lambda a: float(a.get("rhat_max") or 999))
        return (
            f"Pilot panel may be eligible under {best['variant_id']} — verify on extended rerun "
            "before additional real panels."
        )
    best = min(
        (a for a in ablations if a.get("rhat_max") is not None),
        key=lambda a: (float(a["rhat_max"]), int(a.get("divergence_count") or 999)),
        default=None,
    )
    if best:
        return (
            f"No variant cleared evidence bar. Best probe: {best['variant_id']} "
            f"(rhat_max={best.get('rhat_max')}, divergences={best.get('divergence_count')}). "
            "Require converged_diagnostic_only before more real panels."
        )
    return "Do not run additional real panels until a geometry ablation clears convergence gates."


def build_geometry_ablation_artifact(
    *,
    panel_path: str | Path = DEFAULT_PANEL_PATH,
    transform_config_path: str | Path = DEFAULT_TRANSFORM_CONFIG,
    execute_fit: bool = True,
    specs: list[GeometryAblationSpec] | None = None,
) -> dict[str, Any]:
    base_config = load_transform_config(transform_config_path)
    df = load_panel_from_path(panel_path)
    schema = _infer_schema(df, base_config)
    channels = list(schema.channel_columns)

    corr = compute_media_correlation_matrix(df, channels)
    groups = detect_collinear_channel_groups(df, channels, max_abs_corr_threshold=0.95)
    h5i = _load_baseline_h5i()

    ablations = [run_geometry_ablation(s, base_transform_config=base_config, panel_path=panel_path, execute_fit=execute_fit) for s in (specs or default_ablation_specs())]

    return {
        "artifact_id": ARTIFACT_ID,
        "investigation_id": INVESTIGATION_ID,
        "panel_id": PANEL_ID,
        "dataset_snapshot_id": DATASET_SNAPSHOT_ID,
        "panel_path": str(panel_path),
        "baseline_h5i_reference": {
            "diagnostics_artifact": str(DEFAULT_H5I_DIAGNOSTICS),
            "suspected_failure_modes": h5i.get("suspected_failure_modes"),
            "best_h5i_variant": BASELINE_H5I_REF,
        },
        "media_correlation_matrix": corr,
        "detected_collinear_groups": groups,
        "ablation_results": ablations,
        "recommended_next_action": _recommended_next_action(ablations),
        "shadow_run_eligibility_rule": (
            "No additional real-panel shadow runs until converged_diagnostic_only "
            "(rhat_max <= 1.05 and divergence_count == 0) on pilot panel under explicit channel_policy."
        ),
        "label": "RESEARCH ONLY — NOT DECISION GRADE",
        **research_production_flags(),
        "outputs_are_diagnostic_only": True,
    }


def validate_geometry_ablation_artifact(artifact: dict[str, Any]) -> None:
    for key, val in research_production_flags().items():
        if artifact.get(key) is not val:
            raise ValueError(f"artifact.{key} must be {val!r}")
    for forbidden in ("decision_surface", "optimizer_ready_curves", "budget_recommendation"):
        if artifact.get(forbidden) is not None:
            raise ValueError(f"forbidden field present: {forbidden!r}")


def write_geometry_ablation_artifact(
    output_path: str | Path | None = None,
    *,
    execute_fit: bool = True,
) -> dict[str, Any]:
    artifact = build_geometry_ablation_artifact(execute_fit=execute_fit)
    out = Path(output_path or DEFAULT_OUTPUT)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2, default=str) + "\n", encoding="utf-8")
    validate_geometry_ablation_artifact(artifact)
    artifact["artifact_path"] = str(out)
    return artifact


def main() -> int:
    artifact = write_geometry_ablation_artifact(execute_fit=True)
    print(artifact["artifact_path"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
