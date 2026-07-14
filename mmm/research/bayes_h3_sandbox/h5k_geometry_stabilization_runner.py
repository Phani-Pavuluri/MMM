"""Bayes-H5k non-centered / geometry stabilization runner (sample panel only)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mmm.data.schema import validate_panel
from mmm.research.bayes_h3_sandbox.entrypoint import run_sandbox_fit
from mmm.research.bayes_h3_sandbox.fencing import H5_MODEL_SPEC_VERSION
from mmm.research.bayes_h3_sandbox.h5_convergence_diagnostics import (
    DATASET_SNAPSHOT_ID,
    DEFAULT_PANEL_PATH,
    DEFAULT_TRANSFORM_CONFIG,
    PANEL_ID,
)
from mmm.research.bayes_h3_sandbox.h5_geometry_config import (
    HIERARCHY_FIXED_TAU,
    HIERARCHY_FULL_GEO_CHANNEL,
    HIERARCHY_POOLED_CHANNEL,
    LIKELIHOOD_PRESCALED_LOG_OUTCOME,
    PARAMETERIZATION_CENTERED,
    PARAMETERIZATION_NON_CENTERED,
    geometry_record_for_artifact,
)
from mmm.research.bayes_h3_sandbox.h5_real_panel_preprocessing import (
    CHANNEL_POLICY_DROP_COLLINEAR,
    CHANNEL_POLICY_SINGLE,
    apply_channel_policy,
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
from mmm.research.bayes_h3_sandbox.recovery_worlds import SAMPLER_EXTENDED

INVESTIGATION_ID = "INV-H5K_GEOMETRY_STABILIZATION"
ARTIFACT_ID = "BAYES_H5K_GEOMETRY_STABILIZATION_20260601"
DEFAULT_H5J_ARTIFACT = Path(
    "docs/05_validation/archives/BAYES_H5J_COLLINEARITY_GEOMETRY_ABLATIONS_20260601.json"
)
DEFAULT_OUTPUT = Path(f"docs/05_validation/archives/{ARTIFACT_ID}.json")

H5J_BEST_CHANNEL_POLICY = {
    "mode": CHANNEL_POLICY_DROP_COLLINEAR,
    "max_abs_corr_threshold": 0.95,
}
H5J_BEST_VARIANT_ID = "H5J-D-DROP-COLLINEAR-PRESCALE-EXTENDED"

BASE_REAL_PANEL_OVERRIDES = {
    "h5_panel_context": "real_panel",
    "h5_real_panel": True,
}


@dataclass(frozen=True)
class GeometryStabilizationSpec:
    variant_id: str
    parameterization: str
    likelihood_scale_policy: str
    hierarchy_policy: str
    channel_policy: dict[str, Any]
    sampler_overrides: dict[str, Any]
    sampler_profile: str
    geometry_extras: dict[str, Any] | None = None
    notes: str = ""
    skip_fit: bool = False


def default_stabilization_specs() -> list[GeometryStabilizationSpec]:
    extended = dict(SAMPLER_EXTENDED)
    drop = dict(H5J_BEST_CHANNEL_POLICY)
    prescaled = LIKELIHOOD_PRESCALED_LOG_OUTCOME
    full = HIERARCHY_FULL_GEO_CHANNEL
    nc = PARAMETERIZATION_NON_CENTERED
    centered = PARAMETERIZATION_CENTERED

    def geom(
        *,
        parameterization: str,
        likelihood: str,
        hierarchy: str,
        extras: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        cfg: dict[str, Any] = {
            "parameterization": parameterization,
            "likelihood_scale_policy": likelihood,
            "hierarchy_policy": hierarchy,
        }
        if extras:
            cfg.update(extras)
        return cfg

    return [
        GeometryStabilizationSpec(
            variant_id="H5K-A-H5J-BEST-CENTERED-REPLAY",
            parameterization=centered,
            likelihood_scale_policy=prescaled,
            hierarchy_policy=full,
            channel_policy=drop,
            sampler_overrides=extended,
            sampler_profile="extended",
            notes="H5j-D channel policy replay with centered beta (funnel-prone ablation)",
        ),
        GeometryStabilizationSpec(
            variant_id="H5K-B-NON-CENTERED-DROP-COLLINEAR-EXTENDED",
            parameterization=nc,
            likelihood_scale_policy=prescaled,
            hierarchy_policy=full,
            channel_policy=drop,
            sampler_overrides=extended,
            sampler_profile="extended",
            notes="Explicit non-centered label + H5j-D channel policy (legacy default geometry)",
        ),
        GeometryStabilizationSpec(
            variant_id="H5K-C-NON-CENTERED-TARGET-ACCEPT-099",
            parameterization=nc,
            likelihood_scale_policy=prescaled,
            hierarchy_policy=full,
            channel_policy=drop,
            sampler_overrides={**extended, "target_accept": 0.99},
            sampler_profile="extended",
            notes="Non-centered + drop collinear + target_accept=0.99",
        ),
        GeometryStabilizationSpec(
            variant_id="H5K-D-NON-CENTERED-SINGLE-SEARCH",
            parameterization=nc,
            likelihood_scale_policy=prescaled,
            hierarchy_policy=full,
            channel_policy={"mode": CHANNEL_POLICY_SINGLE, "channel": "search"},
            sampler_overrides=extended,
            sampler_profile="extended",
            notes="Non-centered + single search channel (H5j-C geometry probe)",
        ),
        GeometryStabilizationSpec(
            variant_id="H5K-E-POOLED-CHANNEL-ABLATION",
            parameterization=nc,
            likelihood_scale_policy=prescaled,
            hierarchy_policy=HIERARCHY_POOLED_CHANNEL,
            channel_policy=drop,
            sampler_overrides=extended,
            sampler_profile="extended",
            notes="Pooled channel coefficients — no geo×channel partial pooling",
        ),
        GeometryStabilizationSpec(
            variant_id="H5K-F-FIXED-TAU-ABLATION",
            parameterization=nc,
            likelihood_scale_policy=prescaled,
            hierarchy_policy=HIERARCHY_FIXED_TAU,
            channel_policy=drop,
            sampler_overrides=extended,
            sampler_profile="extended",
            geometry_extras={"fixed_tau_value": 0.2},
            notes="Fixed tau_channel=0.2 ablation — removes tau funnel",
        ),
    ]


def _geometry_config_for_spec(spec: GeometryStabilizationSpec) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "parameterization": spec.parameterization,
        "likelihood_scale_policy": spec.likelihood_scale_policy,
        "hierarchy_policy": spec.hierarchy_policy,
    }
    if spec.geometry_extras:
        cfg.update(spec.geometry_extras)
    return cfg


def run_geometry_stabilization(
    spec: GeometryStabilizationSpec,
    *,
    base_transform_config: dict[str, Any],
    panel_path: str | Path = DEFAULT_PANEL_PATH,
    execute_fit: bool = True,
) -> dict[str, Any]:
    geometry_cfg = _geometry_config_for_spec(spec)
    base_row: dict[str, Any] = {
        "variant_id": spec.variant_id,
        "parameterization": spec.parameterization,
        "likelihood_scale_policy": spec.likelihood_scale_policy,
        "hierarchy_policy": spec.hierarchy_policy,
        "channel_policy": spec.channel_policy,
        "sampler_profile": spec.sampler_profile,
        "sampler_settings": spec.sampler_overrides,
        "geometry_config": geometry_record_for_artifact(
            {**geometry_cfg, "explicit": True, "legacy_default": False}
        ),
        "notes": spec.notes,
        **research_production_flags(),
    }
    if spec.skip_fit or not execute_fit:
        return {
            **base_row,
            "rhat_max": None,
            "ess_min": None,
            "divergence_count": None,
            "convergence_status": "not_executed",
            "evidence_promotion_allowed": False,
            "warnings": ["h5:production:block", "h5:evidence:blocked"],
            "posterior_sanity_notes": spec.notes,
            "recommendation": "not_executed",
        }

    transform_config = dict(base_transform_config)
    transform_config["channel_policy"] = dict(spec.channel_policy)

    df = load_panel_from_path(panel_path)
    schema = _infer_schema(df, transform_config)
    df = validate_panel(df, schema, integrity_qa=False, calendar_strict=False)

    try:
        df, schema, transform_config, policy_record = apply_channel_policy(df, schema, transform_config)
    except Exception as exc:
        return {
            **base_row,
            "convergence_status": "not_executed",
            "evidence_promotion_allowed": False,
            "error": str(exc),
            "warnings": ["h5:production:block", "h5:evidence:blocked"],
            "posterior_sanity_notes": f"preprocessing failed: {exc}",
            "recommendation": "failed_preprocessing",
        }

    cfg, _ = _config_from_panel(
        df,
        schema,
        fast_mcmc=False,
        extended_mcmc=spec.sampler_profile == "extended",
    )
    cfg = cfg.model_copy(update={"bayesian": cfg.bayesian.model_copy(update=spec.sampler_overrides)})

    overrides = {
        **BASE_REAL_PANEL_OVERRIDES,
        "media_transforms_by_channel": dict(transform_config["media_transforms_by_channel"]),
        "transform_params_by_channel": dict(transform_config.get("transform_params_by_channel") or {}),
        "h5_transform_mismatch_mode": str(transform_config.get("transform_mismatch_mode", "aligned")),
        "h5_geometry_config": geometry_cfg,
    }
    fit_artifact = run_sandbox_fit(
        cfg,
        schema,
        df,
        sandbox_model_overrides=overrides,
        model_spec_version=H5_MODEL_SPEC_VERSION,
        enable_h5_sandbox=True,
        research_only=True,
    )

    conv = fit_artifact.get("convergence_diagnostics") or {}
    geom_diag = fit_artifact.get("h5_geometry_diagnostics") or {}
    status = classify_convergence_status(
        rhat_max=conv.get("rhat_max"),
        divergence_count=conv.get("divergence_count"),
    )
    warnings = derive_real_panel_transform_warning_codes(transform_config)
    if status != "converged_diagnostic_only":
        warnings = sorted(set(warnings + ["h5:evidence:blocked"]))

    sanity: list[str] = []
    if conv.get("divergence_count"):
        sanity.append(f"{conv['divergence_count']} divergences — check tau/beta/sigma geometry")
    if geom_diag.get("hierarchy_policy") == HIERARCHY_POOLED_CHANNEL:
        sanity.append("pooled channel ablation — geo-specific media effects removed")
    if geom_diag.get("hierarchy_policy") == HIERARCHY_FIXED_TAU:
        sanity.append(f"fixed tau ablation tau={geom_diag.get('fixed_tau_value')}")

    return {
        **base_row,
        "channel_policy_record": policy_record,
        "active_channels": list(schema.channel_columns),
        "rhat_max": conv.get("rhat_max"),
        "ess_min": conv.get("ess_bulk_min"),
        "divergence_count": conv.get("divergence_count"),
        "worst_rhat_parameters": conv.get("worst_rhat_parameters"),
        "convergence_status": status,
        "evidence_promotion_allowed": evidence_promotion_allowed(status),
        "warnings": warnings,
        "posterior_sanity_notes": "; ".join(sanity) if sanity else "no major sanity flags",
        "recommendation": _recommendation(status),
        "geometry_diagnostics_from_fit": geom_diag,
    }


def _recommendation(status: str) -> str:
    if status == "converged_diagnostic_only":
        return "report_only_pass_eligibility_candidate"
    if status == "weak_convergence":
        return "report_only_weak"
    return "report_only_fail"


def _pick_best_variant(variants: list[dict[str, Any]]) -> dict[str, Any] | None:
    scored = [v for v in variants if v.get("rhat_max") is not None]
    if not scored:
        return None
    return min(
        scored,
        key=lambda v: (
            0 if v.get("convergence_status") == "converged_diagnostic_only" else 1,
            float(v["rhat_max"]),
            int(v.get("divergence_count") or 999),
        ),
    )


def _load_h5j_artifact() -> dict[str, Any]:
    if DEFAULT_H5J_ARTIFACT.is_file():
        return json.loads(DEFAULT_H5J_ARTIFACT.read_text(encoding="utf-8"))
    return {}


def build_geometry_stabilization_artifact(
    *,
    panel_path: str | Path = DEFAULT_PANEL_PATH,
    transform_config_path: str | Path = DEFAULT_TRANSFORM_CONFIG,
    execute_fit: bool = True,
    specs: list[GeometryStabilizationSpec] | None = None,
) -> dict[str, Any]:
    base_config = load_transform_config(transform_config_path)
    h5j = _load_h5j_artifact()
    spec_list = specs or default_stabilization_specs()
    variants = [
        run_geometry_stabilization(s, base_transform_config=base_config, panel_path=panel_path, execute_fit=execute_fit)
        for s in spec_list
    ]
    best = _pick_best_variant(variants)
    any_converged = any(v.get("convergence_status") == "converged_diagnostic_only" for v in variants)

    recommendation = (
        "Pilot may be eligible for evidence review under explicit geometry + channel policy."
        if any_converged
        else (
            f"No variant reached converged_diagnostic_only. Best: {best['variant_id']} "
            f"(rhat_max={best.get('rhat_max')}, divergences={best.get('divergence_count')}). "
            "Continue geometry research before additional real panels."
            if best
            else "Continue geometry research — no scored variants."
        )
    )

    return {
        "artifact_id": ARTIFACT_ID,
        "investigation_id": INVESTIGATION_ID,
        "panel_id": PANEL_ID,
        "dataset_snapshot_id": DATASET_SNAPSHOT_ID,
        "panel_path": str(panel_path),
        "source_h5j_artifact": str(DEFAULT_H5J_ARTIFACT),
        "source_h5j_best_variant": H5J_BEST_VARIANT_ID,
        "baseline_channel_policy": H5J_BEST_CHANNEL_POLICY,
        "h5j_summary": {
            "best_variant_id": H5J_BEST_VARIANT_ID,
            "best_rhat_max": 1.02,
            "best_divergence_count": 4,
            "best_status": "weak_convergence",
            "ablation_results_present": bool(h5j.get("ablation_results")),
        },
        "variants": variants,
        "best_variant": best,
        "any_variant_converged_diagnostic_only": any_converged,
        "recommendation": recommendation,
        "shadow_run_eligibility_rule": (
            "No additional real-panel shadow runs until converged_diagnostic_only "
            "(rhat_max <= 1.05 and divergence_count == 0) under explicit channel + geometry policy."
        ),
        "label": "RESEARCH ONLY — NOT DECISION GRADE",
        **research_production_flags(),
        "outputs_are_diagnostic_only": True,
    }


def validate_geometry_stabilization_artifact(artifact: dict[str, Any]) -> None:
    for key, val in research_production_flags().items():
        if artifact.get(key) is not val:
            raise ValueError(f"artifact.{key} must be {val!r}")
    for forbidden in ("decision_surface", "optimizer_ready_curves", "budget_recommendation"):
        if artifact.get(forbidden) is not None:
            raise ValueError(f"forbidden field present: {forbidden!r}")
    if "variants" not in artifact:
        raise ValueError("variants required")


def write_geometry_stabilization_artifact(
    output_path: str | Path | None = None,
    *,
    execute_fit: bool = True,
) -> dict[str, Any]:
    artifact = build_geometry_stabilization_artifact(execute_fit=execute_fit)
    out = Path(output_path or DEFAULT_OUTPUT)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2, default=str) + "\n", encoding="utf-8")
    validate_geometry_stabilization_artifact(artifact)
    artifact["artifact_path"] = str(out)
    return artifact


def main() -> int:
    artifact = write_geometry_stabilization_artifact(execute_fit=True)
    print(artifact["artifact_path"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
