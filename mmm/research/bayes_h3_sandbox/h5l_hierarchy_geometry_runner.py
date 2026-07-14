"""Bayes-H5l hierarchy-faithful geometry refinement runner (sample panel only)."""

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
    BETA_PRIOR_CHANNEL_SCALED,
    BETA_PRIOR_CURRENT,
    BETA_PRIOR_STRONGER,
    HIERARCHY_FIXED_TAU,
    HIERARCHY_FULL_GEO_CHANNEL,
    HIERARCHY_POOLED_CHANNEL,
    HIERARCHY_STRENGTH_FIXED_TAU,
    HIERARCHY_STRENGTH_LEARNED,
    HIERARCHY_STRENGTH_STRONG_REG,
    HIERARCHY_STRENGTH_WEAK_REG,
    LIKELIHOOD_PRESCALED_LOG_OUTCOME,
    PARAMETERIZATION_NON_CENTERED,
    SIGMA_POLICY_CURRENT,
    SIGMA_POLICY_FLOOR,
    SIGMA_POLICY_PRIOR_REGULARIZED,
    TAU_PARAM_CURRENT,
    TAU_PARAM_LOG_TAU,
    TAU_PARAM_NONCENTERED_LOG_TAU,
    evidence_promotion_for_geometry,
    geometry_record_for_artifact,
)
from mmm.research.bayes_h3_sandbox.h5_real_panel_preprocessing import (
    CHANNEL_POLICY_DROP_COLLINEAR,
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
    research_production_flags,
)
from mmm.research.bayes_h3_sandbox.recovery_worlds import SAMPLER_EXTENDED

INVESTIGATION_ID = "INV-H5L_HIERARCHY_GEOMETRY_REFINEMENT"
ARTIFACT_ID = "BAYES_H5L_HIERARCHY_GEOMETRY_REFINEMENT_20260601"
DEFAULT_H5K_ARTIFACT = Path(
    "docs/05_validation/archives/BAYES_H5K_GEOMETRY_STABILIZATION_20260601.json"
)
DEFAULT_OUTPUT = Path(f"docs/05_validation/archives/{ARTIFACT_ID}.json")

H5J_BEST_CHANNEL_POLICY = {
    "mode": CHANNEL_POLICY_DROP_COLLINEAR,
    "max_abs_corr_threshold": 0.95,
}
BASE_REAL_PANEL_OVERRIDES = {
    "h5_panel_context": "real_panel",
    "h5_real_panel": True,
}
DEFAULT_SIGMA_FLOOR = 0.05


@dataclass(frozen=True)
class HierarchyGeometrySpec:
    variant_id: str
    geometry_config: dict[str, Any]
    channel_policy: dict[str, Any]
    sampler_overrides: dict[str, Any]
    sampler_profile: str
    hierarchy_faithful: bool
    ablation_only: bool
    notes: str = ""


def _base_full_hierarchy(**extras: Any) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "parameterization": PARAMETERIZATION_NON_CENTERED,
        "likelihood_scale_policy": LIKELIHOOD_PRESCALED_LOG_OUTCOME,
        "hierarchy_policy": HIERARCHY_FULL_GEO_CHANNEL,
        "tau_parameterization": TAU_PARAM_CURRENT,
        "sigma_policy": SIGMA_POLICY_CURRENT,
        "beta_prior_policy": BETA_PRIOR_CURRENT,
        "hierarchy_strength_policy": HIERARCHY_STRENGTH_LEARNED,
    }
    cfg.update(extras)
    return cfg


def default_hierarchy_specs() -> list[HierarchyGeometrySpec]:
    extended = dict(SAMPLER_EXTENDED)
    drop = dict(H5J_BEST_CHANNEL_POLICY)

    def spec(
        variant_id: str,
        geom: dict[str, Any],
        *,
        faithful: bool,
        ablation: bool,
        notes: str,
    ) -> HierarchyGeometrySpec:
        return HierarchyGeometrySpec(
            variant_id=variant_id,
            geometry_config=geom,
            channel_policy=drop,
            sampler_overrides=extended,
            sampler_profile="extended",
            hierarchy_faithful=faithful,
            ablation_only=ablation,
            notes=notes,
        )

    return [
        spec(
            "H5L-A-H5K-FULL-HIERARCHY-REPLAY",
            _base_full_hierarchy(),
            faithful=True,
            ablation=False,
            notes="H5k-B replay: explicit NC + prescale + drop collinear",
        ),
        spec(
            "H5L-B-NC-SIGMA-FLOOR",
            _base_full_hierarchy(
                sigma_policy=SIGMA_POLICY_FLOOR,
                sigma_floor=DEFAULT_SIGMA_FLOOR,
            ),
            faithful=True,
            ablation=False,
            notes="Non-centered beta + sigma floor on likelihood",
        ),
        spec(
            "H5L-C-NC-STRONGER-TAU-PRIOR",
            _base_full_hierarchy(hierarchy_strength_policy=HIERARCHY_STRENGTH_STRONG_REG),
            faithful=True,
            ablation=False,
            notes="Stronger regularization on tau (smaller HalfNormal scale)",
        ),
        spec(
            "H5L-D-NC-STRONGER-BETA-PRIOR",
            _base_full_hierarchy(beta_prior_policy=BETA_PRIOR_STRONGER),
            faithful=True,
            ablation=False,
            notes="Tighter mu_channel and z_beta priors",
        ),
        spec(
            "H5L-E-NC-SIGMA-FLOOR-STRONG-TAU",
            _base_full_hierarchy(
                sigma_policy=SIGMA_POLICY_FLOOR,
                sigma_floor=DEFAULT_SIGMA_FLOOR,
                hierarchy_strength_policy=HIERARCHY_STRENGTH_STRONG_REG,
            ),
            faithful=True,
            ablation=False,
            notes="Combined sigma floor + stronger tau prior",
        ),
        spec(
            "H5L-F-LOG-TAU-REPARAM",
            _base_full_hierarchy(tau_parameterization=TAU_PARAM_LOG_TAU),
            faithful=True,
            ablation=False,
            notes="Log-scale tau parameterization (positive tau via exp)",
        ),
        spec(
            "H5L-G-NC-LOG-TAU-REPARAM",
            _base_full_hierarchy(tau_parameterization=TAU_PARAM_NONCENTERED_LOG_TAU),
            faithful=True,
            ablation=False,
            notes="Non-centered log-tau hierarchy",
        ),
        spec(
            "H5L-H-NC-WEAK-TAU-SIGMA-REG",
            _base_full_hierarchy(
                hierarchy_strength_policy=HIERARCHY_STRENGTH_WEAK_REG,
                sigma_policy=SIGMA_POLICY_PRIOR_REGULARIZED,
            ),
            faithful=True,
            ablation=False,
            notes="Weak tau reg + tighter sigma prior",
        ),
        spec(
            "H5L-I-NC-CHANNEL-SCALED-BETA",
            _base_full_hierarchy(beta_prior_policy=BETA_PRIOR_CHANNEL_SCALED),
            faithful=True,
            ablation=False,
            notes="Beta offsets scaled by channel media scale",
        ),
        spec(
            "H5L-J-FIXED-TAU-BENCHMARK",
            _base_full_hierarchy(
                hierarchy_policy=HIERARCHY_FIXED_TAU,
                hierarchy_strength_policy=HIERARCHY_STRENGTH_FIXED_TAU,
                fixed_tau_value=0.2,
            ),
            faithful=False,
            ablation=True,
            notes="Benchmark only — not hierarchy-faithful promotion evidence",
        ),
        spec(
            "H5L-K-POOLED-BENCHMARK",
            _base_full_hierarchy(hierarchy_policy=HIERARCHY_POOLED_CHANNEL),
            faithful=False,
            ablation=True,
            notes="Benchmark only — pooled channels, not geo-specific betas",
        ),
    ]


def run_hierarchy_geometry_variant(
    spec: HierarchyGeometrySpec,
    *,
    base_transform_config: dict[str, Any],
    panel_path: str | Path = DEFAULT_PANEL_PATH,
    execute_fit: bool = True,
) -> dict[str, Any]:
    geom = dict(spec.geometry_config)
    row: dict[str, Any] = {
        "variant_id": spec.variant_id,
        "parameterization": geom.get("parameterization"),
        "tau_parameterization": geom.get("tau_parameterization"),
        "sigma_policy": geom.get("sigma_policy"),
        "beta_prior_policy": geom.get("beta_prior_policy"),
        "hierarchy_policy": geom.get("hierarchy_policy"),
        "hierarchy_strength_policy": geom.get("hierarchy_strength_policy"),
        "likelihood_scale_policy": geom.get("likelihood_scale_policy"),
        "channel_policy": spec.channel_policy,
        "sampler_profile": spec.sampler_profile,
        "sampler_settings": spec.sampler_overrides,
        "hierarchy_faithful": spec.hierarchy_faithful,
        "ablation_only": spec.ablation_only,
        "geometry_config": geometry_record_for_artifact(
            {**geom, "hierarchy_faithful": spec.hierarchy_faithful, "ablation_only": spec.ablation_only}
        ),
        "notes": spec.notes,
        **research_production_flags(),
    }

    if not execute_fit:
        return {
            **row,
            "rhat_max": None,
            "ess_min": None,
            "divergence_count": None,
            "convergence_status": "not_executed",
            "evidence_promotion_allowed": False,
            "warnings": ["h5:production:block", "h5:evidence:blocked"],
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
            **row,
            "convergence_status": "not_executed",
            "evidence_promotion_allowed": False,
            "error": str(exc),
            "warnings": ["h5:production:block", "h5:evidence:blocked"],
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
        "h5_geometry_config": geom,
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
    resolved_geom = {**geom}
    resolved_geom["hierarchy_faithful"] = spec.hierarchy_faithful
    resolved_geom["ablation_only"] = spec.ablation_only
    status = classify_convergence_status(
        rhat_max=conv.get("rhat_max"),
        divergence_count=conv.get("divergence_count"),
    )
    promo = evidence_promotion_for_geometry(status, resolved_geom)
    warnings = derive_real_panel_transform_warning_codes(transform_config)
    if not promo:
        warnings = sorted(set(warnings + ["h5:evidence:blocked"]))
    if spec.ablation_only:
        warnings = sorted(set(warnings + ["h5:ablation:benchmark_only"]))

    return {
        **row,
        "channel_policy_record": policy_record,
        "active_channels": list(schema.channel_columns),
        "rhat_max": conv.get("rhat_max"),
        "ess_min": conv.get("ess_bulk_min"),
        "divergence_count": conv.get("divergence_count"),
        "worst_rhat_parameters": conv.get("worst_rhat_parameters"),
        "convergence_status": status,
        "evidence_promotion_allowed": promo,
        "warnings": warnings,
        "recommendation": _recommendation(status, spec.ablation_only),
        "geometry_diagnostics_from_fit": fit_artifact.get("h5_geometry_diagnostics"),
    }


def _recommendation(status: str, ablation_only: bool) -> str:
    if ablation_only:
        return "benchmark_ablation_only"
    if status == "converged_diagnostic_only":
        return "report_only_pass_hierarchy_faithful_candidate"
    if status == "weak_convergence":
        return "report_only_weak"
    return "report_only_fail"


def _pick_best_faithful(variants: list[dict[str, Any]]) -> dict[str, Any] | None:
    faithful = [v for v in variants if v.get("hierarchy_faithful") and v.get("rhat_max") is not None]
    if not faithful:
        return None
    return min(
        faithful,
        key=lambda v: (
            0 if v.get("convergence_status") == "converged_diagnostic_only" else 1,
            float(v["rhat_max"]),
            int(v.get("divergence_count") or 999),
        ),
    )


def build_hierarchy_geometry_artifact(
    *,
    panel_path: str | Path = DEFAULT_PANEL_PATH,
    transform_config_path: str | Path = DEFAULT_TRANSFORM_CONFIG,
    execute_fit: bool = True,
    specs: list[HierarchyGeometrySpec] | None = None,
) -> dict[str, Any]:
    base_config = load_transform_config(transform_config_path)
    h5k = _load_h5k_artifact()
    spec_list = specs or default_hierarchy_specs()
    variants = [
        run_hierarchy_geometry_variant(
            s,
            base_transform_config=base_config,
            panel_path=panel_path,
            execute_fit=execute_fit,
        )
        for s in spec_list
    ]
    faithful_variants = [v for v in variants if v.get("hierarchy_faithful")]
    ablation_benchmarks = [v for v in variants if v.get("ablation_only")]
    best_faithful = _pick_best_faithful(variants)
    any_faithful_converged = any(
        v.get("hierarchy_faithful") and v.get("convergence_status") == "converged_diagnostic_only"
        for v in variants
    )

    return {
        "artifact_id": ARTIFACT_ID,
        "investigation_id": INVESTIGATION_ID,
        "panel_id": PANEL_ID,
        "dataset_snapshot_id": DATASET_SNAPSHOT_ID,
        "panel_path": str(panel_path),
        "sample_panel_metadata": {
            "n_geos": 3,
            "n_weeks": 41,
            "n_rows": 123,
            "channels_baseline": ["search", "social", "tv"],
            "channels_after_drop_collinear": ["search", "social"],
        },
        "source_h5k_artifact": str(DEFAULT_H5K_ARTIFACT),
        "source_h5k_summary": {
            "best_full_hierarchy": "H5K-B (rhat≈1.02, 4 div)",
            "ablation_pass": ["H5K-E pooled", "H5K-F fixed-tau"],
        },
        "h5k_reference": h5k.get("best_variant") if h5k else None,
        "variants": variants,
        "full_hierarchy_variants": faithful_variants,
        "ablation_benchmarks": ablation_benchmarks,
        "best_hierarchy_faithful_variant": best_faithful,
        "any_hierarchy_faithful_converged_diagnostic_only": any_faithful_converged,
        "ablation_variants_not_promotion_evidence": (
            "Pooled and fixed-tau benchmark variants may reach converged_diagnostic_only "
            "but must not be used as production or default H5 research evidence."
        ),
        "recommendation": _artifact_recommendation(any_faithful_converged, best_faithful, variants),
        "shadow_run_eligibility_rule": (
            "No additional real-panel shadow runs until a hierarchy-faithful variant reaches "
            "converged_diagnostic_only under explicit channel + geometry policy."
        ),
        "label": "RESEARCH ONLY — NOT DECISION GRADE",
        **research_production_flags(),
        "outputs_are_diagnostic_only": True,
    }


def _artifact_recommendation(
    any_faithful_converged: bool,
    best_faithful: dict[str, Any] | None,
    variants: list[dict[str, Any]],
) -> str:
    if any_faithful_converged and best_faithful:
        return (
            f"Hierarchy-faithful pilot candidate: {best_faithful['variant_id']} — "
            "verify on holdout policy before more real panels."
        )
    if best_faithful:
        return (
            f"No hierarchy-faithful variant cleared evidence bar. Best faithful probe: "
            f"{best_faithful['variant_id']} (rhat_max={best_faithful.get('rhat_max')}, "
            f"divergences={best_faithful.get('divergence_count')}). "
            "Ablation benchmarks are diagnostic only."
        )
    return "Continue hierarchy-faithful geometry research; ablation passes are not promotion evidence."


def _load_h5k_artifact() -> dict[str, Any]:
    if DEFAULT_H5K_ARTIFACT.is_file():
        return json.loads(DEFAULT_H5K_ARTIFACT.read_text(encoding="utf-8"))
    return {}


def validate_hierarchy_geometry_artifact(artifact: dict[str, Any]) -> None:
    for key, val in research_production_flags().items():
        if artifact.get(key) is not val:
            raise ValueError(f"artifact.{key} must be {val!r}")
    for forbidden in ("decision_surface", "optimizer_ready_curves", "budget_recommendation"):
        if artifact.get(forbidden) is not None:
            raise ValueError(f"forbidden field present: {forbidden!r}")
    if "ablation_variants_not_promotion_evidence" not in artifact:
        raise ValueError("ablation_variants_not_promotion_evidence required")
    for row in artifact.get("variants", []):
        if row.get("ablation_only") and row.get("evidence_promotion_allowed"):
            raise ValueError(f"{row.get('variant_id')}: ablation cannot be promotion allowed")


def write_hierarchy_geometry_artifact(
    output_path: str | Path | None = None,
    *,
    execute_fit: bool = True,
) -> dict[str, Any]:
    artifact = build_hierarchy_geometry_artifact(execute_fit=execute_fit)
    out = Path(output_path or DEFAULT_OUTPUT)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2, default=str) + "\n", encoding="utf-8")
    validate_hierarchy_geometry_artifact(artifact)
    artifact["artifact_path"] = str(out)
    return artifact


def main() -> int:
    artifact = write_hierarchy_geometry_artifact(execute_fit=True)
    print(artifact["artifact_path"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
