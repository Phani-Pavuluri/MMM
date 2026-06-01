"""Bayes-H5 sandbox pilot — research-only validation over H5 worlds."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mmm.config.schema import BayesianBackend
from mmm.research.bayes_h3_sandbox.fencing import H5_MODEL_SPEC_VERSION
from mmm.research.bayes_h3_sandbox.h4_recovery_threshold_policy import evaluate_world_against_policy
from mmm.research.bayes_h3_sandbox.h5_transforms import list_transform_registry
from mmm.research.bayes_h3_sandbox.h5_validation_worlds import (
    H5_WORLD_IDS,
    h5_world_catalog_metadata,
    h5_world_production_flags,
)
from mmm.research.bayes_h3_sandbox.recovery_runner import run_h5_recovery_world
from mmm.research.bayes_h3_sandbox.recovery_worlds import RecoveryWorldSpec, get_recovery_world, recovery_world_config

PILOT_ID = "BAYES_H5_SANDBOX_PILOT_20260601"
PILOT_VERSION = "bayes_h5_sandbox_pilot_v1"
DEFAULT_ARTIFACT_PATH = Path("docs/05_validation/archives/BAYES_H5_SANDBOX_PILOT_20260601.json")


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except Exception:
            pass
    return str(value)


def _backend_metadata(spec: RecoveryWorldSpec, *, fast_mcmc: bool) -> dict[str, Any]:
    cfg = recovery_world_config(spec, fast_mcmc=fast_mcmc)
    b = cfg.bayesian
    return {
        "inference_backend": BayesianBackend.PYMC.value,
        "framework": cfg.framework.value,
        "pooling": cfg.pooling.value,
        "model_form": cfg.model_form.value,
        "draws": int(b.draws),
        "tune": int(b.tune),
        "chains": int(b.chains),
        "target_accept": float(b.target_accept),
        "nuts_seed": int(b.nuts_seed),
        "world_mcmc_seed": int(spec.mcmc_seed),
        "fast_mcmc_profile": fast_mcmc,
    }


def _world_row_from_report(report: dict[str, Any], spec: RecoveryWorldSpec) -> dict[str, Any]:
    rec = report.get("h4_recovery") or report.get("h5_recovery") or {}
    conv = rec.get("convergence_diagnostics") or report.get("convergence_diagnostics") or {}
    rhat = conv.get("rhat_max")
    h5_diag = report.get("h5_transform_diagnostics") or {}
    exp = spec.expected_diagnostic_behavior
    policy = evaluate_world_against_policy(spec.world_id, rec)
    return {
        "world_id": spec.world_id,
        "role": exp.get("role"),
        "h5_classification": exp.get("h5_classification"),
        "generative_transform": exp.get("generative_transform"),
        "fitted_transform_expectation": exp.get("fitted_transform_expectation"),
        "transform_mismatch_mode": exp.get("transform_mismatch_mode"),
        "beta_gc_mae": rec.get("beta_gc_mae"),
        "mu_c_mae": rec.get("mu_c_mae"),
        "beta_gc_coverage_90": rec.get("beta_gc_coverage_90"),
        "shrinkage_ratio_sparse": rec.get("shrinkage_ratio_sparse"),
        "shrinkage_ratio_sparse_vs_true_mu": rec.get("shrinkage_ratio_sparse_vs_true_mu"),
        "h5_transform_diagnostics": _json_safe(h5_diag),
        "h5_diagnostic_warnings": list(rec.get("h5_diagnostic_warnings") or []),
        "conflict_warnings": list(rec.get("conflict_warnings") or []),
        "policy_outcomes": policy,
        "convergence": {
            "rhat_max": rhat,
            "ess_bulk_min": conv.get("ess_bulk_min"),
        },
        "expected_diagnostic_behavior": dict(exp),
        **h5_world_production_flags(),
        "has_decision_surface": report.get("decision_surface") is not None,
        "has_budget_recommendation": report.get("budget_recommendation") is not None,
        "has_optimizer_ready_curves": report.get("optimizer_ready_curves") is not None,
    }


def build_h5_pilot_summary(
    world_rows: list[dict[str, Any]],
    *,
    fast_mcmc: bool = True,
    sampler_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble pilot JSON (research only — no production promotion)."""
    mismatch_rows = [r for r in world_rows if r.get("transform_mismatch_mode") == "intentional_mismatch"]
    aligned_rows = [r for r in world_rows if r.get("transform_mismatch_mode") == "aligned"]
    all_warnings: list[str] = []
    for r in world_rows:
        all_warnings.extend(r.get("h5_diagnostic_warnings") or [])

    return {
        "pilot_id": PILOT_ID,
        "pilot_version": PILOT_VERSION,
        "model_spec_version": H5_MODEL_SPEC_VERSION,
        "label": "RESEARCH ONLY — NOT DECISION GRADE",
        "research_only": True,
        "transform_registry": list_transform_registry(),
        "world_catalog": h5_world_catalog_metadata(),
        "sampler_settings": sampler_metadata or {"fast_mcmc_profile": fast_mcmc},
        "per_world_metrics": world_rows,
        "diagnostic_warnings": sorted(set(all_warnings)),
        "mismatch_worlds": [r["world_id"] for r in mismatch_rows],
        "aligned_worlds": [r["world_id"] for r in aligned_rows],
        "hard_gate": False,
        "production_promotion": False,
        "approved_for_prod": False,
        "prod_decisioning_allowed": False,
        "decision_grade": False,
        "outputs_are_diagnostic_only": True,
        "production_decision_surface": False,
        "note": (
            "H5 sandbox pilot — transform alignment probe only. "
            "INV-071 thresholds remain report-only; no optimizer or DecisionSurface."
        ),
    }


def run_h5_pilot(
    *,
    world_ids: tuple[str, ...] | None = None,
    fast_mcmc: bool = True,
    artifact_path: Path | None = None,
) -> dict[str, Any]:
    """Run fast H5 pilot over validation worlds and optionally write JSON artifact."""
    ids = world_ids or H5_WORLD_IDS
    rows: list[dict[str, Any]] = []
    sampler_meta: dict[str, Any] | None = None
    for wid in ids:
        spec = get_recovery_world(wid)
        report = run_h5_recovery_world(wid, fast_mcmc=fast_mcmc)
        if sampler_meta is None:
            sampler_meta = _backend_metadata(spec, fast_mcmc=fast_mcmc)
        rows.append(_world_row_from_report(report, spec))

    summary = build_h5_pilot_summary(rows, fast_mcmc=fast_mcmc, sampler_metadata=sampler_meta)
    path = artifact_path or DEFAULT_ARTIFACT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(summary), indent=2) + "\n", encoding="utf-8")
    summary["artifact_path"] = str(path)
    return summary


def validate_h5_pilot_schema(summary: dict[str, Any]) -> None:
    """Fail fast if pilot artifact missing required research-only keys."""
    required = {
        "model_spec_version",
        "world_catalog",
        "sampler_settings",
        "per_world_metrics",
        "diagnostic_warnings",
        "hard_gate",
        "production_promotion",
        "approved_for_prod",
        "prod_decisioning_allowed",
    }
    missing = required - set(summary.keys())
    if missing:
        raise ValueError(f"H5 pilot schema missing keys: {sorted(missing)}")
    if summary.get("hard_gate") is not False:
        raise ValueError("hard_gate must be false")
    if summary.get("approved_for_prod") is not False:
        raise ValueError("approved_for_prod must be false")
    for row in summary.get("per_world_metrics") or []:
        for key in ("decision_surface", "optimizer_ready_curves", "budget_recommendation", "recommendation"):
            if row.get(key) is not None:
                raise ValueError(f"forbidden production field {key!r} in per_world_metrics")
