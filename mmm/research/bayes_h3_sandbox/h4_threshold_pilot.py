"""Bayes-H4a recovery threshold pilot — research-only report generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mmm.config.schema import BayesianBackend
from mmm.research.bayes_h3_sandbox.recovery_runner import run_h4_recovery_world
from mmm.research.bayes_h3_sandbox.recovery_worlds import (
    H4_WORLD_IDS,
    RecoveryWorldSpec,
    get_recovery_world,
    recovery_world_config,
)

PILOT_ID = "BAYES_H4_THRESHOLD_PILOT_20260601"
PILOT_VERSION = "bayes_h4_threshold_pilot_v1"
DEFAULT_ARTIFACT_PATH = Path("docs/05_validation/archives/BAYES_H4_THRESHOLD_PILOT_20260601.json")


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


def _known_truth_ref(spec: RecoveryWorldSpec) -> dict[str, Any]:
    return {
        "true_mu_c": dict(spec.true_mu_c),
        "true_tau_c": dict(spec.true_tau_c),
        "true_beta_gc": {g: dict(ch) for g, ch in spec.true_beta_gc.items()},
        "noise_sigma": spec.noise_sigma,
        "geo_order": list(spec.geo_order),
        "channels": list(spec.channels),
    }


def _world_row_from_report(report: dict[str, Any], spec: RecoveryWorldSpec) -> dict[str, Any]:
    rec = report.get("h4_recovery") or {}
    conv = rec.get("convergence_diagnostics") or report.get("convergence_diagnostics") or {}
    rhat = conv.get("rhat_max")
    posterior_indexing = rec.get("posterior_indexing") or {}
    return {
        "world_id": spec.world_id,
        "seed": int(spec.mcmc_seed),
        "beta_gc_mae": rec.get("beta_gc_mae"),
        "mu_c_mae": rec.get("mu_c_mae"),
        "beta_gc_coverage_90": rec.get("beta_gc_coverage_90"),
        "shrinkage_ratio_sparse": rec.get("shrinkage_ratio_sparse"),
        "shrinkage_ratio_sparse_vs_true_mu": rec.get("shrinkage_ratio_sparse_vs_true_mu"),
        "sparse_shrinkage_decomposition": rec.get("sparse_shrinkage_decomposition"),
        "beta_geo_index_order": list(posterior_indexing.get("beta_geo_index_order") or []),
        "channel_index_order": list(posterior_indexing.get("channel_index_order") or []),
        "conflict_warnings": list(rec.get("conflict_warnings") or []),
        "convergence": {
            "rhat_max": rhat,
            "ess_bulk_min": conv.get("ess_bulk_min"),
            "converged_sanity": rhat == rhat and float(rhat) < 1.2 if rhat is not None else None,
        },
        "known_truth_ref": _known_truth_ref(spec),
        "expected_diagnostic_behavior": dict(spec.expected_diagnostic_behavior),
        "research_only": True,
        "approved_for_prod": False,
        "prod_decisioning_allowed": False,
        "production_promotion": False,
        "decision_grade": False,
        "production_decision_surface": report.get("production_decision_surface", False),
        "has_decision_surface": report.get("decision_surface") is not None,
        "has_budget_recommendation": report.get("budget_recommendation") is not None,
        "has_optimizer_ready_curves": report.get("optimizer_ready_curves") is not None,
    }


def build_provisional_threshold_bands(world_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Derive report-only provisional bands from pilot world rows (not promotion gates)."""
    beta_maes = [float(r["beta_gc_mae"]) for r in world_rows if r.get("beta_gc_mae") is not None]
    mu_maes = [float(r["mu_c_mae"]) for r in world_rows if r.get("mu_c_mae") is not None]
    coverages = [float(r["beta_gc_coverage_90"]) for r in world_rows if r.get("beta_gc_coverage_90") is not None]
    shrink = [float(r["shrinkage_ratio_sparse"]) for r in world_rows if r.get("shrinkage_ratio_sparse") is not None]

    bands: dict[str, Any] = {
        "report_only": True,
        "hard_gate": False,
        "production_promotion": False,
        "beta_gc_mae": {
            "mode": "report",
            "pilot_max": max(beta_maes) if beta_maes else None,
            "provisional_warn_above": (max(beta_maes) * 1.5) if beta_maes else None,
            "note": "No hard fail threshold until repeated pilots and extended worlds.",
        },
        "mu_c_mae": {
            "mode": "report",
            "pilot_max": max(mu_maes) if mu_maes else None,
            "provisional_warn_above": (max(mu_maes) * 1.5) if mu_maes else None,
            "note": "Report-only; toy worlds are not calibrated for national-scale μ priors.",
        },
        "beta_gc_coverage_90": {
            "mode": "directional_expectation",
            "pilot_range": [min(coverages), max(coverages)] if coverages else None,
            "note": "Do not require exact 90% on tiny toy panels; use for trend monitoring only.",
        },
        "shrinkage_ratio_sparse": {
            "mode": "directional",
            "expect_lt": 1.0,
            "pilot_values": shrink,
            "note": "Sparse geo posterior should shrink toward μ_c vs generative outlier.",
        },
        "conflict_warnings": {
            "mode": "required_non_empty",
            "applies_to": "WORLD-BAYES-H4-CONFLICTING-EVIDENCE",
            "note": "Conflict world must emit at least one diagnostic conflict warning.",
        },
        "production_flags": {
            "approved_for_prod": False,
            "prod_decisioning_allowed": False,
            "decision_grade": False,
            "note": "Must always remain blocked for Bayes-H4 pilot artifacts.",
        },
    }
    return bands


def build_pilot_summary(
    world_rows: list[dict[str, Any]],
    *,
    pilot_id: str = PILOT_ID,
    backend_profiles: dict[str, Any] | None = None,
    status: str = "complete",
) -> dict[str, Any]:
    """Build deterministic pilot summary from per-world rows (for tests or live runs)."""
    rows = sorted(world_rows, key=lambda r: str(r["world_id"]))
    return _json_safe(
        {
            "pilot_id": pilot_id,
            "pilot_version": PILOT_VERSION,
            "status": status,
            "label": "RESEARCH ONLY — NOT DECISION GRADE",
            "research_only": True,
            "approved_for_prod": False,
            "prod_decisioning_allowed": False,
            "production_promotion": False,
            "decision_grade": False,
            "outputs_are_diagnostic_only": True,
            "backend_defaults": backend_profiles or {},
            "world_ids": [r["world_id"] for r in rows],
            "provisional_thresholds": build_provisional_threshold_bands(rows),
            "worlds": rows,
        }
    )


def run_h4_threshold_pilot(
    world_ids: tuple[str, ...] | None = None,
    *,
    fast_mcmc: bool = True,
) -> dict[str, Any]:
    """Run H4 recovery worlds and aggregate threshold pilot metrics (research only)."""
    ids = world_ids or H4_WORLD_IDS
    rows: list[dict[str, Any]] = []
    backends: dict[str, Any] = {}
    for wid in ids:
        spec = get_recovery_world(wid)
        backends[wid] = _backend_metadata(spec, fast_mcmc=fast_mcmc)
        report = run_h4_recovery_world(wid, fast_mcmc=fast_mcmc)
        rows.append(_world_row_from_report(report, spec))
    return build_pilot_summary(rows, backend_profiles=backends)


def write_h4_threshold_pilot_artifact(
    path: str | Path | None = None,
    summary: dict[str, Any] | None = None,
) -> Path:
    out_path = Path(path or DEFAULT_ARTIFACT_PATH)
    payload = summary if summary is not None else run_h4_threshold_pilot()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True)
    out_path.write_text(text + "\n", encoding="utf-8")
    return out_path


def load_h4_threshold_pilot_artifact(path: str | Path | None = None) -> dict[str, Any]:
    p = Path(path or DEFAULT_ARTIFACT_PATH)
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run Bayes-H4 threshold pilot and write JSON artifact")
    parser.add_argument("--output", type=Path, default=DEFAULT_ARTIFACT_PATH)
    parser.add_argument("--fast-mcmc", action="store_true", default=True)
    args = parser.parse_args()
    out = write_h4_threshold_pilot_artifact(args.output)
    print(json.dumps({"written": str(out), "pilot_id": PILOT_ID}))


if __name__ == "__main__":
    main()
