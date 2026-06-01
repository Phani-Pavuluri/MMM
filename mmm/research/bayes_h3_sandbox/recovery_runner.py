"""Bayes-H4 recovery validation runner for H3 sandbox (research only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from mmm.research.bayes_h3_sandbox.entrypoint import run_sandbox_fit
from mmm.research.bayes_h3_sandbox.labels import validate_research_only_artifact
from mmm.research.bayes_h3_sandbox.recovery_worlds import (
    H4_WORLD_IDS,
    RecoveryWorldSpec,
    get_recovery_world,
    materialize_recovery_bundle,
)
from mmm.research.bayes_h3_sandbox.sparse_shrinkage_metrics import (
    beta_geo_index_order,
    channel_index_order,
    compute_sparse_shrinkage_decomposition,
    posterior_beta_means_by_geo,
    posterior_mu_tau,
)


def _beta_posterior_intervals(
    artifact: dict[str, Any],
    spec: RecoveryWorldSpec,
    *,
    q_low: float = 0.05,
    q_high: float = 0.95,
) -> dict[str, dict[str, tuple[float, float]]]:
    idata = artifact.get("idata")
    if idata is None:
        return {}
    geo_order = beta_geo_index_order(artifact, spec)
    ch_order = channel_index_order(artifact, spec)
    try:
        beta_da = idata.posterior["beta"].stack(sample=("chain", "draw"))
        beta = np.asarray(beta_da.values).reshape(-1, beta_da.shape[-2], beta_da.shape[-1])
    except Exception:
        return {}
    out: dict[str, dict[str, tuple[float, float]]] = {}
    for gi, geo in enumerate(geo_order):
        if gi >= beta.shape[1]:
            continue
        out[geo] = {}
        for ci, ch in enumerate(ch_order):
            if ci >= beta.shape[2]:
                continue
            draws = beta[:, gi, ci]
            out[geo][ch] = (float(np.quantile(draws, q_low)), float(np.quantile(draws, q_high)))
    return out


def compute_conflict_warnings(spec: RecoveryWorldSpec) -> list[str]:
    """Detect calibration stub directions that oppose generative truth (diagnostic only)."""
    warnings: list[str] = []
    for sig in spec.calibration_signals:
        ch = str(sig.get("channel", ""))
        scope = str(sig.get("scope_id", ""))
        if ch not in spec.channels or scope not in spec.true_beta_gc:
            continue
        true_b = spec.true_beta_gc[scope][ch]
        claimed = str(sig.get("claimed_direction", "")).lower()
        if claimed == "negative" and true_b > 0.25:
            warnings.append(
                f"conflict:{sig.get('signal_id')}: claimed negative lift on {scope}/{ch} "
                f"but true_beta_gc={true_b:.3f} is positive"
            )
        if claimed == "positive" and true_b < -0.05:
            warnings.append(
                f"conflict:{sig.get('signal_id')}: claimed positive lift on {scope}/{ch} "
                f"but true_beta_gc={true_b:.3f} is non-positive"
            )
    return warnings


def compute_recovery_metrics(artifact: dict[str, Any], spec: RecoveryWorldSpec) -> dict[str, Any]:
    """Compare sandbox posterior summaries to known truth (diagnostic only)."""
    beta_post = posterior_beta_means_by_geo(artifact, spec)
    mu_post, _tau_post = posterior_mu_tau(artifact, spec)
    intervals = _beta_posterior_intervals(artifact, spec)

    beta_errors: list[float] = []
    mu_errors: list[float] = []
    covered = 0
    total_intervals = 0

    for geo in spec.geo_order:
        for ch in spec.channels:
            true_b = spec.true_beta_gc[geo][ch]
            post_b = beta_post.get(geo, {}).get(ch, float("nan"))
            if post_b == post_b:
                beta_errors.append(abs(post_b - true_b))
            true_mu = spec.true_mu_c[ch]
            post_mu = mu_post.get(ch, float("nan"))
            if post_mu == post_mu:
                mu_errors.append(abs(post_mu - true_mu))
            if geo in intervals and ch in intervals[geo]:
                lo, hi = intervals[geo][ch]
                total_intervals += 1
                if lo <= true_b <= hi:
                    covered += 1

    sparse_decomp: dict[str, Any] | None = None
    shrinkage_ratio_sparse: float | None = None
    shrinkage_ratio_sparse_vs_true_mu: float | None = None
    if spec.sparse_geos:
        sparse_decomp = compute_sparse_shrinkage_decomposition(artifact, spec)
        shrinkage_ratio_sparse = sparse_decomp.get("shrinkage_ratio_sparse")
        shrinkage_ratio_sparse_vs_true_mu = sparse_decomp.get("shrinkage_ratio_sparse_vs_true_mu")

    conflict_warnings = compute_conflict_warnings(spec)
    trust = artifact.get("diagnostic_trust_report") or {}
    if conflict_warnings:
        trust = {**trust, "conflict_warnings": conflict_warnings}

    return {
        "world_id": spec.world_id,
        "known_truth": spec.known_truth,
        "beta_gc_mae": float(np.mean(beta_errors)) if beta_errors else None,
        "mu_c_mae": float(np.mean(mu_errors)) if mu_errors else None,
        "beta_gc_coverage_90": (covered / total_intervals) if total_intervals else None,
        "shrinkage_ratio_sparse": shrinkage_ratio_sparse,
        "shrinkage_ratio_sparse_vs_true_mu": shrinkage_ratio_sparse_vs_true_mu,
        "sparse_shrinkage_decomposition": sparse_decomp,
        "posterior_indexing": {
            "beta_geo_index_order": beta_geo_index_order(artifact, spec),
            "channel_index_order": channel_index_order(artifact, spec),
            "spec_geo_order": list(spec.geo_order),
            "index_order_matches_spec": beta_geo_index_order(artifact, spec) == list(spec.geo_order),
        },
        "conflict_warnings": conflict_warnings,
        "expected_diagnostic_behavior": dict(spec.expected_diagnostic_behavior),
        "convergence_diagnostics": artifact.get("convergence_diagnostics"),
        "diagnostic_trust_excerpt": {
            "trust_report_kind": trust.get("trust_report_kind"),
            "conflict_warnings": conflict_warnings,
        },
        "outputs_are_diagnostic_only": True,
        "approved_for_prod": False,
        "prod_decisioning_allowed": False,
    }


def build_h4_recovery_report(
    spec: RecoveryWorldSpec,
    artifact: dict[str, Any],
) -> dict[str, Any]:
    """Attach H4 recovery metrics to a research-only sandbox artifact."""
    metrics = compute_recovery_metrics(artifact, spec)
    out = dict(artifact)
    out["h4_recovery"] = metrics
    out["world_id"] = spec.world_id
    if metrics.get("conflict_warnings"):
        dtr = dict(out.get("diagnostic_trust_report") or {})
        dtr["conflict_warnings"] = metrics["conflict_warnings"]
        out["diagnostic_trust_report"] = dtr
    validate_research_only_artifact(out)
    return out


def run_h4_recovery_world(
    world_id: str,
    *,
    fast_mcmc: bool = True,
    sampler: dict[str, Any] | None = None,
    nuts_seed: int | None = None,
    panel_seed: int | None = None,
    sandbox_model_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Materialize world, run sandbox fit, compute recovery metrics (research only).

    Does not emit production DecisionSurface, optimizer inputs, or recommendations.
    """
    spec = get_recovery_world(world_id)
    cfg, schema, df = materialize_recovery_bundle(
        spec,
        fast_mcmc=fast_mcmc,
        sampler=sampler,
        nuts_seed=nuts_seed,
        panel_seed=panel_seed,
    )
    overrides = dict(spec.sandbox_model_overrides)
    if sandbox_model_overrides:
        overrides.update(sandbox_model_overrides)
    artifact = run_sandbox_fit(
        cfg,
        schema,
        df,
        geo_hierarchy_mapping=spec.geo_hierarchy,
        calibration_signals_stub=list(spec.calibration_signals),
        sandbox_model_overrides=overrides or None,
    )
    return build_h4_recovery_report(spec, artifact)


def validate_world_catalog() -> dict[str, Any]:
    """Fast validation that all H4 worlds materialize deterministically."""
    results: list[dict[str, Any]] = []
    for wid in H4_WORLD_IDS:
        spec = get_recovery_world(wid)
        df1 = materialize_recovery_bundle(spec)[2]
        df2 = materialize_recovery_bundle(spec)[2]
        ok = df1.equals(df2)
        results.append(
            {
                "world_id": wid,
                "rows": len(df1),
                "deterministic": ok,
                "has_known_truth": bool(spec.known_truth),
            }
        )
    return {
        "status": "pass" if all(r["deterministic"] and r["has_known_truth"] for r in results) else "fail",
        "worlds": results,
    }


def validate_posterior_index_mapping(artifact: dict[str, Any], spec: RecoveryWorldSpec) -> dict[str, Any]:
    """Expose beta/geo/channel index mapping for tests (research only)."""
    geo_order = beta_geo_index_order(artifact, spec)
    ch_order = channel_index_order(artifact, spec)
    beta_post = posterior_beta_means_by_geo(artifact, spec)
    return {
        "beta_geo_index_order": geo_order,
        "channel_index_order": ch_order,
        "beta_keys_by_geo": {g: sorted(beta_post.get(g, {}).keys()) for g in geo_order},
        "spec_geo_order": list(spec.geo_order),
        "spec_channels": list(spec.channels),
    }
