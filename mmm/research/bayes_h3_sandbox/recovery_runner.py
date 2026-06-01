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


def _posterior_beta_means(artifact: dict[str, Any], spec: RecoveryWorldSpec) -> dict[str, dict[str, float]]:
    hier = artifact.get("hierarchy_evidence_diagnostics") or {}
    raw = hier.get("beta_geo_channel_mean") or {}
    out: dict[str, dict[str, float]] = {}
    for gi, geo in enumerate(spec.geo_order):
        ch_map = raw.get(str(gi), raw.get(geo, {}))
        if not isinstance(ch_map, dict):
            continue
        out[geo] = {str(c): float(ch_map.get(c, ch_map.get(str(c), float("nan")))) for c in spec.channels}
    return out


def _posterior_mu_tau(artifact: dict[str, Any], spec: RecoveryWorldSpec) -> tuple[dict[str, float], dict[str, float]]:
    ps = artifact.get("posterior_summary") or {}
    mu = {c: float(ps.get("mu_channel_mean", {}).get(c, float("nan"))) for c in spec.channels}
    tau = {c: float(ps.get("tau_channel_mean", {}).get(c, float("nan"))) for c in spec.channels}
    pool = artifact.get("pooling_diagnostics") or {}
    if pool.get("tau_channel_mean"):
        tau = {c: float(pool["tau_channel_mean"].get(c, tau[c])) for c in spec.channels}
    return mu, tau


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
    try:
        beta = idata.posterior["beta"].stack(sample=("chain", "draw")).values
    except Exception:
        return {}
    out: dict[str, dict[str, tuple[float, float]]] = {}
    for gi, geo in enumerate(spec.geo_order):
        out[geo] = {}
        for ci, ch in enumerate(spec.channels):
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
    beta_post = _posterior_beta_means(artifact, spec)
    mu_post, tau_post = _posterior_mu_tau(artifact, spec)
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

    shrinkage_ratios: list[float] = []
    for geo in spec.sparse_geos:
        for ch in spec.channels:
            true_b = spec.true_beta_gc[geo][ch]
            mu_c = spec.true_mu_c[ch]
            post_b = beta_post.get(geo, {}).get(ch, float("nan"))
            if post_b != post_b:
                continue
            d_true = abs(true_b - mu_c)
            d_post = abs(post_b - mu_c)
            if d_true > 1e-6:
                shrinkage_ratios.append(d_post / d_true)

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
        "shrinkage_ratio_sparse": float(np.mean(shrinkage_ratios)) if shrinkage_ratios else None,
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
) -> dict[str, Any]:
    """
    Materialize world, run sandbox fit, compute recovery metrics (research only).

    Does not emit production DecisionSurface, optimizer inputs, or recommendations.
    """
    spec = get_recovery_world(world_id)
    cfg, schema, df = materialize_recovery_bundle(spec, fast_mcmc=fast_mcmc)
    artifact = run_sandbox_fit(
        cfg,
        schema,
        df,
        geo_hierarchy_mapping=spec.geo_hierarchy,
        calibration_signals_stub=list(spec.calibration_signals),
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
