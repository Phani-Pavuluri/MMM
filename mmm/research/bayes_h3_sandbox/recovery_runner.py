"""Bayes-H4 recovery validation runner for H3 sandbox (research only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from mmm.research.bayes_h3_sandbox.entrypoint import run_sandbox_fit
from mmm.research.bayes_h3_sandbox.fencing import H5_MODEL_SPEC_VERSION
from mmm.research.bayes_h3_sandbox.h5_transforms import transforms_aligned
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


def _panel_max_channel_correlation(df: Any, channels: tuple[str, ...]) -> float | None:
    import pandas as pd

    if not isinstance(df, pd.DataFrame) or len(channels) < 2:
        return None
    cols = [c for c in channels if c in df.columns]
    if len(cols) < 2:
        return None
    corr = df[cols].corr().to_numpy()
    n = corr.shape[0]
    off_diag = [abs(corr[i, j]) for i in range(n) for j in range(n) if i != j]
    return float(max(off_diag)) if off_diag else None


def compute_h4c_diagnostic_warnings(
    artifact: dict[str, Any],
    spec: RecoveryWorldSpec,
    *,
    panel_df: Any | None = None,
) -> list[str]:
    """H4c reliability-map warnings (research only — not production gates)."""
    warnings: list[str] = []
    exp = spec.expected_diagnostic_behavior
    kind = str(exp.get("generative_transform", exp.get("generative_kind", "linear")))

    if exp.get("h5_classification"):
        pass  # H5 worlds use compute_h5_diagnostic_warnings for transform probes
    elif exp.get("transform_mismatch_warning_expected") or kind in ("adstock", "saturation"):
        warnings.append(
            f"h4c:transform_mismatch:{spec.world_id}: generative={kind} vs MVP semi_log on raw standardized media"
        )
    if exp.get("collinearity_warning_expected") and panel_df is not None:
        max_corr = _panel_max_channel_correlation(panel_df, spec.channels)
        if max_corr is not None and max_corr > 0.85:
            warnings.append(f"h4c:collinearity:{spec.world_id}: max_channel_corr={max_corr:.3f}")
    if exp.get("conflict_warning_expected"):
        warnings.extend(compute_conflict_warnings(spec))

    rec = artifact.get("h4_recovery") or {}
    beta_mae = rec.get("beta_gc_mae")
    if exp.get("h4c_classification") == "weak_identification" and beta_mae is not None and float(beta_mae) > 0.35:
        warnings.append(f"h4c:weak_identification:{spec.world_id}: beta_gc_mae={float(beta_mae):.3f}")

    return warnings


def compute_h5_diagnostic_warnings(
    artifact: dict[str, Any],
    spec: RecoveryWorldSpec,
    *,
    panel_df: Any | None = None,
) -> list[str]:
    """H5 transform / weak-ID warnings (research only — not production gates)."""
    warnings: list[str] = []
    exp = spec.expected_diagnostic_behavior
    h5_diag = artifact.get("h5_transform_diagnostics") or {}
    gen = str(exp.get("generative_transform", "linear"))
    fitted = str(exp.get("fitted_transform_id", "identity"))

    if exp.get("transform_mismatch_warning_expected"):
        if h5_diag.get("transform_mismatch_detected"):
            warnings.append(
                f"h5:transform_mismatch:{spec.world_id}: generative={gen} fitted={fitted}"
            )
        else:
            warnings.append(f"h5:transform_mismatch_expected:{spec.world_id}: not detected in fit")
    elif (
        exp.get("transform_mismatch_mode") == "aligned"
        and h5_diag.get("transform_mismatch_detected")
        and not transforms_aligned(gen, fitted)
        and exp.get("h5_classification") != "weak_identification"
    ):
        warnings.append(f"h5:unexpected_transform_mismatch:{spec.world_id}")

    if exp.get("collinearity_warning_expected") and panel_df is not None:
        max_corr = _panel_max_channel_correlation(panel_df, spec.channels)
        if max_corr is not None and max_corr > 0.85:
            warnings.append(f"h5:collinearity:{spec.world_id}: max_channel_corr={max_corr:.3f}")

    rec = artifact.get("h4_recovery") or artifact.get("h5_recovery") or {}
    beta_mae = rec.get("beta_gc_mae")
    if exp.get("h5_classification") == "weak_identification":
        if exp.get("generative_transform") == "weak_signal":
            warnings.append(f"h5:weak_identification:{spec.world_id}: weak_signal_generative")
        elif beta_mae is not None and float(beta_mae) > 0.35:
            warnings.append(f"h5:weak_identification:{spec.world_id}: beta_gc_mae={float(beta_mae):.3f}")

    return warnings


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
    h4c_warnings = compute_h4c_diagnostic_warnings(artifact, spec)
    all_warnings = list(conflict_warnings) + [w for w in h4c_warnings if w not in conflict_warnings]

    interval_widths: list[float] = []
    for geo in spec.geo_order:
        for ch in spec.channels:
            if geo in intervals and ch in intervals[geo]:
                lo, hi = intervals[geo][ch]
                interval_widths.append(float(hi - lo))

    trust = artifact.get("diagnostic_trust_report") or {}
    if all_warnings:
        trust = {**trust, "conflict_warnings": all_warnings, "h4c_diagnostic_warnings": h4c_warnings}

    h4c_class = spec.expected_diagnostic_behavior.get("h4c_classification")

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
        "beta_interval_width_90_mean": float(np.mean(interval_widths)) if interval_widths else None,
        "conflict_warnings": all_warnings,
        "h4c_diagnostic_warnings": h4c_warnings,
        "h4c_classification": h4c_class,
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
    if metrics.get("conflict_warnings") or metrics.get("h4c_diagnostic_warnings"):
        dtr = dict(out.get("diagnostic_trust_report") or {})
        if metrics.get("conflict_warnings"):
            dtr["conflict_warnings"] = metrics["conflict_warnings"]
        if metrics.get("h4c_diagnostic_warnings"):
            dtr["h4c_diagnostic_warnings"] = metrics["h4c_diagnostic_warnings"]
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
    report = build_h4_recovery_report(spec, artifact)
    h4c_warn = compute_h4c_diagnostic_warnings(report, spec, panel_df=df)
    if h4c_warn:
        rec = dict(report.get("h4_recovery") or {})
        existing_h4c = list(rec.get("h4c_diagnostic_warnings") or [])
        rec["h4c_diagnostic_warnings"] = existing_h4c + [w for w in h4c_warn if w not in existing_h4c]
        rec["conflict_warnings"] = list(rec.get("conflict_warnings") or []) + [
            w for w in h4c_warn if w not in rec.get("conflict_warnings", [])
        ]
        report["h4_recovery"] = rec
    return report


def run_h5_recovery_world(
    world_id: str,
    *,
    fast_mcmc: bool = True,
    sampler: dict[str, Any] | None = None,
    nuts_seed: int | None = None,
    panel_seed: int | None = None,
    sandbox_model_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Materialize H5 world, run gated H5 sandbox fit, recovery metrics (research only).
    """
    from mmm.research.bayes_h3_sandbox.h5_validation_worlds import sandbox_overrides_for_h5_world

    spec = get_recovery_world(world_id)
    cfg, schema, df = materialize_recovery_bundle(
        spec,
        fast_mcmc=fast_mcmc,
        sampler=sampler,
        nuts_seed=nuts_seed,
        panel_seed=panel_seed,
    )
    overrides = sandbox_overrides_for_h5_world(spec)
    if sandbox_model_overrides:
        overrides.update(sandbox_model_overrides)
    artifact = run_sandbox_fit(
        cfg,
        schema,
        df,
        geo_hierarchy_mapping=spec.geo_hierarchy,
        calibration_signals_stub=list(spec.calibration_signals),
        sandbox_model_overrides=overrides,
        model_spec_version=H5_MODEL_SPEC_VERSION,
        enable_h5_sandbox=True,
        research_only=True,
    )
    report = build_h4_recovery_report(spec, artifact)
    h5_warn = compute_h5_diagnostic_warnings(report, spec, panel_df=df)
    if h5_warn:
        rec = dict(report.get("h4_recovery") or {})
        existing = list(rec.get("h5_diagnostic_warnings") or [])
        rec["h5_diagnostic_warnings"] = existing + [w for w in h5_warn if w not in existing]
        report["h4_recovery"] = rec
        report["h5_recovery"] = rec
        dtr = dict(report.get("diagnostic_trust_report") or {})
        dtr["h5_diagnostic_warnings"] = rec["h5_diagnostic_warnings"]
        report["diagnostic_trust_report"] = dtr
    report["model_spec_version"] = H5_MODEL_SPEC_VERSION
    report["enable_h5_sandbox"] = True
    return report


def validate_h5_world_catalog() -> dict[str, Any]:
    """Fast validation that all H5 worlds materialize deterministically."""
    from mmm.research.bayes_h3_sandbox.h5_validation_worlds import H5_WORLD_IDS

    results: list[dict[str, Any]] = []
    for wid in H5_WORLD_IDS:
        spec = get_recovery_world(wid)
        df1 = materialize_recovery_bundle(spec)[2]
        df2 = materialize_recovery_bundle(spec)[2]
        exp = spec.expected_diagnostic_behavior
        results.append(
            {
                "world_id": wid,
                "rows": len(df1),
                "deterministic": df1.equals(df2),
                "has_known_truth": bool(spec.known_truth),
                "h5_classification": exp.get("h5_classification"),
                "transform_mismatch_mode": exp.get("transform_mismatch_mode"),
                "approved_for_prod": False,
                "prod_decisioning_allowed": False,
                "hard_gate": False,
            }
        )
    ok = all(
        r["deterministic"] and r["has_known_truth"] and r["h5_classification"] for r in results
    )
    return {"status": "pass" if ok else "fail", "worlds": results}


def validate_h4c_world_catalog() -> dict[str, Any]:
    """Fast validation that all H4c worlds materialize deterministically."""
    from mmm.research.bayes_h3_sandbox.h4c_recovery_worlds import H4C_WORLD_IDS

    results: list[dict[str, Any]] = []
    for wid in H4C_WORLD_IDS:
        spec = get_recovery_world(wid)
        df1 = materialize_recovery_bundle(spec)[2]
        df2 = materialize_recovery_bundle(spec)[2]
        exp = spec.expected_diagnostic_behavior
        results.append(
            {
                "world_id": wid,
                "rows": len(df1),
                "deterministic": df1.equals(df2),
                "has_known_truth": bool(spec.known_truth),
                "h4c_classification": exp.get("h4c_classification"),
            }
        )
    ok = all(r["deterministic"] and r["has_known_truth"] and r["h4c_classification"] for r in results)
    return {
        "status": "pass" if ok else "fail",
        "worlds": results,
    }


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
