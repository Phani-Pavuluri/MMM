"""H6c — Ridge vs Bayes-H5 benchmark harness on production-shaped synthetic worlds."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from mmm.models.ridge_bo.trainer import RidgeBOMMMTrainer
from mmm.research.bayes_h3_sandbox.entrypoint import run_sandbox_fit
from mmm.research.bayes_h3_sandbox.fencing import H5_MODEL_SPEC_VERSION
from mmm.research.bayes_h3_sandbox.recovery_runner import (
    build_h4_recovery_report,
    compute_h5_diagnostic_warnings,
    compute_recovery_metrics,
)
from mmm.research.h6_synthetic.production_shapes import (
    ProductionShapedWorldSpec,
    as_recovery_world_spec,
    forbidden_claims_for_h6_world,
    get_h6_world,
    h6_h5_config,
    h6_panel_schema,
    h6_ridge_config,
    materialize_h6_panel,
    materialize_h6_truth_artifact,
)


def _panel_max_channel_correlation(df: pd.DataFrame, channels: tuple[str, ...]) -> float | None:
    cols = [c for c in channels if c in df.columns]
    if len(cols) < 2:
        return None
    corr = df[cols].corr().to_numpy()
    n = corr.shape[0]
    off = [abs(corr[i, j]) for i in range(n) for j in range(n) if i != j]
    return float(max(off)) if off else None


def _ridge_lift_recovery(
    fit: dict[str, Any],
    spec: ProductionShapedWorldSpec,
    channels: tuple[str, ...],
) -> dict[str, Any]:
    """Coarse lift sign recovery vs known geo-mean beta (diagnostic only)."""
    art = fit.get("artifacts")
    coef = getattr(art, "coef", None) if art is not None else None
    if coef is None:
        return {"lift_recovery_available": False}
    coef = np.asarray(coef).ravel()
    n_ch = len(channels)
    if coef.size < n_ch:
        return {"lift_recovery_available": False}
    ch_coef = coef[:n_ch]
    true_lift = np.array([np.mean([spec.true_beta_gc[g][c] for g in spec.geo_order]) for c in channels])
    sign_match = float(np.mean(np.sign(ch_coef) == np.sign(true_lift)))
    return {
        "lift_recovery_available": True,
        "lift_sign_match_rate_vs_true_beta": sign_match,
        "lift_mae_vs_geo_mean_beta": float(np.mean(np.abs(ch_coef - true_lift))),
    }


def _ridge_coef_recovery(
    fit: dict[str, Any],
    spec: ProductionShapedWorldSpec,
    channels: tuple[str, ...],
) -> dict[str, Any]:
    """Coarse Ridge coefficient vs known beta (on raw media scale — diagnostic only)."""
    art = fit.get("artifacts")
    coef = getattr(art, "coef", None) if art is not None else None
    if coef is None:
        return {"coef_recovery_available": False}
    coef = np.asarray(coef).ravel()
    n_ch = len(channels)
    if coef.size < n_ch:
        return {"coef_recovery_available": False}
    ch_coef = coef[:n_ch]
    mu_true = np.array([spec.true_mu_c[c] for c in channels])
    sign_match = float(np.mean(np.sign(ch_coef) == np.sign(mu_true)))
    return {
        "coef_recovery_available": True,
        "coef_mu_mae_vs_true_mu": float(np.mean(np.abs(ch_coef - mu_true))),
        "coef_sign_match_rate_vs_mu": sign_match,
    }


def run_ridge_h6_benchmark(
    spec: ProductionShapedWorldSpec,
    panel: pd.DataFrame,
    *,
    n_folds: int = 3,
) -> dict[str, Any]:
    """Production Ridge baseline metrics on H6 panel (research environment)."""
    config = h6_ridge_config(spec)
    schema = h6_panel_schema(spec)
    week_col = schema.week_column
    df = panel.copy()
    if week_col in df.columns:
        df[week_col] = pd.to_datetime(df[week_col])

    trainer = RidgeBOMMMTrainer(config, schema)
    fit = trainer.fit(df)

    y = df[schema.target_column].astype(float).values
    try:
        yhat = np.asarray(trainer.predict(df))
        if yhat.size == y.size:
            pred_err = float(np.sqrt(np.mean((y - yhat) ** 2)))
            denom = float(np.sum(np.abs(y)))
            wmape = float(np.sum(np.abs(y - yhat)) / denom) if denom > 0 else None
        else:
            yhat = np.array([])
            pred_err = None
            wmape = None
    except Exception:
        yhat = np.array([])
        pred_err = None
        wmape = None

    fold_rmse: list[float] = []
    geos = df[schema.geo_column].unique()
    if len(geos) >= n_folds and yhat.size == y.size:
        fold_size = max(1, len(geos) // n_folds)
        for f in range(n_folds):
            test_geos = geos[f * fold_size : (f + 1) * fold_size]
            mask = df[schema.geo_column].isin(test_geos).values
            if mask.sum() > 0:
                fold_rmse.append(float(np.sqrt(np.mean((y[mask] - yhat[mask]) ** 2))))

    max_corr = _panel_max_channel_correlation(df, spec.channels)
    coef_rec = _ridge_coef_recovery(fit, spec, spec.channels)
    lift_rec = _ridge_lift_recovery(fit, spec, spec.channels)

    return {
        "model": "ridge_bo",
        "run_environment": config.run_environment.value,
        "prediction_rmse": pred_err,
        "prediction_wmape": wmape,
        "geo_fold_rmse_mean": float(np.mean(fold_rmse)) if fold_rmse else None,
        "geo_fold_stability_n_folds": len(fold_rmse),
        "max_channel_correlation": max_corr,
        "collinearity_sensitive": max_corr is not None and max_corr > 0.85,
        "sparse_channels": list(spec.sparse_channels),
        "ridge_coef_recovery": coef_rec,
        "ridge_lift_recovery": lift_rec,
        "forbidden_claims": forbidden_claims_for_h6_world(spec),
        "approved_for_prod": False,
        "prod_decisioning_allowed": False,
        "optimizer_enabled": False,
        "decision_surface_enabled": False,
        "outputs_are_diagnostic_only": True,
    }


def run_h5_h6_benchmark(
    spec: ProductionShapedWorldSpec,
    panel: pd.DataFrame,
    *,
    fast_mcmc: bool = True,
    enable_h5: bool = True,
) -> dict[str, Any]:
    """Research Bayes-H5 sandbox metrics on same H6 panel."""
    config = h6_h5_config(spec, fast_mcmc=fast_mcmc)
    schema = h6_panel_schema(spec)
    rec_spec = as_recovery_world_spec(spec)
    df = panel.copy()
    week_col = schema.week_column
    if week_col in df.columns:
        df[week_col] = pd.to_datetime(df[week_col])

    artifact = run_sandbox_fit(
        config,
        schema,
        df,
        enable_h5_sandbox=enable_h5,
        model_spec_version=H5_MODEL_SPEC_VERSION if enable_h5 else None,
    )
    report = build_h4_recovery_report(rec_spec, artifact)
    recovery = compute_recovery_metrics(report, rec_spec)
    h5_warnings = compute_h5_diagnostic_warnings(report, rec_spec, panel_df=df)
    conv = report.get("convergence_diagnostics") or {}

    transform_mismatch = any("transform_mismatch" in w for w in h5_warnings)
    div = conv.get("divergences")
    rhat = conv.get("rhat_max")
    if div is not None and int(div) > 0:
        convergence_status = "failed_divergences"
    elif rhat is not None and float(rhat) > 1.05:
        convergence_status = "failed_rhat"
    else:
        convergence_status = "converged_diagnostic_only"

    return {
        "model": "bayes_h5_sandbox",
        "run_environment": config.run_environment.value,
        "model_spec_version": H5_MODEL_SPEC_VERSION if enable_h5 else None,
        "convergence_status": convergence_status,
        "convergence": {
            "divergences": div,
            "rhat_max": rhat,
            "ess_bulk_min": conv.get("ess_bulk_min"),
        },
        "h5_recovery": recovery,
        "h5_diagnostic_warnings": h5_warnings,
        "warning_count": len(h5_warnings),
        "transform_mismatch_warning": transform_mismatch,
        "evidence_promotion_allowed": False,
        "approved_for_prod": False,
        "prod_decisioning_allowed": False,
        "outputs_are_diagnostic_only": True,
    }


def run_h6_benchmark_pair(
    world_id: str,
    *,
    fast_mcmc: bool = True,
    run_h5: bool = True,
    panel_seed: int | None = None,
) -> dict[str, Any]:
    """Run Ridge + H5 on the same production-shaped synthetic world."""
    spec = get_h6_world(world_id)
    panel = materialize_h6_panel(spec, panel_seed=panel_seed)
    ridge = run_ridge_h6_benchmark(spec, panel)
    h5: dict[str, Any] | None = None
    if run_h5:
        h5 = run_h5_h6_benchmark(spec, panel, fast_mcmc=fast_mcmc)
    return {
        "world_id": world_id,
        "lane": "bayes_h6_synthetic_benchmark",
        "truth_artifact": materialize_h6_truth_artifact(spec),
        "panel_rows": len(panel),
        "ridge_benchmark": ridge,
        "h5_benchmark": h5,
        "production_flags": {
            "approved_for_prod": False,
            "prod_decisioning_allowed": False,
            "optimizer_enabled": False,
            "decision_surface_enabled": False,
            "recommendations_enabled": False,
            "ridge_remains_production_baseline": True,
            "bayes_h5_research_only": True,
        },
        "outputs_are_diagnostic_only": True,
    }


def build_h6_benchmark_artifact(
    world_id: str,
    *,
    fast_mcmc: bool = True,
    run_h5: bool = True,
) -> dict[str, Any]:
    """Build a governed H6 benchmark artifact suitable for validation archives."""
    out = run_h6_benchmark_pair(world_id, fast_mcmc=fast_mcmc, run_h5=run_h5)
    out["artifact_kind"] = "BAYES_H6_SYNTHETIC_BENCHMARK"
    return out


def build_h6_confounding_comparison(
    world_ids: tuple[str, ...] | None = None,
    *,
    fast_mcmc: bool = True,
    run_h5: bool = False,
) -> dict[str, Any]:
    """H6d — compare full vs omitted vs media-correlated control stress worlds."""
    from mmm.research.h6_synthetic.production_shapes import list_h6_confounding_world_ids

    ids = world_ids or list_h6_confounding_world_ids()
    comparisons: list[dict[str, Any]] = []
    for wid in ids:
        spec = get_h6_world(wid)
        panel = materialize_h6_panel(spec)
        ridge = run_ridge_h6_benchmark(spec, panel)
        entry: dict[str, Any] = {
            "world_id": wid,
            "stress_variant": spec.stress_variant,
            "vertical_id": spec.vertical_id,
            "active_controls": list(spec.active_controls),
            "omitted_controls": spec.control_truth.get("omitted_controls") or [],
            "forbidden_claims": list(
                spec.expected_diagnostic_behavior.get("forbidden_claims_when_controls_missing") or []
            ),
            "ridge_benchmark": ridge,
        }
        if run_h5:
            entry["h5_benchmark"] = run_h5_h6_benchmark(spec, panel, fast_mcmc=fast_mcmc)
        comparisons.append(entry)

    return {
        "artifact_kind": "BAYES_H6_CONFOUNDING_STRESS",
        "lane": "bayes_h6_synthetic",
        "world_comparisons": comparisons,
        "interpretation_notes": [
            "Compare Ridge coef sign/recovery across stress variants — media must not absorb omitted control lift.",
            "When required controls are omitted, forbidden_claims apply; no incrementality promotion.",
        ],
        "production_flags": {
            "approved_for_prod": False,
            "prod_decisioning_allowed": False,
            "recommendations_enabled": False,
        },
        "outputs_are_diagnostic_only": True,
    }
