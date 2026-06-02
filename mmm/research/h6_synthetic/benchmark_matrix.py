"""H6f — Ridge vs H5 synthetic benchmark matrix and confounding summary (research only)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from mmm.research.h6_synthetic.benchmark_harness import (
    run_h5_h6_benchmark,
    run_ridge_h6_benchmark,
)
from mmm.research.h6_synthetic.production_shapes import (
    H6_PILOT_WORLD_IDS,
    ProductionShapedWorldSpec,
    forbidden_claims_for_h6_world,
    get_h6_world,
    list_h6_confounding_world_ids,
    materialize_h6_panel,
    materialize_h6_truth_artifact,
)

H6F_ARTIFACT_KIND_MATRIX = "BAYES_H6F_RIDGE_H5_SYNTHETIC_BENCHMARK_MATRIX"
H6F_ARTIFACT_KIND_CONFOUNDING = "BAYES_H6F_CONTROL_CONFOUNDING_SUMMARY"

H6F_PRODUCTION_FLAGS: dict[str, Any] = {
    "approved_for_prod": False,
    "prod_decisioning_allowed": False,
    "optimizer_enabled": False,
    "decision_surface_enabled": False,
    "recommendations_enabled": False,
    "ridge_remains_production_baseline": True,
    "bayes_h5_research_only": True,
    "production_bayes_blocked": True,
}

H6F_H5_WORLDS_DEFAULT: tuple[str, ...] = (
    "WORLD-H6-PILOT-RETAIL-FULL-CONTROLS",
    "WORLD-H6-PILOT-RETAIL-MEDIA-CORRELATED-CONTROLS",
)


def _sparse_channel_diagnostics(
    panel: pd.DataFrame,
    spec: ProductionShapedWorldSpec,
) -> dict[str, Any]:
    out: dict[str, Any] = {"by_channel": {}, "sparse_channels_flagged": []}
    geo_col = "geo_id"
    for ch in spec.sparse_channels:
        if ch not in panel.columns:
            continue
        nz_share = float((panel[ch] <= 0.5).mean())
        geo_active = panel.groupby(geo_col)[ch].apply(lambda s: float((s > 0.5).any()))
        out["by_channel"][ch] = {
            "near_zero_share": nz_share,
            "geos_with_activity_share": float(geo_active.mean()),
        }
        if nz_share > 0.5:
            out["sparse_channels_flagged"].append(ch)
    return out


def _prediction_metrics(y: np.ndarray, yhat: np.ndarray) -> dict[str, Any]:
    if yhat.size != y.size or y.size == 0:
        return {"rmse": None, "wmape": None}
    rmse = float(np.sqrt(np.mean((y - yhat) ** 2)))
    denom = float(np.sum(np.abs(y)))
    wmape = float(np.sum(np.abs(y - yhat)) / denom) if denom > 0 else None
    return {"rmse": rmse, "wmape": wmape}


def _world_descriptor(spec: ProductionShapedWorldSpec) -> dict[str, Any]:
    ct = spec.control_truth
    return {
        "world_id": spec.world_id,
        "vertical": spec.vertical_id,
        "control_variant": spec.stress_variant,
        "scale": spec.scale,
        "geos": spec.n_geos,
        "weeks": spec.n_weeks,
        "channels": list(spec.channels),
        "known_truth_summary": {
            "noise_sigma": spec.noise_sigma,
            "true_mu_c": dict(spec.true_mu_c),
            "sparse_channels": list(spec.sparse_channels),
            "national_channels": list(spec.national_channels),
            "collinearity_blocks": {k: list(v) for k, v in spec.collinearity_blocks.items()},
        },
        "transform_truth": dict(spec.transform_truth),
        "controls": {
            "present": list(spec.active_controls),
            "omitted": list(ct.get("omitted_controls") or []),
            "omitted_required": list(ct.get("omitted_required_controls") or []),
            "mis_specified": spec.stress_variant == "mis_specified_controls",
            "media_correlated": spec.stress_variant == "media_correlated_controls",
        },
        "forbidden_claims": forbidden_claims_for_h6_world(spec),
    }


def _ridge_matrix_metrics(
    spec: ProductionShapedWorldSpec,
    panel: pd.DataFrame,
    ridge: dict[str, Any],
) -> dict[str, Any]:
    coef = ridge.get("ridge_coef_recovery") or {}
    return {
        "prediction_rmse": ridge.get("prediction_rmse"),
        "prediction_wmape": ridge.get("prediction_wmape"),
        "geo_fold_rmse_mean": ridge.get("geo_fold_rmse_mean"),
        "geo_fold_stability_n_folds": ridge.get("geo_fold_stability_n_folds"),
        "coefficient_recovery": coef,
        "lift_recovery": ridge.get("ridge_lift_recovery") or {},
        "sign_plausibility": {
            "coef_sign_match_rate_vs_mu": coef.get("coef_sign_match_rate_vs_mu"),
        },
        "collinearity": {
            "max_channel_correlation": ridge.get("max_channel_correlation"),
            "collinearity_sensitive": ridge.get("collinearity_sensitive"),
        },
        "sparse_channel_diagnostics": _sparse_channel_diagnostics(panel, spec),
        "forbidden_claims": forbidden_claims_for_h6_world(spec),
        "recommendations_emitted": False,
        "optimizer_emitted": False,
        "decision_surface_emitted": False,
    }


def _h5_matrix_metrics(h5: dict[str, Any] | None) -> dict[str, Any] | None:
    if h5 is None:
        return None
    conv = h5.get("convergence") or {}
    recovery = h5.get("h5_recovery") or {}
    warnings = list(h5.get("h5_diagnostic_warnings") or [])
    transform_mismatch = any("transform_mismatch" in w for w in warnings)
    div = conv.get("divergences")
    rhat = conv.get("rhat_max")
    if div is not None and int(div) > 0:
        convergence_status = "failed_divergences"
    elif rhat is not None and float(rhat) > 1.05:
        convergence_status = "failed_rhat"
    else:
        convergence_status = "converged_diagnostic_only"

    return {
        "convergence_status": convergence_status,
        "rhat_max": rhat,
        "divergence_count": div,
        "beta_gc_mae": recovery.get("beta_gc_mae"),
        "mu_c_mae": recovery.get("mu_c_mae"),
        "beta_gc_coverage_90": recovery.get("beta_gc_coverage_90"),
        "warning_count": h5.get("warning_count"),
        "transform_mismatch_warning": transform_mismatch,
        "h5_diagnostic_warnings": warnings,
        "evidence_promotion_allowed": False,
        "approved_for_prod": False,
        "prod_decisioning_allowed": False,
        "outputs_are_diagnostic_only": True,
    }


def build_h6f_world_row(
    world_id: str,
    *,
    run_h5: bool = False,
    fast_mcmc: bool = True,
) -> dict[str, Any]:
    spec = get_h6_world(world_id)
    panel = materialize_h6_panel(spec)
    ridge_raw = run_ridge_h6_benchmark(spec, panel)
    h5_raw = (
        run_h5_h6_benchmark(spec, panel, fast_mcmc=fast_mcmc) if run_h5 else None
    )
    return {
        **_world_descriptor(spec),
        "panel_rows": len(panel),
        "ridge": _ridge_matrix_metrics(spec, panel, ridge_raw),
        "h5": _h5_matrix_metrics(h5_raw),
        "h5_ran": run_h5,
    }


def build_h6f_benchmark_matrix(
    world_ids: tuple[str, ...] | None = None,
    *,
    h5_world_ids: tuple[str, ...] | None = None,
    fast_mcmc: bool = True,
) -> dict[str, Any]:
    """Build combined H6f benchmark matrix across pilot worlds."""
    ids = world_ids or H6_PILOT_WORLD_IDS
    if h5_world_ids is None:
        h5_ids = set(H6F_H5_WORLDS_DEFAULT)
    else:
        h5_ids = set(h5_world_ids)
    rows = [
        build_h6f_world_row(wid, run_h5=wid in h5_ids, fast_mcmc=fast_mcmc) for wid in ids
    ]
    return {
        "artifact_kind": H6F_ARTIFACT_KIND_MATRIX,
        "lane": "bayes_h6f_synthetic_benchmark_matrix",
        "milestone": "H6f",
        "worlds_tested": list(ids),
        "h5_worlds_ran": [w for w in ids if w in h5_ids],
        "matrix_rows": rows,
        "interpretation_axes": [
            "model_issue",
            "transform_specification_issue",
            "control_omission_issue",
            "identification_issue",
        ],
        "production_flags": dict(H6F_PRODUCTION_FLAGS),
        "outputs_are_diagnostic_only": True,
        "synthetic_success_is_not_real_world_proof": True,
    }


def _confounding_interpretation(
    comparisons: list[dict[str, Any]],
) -> dict[str, Any]:
    full = next((c for c in comparisons if c.get("stress_variant") == "full_controls"), None)
    omitted = next((c for c in comparisons if c.get("stress_variant") == "omitted_controls"), None)
    media_corr = next(
        (c for c in comparisons if c.get("stress_variant") == "media_correlated_controls"), None
    )

    def _coef_mae(c: dict[str, Any] | None) -> float | None:
        if not c:
            return None
        rec = (c.get("ridge") or {}).get("coefficient_recovery") or {}
        return rec.get("coef_mu_mae_vs_true_mu")

    full_mae = _coef_mae(full)
    omit_mae = _coef_mae(omitted)
    media_mae = _coef_mae(media_corr)

    ridge_degrades_omitted = (
        full_mae is not None
        and omit_mae is not None
        and omit_mae > full_mae * 1.05
    )

    return {
        "ridge_coef_recovery_degrades_under_omitted_controls": ridge_degrades_omitted,
        "full_controls_coef_mu_mae": full_mae,
        "omitted_controls_coef_mu_mae": omit_mae,
        "media_correlated_coef_mu_mae": media_mae,
        "ridge_may_attribute_omitted_control_to_media": ridge_degrades_omitted,
        "forbidden_claims_required_when_controls_missing": bool(
            omitted and omitted.get("forbidden_claims")
        ),
        "vertical_controls_necessary_for_credible_mmm": True,
        "mis_specified_controls_in_pilot_registry": False,
        "notes": [
            "Compare coef_mu_mae and sign_match across stress variants.",
            "Omitted required controls must emit forbidden_claims — no incrementality promotion.",
            "Media-correlated controls stress tests false attribution risk to national TV.",
        ],
    }


def build_h6f_control_confounding_summary(
    world_ids: tuple[str, ...] | None = None,
    *,
    run_h5: bool = False,
    fast_mcmc: bool = True,
) -> dict[str, Any]:
    """H6f confounding summary across control-stress pilot worlds."""
    ids = world_ids or list_h6_confounding_world_ids()
    comparisons: list[dict[str, Any]] = []
    for wid in ids:
        spec = get_h6_world(wid)
        panel = materialize_h6_panel(spec)
        ridge_raw = run_ridge_h6_benchmark(spec, panel)
        row: dict[str, Any] = {
            **_world_descriptor(spec),
            "ridge": _ridge_matrix_metrics(spec, panel, ridge_raw),
            "truth_artifact": materialize_h6_truth_artifact(spec),
        }
        if run_h5:
            h5_raw = run_h5_h6_benchmark(spec, panel, fast_mcmc=fast_mcmc)
            row["h5"] = _h5_matrix_metrics(h5_raw)
        comparisons.append(row)

    return {
        "artifact_kind": H6F_ARTIFACT_KIND_CONFOUNDING,
        "lane": "bayes_h6f_control_confounding",
        "milestone": "H6f",
        "world_comparisons": comparisons,
        "interpretation": _confounding_interpretation(comparisons),
        "production_flags": dict(H6F_PRODUCTION_FLAGS),
        "outputs_are_diagnostic_only": True,
    }
