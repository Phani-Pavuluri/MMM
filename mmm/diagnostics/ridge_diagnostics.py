"""H7 — Ridge production diagnostic hardening (decision-safety metadata only)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from mmm.config.schema import Framework, MMMConfig, RunEnvironment
from mmm.config.vertical_control_profiles import VerticalControlProfile, resolve_vertical_profile
from mmm.data.schema import PanelSchema
from mmm.features.design_matrix import build_design_matrix
from mmm.models.ridge_bo.trainer import RidgeBOMMMTrainer
from mmm.validation.cv import auto_cv_mode

REPORT_VERSION = "mmm_ridge_production_diagnostics_v1"

SPARSE_NEAR_ZERO_THRESHOLD = 0.99
COLLINEARITY_WEAK_ID_THRESHOLD = 0.85
COLLINEARITY_GROUP_THRESHOLD = 0.95

FORBIDDEN_OUTPUT_FIELDS = frozenset(
    {
        "decision_surface",
        "optimizer_ready_curves",
        "budget_recommendation",
        "recommendation",
        "production_decision_surface",
        "optimizer_output",
    }
)


def _max_abs_channel_correlation(df: pd.DataFrame, channels: tuple[str, ...]) -> float:
    cols = [c for c in channels if c in df.columns]
    if len(cols) < 2:
        return 0.0
    corr = df[cols].corr().to_numpy()
    n = corr.shape[0]
    off = [abs(corr[i, j]) for i in range(n) for j in range(n) if i != j]
    return float(max(off)) if off else 0.0


def _collinear_groups(
    df: pd.DataFrame,
    channels: tuple[str, ...],
    *,
    threshold: float = COLLINEARITY_GROUP_THRESHOLD,
) -> list[dict[str, Any]]:
    chs = [c for c in channels if c in df.columns]
    if len(chs) < 2:
        return []
    media = df[chs].to_numpy(dtype=float)
    corr = np.corrcoef(media.T)
    n = len(chs)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j in range(i + 1, n):
            if abs(float(corr[i, j])) >= threshold:
                union(i, j)

    groups_map: dict[int, list[str]] = {}
    for idx, ch in enumerate(chs):
        groups_map.setdefault(find(idx), []).append(ch)
    return [
        {"channels": sorted(members), "max_abs_corr_threshold": threshold}
        for members in groups_map.values()
        if len(members) >= 2
    ]


def build_ridge_transform_diagnostics(
    config: MMMConfig,
    schema: PanelSchema,
    fit_result: dict[str, Any] | None,
    *,
    panel: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Report Ridge transform path: raw media, transformed features, BO-selected hyperparameters."""
    warnings: list[str] = []
    art = (fit_result or {}).get("artifacts")
    best = getattr(art, "best_params", None) if art is not None else None
    best = best if isinstance(best, dict) else {}

    transform_config = {
        "adstock_type": str(config.transforms.adstock),
        "saturation_type": str(config.transforms.saturation),
        "framework": config.framework.value,
    }
    selected = {
        "decay": best.get("decay"),
        "hill_half": best.get("hill_half"),
        "hill_slope": best.get("hill_slope"),
        "log_alpha": best.get("log_alpha"),
        "transform_search_selected": bool(best),
    }
    metadata_complete = all(
        best.get(k) is not None for k in ("decay", "hill_half", "hill_slope")
    )
    if not metadata_complete:
        warnings.append("ridge_transform:missing_best_params_metadata")

    raw_media_columns = list(schema.channel_columns)
    transformed_media_columns = list(schema.channel_columns)
    post_transform_summary: dict[str, Any] = {}

    if panel is not None and best.get("decay") is not None:
        try:
            bundle = build_design_matrix(
                panel,
                schema,
                config,
                decay=float(best["decay"]),
                hill_half=float(best["hill_half"]),
                hill_slope=float(best["hill_slope"]),
            )
            lineage = bundle.feature_lineage or {}
            post_transform_summary = {
                "n_features": int(bundle.X.shape[1]),
                "feature_lineage_keys": sorted(lineage.keys())[:20],
            }
            transformed_media_columns = list(
                lineage.get("channel_feature_names") or schema.channel_columns
            )
        except Exception as exc:
            warnings.append(f"ridge_transform:design_matrix_probe_failed:{exc}")

    return {
        "transform_config": transform_config,
        "selected_adstock_saturation": selected,
        "raw_media_columns": raw_media_columns,
        "transformed_media_columns": transformed_media_columns,
        "post_transform_summary": post_transform_summary,
        "metadata_complete": metadata_complete,
        "warnings": warnings,
    }


def build_control_completeness_diagnostics(
    schema: PanelSchema,
    *,
    vertical_id: str | None = None,
    profile: VerticalControlProfile | None = None,
    media_correlated_controls: bool = False,
) -> dict[str, Any]:
    """Compare panel controls to vertical recommendations."""
    present = set(schema.control_columns)
    prof = profile or resolve_vertical_profile(vertical_id)
    if prof is None:
        warnings: list[str] = ["control_completeness:no_vertical_profile"]
        if vertical_id:
            warnings.append(f"control_completeness:unknown_vertical:{vertical_id}")
        return {
            "vertical_id": vertical_id,
            "vertical_profile_known": False,
            "required_controls_by_vertical": [],
            "optional_controls_by_vertical": [],
            "controls_present": sorted(present),
            "missing_controls": [],
            "missing_required_controls": [],
            "omitted_control_risk": False,
            "media_correlated_controls": media_correlated_controls,
            "warnings": warnings,
        }

    required = set(prof.required_controls)
    optional = set(prof.optional_controls)
    missing_required = sorted(required - present)
    missing_optional = sorted(optional - present)
    missing_all = sorted((required | optional) - present)

    omitted_risk = bool(missing_required)
    warnings: list[str] = []
    if missing_required:
        warnings.append(f"control_completeness:missing_required:{missing_required}")
    if media_correlated_controls:
        warnings.append("control_completeness:media_correlated_controls_confounding_risk")

    return {
        "vertical_id": prof.vertical_id,
        "vertical_profile_known": True,
        "required_controls_by_vertical": list(prof.required_controls),
        "optional_controls_by_vertical": list(prof.optional_controls),
        "controls_present": sorted(present),
        "missing_controls": missing_all,
        "missing_required_controls": missing_required,
        "missing_optional_controls": missing_optional,
        "omitted_control_risk": omitted_risk,
        "media_correlated_controls": media_correlated_controls,
        "warnings": warnings,
    }


def build_collinearity_diagnostics(
    panel: pd.DataFrame,
    schema: PanelSchema,
    *,
    calibration_evidence_available: bool = False,
) -> dict[str, Any]:
    channels = tuple(schema.channel_columns)
    max_corr = _max_abs_channel_correlation(panel, channels)
    groups = _collinear_groups(panel, channels)
    weak_id = max_corr >= COLLINEARITY_WEAK_ID_THRESHOLD
    warnings: list[str] = []
    if weak_id:
        warnings.append(f"collinearity:weak_identification_risk:max_abs_corr={max_corr:.3f}")
    return {
        "max_abs_correlation": max_corr,
        "weak_identification_risk": weak_id,
        "collinear_channel_groups": groups,
        "calibration_evidence_available": calibration_evidence_available,
        "warnings": warnings,
    }


def build_sparse_channel_diagnostics(
    panel: pd.DataFrame,
    schema: PanelSchema,
) -> dict[str, Any]:
    by_channel: dict[str, Any] = {}
    extreme: list[str] = []
    for ch in schema.channel_columns:
        if ch not in panel.columns:
            continue
        col = panel[ch].to_numpy(dtype=float)
        near_zero_share = float(np.mean(col < 1e-6))
        by_channel[ch] = {
            "near_zero_share": near_zero_share,
            "zero_share": float(np.mean(col <= 0.0)),
        }
        if near_zero_share >= SPARSE_NEAR_ZERO_THRESHOLD:
            extreme.append(ch)

    warnings: list[str] = []
    for ch in extreme:
        warnings.append(f"sparse_channel:extreme_near_zero:{ch}:{SPARSE_NEAR_ZERO_THRESHOLD}")

    return {
        "near_zero_threshold": SPARSE_NEAR_ZERO_THRESHOLD,
        "by_channel": by_channel,
        "sparse_channel_extreme": extreme,
        "silent_drop_occurred": False,
        "warnings": warnings,
    }


def build_fold_stability_diagnostics(
    panel: pd.DataFrame,
    schema: PanelSchema,
    config: MMMConfig,
    trainer: RidgeBOMMMTrainer | None = None,
    *,
    n_folds: int = 3,
) -> dict[str, Any]:
    """Geo-fold RMSE stability using trained Ridge predictor when available."""
    y = panel[schema.target_column].astype(float).values
    yhat: np.ndarray | None = None
    if trainer is not None:
        try:
            yhat = np.asarray(trainer.predict(panel))
        except Exception:
            yhat = None

    fold_rmse: list[float] = []
    geos = panel[schema.geo_column].unique()
    if yhat is not None and yhat.size == y.size and len(geos) >= n_folds:
        fold_size = max(1, len(geos) // n_folds)
        for f in range(n_folds):
            test_geos = geos[f * fold_size : (f + 1) * fold_size]
            mask = panel[schema.geo_column].isin(test_geos).values
            if mask.sum() > 0:
                fold_rmse.append(float(np.sqrt(np.mean((y[mask] - yhat[mask]) ** 2))))

    cv = auto_cv_mode(panel, schema, config.cv)
    splits = cv.split(panel, schema)

    return {
        "n_cv_splits": len(splits),
        "geo_fold_rmse": fold_rmse,
        "geo_fold_rmse_mean": float(np.mean(fold_rmse)) if fold_rmse else None,
        "geo_fold_rmse_std": float(np.std(fold_rmse)) if len(fold_rmse) > 1 else None,
        "fold_stability_ok": len(fold_rmse) >= 2
        and (float(np.std(fold_rmse)) / (float(np.mean(fold_rmse)) + 1e-9) < 0.5),
        "warnings": [] if fold_rmse else ["fold_stability:insufficient_geo_folds"],
    }


def build_coefficient_stability_diagnostics(
    fit_result: dict[str, Any] | None,
    schema: PanelSchema,
) -> dict[str, Any]:
    art = (fit_result or {}).get("artifacts")
    coef = getattr(art, "coef", None) if art is not None else None
    if coef is None:
        return {"available": False, "warnings": ["coefficient_stability:no_coef"]}
    coef = np.asarray(coef).ravel()
    n_media = len(schema.channel_columns)
    media_coef = coef[:n_media] if coef.size >= n_media else coef
    signs = {ch: float(np.sign(c)) if c != 0 else 0.0 for ch, c in zip(schema.channel_columns, media_coef)}
    return {
        "available": True,
        "media_coef_by_channel": {
            ch: float(media_coef[i]) for i, ch in enumerate(schema.channel_columns) if i < len(media_coef)
        },
        "sign_by_channel": signs,
        "warnings": [],
    }


def build_sign_plausibility_diagnostics(
    coef_diag: dict[str, Any],
    *,
    known_mu: dict[str, float] | None = None,
) -> dict[str, Any]:
    if not coef_diag.get("available") or not known_mu:
        return {"available": bool(known_mu), "sign_match_rate": None, "warnings": []}
    signs = coef_diag.get("sign_by_channel") or {}
    matches = []
    for ch, true_mu in known_mu.items():
        if ch in signs and true_mu != 0:
            matches.append(signs[ch] == np.sign(true_mu))
    rate = float(np.mean(matches)) if matches else None
    warnings = []
    if rate is not None and rate < 0.5:
        warnings.append(f"sign_plausibility:low_match_rate:{rate:.2f}")
    return {"available": True, "sign_match_rate_vs_known_mu": rate, "warnings": warnings}


def build_response_curve_plausibility_diagnostics(
    transform_diag: dict[str, Any],
) -> dict[str, Any]:
    selected = transform_diag.get("selected_adstock_saturation") or {}
    warnings: list[str] = []
    decay = selected.get("decay")
    hill_slope = selected.get("hill_slope")
    if decay is not None and (decay < 0.05 or decay > 0.95):
        warnings.append("response_curve:decay_outside_typical_range")
    if hill_slope is not None and (hill_slope < 0.5 or hill_slope > 8.0):
        warnings.append("response_curve:hill_slope_outside_typical_range")
    return {
        "decay": decay,
        "hill_half": selected.get("hill_half"),
        "hill_slope": hill_slope,
        "warnings": warnings,
    }


def build_lift_simulation_stability_diagnostics(
    fit_result: dict[str, Any] | None,
) -> dict[str, Any]:
    """Placeholder for lift-simulation stability — diagnostic only, no optimizer."""
    detail = (fit_result or {}).get("best_detail") or {}
    return {
        "lift_simulation_run": False,
        "note": "Lift simulation stability is report-only; optimizer not invoked.",
        "best_score_available": (fit_result or {}).get("best_score") is not None,
        "calibration_detail_keys": sorted(detail.keys())[:15] if isinstance(detail, dict) else [],
        "warnings": [],
    }


def build_forbidden_claims_for_ridge(
    *,
    control_diag: dict[str, Any],
    collinearity_diag: dict[str, Any],
    sparse_diag: dict[str, Any],
    calibration_evidence_available: bool = False,
) -> list[str]:
    claims: list[str] = []

    if control_diag.get("omitted_control_risk"):
        claims.extend(
            [
                "no_clean_media_attribution_claim",
                "no_channel_level_causal_claim_without_caveat",
                "no_budget_reallocation_claim_based_only_on_this_run",
            ]
        )
    if control_diag.get("media_correlated_controls"):
        claims.append("media_correlated_controls_may_inflate_attribution")

    for ch in sparse_diag.get("sparse_channel_extreme") or []:
        claims.append(f"no_separate_channel_effect_claim_for_{ch}")

    if collinearity_diag.get("weak_identification_risk"):
        if not calibration_evidence_available:
            claims.append("no_clean_separate_channel_effect_claim_without_external_calibration")
        for grp in collinearity_diag.get("collinear_channel_groups") or []:
            members = grp.get("channels") or []
            if len(members) >= 2:
                claims.append(
                    f"collinear_group_{'_'.join(members[:2])}:forbid_isolated_channel_lift_claims"
                )

    return sorted(set(claims))


def _diagnostic_severity(
    *,
    warnings: list[str],
    forbidden_claims: list[str],
    omitted_control_risk: bool,
) -> str:
    if omitted_control_risk or len(forbidden_claims) >= 3:
        return "high"
    if forbidden_claims or len(warnings) >= 3:
        return "medium"
    if warnings:
        return "low"
    return "none"


def _production_flags(config: MMMConfig) -> dict[str, Any]:
    prod = config.run_environment == RunEnvironment.PROD
    return {
        "approved_for_prod": False,
        "prod_decisioning_allowed": prod,
        "optimizer_enabled": False,
        "decision_surface_enabled": False,
        "recommendations_enabled": False,
        "ridge_remains_production_baseline": True,
        "bayes_h5_research_only": True,
        "production_bayes_blocked": True,
        "diagnostics_are_not_hard_gates": True,
        "allowed_outputs": ["coefficients", "diagnostics", "trust_warnings", "model_metrics"],
        "forbidden_outputs": sorted(FORBIDDEN_OUTPUT_FIELDS),
    }


def compose_ridge_diagnostic_report(
    panel: pd.DataFrame,
    schema: PanelSchema,
    config: MMMConfig,
    fit_result: dict[str, Any] | None = None,
    *,
    trainer: RidgeBOMMMTrainer | None = None,
    vertical_id: str | None = None,
    model_id: str = "ridge_bo",
    run_id: str | None = None,
    dataset_snapshot_id: str | None = None,
    calibration_evidence_available: bool = False,
    media_correlated_controls: bool = False,
    known_truth: dict[str, Any] | None = None,
    world_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compose governed Ridge production diagnostic report (H7)."""
    if config.framework != Framework.RIDGE_BO:
        raise ValueError("compose_ridge_diagnostic_report requires framework=ridge_bo")

    transform_diag = build_ridge_transform_diagnostics(config, schema, fit_result, panel=panel)
    control_diag = build_control_completeness_diagnostics(
        schema,
        vertical_id=vertical_id,
        media_correlated_controls=media_correlated_controls,
    )
    collinearity_diag = build_collinearity_diagnostics(
        panel,
        schema,
        calibration_evidence_available=calibration_evidence_available,
    )
    sparse_diag = build_sparse_channel_diagnostics(panel, schema)
    fold_diag = build_fold_stability_diagnostics(panel, schema, config, trainer)
    coef_diag = build_coefficient_stability_diagnostics(fit_result, schema)
    known_mu = (known_truth or {}).get("true_mu_c")
    sign_diag = build_sign_plausibility_diagnostics(coef_diag, known_mu=known_mu)
    response_diag = build_response_curve_plausibility_diagnostics(transform_diag)
    lift_diag = build_lift_simulation_stability_diagnostics(fit_result)

    forbidden_claims = build_forbidden_claims_for_ridge(
        control_diag=control_diag,
        collinearity_diag=collinearity_diag,
        sparse_diag=sparse_diag,
        calibration_evidence_available=calibration_evidence_available,
    )

    all_warnings: list[str] = []
    for block in (
        transform_diag,
        control_diag,
        collinearity_diag,
        sparse_diag,
        fold_diag,
        coef_diag,
        sign_diag,
        response_diag,
        lift_diag,
    ):
        all_warnings.extend(block.get("warnings") or [])

    recovery_block: dict[str, Any] | None = None
    if known_truth and coef_diag.get("available"):
        mu = known_truth.get("true_mu_c") or {}
        media_coef = coef_diag.get("media_coef_by_channel") or {}
        errors = [abs(float(media_coef[ch]) - float(mu[ch])) for ch in mu if ch in media_coef]
        recovery_block = {
            "known_truth_available": True,
            "coef_mu_mae_vs_true_mu": float(np.mean(errors)) if errors else None,
            "sign_match_rate_vs_known_mu": sign_diag.get("sign_match_rate_vs_known_mu"),
        }

    report: dict[str, Any] = {
        "report_version": REPORT_VERSION,
        "artifact_kind": "RIDGE_PRODUCTION_DIAGNOSTICS",
        "milestone": "H7",
        "model_id": model_id,
        "run_id": run_id or config.data.data_version_id,
        "dataset_snapshot_id": dataset_snapshot_id or config.data.data_version_id,
        "run_environment": config.run_environment.value,
        "media_columns": list(schema.channel_columns),
        "control_columns": list(schema.control_columns),
        "transform_diagnostics": transform_diag,
        "control_completeness": control_diag,
        "collinearity": collinearity_diag,
        "sparse_channels": sparse_diag,
        "fold_stability": fold_diag,
        "coefficient_stability": coef_diag,
        "sign_plausibility": sign_diag,
        "response_curve_plausibility": response_diag,
        "lift_simulation_stability": lift_diag,
        "truth_recovery": recovery_block,
        "forbidden_claims": forbidden_claims,
        "warnings": sorted(set(all_warnings)),
        "production_flags": _production_flags(config),
        "outputs_are_diagnostic_only": True,
    }
    if world_metadata:
        report["world_metadata"] = world_metadata

    from mmm.diagnostics.calibration_signal_ingestion import build_evidence_attachment_lineage

    report["evidence_attachment_lineage"] = build_evidence_attachment_lineage(
        report, attempted=False, source_type="none"
    )

    for forbidden in FORBIDDEN_OUTPUT_FIELDS:
        if forbidden in report and report.get(forbidden):
            raise ValueError(f"ridge diagnostics must not emit {forbidden!r}")

    from mmm.diagnostics.ridge_severity_policy import apply_severity_policy_to_report

    return apply_severity_policy_to_report(report)


def attach_ridge_diagnostics_to_extension_report(
    extension_report: dict[str, Any],
    panel: pd.DataFrame,
    schema: PanelSchema,
    config: MMMConfig,
    fit_result: dict[str, Any],
    *,
    trainer: RidgeBOMMMTrainer | None = None,
    vertical_id: str | None = None,
    calibration_evidence_available: bool = False,
    calibration_signals: list[dict[str, Any]] | None = None,
    calibration_signals_path: str | None = None,
) -> dict[str, Any]:
    """Merge H7 diagnostics into an extension report without changing optimizer behavior."""
    if config.framework != Framework.RIDGE_BO:
        return extension_report
    report = compose_ridge_diagnostic_report(
        panel,
        schema,
        config,
        fit_result,
        trainer=trainer,
        vertical_id=vertical_id,
        calibration_evidence_available=calibration_evidence_available,
    )
    from mmm.diagnostics.calibration_signal_ingestion import ingest_calibration_signals_into_report

    report = ingest_calibration_signals_into_report(
        report,
        signals=calibration_signals,
        signals_path=calibration_signals_path,
    )
    out = dict(extension_report)
    out["ridge_production_diagnostics_report"] = report
    return out
