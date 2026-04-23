"""Post-fit validation bundle and release-severity gate (Phase 2 — beyond CV score)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from mmm.config.schema import Framework, MMMConfig, ModelForm
from mmm.data.schema import PanelSchema


def _residual_lag1_autocorr(resid: np.ndarray) -> float:
    if resid.size < 4:
        return 0.0
    z = resid - np.mean(resid)
    v = float(np.var(z) + 1e-12)
    return float(np.mean(z[1:] * z[:-1]) / v)


def compute_post_fit_validation_bundle(
    *,
    panel: pd.DataFrame,
    schema: PanelSchema,
    config: MMMConfig,
    fit_out: dict[str, Any],
    yhat: np.ndarray,
) -> dict[str, Any]:
    """
    Lightweight structural checks after a single final fit (not a substitute for causal validation).

    Emits ``release_gate_severity`` ∈ {ok, warn, block} using conservative defaults suitable for screening.
    """
    y = panel[schema.target_column].to_numpy(dtype=float)
    yhat_ = np.asarray(yhat, dtype=float).reshape(-1)
    if y.size != yhat_.size:
        return {
            "policy_version": "post_fit_validation_v1",
            "error": "length_mismatch_y_yhat",
            "release_gate_severity": "block",
            "release_gate_reasons": ["y_yhat_length_mismatch"],
        }

    if config.framework != Framework.RIDGE_BO:
        return {
            "policy_version": "post_fit_validation_v1",
            "skipped": True,
            "reason": "ridge_bo_only_bundle_in_this_release",
            "release_gate_severity": "ok",
            "release_gate_reasons": [],
        }

    if config.model_form == ModelForm.SEMI_LOG:
        yt = np.log(np.maximum(y, 1e-12))
        yp = np.log(np.maximum(yhat_, 1e-12))
        resid = yt - yp
        kpi_sanity = {
            "mean_y_level": float(np.mean(y)),
            "mean_yhat_level": float(np.mean(yhat_)),
            "log_mean_ratio": float(np.log((np.mean(y) + 1e-12) / (np.mean(yhat_) + 1e-12))),
        }
    else:
        resid = np.log(np.maximum(y, 1e-12)) - np.log(np.maximum(yhat_, 1e-12))
        kpi_sanity = {
            "mean_log_y": float(np.mean(np.log(np.maximum(y, 1e-12)))),
            "mean_log_yhat": float(np.mean(np.log(np.maximum(yhat_, 1e-12)))),
        }

    ac1 = _residual_lag1_autocorr(resid)
    gv = config.extensions.governance
    ac_block = float(gv.post_fit_residual_autocorr_block_abs)
    oot_warn = float(gv.post_fit_oot_mae_ratio_warn)
    oot_block = float(gv.post_fit_oot_mae_ratio_block)
    obj_cv_warn = float(gv.post_fit_objective_trial_cv_warn)
    kpi_warn = float(gv.post_fit_kpi_log_mean_ratio_warn_abs)

    # OOT: last 20% of rows (panel is sorted geo, week for modeling — tail ≈ later time).
    n = len(panel)
    oot_ratio = None
    oot_mae = None
    train_mae = None
    n_tail = int(max(1, round(0.2 * n)))
    if n >= 10 and n_tail >= 1 and n - n_tail >= 3:
        is_tr = np.zeros(n, dtype=bool)
        is_tr[: n - n_tail] = True
        is_oot = ~is_tr
        train_mae = float(np.mean(np.abs(resid[is_tr])))
        oot_mae = float(np.mean(np.abs(resid[is_oot])))
        oot_ratio = float(oot_mae / (train_mae + 1e-12))

    # Objective landscape spread across top trials (screening for unstable BO surface).
    obj_cv = None
    art = fit_out.get("artifacts")
    if art is not None and getattr(art, "leaderboard", None):
        totals = [float(d["total"]) for d in art.leaderboard if isinstance(d, dict) and "total" in d][:15]
        if len(totals) >= 3:
            obj_cv = float(np.std(totals) / (abs(float(np.mean(totals))) + 1e-9))

    reasons: list[str] = []
    sev = "ok"
    if abs(ac1) > ac_block:
        sev = "block"
        reasons.append(f"residual_lag1_autocorr_abs={abs(ac1):.3f}_exceeds_{ac_block}")
    if oot_ratio is not None:
        if oot_ratio > oot_block:
            sev = "block"
            reasons.append(f"oot_mae_ratio={oot_ratio:.3f}_exceeds_{oot_block}")
        elif oot_ratio > oot_warn and sev != "block":
            sev = "warn"
            reasons.append(f"oot_mae_ratio={oot_ratio:.3f}_exceeds_{oot_warn}")
    if obj_cv is not None and obj_cv > obj_cv_warn and sev == "ok":
        sev = "warn"
        reasons.append(f"objective_top_trial_total_cv={obj_cv:.3f}_exceeds_{obj_cv_warn}")

    if config.model_form == ModelForm.SEMI_LOG and "log_mean_ratio" in kpi_sanity:
        if abs(kpi_sanity["log_mean_ratio"]) > kpi_warn and sev != "block":
            sev = "warn"
            reasons.append(f"kpi_mean_level_vs_yhat_log_ratio_exceeds_{kpi_warn}")

    return {
        "policy_version": "post_fit_validation_v1",
        "model_form": config.model_form.value,
        "link_notes": "semi_log uses log(resid) on y vs yhat; log_log uses log differences on log inputs.",
        "residual_lag1_autocorr": ac1,
        "oot_holdout": {
            "method": "last_20_percent_panel_rows_time_ordered_panel",
            "n_rows_train": int(n - n_tail) if n >= 10 else 0,
            "n_rows_oot": int(n_tail) if n >= 10 else 0,
            "train_mae_modeling_residual": train_mae,
            "oot_mae_modeling_residual": oot_mae,
            "oot_to_train_mae_ratio": oot_ratio,
        },
        "kpi_scale_sanity": kpi_sanity,
        "objective_landscape_top_trials_cv": obj_cv,
        "release_gate_severity": sev,
        "release_gate_reasons": reasons,
        "thresholds_applied": {
            "post_fit_residual_autocorr_block_abs": ac_block,
            "post_fit_oot_mae_ratio_warn": oot_warn,
            "post_fit_oot_mae_ratio_block": oot_block,
            "post_fit_objective_trial_cv_warn": obj_cv_warn,
            "post_fit_kpi_log_mean_ratio_warn_abs": kpi_warn,
        },
    }
