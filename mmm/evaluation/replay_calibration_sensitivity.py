"""Diagnostic: replay calibration sensitivity to full-panel refit (no BO objective changes)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from mmm.calibration.replay_lift import aggregate_replay_calibration_loss
from mmm.config.schema import Framework, MMMConfig
from mmm.data.schema import PanelSchema
from mmm.features.design_matrix import build_design_matrix
from mmm.models.ridge_bo.ridge import fit_ridge, predict_ridge


def build_replay_calibration_sensitivity(
    panel: pd.DataFrame,
    schema: PanelSchema,
    config: MMMConfig,
    fit_out: dict[str, Any],
    *,
    holdout_fraction: float = 0.25,
) -> dict[str, Any]:
    """
    Compare replay loss when predict_level uses a full-panel refit vs a train-mask refit.

    Diagnostic only — does not change Ridge BO trial scoring.
    """
    if config.framework != Framework.RIDGE_BO or not fit_out.get("artifacts"):
        return {"status": "skipped", "reason": "ridge_artifacts_required"}
    if not config.calibration.use_replay_calibration or not config.calibration.replay_units_path:
        return {"status": "skipped", "reason": "replay_calibration_not_configured"}

    from mmm.calibration.units_io import load_calibration_units_from_json

    units = load_calibration_units_from_json(config.calibration.replay_units_path)
    if not units:
        return {"status": "skipped", "reason": "no_replay_units"}

    art = fit_out["artifacts"]
    bp = art.best_params
    n = len(panel)
    if n < 12:
        return {"status": "skipped", "reason": "insufficient_rows", "n_rows": n}

    rng = np.random.default_rng(int(config.random_seed or 0))
    idx = np.arange(n)
    rng.shuffle(idx)
    holdout_n = max(4, int(n * holdout_fraction))
    train_idx = idx[holdout_n:]
    train_df = panel.iloc[train_idx].reset_index(drop=True)
    full_df = panel

    def _fit_predict(df_fit: pd.DataFrame, df_pred: pd.DataFrame) -> np.ndarray:
        bundle = build_design_matrix(
            df_fit,
            schema,
            config,
            decay=float(bp["decay"]),
            hill_half=float(bp["hill_half"]),
            hill_slope=float(bp["hill_slope"]),
        )
        coef, intercept = fit_ridge(bundle.X, bundle.y_modeling, float(10 ** float(bp.get("log_alpha", 0))))
        b_pred = build_design_matrix(
            df_pred,
            schema,
            config,
            decay=float(bp["decay"]),
            hill_half=float(bp["hill_half"]),
            hill_slope=float(bp["hill_slope"]),
        )
        return np.exp(predict_ridge(b_pred.X, coef, intercept))

    def predict_full(dfp: pd.DataFrame) -> np.ndarray:
        return _fit_predict(full_df, dfp)

    def predict_train(dfp: pd.DataFrame) -> np.ndarray:
        return _fit_predict(train_df, dfp)

    loss_full, meta_full = aggregate_replay_calibration_loss(
        units, predict_full, schema=schema, target_col=schema.target_column, config=config
    )
    loss_train, meta_train = aggregate_replay_calibration_loss(
        units, predict_train, schema=schema, target_col=schema.target_column, config=config
    )
    denom = max(abs(float(loss_full)), 1e-9)
    rel = abs(float(loss_train) - float(loss_full)) / denom
    return {
        "status": "ok",
        "diagnostic_only": True,
        "policy_note": "BO objective unchanged; use to assess replay overfit to full-panel refit.",
        "replay_loss_full_panel_refit": float(loss_full),
        "replay_loss_train_mask_refit": float(loss_train),
        "relative_shift": float(rel),
        "holdout_fraction": holdout_fraction,
        "n_units": len(units),
        "meta_full": meta_full,
        "meta_train": meta_train,
        "interpretation": (
            "large_relative_shift suggests replay objective is sensitive to in-sample refit"
            if rel > 0.2
            else "replay loss stable between full-panel and train-mask refit at this split"
        ),
    }
