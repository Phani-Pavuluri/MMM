"""Post-fit replay holdout diagnostics (does not change BO unless split enabled in trainer)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from mmm.calibration.replay_lift import aggregate_replay_calibration_loss
from mmm.calibration.replay_units_resolve import resolve_replay_unit_sets
from mmm.config.schema import Framework, MMMConfig
from mmm.data.schema import PanelSchema
from mmm.features.design_matrix import build_design_matrix
from mmm.models.ridge_bo.ridge import predict_ridge


def build_replay_holdout_validation(
    panel_sorted: pd.DataFrame,
    schema: PanelSchema,
    config: MMMConfig,
    fit_out: dict[str, Any],
) -> dict[str, Any]:
    """Emit ``replay_holdout_validation`` artifact section."""
    if not config.calibration.use_replay_calibration:
        return {"status": "skipped", "reason": "replay_calibration_not_configured"}
    if not config.calibration.use_replay_holdout_split:
        train_units, _, split_meta = resolve_replay_unit_sets(config, schema)
        return {
            "status": "disabled",
            "reason": "replay_holdout_split_disabled",
            "holdout_not_available_reason": "replay_holdout_split_disabled",
            "n_train_replay_units": len(train_units),
            "n_holdout_replay_units": 0,
            "train_replay_loss": None,
            "holdout_replay_loss": None,
            "sensitivity_warning": False,
            "split_meta": split_meta,
        }
    if config.framework != Framework.RIDGE_BO or not fit_out.get("artifacts"):
        return {"status": "skipped", "reason": "ridge_artifacts_required"}

    train_units, holdout_units, split_meta = resolve_replay_unit_sets(config, schema)
    base: dict[str, Any] = {
        "status": "ok",
        "diagnostic_only": True,
        "policy_note": "Holdout replay loss is never used in the BO objective when split is enabled.",
        "n_train_replay_units": len(train_units),
        "n_holdout_replay_units": len(holdout_units),
        "split_meta": split_meta,
    }
    reason = split_meta.get("holdout_not_available_reason")
    if reason:
        base["status"] = "holdout_not_available"
        base["holdout_not_available_reason"] = reason
        base["train_replay_loss"] = None
        base["holdout_replay_loss"] = None
        base["sensitivity_warning"] = False
        return base

    if not train_units:
        base["status"] = "skipped"
        base["holdout_not_available_reason"] = "no_train_replay_units"
        return base

    art = fit_out["artifacts"]

    def predict_level(dfp: pd.DataFrame) -> np.ndarray:
        b = build_design_matrix(
            dfp,
            schema,
            config,
            decay=art.best_params["decay"],
            hill_half=art.best_params["hill_half"],
            hill_slope=art.best_params["hill_slope"],
        )
        ylog = predict_ridge(b.X, art.coef, art.intercept)
        return np.exp(ylog)

    train_loss, _train_meta = aggregate_replay_calibration_loss(
        train_units,
        predict_level,
        schema=schema,
        target_col=schema.target_column,
        config=config,
    )
    base["train_replay_loss"] = float(train_loss)
    if not holdout_units:
        base["holdout_replay_loss"] = None
        base["sensitivity_warning"] = False
        base["holdout_not_available_reason"] = (
            split_meta.get("holdout_not_available_reason") or "holdout_split_disabled"
        )
        return base

    holdout_loss, _hold_meta = aggregate_replay_calibration_loss(
        holdout_units,
        predict_level,
        schema=schema,
        target_col=schema.target_column,
        config=config,
    )
    base["holdout_replay_loss"] = float(holdout_loss)
    denom = max(abs(float(train_loss)), 1e-9)
    rel = abs(float(holdout_loss) - float(train_loss)) / denom
    base["relative_holdout_shift"] = float(rel)
    base["sensitivity_warning"] = bool(rel > 0.2)
    if base["sensitivity_warning"]:
        base["interpretation"] = "holdout replay loss diverges from train replay loss — review calibration overfit"
    else:
        base["interpretation"] = "holdout replay loss stable vs train at this split"
    return base
