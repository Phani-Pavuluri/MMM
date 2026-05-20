"""Extension-time replay calibration (uses full-fit coefficients when available)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from mmm.config.schema import Framework, MMMConfig
from mmm.data.schema import PanelSchema
from mmm.features.design_matrix import build_design_matrix
from mmm.models.ridge_bo.ridge import predict_ridge


def compute_replay_calibration_metrics(
    panel_sorted: pd.DataFrame,
    schema: PanelSchema,
    config: MMMConfig,
    fit_out: dict[str, Any],
) -> tuple[float | None, dict[str, Any], bool]:
    """
    Returns ``(loss_or_none, meta, is_replay)``.

    ``loss`` is mean standardized squared error in replay units; ``None`` if not configured
    or not Ridge / missing artifacts / no units.
    """
    cal = config.calibration
    if not cal.use_replay_calibration:
        return None, {}, False
    if not (cal.replay_units_path or cal.train_replay_units_path):
        return None, {}, False
    if config.framework != Framework.RIDGE_BO or not fit_out.get("artifacts"):
        return None, {}, False
    from mmm.calibration.replay_lift import aggregate_replay_calibration_loss
    from mmm.calibration.replay_units_resolve import resolve_replay_unit_sets

    train_units, holdout_units, split_meta = resolve_replay_unit_sets(config, schema)
    units = train_units
    if not units:
        return None, {"reason": "no_units_loaded", "split_meta": split_meta}, False
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

    loss, meta = aggregate_replay_calibration_loss(
        units,
        predict_level,
        schema=schema,
        target_col=schema.target_column,
        config=config,
    )
    if isinstance(meta, dict):
        meta = {**meta, "split_meta": split_meta, "n_holdout_replay_units": len(holdout_units)}
    return float(loss), meta, True
