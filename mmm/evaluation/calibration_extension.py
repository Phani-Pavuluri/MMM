"""Extension-time replay calibration (uses full-fit coefficients when available)."""

from __future__ import annotations

from pathlib import Path
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
    if not config.calibration.use_replay_calibration:
        return None, {}, False
    if config.framework != Framework.RIDGE_BO or not fit_out.get("artifacts"):
        return None, {}, False

    from mmm.calibration.evidence_replay import (
        aggregate_weighted_evidence_replay_loss,
        build_evidence_weighted_replay_summary,
        prepare_evidence_replay,
        uses_legacy_replay,
        uses_weighted_evidence_replay,
    )
    from mmm.calibration.replay_lift import aggregate_replay_calibration_loss
    from mmm.calibration.replay_prod_gate import assert_replay_production_ready
    from mmm.calibration.units_io import load_calibration_units_from_json

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

    if uses_weighted_evidence_replay(config):
        prepared = prepare_evidence_replay(config, panel_sorted, schema)
        if not prepared.used:
            return None, {"reason": "no_compatible_evidence_units", "rejected": prepared.rejected}, False
        loss, meta = aggregate_weighted_evidence_replay_loss(
            prepared.used,
            predict_level,
            schema=schema,
            target_col=schema.target_column,
            config=config,
        )
        summary = build_evidence_weighted_replay_summary(prepared, meta)
        meta["evidence_weighted_replay_summary"] = summary
        assert_replay_production_ready(
            config,
            [e.unit for e in prepared.used],
            schema=schema,
            evidence_summary=summary,
        )
        return float(loss), meta, True

    if not uses_legacy_replay(config):
        return None, {"reason": "replay_not_configured"}, False

    from mmm.calibration.replay_frames import normalize_replay_units_to_full_panel

    units = load_calibration_units_from_json(Path(config.calibration.replay_units_path))
    units, legacy_warnings = normalize_replay_units_to_full_panel(panel_sorted, schema, units)
    assert_replay_production_ready(config, units, schema=schema)
    if not units:
        return None, {"reason": "no_units_loaded"}, False

    loss, meta = aggregate_replay_calibration_loss(
        units,
        predict_level,
        schema=schema,
        target_col=schema.target_column,
        config=config,
    )
    meta["replay_mode_used"] = "legacy"
    if legacy_warnings:
        meta["legacy_replay_upgrade_warnings"] = legacy_warnings
    return float(loss), meta, True
