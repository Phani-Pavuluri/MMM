"""Replay spend paths through the same predict path as training (explicit estimand only)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from mmm.calibration.contracts import CalibrationUnit
from mmm.calibration.replay_estimand import ReplayEstimandSpec, aggregate_level_delta_masked
from mmm.calibration.replay_prod_gate import validate_replay_units_economics_alignment
from mmm.config.schema import MMMConfig
from mmm.data.schema import PanelSchema


def implied_lift_from_counterfactual(
    *,
    panel_observed: pd.DataFrame,
    panel_counterfactual: pd.DataFrame,
    predict_fn: Callable[[pd.DataFrame], np.ndarray],
    schema: PanelSchema,
    estimand: ReplayEstimandSpec,
) -> dict[str, Any]:
    """
    Compare model predictions on observed vs counterfactual spend panels under **explicit** estimand.

    Implied lift is aggregated per ``estimand.aggregation`` over rows selected by geo + week window.
    """
    if len(panel_observed) != len(panel_counterfactual):
        raise ValueError("observed and counterfactual panels must have the same length for index-aligned replay")
    yhat_obs = predict_fn(panel_observed)
    yhat_cf = predict_fn(panel_counterfactual)
    if estimand.target_kpi_column != schema.target_column:
        raise ValueError(
            f"replay_estimand.target_kpi_column {estimand.target_kpi_column!r} must match schema.target_column "
            f"{schema.target_column!r}"
        )
    implied, agg_meta = aggregate_level_delta_masked(yhat_obs, yhat_cf, panel_observed, schema, estimand)
    return {
        "implied_mean_delta": implied,
        "n_eval": int(agg_meta.get("n_eval_rows", 0)),
        "estimand": estimand.to_json(),
    }


def aggregate_replay_calibration_loss(
    units: list[CalibrationUnit],
    predict_fn: Callable[[pd.DataFrame], np.ndarray],
    *,
    schema: PanelSchema,
    target_col: str,
    config: MMMConfig | None = None,  # reserved for future environment-specific policy hooks
) -> tuple[float, dict[str, Any]]:
    """
    Mean squared standardized error: mean((implied_delta - observed_lift)^2 / SE^2).

    **Requires** ``CalibrationUnit.replay_estimand`` (serialized :class:`ReplayEstimandSpec`) on every
    unit with frames — no implicit full-panel mean (fail-closed for decision safety).
    """
    if target_col != schema.target_column:
        raise ValueError("target_col must match schema.target_column for replay KPI alignment")
    if config is not None:
        validate_replay_units_economics_alignment(config, schema, units)
    z2: list[float] = []
    meta_units: list[dict[str, Any]] = []
    _ = config
    for u in units:
        if u.observed_spend_frame is None or u.counterfactual_spend_frame is None or u.observed_lift is None:
            continue
        if not u.replay_estimand:
            raise ValueError(
                f"replay unit {u.unit_id!r}: replay_estimand is required (explicit geo/time window + aggregation); "
                "implicit full-panel mean is not supported"
            )
        spec = ReplayEstimandSpec.from_dict(u.replay_estimand)
        r = implied_lift_from_counterfactual(
            panel_observed=u.observed_spend_frame,
            panel_counterfactual=u.counterfactual_spend_frame,
            predict_fn=predict_fn,
            schema=schema,
            estimand=spec,
        )
        implied = float(r["implied_mean_delta"])
        se = float(u.lift_se) if u.lift_se is not None and u.lift_se > 0 else 1.0
        diff = implied - float(u.observed_lift)
        z2.append(float((diff**2) / (se**2 + 1e-12)))
        meta_units.append(
            {
                "unit_id": u.unit_id,
                "implied_delta": implied,
                "observed_lift": u.observed_lift,
                "se": se,
                "replay_estimand": spec.to_json(),
            }
        )
    if not z2:
        return 0.0, {"n_units": 0, "units": meta_units}
    arr = np.array(z2, dtype=float)
    return float(np.mean(arr)), {
        "n_units": len(z2),
        "mean_standardized_sq_error": float(np.mean(arr)),
        "units": meta_units,
        "mean_lift_se": float(np.mean([u.lift_se for u in units if u.lift_se and u.lift_se > 0] or [1.0])),
    }


def replay_unit_placeholder(unit: CalibrationUnit) -> dict[str, Any]:
    """Return structure for JSON when frames are not yet wired."""
    return {
        "unit_id": unit.unit_id,
        "treated_channel_names": unit.treated_channel_names,
        "replay_ready": unit.observed_spend_frame is not None and unit.counterfactual_spend_frame is not None,
    }
