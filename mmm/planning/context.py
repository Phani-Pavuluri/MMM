"""Fitted-model context for full-panel simulation (Ridge path)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from mmm.config.schema import MMMConfig
from mmm.data.schema import PanelSchema


@dataclass
class RidgeFitContext:
    """Panel + schema + resolved Ridge BO parameters for counterfactual μ evaluation."""

    panel: pd.DataFrame
    schema: PanelSchema
    config: MMMConfig
    best_params: dict[str, float]
    coef: np.ndarray
    intercept: np.ndarray


def ridge_context_from_fit(
    panel: pd.DataFrame,
    schema: PanelSchema,
    config: MMMConfig,
    fit_out: dict[str, Any],
) -> RidgeFitContext:
    art = fit_out.get("artifacts")
    if art is None:
        raise ValueError("fit_out must contain Ridge BO artifacts for full-model simulation")
    coef = np.asarray(art.coef, dtype=float).ravel()
    intercept = np.asarray(art.intercept, dtype=float).ravel()
    return RidgeFitContext(
        panel=panel,
        schema=schema,
        config=config,
        best_params=dict(art.best_params),
        coef=coef,
        intercept=intercept,
    )


def ridge_fit_summary_from_artifacts(art: Any) -> dict[str, Any]:
    """JSON-serializable ridge fit blob for extension reports / CLI."""
    return {
        "best_params": dict(art.best_params),
        "coef": np.asarray(art.coef, dtype=float).tolist(),
        "intercept": np.asarray(art.intercept, dtype=float).tolist(),
    }


def ridge_context_from_summary(
    panel: pd.DataFrame,
    schema: PanelSchema,
    config: MMMConfig,
    summary: dict[str, Any],
) -> RidgeFitContext:
    """Rebuild :class:`RidgeFitContext` from ``ridge_fit_summary`` JSON."""
    bp = summary.get("best_params")
    if not isinstance(bp, dict):
        raise ValueError("ridge_fit_summary.best_params missing")
    coef = np.asarray(summary["coef"], dtype=float).ravel()
    intercept = np.asarray(summary["intercept"], dtype=float).ravel()
    return RidgeFitContext(
        panel=panel,
        schema=schema,
        config=config,
        best_params={str(k): float(v) for k, v in bp.items()},
        coef=coef,
        intercept=intercept,
    )
