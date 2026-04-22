"""Shared helpers to build the same media design matrix as fit/decomposition."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from mmm.config.schema import Framework, MMMConfig
from mmm.data.schema import PanelSchema
from mmm.features.design_matrix import DesignMatrixBundle, build_design_matrix, media_design_matrix


def transform_params_from_fit_or_defaults(
    config: MMMConfig, fit_out: dict[str, Any] | None
) -> dict[str, float]:
    if config.framework == Framework.RIDGE_BO and fit_out and fit_out.get("artifacts") is not None:
        bp = fit_out["artifacts"].best_params
        return {"decay": bp["decay"], "hill_half": bp["hill_half"], "hill_slope": bp["hill_slope"]}
    return {
        "decay": float(config.transforms.adstock_params.get("decay", 0.5)),
        "hill_half": float(config.transforms.saturation_params.get("half_max", 1.0)),
        "hill_slope": float(config.transforms.saturation_params.get("slope", 2.0)),
    }


def build_extension_design_bundle(
    panel_sorted: pd.DataFrame,
    schema: PanelSchema,
    config: MMMConfig,
    fit_out: dict[str, Any] | None,
) -> tuple[DesignMatrixBundle, np.ndarray]:
    """
    Single entry for extensions: same ``build_design_matrix`` as Ridge trainer.

    Returns ``(bundle, X_media)`` with ``X_media`` = media columns only.
    """
    tp = transform_params_from_fit_or_defaults(config, fit_out)
    bundle = build_design_matrix(
        panel_sorted,
        schema,
        config,
        decay=tp["decay"],
        hill_half=tp["hill_half"],
        hill_slope=tp["hill_slope"],
    )
    return bundle, media_design_matrix(bundle, schema)
