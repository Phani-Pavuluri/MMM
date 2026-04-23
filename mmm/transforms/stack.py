"""Apply transform stack to panel data."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from mmm.config.schema import MMMConfig, ModelForm, TransformConfig
from mmm.data.schema import PanelSchema
from mmm.transforms.registry import apply_adstock_saturation_series


def model_form_guidance(model_form: ModelForm, transform_cfg: TransformConfig) -> None:
    """Emit warnings for risky transform + model form pairings."""
    if model_form == ModelForm.LOG_LOG and transform_cfg.saturation in {"hill", "logistic"}:
        warnings.warn(
            "log-log with strong saturation may double-compress media; prefer semi-log + Hill.",
            stacklevel=2,
        )


def build_channel_features_from_params(
    df: pd.DataFrame,
    schema: PanelSchema,
    transform_cfg: TransformConfig,
    *,
    decay: float | None = None,
    hill_half: float | None = None,
    hill_slope: float | None = None,
    modeling_config: MMMConfig | None = None,
) -> np.ndarray:
    """Build (n_rows, n_channels) transformed media matrix with explicit hyperparameters."""
    from mmm.contracts.canonical_transforms import assert_canonical_media_stack_for_modeling
    from mmm.transforms.adstock.geometric import GeometricAdstock
    from mmm.transforms.saturation.hill import HillSaturation

    if modeling_config is not None:
        assert_canonical_media_stack_for_modeling(modeling_config)

    if transform_cfg.adstock != "geometric":
        raise NotImplementedError("Only geometric adstock in fast path for BO")
    if transform_cfg.saturation != "hill":
        raise NotImplementedError("Only hill saturation in fast path for BO")
    decay = 0.5 if decay is None else float(decay)
    hill_half = 1.0 if hill_half is None else float(hill_half)
    hill_slope = 2.0 if hill_slope is None else float(hill_slope)
    ad = GeometricAdstock(decay)
    sat = HillSaturation(half_max=hill_half, slope=hill_slope)

    mats: list[np.ndarray] = []
    grouped = df.groupby(schema.geo_column, sort=False)
    for _, g in grouped:
        g = g.sort_values(schema.week_column)
        block = []
        for ch in schema.channel_columns:
            arr = g[ch].to_numpy(dtype=float)
            block.append(apply_adstock_saturation_series(arr, ad, sat))
        mats.append(np.column_stack(block))
    return np.vstack(mats)
