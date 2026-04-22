"""E5: baseline benchmarks vs main model metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from mmm.data.schema import PanelSchema
from mmm.evaluation.metrics import wmape
from mmm.models.ridge_bo.ridge import fit_ridge, predict_ridge
from mmm.utils.math import safe_log


@dataclass
class BaselineComparisonReport:
    mae_main: float
    mae_no_media: float
    mae_simple_ridge_raw: float
    mae_semilog_linear: float
    beats_baselines: bool
    details: dict[str, float]
    mae_shuffled_media: float | None = None
    beats_shuffled_media: bool | None = None
    signal_may_be_spurious_timing: bool = False

    def to_json(self) -> dict:
        out = {
            "mae_main": self.mae_main,
            "mae_no_media": self.mae_no_media,
            "mae_simple_ridge_raw": self.mae_simple_ridge_raw,
            "mae_semilog_linear": self.mae_semilog_linear,
            "beats_baselines": self.beats_baselines,
            "details": self.details,
            "signal_may_be_spurious_timing": self.signal_may_be_spurious_timing,
        }
        if self.mae_shuffled_media is not None:
            out["mae_shuffled_media"] = self.mae_shuffled_media
        if self.beats_shuffled_media is not None:
            out["beats_shuffled_media"] = self.beats_shuffled_media
        out["baseline_definitions"] = {
            "no_media": "intercept-only on log(target)",
            "simple_ridge_raw": "log1p(raw channel spends), no adstock/saturation",
            "semilog_linear_on_transformed_media": "same adstock+Hill path as main model (design matrix media block)",
            "shuffled_media": (
                "same transforms as main model on spend permuted within geo (timing destroyed)"
                if self.mae_shuffled_media is not None
                else "not_run",
            ),
        }
        return out


def media_shuffled_within_geo(
    df: pd.DataFrame,
    schema: PanelSchema,
    *,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Return a copy with each channel's spend permuted independently within ``geo_column``."""
    out = df.copy()
    gcol = schema.geo_column
    for _, sub in df.groupby(gcol, sort=False):
        idx = sub.index
        for ch in schema.channel_columns:
            vals = sub[ch].to_numpy(dtype=float)
            out.loc[idx, ch] = vals[rng.permutation(len(vals))]
    return out


def run_baselines(
    df: pd.DataFrame,
    schema: PanelSchema,
    yhat_main: np.ndarray,
    X_media: np.ndarray,
    *,
    rng: np.random.Generator | None = None,
    X_media_shuffled_same_transform: np.ndarray | None = None,
) -> BaselineComparisonReport:
    y = df[schema.target_column].to_numpy(dtype=float)
    yl = safe_log(y)
    mae_main = float(np.mean(np.abs(y - yhat_main)))

    intercept_only = np.ones((len(df), 1))
    c0, i0 = fit_ridge(intercept_only, yl, alpha=10.0)
    y0 = np.exp(predict_ridge(intercept_only, c0, i0))
    mae_nm = float(np.mean(np.abs(y - y0)))

    raw = df[list(schema.channel_columns)].to_numpy(dtype=float)
    raw = np.log1p(np.maximum(raw, 0.0))
    c1, i1 = fit_ridge(raw, yl, alpha=10.0)
    y1 = np.exp(predict_ridge(raw, c1, i1))
    mae_sr = float(np.mean(np.abs(y - y1)))

    # semi-log without saturation: use X_media as already transformed (caller passes saturated)
    c2, i2 = fit_ridge(X_media, yl, alpha=10.0)
    y2 = np.exp(predict_ridge(X_media, c2, i2))
    mae_sl = float(np.mean(np.abs(y - y2)))

    margin = 0.02 * max(mae_nm, 1e-9)
    beats = mae_main + margin < min(mae_nm, mae_sr, mae_sl)
    mae_shuf: float | None = None
    beats_shuf: bool | None = None
    spurious = False
    if rng is not None:
        if X_media_shuffled_same_transform is not None:
            X_shuf = X_media_shuffled_same_transform
        else:
            df_shuf = media_shuffled_within_geo(df, schema, rng=rng)
            X_shuf = df_shuf[list(schema.channel_columns)].to_numpy(dtype=float)
            X_shuf = np.log1p(np.maximum(X_shuf, 0.0))
        c3, i3 = fit_ridge(X_shuf, yl, alpha=10.0)
        y3 = np.exp(predict_ridge(X_shuf, c3, i3))
        mae_shuf = float(np.mean(np.abs(y - y3)))
        beats_shuf = mae_main + margin < mae_shuf
        spurious = not beats_shuf
    return BaselineComparisonReport(
        mae_main=mae_main,
        mae_no_media=mae_nm,
        mae_simple_ridge_raw=mae_sr,
        mae_semilog_linear=mae_sl,
        beats_baselines=beats,
        details={
            "wmape_main": wmape(y, yhat_main),
            "wmape_no_media": wmape(y, y0),
        },
        mae_shuffled_media=mae_shuf,
        beats_shuffled_media=beats_shuf,
        signal_may_be_spurious_timing=spurious,
    )
