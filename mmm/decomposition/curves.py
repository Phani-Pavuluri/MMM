"""Response and marginal ROI curves from transformed spend grids."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mmm.transforms.adstock.geometric import GeometricAdstock
from mmm.transforms.registry import apply_adstock_saturation_series
from mmm.transforms.saturation.hill import HillSaturation


@dataclass
class ResponseCurve:
    spend_grid: np.ndarray
    response: np.ndarray
    marginal_roi: np.ndarray


def build_curve_for_channel(
    spend_grid: np.ndarray,
    *,
    decay: float,
    hill_half: float,
    hill_slope: float,
    beta: float,
    model_form: str,
    horizon_weeks: int = 52,
) -> ResponseCurve:
    """Single-channel steady-state curve: apply adstock to a constant spend path of length ``horizon_weeks``."""
    ad = GeometricAdstock(decay)
    sat = HillSaturation(half_max=hill_half, slope=hill_slope)
    resp = []
    for s in spend_grid:
        series = np.full(max(2, int(horizon_weeks)), float(s))
        x = apply_adstock_saturation_series(series, ad, sat)[-1]
        if model_form == "log_log":
            from mmm.utils.math import safe_log

            x = float(safe_log(np.array([x]))[0])
        else:
            x = float(x)
        resp.append(beta * x)
    resp = np.asarray(resp)
    # np.gradient(..., edge_order=2) needs len(grid) >= 3
    edge = 2 if spend_grid.size >= 3 else 1
    mroi = np.gradient(resp, spend_grid, edge_order=edge)
    return ResponseCurve(spend_grid=spend_grid, response=resp, marginal_roi=mroi)
