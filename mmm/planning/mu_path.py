"""
Full-panel μ path for decision Δμ (constant, piecewise, per-geo spend, optional control overlays).

Uses the same design matrix as training: **all channels**, **controls**, and **recursive adstock**
on the full sorted panel.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd

from mmm.config.schema import MMMConfig
from mmm.data.panel_order import sort_panel_for_modeling
from mmm.data.schema import PanelSchema
from mmm.features.design_matrix import build_design_matrix
from mmm.models.ridge_bo.ridge import predict_ridge
from mmm.planning.control_overlay import ControlOverlaySpec, apply_control_overlay
from mmm.planning.spend_path import PiecewiseSpendPath, counterfactual_piecewise_spend_panel

DeltaMuAggregation = Literal["global_row_mean", "geo_mean_then_global_mean"]


def counterfactual_constant_spend_panel(
    panel: pd.DataFrame,
    schema: PanelSchema,
    spend_by_channel: dict[str, float],
) -> pd.DataFrame:
    """Return a copy of ``panel`` with each channel spend column set to its scalar counterfactual level."""
    out = sort_panel_for_modeling(panel.copy(), schema)
    for ch in schema.channel_columns:
        out[ch] = float(spend_by_channel.get(ch, float(out[ch].mean())))
    return out


def counterfactual_geo_channel_spend_panel(
    panel: pd.DataFrame,
    schema: PanelSchema,
    spend_by_geo_channel: dict[str, dict[str, float]],
) -> pd.DataFrame:
    """Per-row channel spends from ``spend_by_geo_channel[geo][channel]``; geos missing a key keep observed."""
    out = sort_panel_for_modeling(panel.copy(), schema)
    gcol = schema.geo_column
    for ch in schema.channel_columns:
        out[ch] = out[ch].astype(float)
    for i in range(len(out)):
        g = str(out.iloc[i][gcol])
        row = spend_by_geo_channel.get(g)
        if not row:
            continue
        for ch in schema.channel_columns:
            if ch in row:
                out.iloc[i, out.columns.get_loc(ch)] = float(row[ch])
    return out


def _aggregate_mu(
    mu: np.ndarray,
    df_cf: pd.DataFrame,
    schema: PanelSchema,
    aggregation: DeltaMuAggregation,
) -> tuple[float, str]:
    if aggregation == "global_row_mean":
        return float(np.mean(mu)), "mean_over_all_rows_equal_weight"
    gcol = schema.geo_column
    tmp = pd.DataFrame({gcol: df_cf[gcol].astype(str).to_numpy(), "_m": mu})
    per_geo = tmp.groupby(gcol, sort=False)["_m"].mean()
    return float(per_geo.mean()), "mean_of_per_geo_row_mean_mu"


def counterfactual_design_matrix(
    panel: pd.DataFrame,
    schema: PanelSchema,
    config: MMMConfig,
    *,
    best_params: dict[str, float],
    spend_by_channel: dict[str, float] | None = None,
    spend_path: PiecewiseSpendPath | None = None,
    spend_by_geo_channel: dict[str, dict[str, float]] | None = None,
    control_overlay: ControlOverlaySpec | None = None,
) -> tuple[np.ndarray, pd.DataFrame, str]:
    """
    Build **X** on the counterfactual panel (spend path + optional control overlay + adstock design).

    Exactly one spend mode must be set, matching :func:`mean_mu_and_kpi_summary`.
    """
    modes = [spend_path is not None, spend_by_geo_channel is not None, spend_by_channel is not None]
    if sum(modes) != 1:
        raise ValueError("set exactly one of spend_by_channel, spend_path, or spend_by_geo_channel")
    if spend_path is not None:
        df = counterfactual_piecewise_spend_panel(panel, schema, spend_path)
        path_note = "piecewise_calendar_week_channel_overwrites"
    elif spend_by_geo_channel is not None:
        df = counterfactual_geo_channel_spend_panel(panel, schema, spend_by_geo_channel)
        path_note = "per_geo_channel_levels_constant_in_time"
    else:
        df = counterfactual_constant_spend_panel(panel, schema, spend_by_channel or {})
        path_note = "constant_channel_levels_all_weeks"
    df = apply_control_overlay(df, schema, control_overlay)
    decay = float(best_params["decay"])
    hill_half = float(best_params["hill_half"])
    hill_slope = float(best_params["hill_slope"])
    bundle = build_design_matrix(df, schema, config, decay=decay, hill_half=hill_half, hill_slope=hill_slope)
    return bundle.X, df, path_note


def aggregate_mean_mu_draws_hierarchical(
    X: np.ndarray,
    geo_idx: np.ndarray,
    alpha_draws: np.ndarray,
    beta_draws: np.ndarray,
    df_cf: pd.DataFrame,
    schema: PanelSchema,
    aggregation: DeltaMuAggregation,
) -> np.ndarray:
    """
    Same aggregation policy as :func:`aggregate_mean_mu_draws`, but with **per-geo** linear coefficients.

    ``alpha_draws`` is ``(n_draws, n_geo)``, ``beta_draws`` is ``(n_draws, n_geo, n_coef)``, aligned with
    PyMC-style partial / no pooling posteriors. Row ``i`` uses ``geo_idx[i]`` to index draws.
    """
    X = np.asarray(X, dtype=float)
    geo_idx = np.asarray(geo_idx, dtype=int)
    alpha_draws = np.asarray(alpha_draws, dtype=float)
    beta_draws = np.asarray(beta_draws, dtype=float)
    if alpha_draws.ndim != 2 or beta_draws.ndim != 3:
        raise ValueError("alpha_draws must be (n_draws, n_geo) and beta_draws (n_draws, n_geo, n_coef)")
    s, g, p = beta_draws.shape
    if alpha_draws.shape != (s, g):
        raise ValueError("alpha_draws shape must match beta_draws[:, :, 0] geo dimension")
    if X.shape[1] != p:
        raise ValueError(f"X has {X.shape[1]} cols but beta_draws has {p}")
    n = X.shape[0]
    if geo_idx.shape != (n,):
        raise ValueError("geo_idx must be (n_rows,) aligned with X")
    mu = np.zeros((n, s), dtype=float)
    for draw in range(s):
        a_row = alpha_draws[draw, geo_idx]
        b_row = beta_draws[draw, geo_idx, :]
        mu[:, draw] = a_row + np.sum(X * b_row, axis=1)
    if aggregation == "global_row_mean":
        return np.mean(mu, axis=0)
    gcol = schema.geo_column
    g = df_cf[gcol].astype(str).to_numpy()
    _, inv = np.unique(g, return_inverse=True)
    gcount = int(inv.max()) + 1 if inv.size else 1
    out = np.zeros(s, dtype=float)
    for draw in range(s):
        col = mu[:, draw]
        per_geo = np.bincount(inv, weights=col, minlength=gcount) / np.maximum(
            np.bincount(inv, minlength=gcount), 1
        )
        out[draw] = float(np.mean(per_geo))
    return out


def aggregate_mean_mu_draws(
    X: np.ndarray,
    coef_draws: np.ndarray,
    intercept_draws: np.ndarray | None,
    df_cf: pd.DataFrame,
    schema: PanelSchema,
    aggregation: DeltaMuAggregation,
) -> np.ndarray:
    """
    Posterior mean-μ on the modeling scale for each draw (same aggregation policy as point μ).

    ``coef_draws`` is ``(n_draws, n_coef)`` aligned with ``X @ coef`` (Ridge BO layout).
    """
    coef_draws = np.asarray(coef_draws, dtype=float)
    if coef_draws.ndim != 2:
        raise ValueError("coef_draws must be 2d (n_draws, n_features)")
    n_draws, p = coef_draws.shape
    if X.shape[1] != p:
        raise ValueError(f"X has {X.shape[1]} cols but coef_draws has {p}")
    if intercept_draws is None:
        intc = np.zeros(n_draws, dtype=float)
    else:
        intc = np.asarray(intercept_draws, dtype=float).reshape(-1)
        if intc.size == 1 and n_draws > 1:
            intc = np.full(n_draws, float(intc[0]), dtype=float)
        elif intc.size != n_draws:
            raise ValueError("intercept_draws must have length 1 or n_draws")
    mu = X @ coef_draws.T + intc
    if aggregation == "global_row_mean":
        return np.mean(mu, axis=0)
    gcol = schema.geo_column
    g = df_cf[gcol].astype(str).to_numpy()
    _, inv = np.unique(g, return_inverse=True)
    gcount = int(inv.max()) + 1 if inv.size else 1
    _, S = mu.shape
    out = np.zeros(S, dtype=float)
    for s in range(S):
        col = mu[:, s]
        per_geo = np.bincount(inv, weights=col, minlength=gcount) / np.maximum(
            np.bincount(inv, minlength=gcount), 1
        )
        out[s] = float(np.mean(per_geo))
    return out


def mean_mu_and_kpi_summary(
    panel: pd.DataFrame,
    schema: PanelSchema,
    config: MMMConfig,
    *,
    best_params: dict[str, float],
    coef: np.ndarray,
    intercept: np.ndarray,
    spend_by_channel: dict[str, float] | None = None,
    spend_path: PiecewiseSpendPath | None = None,
    spend_by_geo_channel: dict[str, dict[str, float]] | None = None,
    control_overlay: ControlOverlaySpec | None = None,
    delta_mu_aggregation: DeltaMuAggregation = "global_row_mean",
) -> dict[str, Any]:
    """
    Gaussian-mean μ on the **modeling** scale and level-KPI summaries.

    Exactly **one** of ``spend_by_channel``, ``spend_path``, or ``spend_by_geo_channel`` must be set.
    ``control_overlay`` rewrites non-spend columns on the counterfactual panel before the design matrix.
    """
    X, df, path_note = counterfactual_design_matrix(
        panel,
        schema,
        config,
        best_params=best_params,
        spend_by_channel=spend_by_channel,
        spend_path=spend_path,
        spend_by_geo_channel=spend_by_geo_channel,
        control_overlay=control_overlay,
    )
    mu = predict_ridge(X, coef, intercept)
    mean_mu, agg_sem = _aggregate_mu(mu, df, schema, delta_mu_aggregation)
    kpi_level = np.exp(mu)
    if delta_mu_aggregation == "global_row_mean":
        mean_kpi = float(np.mean(kpi_level))
    else:
        gcol = schema.geo_column
        tmp = pd.DataFrame({gcol: df[gcol].astype(str).to_numpy(), "_k": kpi_level})
        mean_kpi = float(tmp.groupby(gcol, sort=False)["_k"].mean().mean())
    return {
        "mean_mu_modeling": mean_mu,
        "mean_kpi_level": mean_kpi,
        "n_rows": int(len(mu)),
        "aggregation": agg_sem,
        "delta_mu_aggregation": delta_mu_aggregation,
        "spend_path_kind": path_note,
        "geo_logic": "full sorted panel; recursive adstock on overwritten spend columns",
        "control_overlay_applied": control_overlay is not None and bool(control_overlay.rows),
    }
