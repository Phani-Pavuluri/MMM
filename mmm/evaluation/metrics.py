"""Forecast metrics for CV and objectives."""

from __future__ import annotations

import numpy as np

from mmm.config.schema import FitMetric


def wmape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-9) -> float:
    return float(np.abs(y_true - y_pred).sum() / (np.abs(y_true).sum() + eps))


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-9) -> float:
    return float(np.mean(np.abs((y_true - y_pred) / (y_true + eps))))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def fit_metric(name: FitMetric, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if name == FitMetric.WMAPE:
        return wmape(y_true, y_pred)
    if name == FitMetric.MAPE:
        return mape(y_true, y_pred)
    if name == FitMetric.RMSE:
        return rmse(y_true, y_pred)
    if name == FitMetric.MAE:
        return mae(y_true, y_pred)
    raise ValueError(name)
