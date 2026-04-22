"""Ridge regression via normal equations (numpy only)."""

from __future__ import annotations

import numpy as np


def fit_ridge(X: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (coef excluding intercept, intercept). X should not include intercept column."""
    n, p = X.shape
    X_ = np.column_stack([np.ones(n), X])
    p1 = p + 1
    reg = alpha * np.eye(p1)
    reg[0, 0] = 0.0  # do not penalize intercept
    XtX = X_.T @ X_ + reg
    Xty = X_.T @ y
    beta = np.linalg.solve(XtX, Xty)
    intercept = beta[0]
    coef = beta[1:]
    return coef, np.array([intercept])


def predict_ridge(X: np.ndarray, coef: np.ndarray, intercept: np.ndarray) -> np.ndarray:
    return X @ coef + intercept[0]
