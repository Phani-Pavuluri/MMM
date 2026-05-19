"""Ridge+BO objective uses intercept baseline (not self-referential predictive norm)."""

from __future__ import annotations

import numpy as np

from mmm.config.schema import FitMetric
from mmm.models.ridge_bo.objective import (
    build_composite,
    intercept_only_predictive_baseline,
    predictive_loss,
)


def test_intercept_baseline_stable_across_identical_folds() -> None:
    folds = [np.array([0.0, 0.1, 0.2]), np.array([0.05, 0.15, 0.25])]
    b1 = intercept_only_predictive_baseline(folds, FitMetric.WMAPE)
    b2 = intercept_only_predictive_baseline(folds, FitMetric.WMAPE)
    assert b1 == b2
    assert b1 > 0


def test_build_composite_uses_baseline_not_self_norm() -> None:
    y_true = [np.array([0.0, 0.2, 0.4])]
    y_pred_good = [np.array([0.01, 0.21, 0.39])]
    y_pred_bad = [np.array([0.5, 0.7, 0.9])]
    baseline = intercept_only_predictive_baseline(y_true, FitMetric.WMAPE)
    from mmm.config.schema import CompositeObjectiveConfig

    cfg = CompositeObjectiveConfig()
    _, raw_good, norm_good, rep_good = build_composite(
        y_true_folds=y_true,
        y_pred_folds=y_pred_good,
        metric=FitMetric.WMAPE,
        coef_mat=np.zeros((1, 2)),
        decay=0.5,
        hill_half=1.0,
        hill_slope=2.0,
        log_alpha=0.0,
        calibration_details=None,
        cfg=cfg,
        baseline_predictive=baseline,
    )
    _, raw_bad, norm_bad, _ = build_composite(
        y_true_folds=y_true,
        y_pred_folds=y_pred_bad,
        metric=FitMetric.WMAPE,
        coef_mat=np.zeros((1, 2)),
        decay=0.5,
        hill_half=1.0,
        hill_slope=2.0,
        log_alpha=0.0,
        calibration_details=None,
        cfg=cfg,
        baseline_predictive=baseline,
    )
    assert raw_good.predictive < raw_bad.predictive
    assert norm_good.predictive <= norm_bad.predictive
    assert rep_good.get("baseline_predictive_used") == baseline
    assert baseline != predictive_loss(y_true, y_pred_good, FitMetric.WMAPE)
