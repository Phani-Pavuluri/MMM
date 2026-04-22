"""Composite objective for Ridge + BO — components are explicit and logged."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from mmm.config.schema import CompositeObjectiveConfig, FitMetric, NormalizationProfile
from mmm.evaluation.metrics import fit_metric
from mmm.evaluation.normalization_policy import (
    describe_objective_normalization,
    normalize_objective_vector,
)


@dataclass
class ObjectiveComponents:
    predictive: float
    calibration: float
    stability: float
    plausibility: float
    complexity: float

    def weighted_total(self, weights: CompositeObjectiveConfig) -> float:
        w = weights.weights
        return (
            w.predictive * self.predictive
            + w.calibration * self.calibration
            + w.stability * self.stability
            + w.plausibility * self.plausibility
            + w.complexity * self.complexity
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "predictive": self.predictive,
            "calibration": self.calibration,
            "stability": self.stability,
            "plausibility": self.plausibility,
            "complexity": self.complexity,
        }


def normalize_components(raw: ObjectiveComponents) -> ObjectiveComponents:
    """Scale each to ~[0,1] using soft normalization (``research`` profile)."""
    t = normalize_objective_vector(
        (raw.predictive, raw.calibration, raw.stability, raw.plausibility, raw.complexity),
        NormalizationProfile.RESEARCH,
    )
    return ObjectiveComponents(*t)


def predictive_loss(
    y_true_log: list[np.ndarray],
    y_pred_log: list[np.ndarray],
    metric: FitMetric,
) -> float:
    vals = []
    for yt, yp in zip(y_true_log, y_pred_log, strict=True):
        if metric in {FitMetric.WMAPE, FitMetric.MAPE}:
            yt_ = np.exp(yt)
            yp_ = np.exp(yp)
        else:
            yt_, yp_ = yt, yp
        vals.append(fit_metric(metric, yt_, yp_))
    return float(np.mean(vals)) if vals else 0.0


def stability_penalty(coef_mat: np.ndarray) -> float:
    """coef_mat shape (n_folds, n_features) — variance across folds."""
    if coef_mat.size == 0:
        return 0.0
    return float(np.mean(np.var(coef_mat, axis=0)))


def plausibility_penalty(
    decay: float,
    hill_slope: float,
    coef: np.ndarray,
) -> float:
    pen = 0.0
    if decay < 0.05 or decay > 0.95:
        pen += 1.0
    if hill_slope < 0.5 or hill_slope > 8.0:
        pen += 1.0
    if np.any(coef < -1e-6):
        pen += float(np.mean(np.maximum(-coef, 0.0)))
    return pen


def complexity_penalty(decay: float, hill_half: float, hill_slope: float, log_alpha: float) -> float:
    vec = np.array([decay, hill_half / 5.0, hill_slope / 5.0, (log_alpha + 4) / 8.0])
    return float(np.linalg.norm(vec))


def _weight_sensitivity_one_at_a_time(norm: ObjectiveComponents, cfg: CompositeObjectiveConfig) -> list[dict[str, Any]]:
    """±10% single-weight bump on each axis (normalized components held fixed)."""
    base = float(norm.weighted_total(cfg))
    rows: list[dict[str, Any]] = []
    for fname in ("predictive", "calibration", "stability", "plausibility", "complexity"):
        cur = float(getattr(cfg.weights, fname))
        bumped = cfg.model_copy(update={"weights": cfg.weights.model_copy(update={fname: cur * 1.1})})
        alt = float(norm.weighted_total(bumped))
        rows.append(
            {
                "weight_field": fname,
                "bump_factor": 1.1,
                "weighted_total": alt,
                "delta_vs_base": alt - base,
            }
        )
    return rows


def build_composite(
    *,
    y_true_folds: list[np.ndarray],
    y_pred_folds: list[np.ndarray],
    metric: FitMetric,
    coef_mat: np.ndarray,
    decay: float,
    hill_half: float,
    hill_slope: float,
    log_alpha: float,
    calibration_details: dict | None,
    cfg: CompositeObjectiveConfig,
    baseline_predictive: float | None = None,
) -> tuple[float, ObjectiveComponents, ObjectiveComponents, dict[str, Any]]:
    pred_loss = predictive_loss(y_true_folds, y_pred_folds, metric)
    cal = 0.0
    if calibration_details:
        cal = float(calibration_details.get("loss", 0.0))
    stab = stability_penalty(coef_mat)
    last_coef = coef_mat[-1] if len(coef_mat) else np.array([])
    plaus = plausibility_penalty(decay, hill_slope, last_coef)
    comp = complexity_penalty(decay, hill_half, hill_slope, log_alpha)
    raw = ObjectiveComponents(pred_loss, cal, stab, plaus, comp)
    norm_t = normalize_objective_vector(
        (raw.predictive, raw.calibration, raw.stability, raw.plausibility, raw.complexity),
        cfg.normalization_profile,
        baseline_predictive=baseline_predictive,
        calibration_details=calibration_details,
    )
    norm = ObjectiveComponents(*norm_t)
    total = norm.weighted_total(cfg)
    w = cfg.weights
    norm_report = describe_objective_normalization(
        (raw.predictive, raw.calibration, raw.stability, raw.plausibility, raw.complexity),
        cfg.normalization_profile,
        baseline_predictive=baseline_predictive,
        calibration_details=calibration_details,
        normalized=norm_t,
    )
    norm_report["composite_weights"] = {
        "predictive": w.predictive,
        "calibration": w.calibration,
        "stability": w.stability,
        "plausibility": w.plausibility,
        "complexity": w.complexity,
    }
    norm_report["weighted_total"] = float(total)
    norm_report["weighted_total_formula"] = "sum(weight_k * normalized_component_k)"
    norm_report["weight_one_at_a_time_10pct_bump"] = _weight_sensitivity_one_at_a_time(norm, cfg)
    return total, raw, norm, norm_report
