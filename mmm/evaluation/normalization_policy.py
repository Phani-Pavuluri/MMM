"""Profile-based scaling for composite objective terms (see roadmap §3)."""

from __future__ import annotations

from typing import Any

import numpy as np

from mmm.config.schema import NormalizationProfile

OBJECTIVE_NORMALIZATION_VERSION = "objective_norm_v2_decoupled_piecewise"

# Legacy coupled research rule (pre-v2) retained only for explicit DEBUG-style experiments if needed.
def _soft_normalize_vec(v: np.ndarray) -> np.ndarray:
    scale = np.maximum(v, 1e-6)
    return scale / scale.mean()


def _normalize_piecewise_decoupled(
    raw: tuple[float, float, float, float, float],
    *,
    baseline_predictive: float | None,
    calibration_details: dict | None,
) -> tuple[float, float, float, float, float]:
    """Per-component scaling — same piecewise map as ``STRICT_PROD`` (no cross-component mean coupling)."""
    pred, cal, stab, plaus, comp = raw
    base = baseline_predictive
    if base is None or base <= 0:
        base = max(pred, 1e-9)
    pred_n = float(np.clip(pred / base, 0.0, 50.0))
    se_scale = None
    if calibration_details:
        se_scale = calibration_details.get("mean_lift_se")
    if se_scale is not None and float(se_scale) > 0:
        cal_n = float(cal / float(se_scale))
    else:
        cal_n = float(np.clip(cal, 0.0, 50.0))
    stab_n = float(stab / (0.05 + stab))
    plaus_n = float(np.minimum(plaus, 1.0))
    comp_n = float(np.clip(comp / 5.0, 0.0, 5.0))
    return (pred_n, cal_n, stab_n, plaus_n, comp_n)


def describe_objective_normalization(
    raw: tuple[float, float, float, float, float],
    profile: NormalizationProfile,
    *,
    baseline_predictive: float | None = None,
    calibration_details: dict | None = None,
    normalized: tuple[float, float, float, float, float],
    component_names: tuple[str, ...] = (
        "predictive",
        "calibration",
        "stability",
        "plausibility",
        "complexity",
    ),
) -> dict[str, Any]:
    """
    Inspectable normalization profile for leaderboards / artifacts.

    Documents raw vs normalized vectors and the rule applied so a winning trial is auditable.
    """
    pred, cal, stab, plaus, comp = raw
    n_pred, n_cal, n_stab, n_plaus, n_comp = normalized
    base = baseline_predictive
    if (base is None or base <= 0) and profile == NormalizationProfile.STRICT_PROD:
        base = max(pred, 1e-9)
    report: dict[str, Any] = {
        "profile": profile.value,
        "normalization_version": OBJECTIVE_NORMALIZATION_VERSION,
        "component_names": list(component_names),
        "raw_vector": [pred, cal, stab, plaus, comp],
        "normalized_vector": [n_pred, n_cal, n_stab, n_plaus, n_comp],
        "baseline_predictive_used": float(base) if base is not None and base > 0 else None,
        "calibration_mean_lift_se": calibration_details.get("mean_lift_se") if calibration_details else None,
    }
    if profile == NormalizationProfile.DEBUG:
        report["rule"] = "identity_no_scaling"
    elif profile == NormalizationProfile.RESEARCH:
        report["rule"] = (
            "research_decoupled_piecewise_same_formula_as_strict_prod "
            f"({OBJECTIVE_NORMALIZATION_VERSION}); no cross-component mean coupling"
        )
    else:
        report["rule"] = "strict_prod_piecewise_predictive_over_baseline_calibration_se_scaled"
        report["piecewise"] = {
            "predictive": "clip(pred / baseline_predictive, 0, 50)",
            "calibration": "cal / mean_lift_se if se>0 else clip(cal,0,50)",
            "stability": "stab / (0.05 + stab)",
            "plausibility": "min(plaus, 1)",
            "complexity": "clip(comp/5, 0, 5)",
        }
    return report


def normalize_objective_vector(
    raw: tuple[float, float, float, float, float],
    profile: NormalizationProfile,
    *,
    baseline_predictive: float | None = None,
    calibration_details: dict | None = None,
) -> tuple[float, float, float, float, float]:
    """
    Map (predictive, calibration, stability, plausibility, complexity) to weighted scale.

    ``baseline_predictive`` should match the same units as the predictive term
    (e.g. WMAPE of an intercept-only baseline).
    """
    pred, cal, stab, plaus, comp = raw
    if profile == NormalizationProfile.DEBUG:
        return pred, cal, stab, plaus, comp
    if profile in (NormalizationProfile.RESEARCH, NormalizationProfile.STRICT_PROD):
        return _normalize_piecewise_decoupled(
            raw,
            baseline_predictive=baseline_predictive,
            calibration_details=calibration_details,
        )
    raise ValueError(f"unsupported normalization profile: {profile!r}")
