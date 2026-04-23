"""Match experiments to model scopes — config surface must align with enforced semantics (see trace)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from mmm.calibration.admissibility import experiment_admissibility_violations
from mmm.calibration.quality import experiment_quality_score
from mmm.calibration.schema import ExperimentObservation

if TYPE_CHECKING:
    from mmm.config.schema import RunEnvironment

# Declared in calibration.match_levels — anything else is a parse-time error.
SUPPORTED_CALIBRATION_MATCH_LEVELS: frozenset[str] = frozenset(
    {"geo", "channel", "time_window", "device", "product"}
)


def validate_calibration_match_levels(match_levels: list[str]) -> None:
    bad = [x for x in match_levels if x not in SUPPORTED_CALIBRATION_MATCH_LEVELS]
    if bad:
        raise ValueError(
            f"calibration.match_levels has unsupported entries {bad!r}; "
            f"allowed={sorted(SUPPORTED_CALIBRATION_MATCH_LEVELS)}"
        )


@dataclass
class MatchedExperiment:
    obs: ExperimentObservation
    weight: float
    quality_score: float = 1.0


@dataclass
class MatchExperimentsResult:
    matched: list[MatchedExperiment]
    trace: dict[str, Any]


def _to_ts(v: str | None) -> pd.Timestamp | None:
    if v is None or str(v).strip() == "":
        return None
    t = pd.to_datetime(v, errors="coerce")
    if pd.isna(t):
        return None
    return t  # type: ignore[return-value]


def _time_window_ok(
    ex: ExperimentObservation,
    *,
    panel_week_min: pd.Timestamp | None,
    panel_week_max: pd.Timestamp | None,
) -> bool:
    if ex.start_week is None and ex.end_week is None:
        return True
    if panel_week_min is None or panel_week_max is None:
        return False
    obs_start = _to_ts(ex.start_week) if ex.start_week else panel_week_min
    obs_end = _to_ts(ex.end_week) if ex.end_week else panel_week_max
    if obs_start is None or obs_end is None:
        return False
    return not (obs_end < panel_week_min or obs_start > panel_week_max)


def match_experiments_with_trace(
    experiments: list[ExperimentObservation],
    *,
    available_geos: set[str] | None,
    available_channels: set[str],
    match_levels: list[str],
    apply_quality: bool = True,
    panel_week_min: pd.Timestamp | None = None,
    panel_week_max: pd.Timestamp | None = None,
    allowed_devices: set[str] | None = None,
    allowed_products: set[str] | None = None,
    run_environment: RunEnvironment | None = None,
) -> MatchExperimentsResult:
    """
    Filter experiments and emit an operator-facing **matching trace** (requested vs applied semantics).

    Semantics (must stay aligned with ``CalibrationConfig.match_levels`` documentation):

    - **channel**: always enforced — ``ex.channel`` must be in ``available_channels``.
    - **geo**: when ``geo`` is in ``match_levels``, if ``ex.geo_id`` is set it must lie in
      ``available_geos``. Rows with **no** ``geo_id`` still pass (national / pooled experiments);
      this is recorded as a semantic note, not silent.
    - **time_window**: when in ``match_levels``, observations with **no** ``start_week``/``end_week``
      pass. Observations with dates require overlap with ``[panel_week_min, panel_week_max]``; if
      panel bounds are missing, dated observations **do not** match (strict).
    - **device** / **product**: when in ``match_levels`` and the observation sets the field, the value
      must appear in the corresponding allowed set. If the allowed set is ``None`` while the field
      is set, the observation does **not** match (strict; not a silent ignore).
    """
    validate_calibration_match_levels(match_levels)
    matched: list[MatchedExperiment] = []
    rejections: dict[str, int] = {}
    warnings: list[str] = []

    missing_geo_id = sum(1 for ex in experiments if ex.geo_id is None or str(ex.geo_id).strip() == "")
    if "geo" in match_levels and missing_geo_id:
        warnings.append(
            f"geo_in_match_levels_but_{missing_geo_id}_experiments_missing_geo_id: "
            "those_rows_are_not_geo_filtered; only_rows_with_geo_id_are_checked_vs_panel_geos"
        )

    if "time_window" in match_levels and (panel_week_min is None or panel_week_max is None):
        warnings.append(
            "time_window_in_match_levels_but_panel_week_bounds_missing: "
            "dated_experiments_will_not_match_until_bounds_supplied"
        )

    if "device" in match_levels and allowed_devices is None:
        warnings.append(
            "device_in_match_levels_but_allowed_devices_not_supplied: "
            "experiments_with_device_set_will_not_match"
        )

    if "product" in match_levels and allowed_products is None:
        warnings.append(
            "product_in_match_levels_but_allowed_products_not_supplied: "
            "experiments_with_product_set_will_not_match"
        )

    for ex in experiments:
        adm = experiment_admissibility_violations(ex, run_environment=run_environment)
        if adm:
            key = "admissibility:" + adm[0]
            rejections[key] = rejections.get(key, 0) + 1
            continue
        if ex.channel not in available_channels:
            rejections["channel_not_in_panel_scope"] = rejections.get("channel_not_in_panel_scope", 0) + 1
            continue
        if "geo" in match_levels and ex.geo_id and available_geos and ex.geo_id not in available_geos:
            rejections["geo_not_in_training_geos"] = rejections.get("geo_not_in_training_geos", 0) + 1
            continue
        if "time_window" in match_levels and not _time_window_ok(
            ex, panel_week_min=panel_week_min, panel_week_max=panel_week_max
        ):
            rejections["time_window_no_overlap_or_unparseable_or_missing_bounds"] = (
                rejections.get("time_window_no_overlap_or_unparseable_or_missing_bounds", 0) + 1
            )
            continue
        if "device" in match_levels and ex.device:
            if allowed_devices is None:
                rejections["device_set_but_no_allowed_devices_supplied"] = (
                    rejections.get("device_set_but_no_allowed_devices_supplied", 0) + 1
                )
                continue
            if ex.device not in allowed_devices:
                rejections["device_not_in_allowed_set"] = rejections.get("device_not_in_allowed_set", 0) + 1
                continue
        if "product" in match_levels and ex.product:
            if allowed_products is None:
                rejections["product_set_but_no_allowed_products_supplied"] = (
                    rejections.get("product_set_but_no_allowed_products_supplied", 0) + 1
                )
                continue
            if ex.product not in allowed_products:
                rejections["product_not_in_allowed_set"] = rejections.get("product_not_in_allowed_set", 0) + 1
                continue
        se = ex.lift_se if ex.lift_se and ex.lift_se > 0 else None
        se_eff = float(se) if se is not None else 1.0
        se_eff = max(se_eff, 1e-4)
        inv_se = 1.0 / se_eff
        q = experiment_quality_score(ex) if apply_quality else 1.0
        weight = float(inv_se * q)
        matched.append(MatchedExperiment(obs=ex, weight=weight, quality_score=q))

    applied_levels = ["channel", *[lv for lv in match_levels if lv != "channel"]]
    trace: dict[str, Any] = {
        "matching_version": "mmm_calibration_matching_v2",
        "run_environment": getattr(run_environment, "value", run_environment) if run_environment is not None else None,
        "requested_match_levels": list(match_levels),
        "applied_match_levels": applied_levels,
        "semantics": {
            "channel": "always_enforced_experiment_channel_must_be_in_panel_channel_columns",
            "geo": (
                "when_geo_in_levels_only_rows_with_non_null_geo_id_are_checked_against_available_geos"
                ";null_geo_id_rows_are_not_excluded_by_geo_rule"
            ),
            "time_window": (
                "when_time_window_in_levels_rows_without_dates_pass_rows_with_dates_require_overlap"
                ";missing_panel_bounds_causes_dated_rows_to_fail"
            ),
            "device": (
                "when_device_in_levels_rows_without_device_pass_rows_with_device_require_allowed_devices"
            ),
            "product": (
                "when_product_in_levels_rows_without_product_pass_rows_with_product_require_allowed_products"
            ),
        },
        "n_input_experiments": len(experiments),
        "n_matched": len(matched),
        "rejections": rejections,
        "warnings": warnings,
        "panel_week_bounds": {
            "min": None if panel_week_min is None else str(panel_week_min),
            "max": None if panel_week_max is None else str(panel_week_max),
        },
        "available_geos_supplied": available_geos is not None,
        "allowed_devices_supplied": allowed_devices is not None,
        "allowed_products_supplied": allowed_products is not None,
    }
    return MatchExperimentsResult(matched=matched, trace=trace)


def match_experiments(
    experiments: list[ExperimentObservation],
    *,
    available_geos: set[str] | None,
    available_channels: set[str],
    match_levels: list[str],
    apply_quality: bool = True,
    panel_week_min: pd.Timestamp | None = None,
    panel_week_max: pd.Timestamp | None = None,
    allowed_devices: set[str] | None = None,
    allowed_products: set[str] | None = None,
    run_environment: RunEnvironment | None = None,
) -> list[MatchedExperiment]:
    return match_experiments_with_trace(
        experiments,
        available_geos=available_geos,
        available_channels=available_channels,
        match_levels=match_levels,
        apply_quality=apply_quality,
        panel_week_min=panel_week_min,
        panel_week_max=panel_week_max,
        allowed_devices=allowed_devices,
        allowed_products=allowed_products,
        run_environment=run_environment,
    ).matched


def compute_experiment_weight_audit(matched: list[MatchedExperiment]) -> dict[str, Any]:
    """
    Diagnostics for inverse-SE × quality weights: normalized shares and dominance warnings.

    Does not mutate ``matched``; use before aggregating calibration losses in research notebooks.
    """
    if not matched:
        return {"n": 0, "normalized_weights": [], "raw_weights": [], "dominance_warnings": []}
    w = np.array([m.weight for m in matched], dtype=float)
    s = float(w.sum())
    if s <= 0:
        return {
            "n": len(matched),
            "normalized_weights": [],
            "raw_weights": w.tolist(),
            "dominance_warnings": ["zero_total_weight"],
        }
    p = (w / s).tolist()
    warnings: list[str] = []
    mx = float(w.max() / s)
    if mx > 0.65:
        warnings.append("single_experiment_inverse_se_share_exceeds_0_65")
    if mx > 0.5:
        warnings.append("single_experiment_inverse_se_share_exceeds_0_50")
    return {
        "n": len(matched),
        "normalized_weights": p,
        "raw_weights": w.tolist(),
        "max_inverse_se_share": mx,
        "dominance_warnings": warnings,
    }
