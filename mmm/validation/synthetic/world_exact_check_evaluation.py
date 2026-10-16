"""World-parameterized evaluation of existing exact transform checks.

Calls the same shared registry predicate implementations as the
zero-argument CI/runtime paths (which return None), and the shared
measurement helpers for diagnostic numbers. Diagnostic evidence only:
emitted rows never enter certification levels, readiness, thresholds,
or gates.
"""

from __future__ import annotations

import math
from typing import Any

from mmm.governance.synthetic_certification import (
    _check_geometric_adstock_carryover,
    _check_geometric_adstock_design_matrix,
    _check_hill_monotone_design_matrix,
    _check_hill_saturation_analytic,
    _measure_carryover,
    _measure_design_carryover,
    _measure_hill_analytic,
    _measure_hill_monotone,
)

WORLD_CHECK_ORDER: tuple[str, ...] = (
    "geometric_adstock_carryover",
    "hill_saturation_analytic",
    "geometric_adstock_design_matrix",
    "hill_saturation_monotone_design_matrix",
)

_CHECK_FNS = {
    "geometric_adstock_carryover": _check_geometric_adstock_carryover,
    "hill_saturation_analytic": _check_hill_saturation_analytic,
    "geometric_adstock_design_matrix": _check_geometric_adstock_design_matrix,
    "hill_saturation_monotone_design_matrix": _check_hill_monotone_design_matrix,
}

_MEASURE_FNS = {
    "geometric_adstock_carryover": _measure_carryover,
    "hill_saturation_analytic": _measure_hill_analytic,
    "geometric_adstock_design_matrix": _measure_design_carryover,
    "hill_saturation_monotone_design_matrix": _measure_hill_monotone,
}

_PARAM_FIELDS = {
    "geometric_adstock_carryover": ("adstock_decay_by_channel",),
    "hill_saturation_analytic": ("hill_half_max_by_channel", "hill_slope_by_channel"),
    "geometric_adstock_design_matrix": ("adstock_decay_by_channel",),
    "hill_saturation_monotone_design_matrix": ("hill_half_max_by_channel", "hill_slope_by_channel"),
}

_CALL_KWARGS = {
    "geometric_adstock_carryover": lambda p: {"decay": p["adstock_decay_by_channel"]},
    "hill_saturation_analytic": lambda p: {
        "half_max": p["hill_half_max_by_channel"],
        "slope": p["hill_slope_by_channel"],
    },
    "geometric_adstock_design_matrix": lambda p: {"decay": p["adstock_decay_by_channel"]},
    "hill_saturation_monotone_design_matrix": lambda p: {
        "hill_half": p["hill_half_max_by_channel"],
        "hill_slope": p["hill_slope_by_channel"],
    },
}

# Row-level scalar selectors: (expected_key, actual_key, tolerance_key).
# Ordering predicates report the compared pair; tolerance is None where
# the predicate is an ordering rather than a closed-form equality.
_ROW_SCALARS = {
    "geometric_adstock_carryover": ("expected", "week1_carryover", "tolerance"),
    "hill_saturation_analytic": ("expected", "measured", "tolerance"),
    "geometric_adstock_design_matrix": ("zeroed_response", "impulse_response", None),
    "hill_saturation_monotone_design_matrix": ("floor", "min_diff", None),
}


def _finite_number(value: Any) -> bool:
    """Finite check that never converts: oversized ints fail closed."""
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return abs(value) < 2**1024
    if isinstance(value, float):
        return math.isfinite(value)
    return False


def _check_channels(
    transform: dict[str, Any], fields: tuple[str, ...]
) -> tuple[list[str], str | None]:
    """Resolve the channel set from the primary field; coverage must match exactly."""
    primary = transform.get(fields[0])
    if not isinstance(primary, dict) or not primary:
        return [], f"missing or empty {fields[0]}"
    channels = sorted(c for c in primary if isinstance(c, str) and c)
    if not channels:
        return [], f"missing or empty {fields[0]}"
    for field in fields[1:]:
        raw = transform.get(field)
        if not isinstance(raw, dict) or sorted(raw) != channels:
            return [], f"channel coverage mismatch in {field}"
    return channels, None


def _resolve_channel(
    transform: dict[str, Any], fields: tuple[str, ...], channel: str
) -> tuple[dict[str, float] | None, str | None]:
    """Resolve one channel's finite values, or a fail-closed reason."""
    try:
        params: dict[str, float] = {}
        for field in fields:
            number = transform[field][channel]
            if not _finite_number(number):
                return None, f"non-finite {field} for channel {channel!r}"
            params[field] = float(number)
        return params, None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _evaluate_channel(
    check_id: str, channel: str, params: dict[str, float] | None, error: str | None
) -> dict[str, Any]:
    if error is not None or params is None:
        return {
            "channel": channel,
            "expected": "valid recorded world parameters",
            "actual": error or "unresolved parameters",
            "status": "fail",
        }
    try:
        kwargs = _CALL_KWARGS[check_id](params)
        _CHECK_FNS[check_id](**kwargs)
        measured = _MEASURE_FNS[check_id](**kwargs)
        expected_key, actual_key, tolerance_key = _ROW_SCALARS[check_id]
        tolerance = measured[tolerance_key] if tolerance_key is not None else None
        if check_id == "hill_saturation_monotone_design_matrix":
            expected: Any = -1e-9
        else:
            expected = measured[expected_key]
        return {
            "channel": channel,
            "expected": expected,
            "actual": measured[actual_key],
            "tolerance": tolerance,
            "status": "pass",
        }
    except Exception as exc:
        return {
            "channel": channel,
            "expected": "predicate holds within inherited tolerance",
            "actual": f"{type(exc).__name__}: {exc}",
            "tolerance": None,
            "status": "fail",
        }


def _evaluate_one(check_id: str, world_id: str, truth: Any) -> dict[str, Any]:
    try:
        transform = truth.get("transform_truth") if isinstance(truth, dict) else None
        if not isinstance(transform, dict) or not transform:
            raise ValueError("missing transform_truth mapping")
        channels, coverage_error = _check_channels(transform, _PARAM_FIELDS[check_id])
        if coverage_error is not None:
            raise ValueError(coverage_error)
        channel_results = [
            _evaluate_channel(check_id, channel, *_resolve_channel(transform, _PARAM_FIELDS[check_id], channel))
            for channel in channels
        ]
    except Exception as exc:
        return {
            "check_id": check_id,
            "world_id": world_id,
            "status": "fail",
            "expected": "valid recorded world parameters",
            "actual": f"{type(exc).__name__}: {exc}",
            "tolerance": None,
            "message": f"{type(exc).__name__}: {exc}",
            "skip_reason": None,
            "evidence_refs": {"transform_truth_fields": list(_PARAM_FIELDS[check_id])},
            "channels": [],
        }
    failed = [c for c in channel_results if c["status"] == "fail"]
    binding = failed[0] if failed else channel_results[0]
    return {
        "check_id": check_id,
        "world_id": world_id,
        "status": "fail" if failed else "pass",
        "expected": binding["expected"],
        "actual": binding["actual"],
        "tolerance": binding.get("tolerance"),
        "message": f"binding channel {binding['channel']}: {binding['status']}",
        "skip_reason": None,
        "evidence_refs": {
            "transform_truth_fields": list(_PARAM_FIELDS[check_id]),
            "world_values": {
                field: transform[field] for field in _PARAM_FIELDS[check_id]
                if isinstance(transform.get(field), dict)
            },
        },
        "channels": channel_results,
    }


def evaluate_world_exact_checks(world_id: str, truth: Any) -> list[dict[str, Any]]:
    """Evaluate covered checks against recorded world parameters.

    Deterministic: fixed check order, lexical channel order. Never
    raises — every failure mode yields fail rows.
    """
    try:
        return [_evaluate_one(check_id, world_id, truth) for check_id in WORLD_CHECK_ORDER]
    except Exception as exc:
        return [{
            "check_id": check_id,
            "world_id": world_id,
            "status": "fail",
            "expected": "valid recorded world parameters",
            "actual": f"{type(exc).__name__}: {exc}",
            "tolerance": None,
            "message": f"{type(exc).__name__}: {exc}",
            "skip_reason": None,
            "evidence_refs": {},
            "channels": [],
        } for check_id in WORLD_CHECK_ORDER]
