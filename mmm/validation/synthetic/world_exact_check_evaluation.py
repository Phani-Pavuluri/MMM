"""World-parameterized evaluation of existing exact transform checks.

Calls the same shared registry predicate implementations as the
zero-argument CI/runtime paths, passing recorded world parameters.
Diagnostic evidence only: emitted rows never enter certification
levels, readiness, thresholds, or gates.
"""

from __future__ import annotations

import math
from typing import Any

from mmm.governance.synthetic_certification import (
    _check_geometric_adstock_carryover,
    _check_geometric_adstock_design_matrix,
    _check_hill_monotone_design_matrix,
    _check_hill_saturation_analytic,
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


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _channel_params(truth: Any, check_id: str) -> tuple[list[str], dict[str, dict[str, float]], str | None]:
    """Resolve per-channel parameters or a fail-closed error reason."""
    if not isinstance(truth, dict):
        return [], {}, "world truth is not a mapping"
    transform = truth.get("transform_truth")
    if not isinstance(transform, dict):
        return [], {}, "missing transform_truth mapping"
    fields = _PARAM_FIELDS[check_id]
    maps: dict[str, dict[str, float]] = {}
    for field in fields:
        raw = transform.get(field)
        if not isinstance(raw, dict) or not raw:
            return [], {}, f"missing or empty {field}"
        values: dict[str, float] = {}
        for channel, number in raw.items():
            if not isinstance(channel, str) or not channel or not _finite_number(number):
                return [], {}, f"invalid entry in {field} for channel {channel!r}"
            values[channel] = float(number)
        maps[field] = values
    channels = sorted(maps[fields[0]])
    for field in fields[1:]:
        if sorted(maps[field]) != channels:
            return [], {}, f"channel coverage mismatch in {field}"
    params = {channel: {field: maps[field][channel] for field in fields} for channel in channels}
    return channels, params, None


def _evaluate_one(check_id: str, world_id: str, truth: Any) -> dict[str, Any]:
    fn = _CHECK_FNS[check_id]
    channels, params, error = _channel_params(truth, check_id)
    evidence_refs: dict[str, Any] = {"transform_truth_fields": list(_PARAM_FIELDS[check_id])}
    if error is not None:
        return {
            "check_id": check_id,
            "world_id": world_id,
            "status": "fail",
            "expected": "valid recorded world parameters",
            "actual": error,
            "tolerance": None,
            "message": error,
            "skip_reason": None,
            "evidence_refs": evidence_refs,
            "channels": [],
        }
    channel_results = []
    for channel in channels:
        try:
            measured = fn(**_CALL_KWARGS[check_id](params[channel]))
            channel_results.append({
                "channel": channel,
                "expected": "predicate holds within inherited tolerance",
                "actual": measured,
                "status": "pass",
            })
        except Exception as exc:  # fail closed with the exact reason
            channel_results.append({
                "channel": channel,
                "expected": "predicate holds within inherited tolerance",
                "actual": f"{type(exc).__name__}: {exc}",
                "status": "fail",
            })
    failed = [c for c in channel_results if c["status"] == "fail"]
    binding = failed[0] if failed else channel_results[0]
    evidence_refs["world_values"] = params
    return {
        "check_id": check_id,
        "world_id": world_id,
        "status": "fail" if failed else "pass",
        "expected": binding["expected"],
        "actual": binding["actual"],
        "tolerance": "inherited per-check tolerance (1e-12 analytic, -1e-9 monotone floor)",
        "message": f"binding channel {binding['channel']}: {binding['status']}",
        "skip_reason": None,
        "evidence_refs": evidence_refs,
        "channels": channel_results,
    }


def evaluate_world_exact_checks(world_id: str, truth: Any) -> list[dict[str, Any]]:
    """Evaluate covered checks against recorded world parameters.

    Deterministic: fixed check order, lexical channel order. Never
    raises for malformed input — malformed inputs yield fail rows.
    """
    return [_evaluate_one(check_id, world_id, truth) for check_id in WORLD_CHECK_ORDER]
