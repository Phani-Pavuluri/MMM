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


def _safe_evidence(value: Any) -> Any:
    """JSON-safe evidence: non-finite/outside-standard numbers become explanatory strings."""
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value if abs(value) < 2**1024 else f"non-serializable integer with {value.bit_length()} bits"
    if isinstance(value, float):
        return value if math.isfinite(value) else f"non-finite float: {value!r}"
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return {str(k): _safe_evidence(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_evidence(v) for v in value]
    return repr(value)


def _declared_channels(truth: Any) -> tuple[list[str] | None, str | None]:
    """Declared world channel inventory, or a fail-closed reason."""
    media = truth.get("media_truth") if isinstance(truth, dict) else None
    if not isinstance(media, dict):
        return None, "missing media_truth mapping for declared channel inventory"
    channels = media.get("channels")
    if not isinstance(channels, list) or not channels:
        return None, "missing or empty declared channel inventory"
    if any(not isinstance(c, str) or not c for c in channels):
        return None, "invalid declared channel identifier in inventory"
    if len(set(channels)) != len(channels):
        return None, "duplicate declared channel identifier in inventory"
    return list(channels), None


def _evaluate_channel(
    check_id: str, channel: Any, transform: dict[str, Any], declared: list[str]
) -> dict[str, Any]:
    if not isinstance(channel, str) or not channel:
        return {
            "channel": channel if isinstance(channel, str) else repr(channel),
            "expected": "valid channel identifier",
            "actual": f"malformed channel identifier: {channel!r}",
            "tolerance": None,
            "status": "fail",
        }
    if channel not in declared:
        return {
            "channel": channel,
            "expected": "channel declared in world inventory",
            "actual": f"unknown extra parameter channel {channel!r} not in declared inventory",
            "tolerance": None,
            "status": "fail",
        }
    try:
        params: dict[str, float] = {}
        for field in _PARAM_FIELDS[check_id]:
            raw = transform.get(field)
            if not isinstance(raw, dict) or channel not in raw:
                return {
                    "channel": channel,
                    "expected": "required recorded parameter present",
                    "actual": f"channel coverage mismatch: missing {field} for channel {channel!r}",
                    "tolerance": None,
                    "status": "fail",
                }
            number = raw[channel]
            if not _finite_number(number):
                return {
                    "channel": channel,
                    "expected": "finite recorded parameter",
                    "actual": f"non-finite {field} for channel {channel!r}",
                    "tolerance": None,
                    "status": "fail",
                }
            params[field] = float(number)
        kwargs = _CALL_KWARGS[check_id](params)
        _CHECK_FNS[check_id](**kwargs)
        measured = _MEASURE_FNS[check_id](**kwargs)
        expected_key, actual_key, tolerance_key = _ROW_SCALARS[check_id]
        if check_id == "hill_saturation_monotone_design_matrix":
            expected: Any = -1e-9
        else:
            expected = measured[expected_key]
        return {
            "channel": channel,
            "expected": expected,
            "actual": measured[actual_key],
            "tolerance": measured[tolerance_key] if tolerance_key is not None else None,
            "status": "pass",
            "measured": measured,
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
        declared, inventory_error = _declared_channels(truth)
        if inventory_error is not None or declared is None:
            raise ValueError(inventory_error or "invalid declared channel inventory")
        universe: list[Any] = list(declared)
        for field in _PARAM_FIELDS[check_id]:
            raw = transform.get(field)
            if raw is None:
                raise ValueError(f"missing {field} parameter map")
            if not isinstance(raw, dict):
                raise ValueError(f"malformed {field} parameter map")
            for key in raw:
                if key not in universe:
                    universe.append(key)
        ordered = [c for c in declared if c in universe] + sorted(
            [c for c in universe if c not in declared], key=repr
        )
        channel_results = [_evaluate_channel(check_id, channel, transform, declared) for channel in ordered]
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
        "message": f"binding channel {binding['channel']}: {binding['status']} — {binding['actual']}",
        "skip_reason": None,
        "evidence_refs": {
            "transform_truth_fields": list(_PARAM_FIELDS[check_id]),
            "world_values": _safe_evidence({field: transform[field] for field in _PARAM_FIELDS[check_id]
                                            if isinstance(transform.get(field), dict)}),
            "measured": {c["channel"]: c.get("measured") for c in channel_results if "measured" in c},
            "declared_channels": declared,
        },
        "channels": [{k: v for k, v in c.items() if k != "measured"} for c in channel_results],
    }


def evaluate_world_exact_checks(world_id: str, truth: Any) -> list[dict[str, Any]]:
    """Evaluate covered checks against recorded world parameters.

    Deterministic: fixed check order, declared-then-lexical channel
    order. Never raises — every failure mode yields fail rows.
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
