"""E10: validate response curves before optimization."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from mmm.decomposition.curves import ResponseCurve


@dataclass
class ResponseDiagnostics:
    monotone: bool
    max_gradient_jump: float
    safe_for_optimization: bool
    flags: list[str] = field(default_factory=list)

    def to_json(self) -> dict:
        return {
            "monotone": self.monotone,
            "max_gradient_jump": self.max_gradient_jump,
            "safe_for_optimization": self.safe_for_optimization,
            "flags": self.flags,
        }


def diagnose_response_curve(curve: ResponseCurve) -> ResponseDiagnostics:
    r = curve.response
    g = curve.marginal_roi
    mono = bool(np.all(np.diff(r) >= -1e-7)) if len(r) > 2 else True
    jump = float(np.max(np.abs(np.diff(g)))) if len(g) > 1 else 0.0
    flags: list[str] = []
    if not mono:
        flags.append("non_monotone_response")
    if jump > 50 * (np.nanmax(np.abs(g)) + 1e-9):
        flags.append("sharp_mroi_cliff")
    safe = mono and not flags
    return ResponseDiagnostics(
        monotone=mono,
        max_gradient_jump=jump,
        safe_for_optimization=safe,
        flags=flags,
    )
