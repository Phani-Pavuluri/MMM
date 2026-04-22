"""Extra response-curve checks before optimization (roadmap §7)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mmm.decomposition.curves import ResponseCurve


@dataclass
class CurveStressReport:
    max_second_difference: float
    fd_relative_change: float
    numerically_unstable_for_sqp: bool

    def to_json(self) -> dict:
        return {
            "max_second_difference": self.max_second_difference,
            "fd_relative_change": self.fd_relative_change,
            "numerically_unstable_for_sqp": self.numerically_unstable_for_sqp,
        }


def stress_test_curve(curve: ResponseCurve, *, fd_eps: float = 1e-3) -> CurveStressReport:
    r = np.asarray(curve.response, dtype=float)
    if len(r) < 3:
        return CurveStressReport(0.0, 0.0, False)
    d2 = np.abs(np.diff(r, n=2))
    max_d2 = float(np.max(d2)) if len(d2) else 0.0
    g = np.asarray(curve.marginal_roi, dtype=float)
    unstable = False
    fd_rel = 0.0
    if len(g) > 2:
        g_pert = g * (1.0 + fd_eps)
        fd_rel = float(np.max(np.abs(g_pert - g) / (np.abs(g) + 1e-9)))
        if fd_rel > 5.0 or max_d2 > 1e3 * (np.nanstd(r) + 1e-9):
            unstable = True
    return CurveStressReport(
        max_second_difference=max_d2,
        fd_relative_change=fd_rel,
        numerically_unstable_for_sqp=unstable,
    )
