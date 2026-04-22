"""E6: falsification checks for spurious attribution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from mmm.config.extensions import FalsificationConfig
from mmm.data.schema import PanelSchema
from mmm.models.ridge_bo.ridge import fit_ridge


@dataclass
class FalsificationReport:
    placebo_channel_coef_mean: float
    flags: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {"placebo_channel_coef_mean": self.placebo_channel_coef_mean, "flags": self.flags}


class FalsificationEngine:
    def __init__(self, schema: PanelSchema, cfg: FalsificationConfig) -> None:
        self.schema = schema
        self.cfg = cfg

    def run(self, X_media: np.ndarray, y_log: np.ndarray, rng: np.random.Generator) -> FalsificationReport:
        flags: list[str] = []
        if not self.cfg.enabled or self.cfg.placebo_draws <= 0:
            return FalsificationReport(0.0, flags)
        coefs = []
        n = X_media.shape[0]
        for _ in range(self.cfg.placebo_draws):
            noise = rng.normal(0, 1.0, size=(n, 1))
            Xn = np.hstack([X_media, noise])
            c, _ = fit_ridge(Xn, y_log, alpha=5.0)
            coefs.append(float(c[-1]))
        mean_noise = float(np.mean(np.abs(coefs)))
        cref, _ = fit_ridge(X_media, y_log, alpha=5.0)
        scale = float(np.mean(np.abs(cref)) + 1e-9)
        if mean_noise > 0.05 * scale:
            flags.append("spurious_attribution_risk: placebo channel absorbs signal")
        return FalsificationReport(placebo_channel_coef_mean=mean_noise, flags=flags)
