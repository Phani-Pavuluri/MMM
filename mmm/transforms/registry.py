"""Transform registry + stack builder."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from mmm.config.schema import TransformConfig
from mmm.transforms.adstock.geometric import GeometricAdstock
from mmm.transforms.adstock.weibull import WeibullAdstock
from mmm.transforms.saturation.hill import HillSaturation
from mmm.transforms.saturation.log_sat import LogSaturation
from mmm.transforms.saturation.logistic import LogisticSaturation

AdstockFactory = Callable[[], object]
SaturationFactory = Callable[[], object]


@dataclass
class TransformRegistry:
    adstocks: dict[str, AdstockFactory]
    saturations: dict[str, SaturationFactory]

    @staticmethod
    def default() -> TransformRegistry:
        return TransformRegistry(
            adstocks={
                "geometric": lambda: GeometricAdstock(0.5),
                "weibull": lambda: WeibullAdstock(),
            },
            saturations={
                "hill": lambda: HillSaturation(),
                "log": lambda: LogSaturation(),
                "logistic": lambda: LogisticSaturation(),
            },
        )


def build_default_stack(cfg: TransformConfig) -> tuple[object, object]:
    reg = TransformRegistry.default()
    if cfg.adstock not in reg.adstocks:
        raise ValueError(f"Unknown adstock {cfg.adstock}")
    if cfg.saturation not in reg.saturations:
        raise ValueError(f"Unknown saturation {cfg.saturation}")
    ad = reg.adstocks[cfg.adstock]()
    sat = reg.saturations[cfg.saturation]()
    # apply config params if constructors supported — simple mapping
    if cfg.adstock == "geometric" and cfg.adstock_params:
        decay = float(cfg.adstock_params.get("decay", 0.5))
        ad = GeometricAdstock(decay)
    if cfg.adstock == "weibull" and cfg.adstock_params:
        ad = WeibullAdstock(
            shape=float(cfg.adstock_params.get("shape", 1.5)),
            scale=float(cfg.adstock_params.get("scale", 2.0)),
            max_lag=int(cfg.adstock_params.get("max_lag", 12)),
        )
    if cfg.saturation == "hill" and cfg.saturation_params:
        ad_params = cfg.saturation_params  # noqa: SIM114
        sat = HillSaturation(
            half_max=float(ad_params.get("half_max", 1.0)),
            slope=float(ad_params.get("slope", 2.0)),
        )
    elif cfg.saturation == "log" and cfg.saturation_params:
        sat = LogSaturation(scale=float(cfg.saturation_params.get("scale", 1.0)))
    elif cfg.saturation == "logistic" and cfg.saturation_params:
        sat = LogisticSaturation(
            midpoint=float(cfg.saturation_params.get("midpoint", 1.0)),
            growth=float(cfg.saturation_params.get("growth", 1.0)),
        )
    return ad, sat


def apply_adstock_saturation_series(
    spend: np.ndarray, adstock: object, saturation: object
) -> np.ndarray:
    """Apply adstock along time then saturation pointwise."""
    x = np.asarray(spend, dtype=float).ravel()
    x_ad = adstock.fit(x).transform(x)  # type: ignore[attr-defined]
    return saturation.fit(x_ad).transform(x_ad)  # type: ignore[attr-defined]
