"""Synthetic geo-week panels for tests and benchmarks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from mmm.data.schema import PanelSchema
from mmm.transforms.adstock.geometric import GeometricAdstock
from mmm.transforms.registry import apply_adstock_saturation_series
from mmm.transforms.saturation.hill import HillSaturation


@dataclass
class SyntheticGeoPanelSpec:
    n_geos: int = 4
    n_weeks: int = 80
    channels: tuple[str, ...] = ("search", "social", "tv")
    decay: float = 0.6
    betas: tuple[float, ...] = (0.4, 0.35, 0.25)
    noise: float = 0.05


def generate_geo_panel(spec: SyntheticGeoPanelSpec, seed: int = 0) -> tuple[pd.DataFrame, PanelSchema]:
    rng = np.random.default_rng(seed)
    weeks = pd.date_range("2022-01-03", periods=spec.n_weeks, freq="W-MON")
    ad = GeometricAdstock(spec.decay)
    sat = HillSaturation(1.0, 2.0)
    rows: list[dict] = []
    for g in range(spec.n_geos):
        base = 100 + g * 5
        spend_mat = rng.lognormal(mean=0.0, sigma=0.35, size=(spec.n_weeks, len(spec.channels)))
        x_media = np.zeros_like(spend_mat)
        for j in range(len(spec.channels)):
            x_media[:, j] = apply_adstock_saturation_series(spend_mat[:, j], ad, sat)
        for t in range(spec.n_weeks):
            log_y = np.log(base) + float(x_media[t] @ np.array(spec.betas)) + rng.normal(0, spec.noise)
            row = {
                "geo_id": f"G{g}",
                "week_start_date": weeks[t],
                "revenue": float(np.exp(log_y)),
            }
            for j, ch in enumerate(spec.channels):
                row[ch] = float(spend_mat[t, j])
            rows.append(row)
    df = pd.DataFrame(rows)
    schema = PanelSchema(
        geo_column="geo_id",
        week_column="week_start_date",
        target_column="revenue",
        channel_columns=spec.channels,
        control_columns=(),
    )
    return df, schema


def known_dgp_parameters(spec: SyntheticGeoPanelSpec) -> dict[str, float | tuple[str, ...]]:
    """E15: expose ground-truth DGP knobs for recovery / regression tests."""
    return {
        "decay": spec.decay,
        "betas": spec.betas,
        "channels": spec.channels,
        "noise_std": spec.noise,
    }


def generate_experiment_csv(path: str, channels: tuple[str, ...]) -> None:
    import csv

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["experiment_id", "geo_id", "channel", "lift", "lift_se"],
        )
        w.writeheader()
        for c in channels:
            w.writerow(
                {
                    "experiment_id": "E1",
                    "geo_id": "G0",
                    "channel": c,
                    "lift": 0.05,
                    "lift_se": 0.01,
                }
            )
