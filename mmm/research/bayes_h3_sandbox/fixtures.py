"""Deterministic toy data for Bayes-H3 research sandbox MVP tests."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from mmm.config.schema import BayesianBackend, Framework, MMMConfig, ModelForm, PoolingMode, RunEnvironment
from mmm.data.schema import PanelSchema

TOY_GEOS = ("dma_a", "dma_b")
TOY_CHANNELS = ("tv", "search")
TOY_WEEKS = 12
TOY_SEED = 42

TOY_GEO_HIERARCHY: dict[str, dict[str, str]] = {
    "dma_a": {"state": "state_1", "region": "region_west"},
    "dma_b": {"state": "state_1", "region": "region_west"},
}

TOY_CALIBRATION_SIGNAL_STUB: list[dict[str, Any]] = [
    {
        "signal_id": "cal-sandbox-stub-001",
        "scope_type": "dma",
        "scope_id": "dma_a",
        "channel": "tv",
        "reserved_for_h2_alignment": True,
        "note": "MVP placeholder — not used in likelihood",
    }
]


def toy_sandbox_panel(*, seed: int = TOY_SEED) -> pd.DataFrame:
    """Small geo-week panel with two media channels (deterministic)."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for geo in TOY_GEOS:
        for week in range(TOY_WEEKS):
            tv = float(rng.uniform(2.0, 8.0))
            search = float(rng.uniform(1.0, 6.0))
            # Mild geo-specific effects; outcome on level scale (semi-log fit uses log internally).
            y = 50.0 + (3.0 if geo == "dma_a" else 1.5) * tv + 2.0 * search + float(rng.normal(0, 0.5))
            rows.append(
                {
                    "geo_id": geo,
                    "week": week,
                    "y": max(y, 1.0),
                    "tv": tv,
                    "search": search,
                }
            )
    return pd.DataFrame(rows)


def toy_sandbox_schema() -> PanelSchema:
    return PanelSchema("geo_id", "week", "y", TOY_CHANNELS)


def toy_sandbox_config(*, fast_mcmc: bool = True) -> MMMConfig:
    """Research-only config for sandbox MVP (not production)."""
    bayesian: dict[str, Any] = {
        "backend": BayesianBackend.PYMC,
        "nuts_seed": TOY_SEED,
        "prior_predictive_draws": 0,
        "posterior_predictive_draws": 0,
    }
    if fast_mcmc:
        bayesian.update({"draws": 200, "tune": 200, "chains": 2, "target_accept": 0.92})
    return MMMConfig(
        framework=Framework.BAYESIAN,
        run_environment=RunEnvironment.RESEARCH,
        model_form=ModelForm.SEMI_LOG,
        pooling=PoolingMode.PARTIAL,
        data={
            "path": None,
            "geo_column": "geo_id",
            "week_column": "week",
            "target_column": "y",
            "channel_columns": list(TOY_CHANNELS),
            "control_columns": [],
        },
        bayesian=bayesian,
    )


def toy_sandbox_bundle(*, fast_mcmc: bool = True) -> tuple[MMMConfig, PanelSchema, pd.DataFrame]:
    return toy_sandbox_config(fast_mcmc=fast_mcmc), toy_sandbox_schema(), toy_sandbox_panel()
