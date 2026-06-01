"""Deterministic Bayes-H4 recovery worlds for H3 sandbox validation (research only)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from mmm.config.schema import BayesianBackend, Framework, MMMConfig, ModelForm, PoolingMode, RunEnvironment
from mmm.data.schema import PanelSchema

WORLD_BAYES_H4_SIMPLE_POOLING = "WORLD-BAYES-H4-SIMPLE-POOLING"
WORLD_BAYES_H4_SPARSE_GEO = "WORLD-BAYES-H4-SPARSE-GEO"
WORLD_BAYES_H4_CONFLICTING_EVIDENCE = "WORLD-BAYES-H4-CONFLICTING-EVIDENCE"

H4_WORLD_IDS: tuple[str, ...] = (
    WORLD_BAYES_H4_SIMPLE_POOLING,
    WORLD_BAYES_H4_SPARSE_GEO,
    WORLD_BAYES_H4_CONFLICTING_EVIDENCE,
)

H4_SEED = 4400
H4_CHANNELS = ("tv", "search")
H4_WEEKS_FULL = 10

SAMPLER_FAST: dict[str, Any] = {"draws": 200, "tune": 200, "chains": 2, "target_accept": 0.92}
SAMPLER_EXTENDED: dict[str, Any] = {"draws": 600, "tune": 600, "chains": 4, "target_accept": 0.95}


@dataclass(frozen=True)
class RecoveryWorldSpec:
    """Known-truth synthetic world for Bayes-H4 recovery checks."""

    world_id: str
    geo_order: tuple[str, ...]
    channels: tuple[str, ...]
    weeks_by_geo: dict[str, int]
    noise_sigma: float
    true_mu_c: dict[str, float]
    true_tau_c: dict[str, float]
    true_beta_gc: dict[str, dict[str, float]]
    true_alpha_g: dict[str, float]
    geo_hierarchy: dict[str, dict[str, str]]
    calibration_signals: tuple[dict[str, Any], ...] = ()
    sparse_geos: tuple[str, ...] = ()
    expected_diagnostic_behavior: dict[str, Any] = field(default_factory=dict)
    mcmc_seed: int = H4_SEED

    @property
    def known_truth(self) -> dict[str, Any]:
        return {
            "true_mu_c": dict(self.true_mu_c),
            "true_tau_c": dict(self.true_tau_c),
            "true_beta_gc": {g: dict(ch) for g, ch in self.true_beta_gc.items()},
            "true_alpha_g": dict(self.true_alpha_g),
            "noise_sigma": self.noise_sigma,
        }


def _standardized_media(rng: np.random.Generator, n: int) -> np.ndarray:
    x = rng.uniform(1.0, 5.0, size=n)
    return (x - x.mean()) / (x.std() + 1e-6)


def materialize_recovery_panel(spec: RecoveryWorldSpec, *, panel_seed: int | None = None) -> pd.DataFrame:
    """Build observed panel from known truth (deterministic given spec + panel_seed)."""
    rng = np.random.default_rng(panel_seed if panel_seed is not None else spec.mcmc_seed)
    rows: list[dict[str, Any]] = []
    for geo in spec.geo_order:
        n_weeks = spec.weeks_by_geo[geo]
        xs = {ch: _standardized_media(rng, n_weeks) for ch in spec.channels}
        alpha = spec.true_alpha_g[geo]
        betas = np.array([spec.true_beta_gc[geo][c] for c in spec.channels])
        x_mat = np.column_stack([xs[ch] for ch in spec.channels])
        log_y = alpha + x_mat @ betas + rng.normal(0.0, spec.noise_sigma, size=n_weeks)
        y = np.exp(log_y)
        for t in range(n_weeks):
            row: dict[str, Any] = {"geo_id": geo, "week": t, "y": float(max(y[t], 1e-3))}
            for ch in spec.channels:
                # Panel QA requires non-negative spend/exposure columns.
                row[ch] = float(abs(xs[ch][t]) + 0.5)
            rows.append(row)
    return pd.DataFrame(rows)


def recovery_world_config(
    spec: RecoveryWorldSpec,
    *,
    fast_mcmc: bool = True,
    sampler: dict[str, Any] | None = None,
    nuts_seed: int | None = None,
) -> MMMConfig:
    bayesian: dict[str, Any] = {
        "backend": BayesianBackend.PYMC,
        "nuts_seed": int(nuts_seed if nuts_seed is not None else spec.mcmc_seed),
        "prior_predictive_draws": 0,
        "posterior_predictive_draws": 0,
    }
    if sampler is not None:
        bayesian.update(sampler)
    elif fast_mcmc:
        bayesian.update(SAMPLER_FAST)
    else:
        bayesian.update(SAMPLER_EXTENDED)
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
            "channel_columns": list(spec.channels),
            "control_columns": [],
        },
        bayesian=bayesian,
    )


def recovery_world_schema(spec: RecoveryWorldSpec) -> PanelSchema:
    return PanelSchema("geo_id", "week", "y", spec.channels)


def materialize_recovery_bundle(
    spec: RecoveryWorldSpec,
    *,
    fast_mcmc: bool = True,
    sampler: dict[str, Any] | None = None,
    nuts_seed: int | None = None,
    panel_seed: int | None = None,
) -> tuple[MMMConfig, PanelSchema, pd.DataFrame]:
    return (
        recovery_world_config(spec, fast_mcmc=fast_mcmc, sampler=sampler, nuts_seed=nuts_seed),
        recovery_world_schema(spec),
        materialize_recovery_panel(spec, panel_seed=panel_seed),
    )


def _simple_pooling_world() -> RecoveryWorldSpec:
    geos = ("dma_a", "dma_b", "dma_c")
    mu = {"tv": 0.35, "search": 0.22}
    tau = {"tv": 0.08, "search": 0.06}
    beta = {
        "dma_a": {"tv": 0.38, "search": 0.20},
        "dma_b": {"tv": 0.33, "search": 0.24},
        "dma_c": {"tv": 0.36, "search": 0.21},
    }
    return RecoveryWorldSpec(
        world_id=WORLD_BAYES_H4_SIMPLE_POOLING,
        geo_order=geos,
        channels=H4_CHANNELS,
        weeks_by_geo={g: H4_WEEKS_FULL for g in geos},
        noise_sigma=0.12,
        true_mu_c=mu,
        true_tau_c=tau,
        true_beta_gc=beta,
        true_alpha_g={g: 3.0 for g in geos},
        geo_hierarchy={g: {"state": "s1", "region": "r1"} for g in geos},
        calibration_signals=(),
        expected_diagnostic_behavior={
            "pooling_strength": "low",
            "recovery_expectation": "beta_gc_near_mu_c",
        },
    )


def _sparse_geo_world() -> RecoveryWorldSpec:
    geos = ("dma_dense_a", "dma_dense_b", "dma_sparse")
    mu = {"tv": 0.30, "search": 0.18}
    tau = {"tv": 0.25, "search": 0.20}
    beta = {
        "dma_dense_a": {"tv": 0.32, "search": 0.17},
        "dma_dense_b": {"tv": 0.29, "search": 0.19},
        "dma_sparse": {"tv": 0.85, "search": 0.05},
    }
    return RecoveryWorldSpec(
        world_id=WORLD_BAYES_H4_SPARSE_GEO,
        geo_order=geos,
        channels=H4_CHANNELS,
        weeks_by_geo={"dma_dense_a": H4_WEEKS_FULL, "dma_dense_b": H4_WEEKS_FULL, "dma_sparse": 3},
        noise_sigma=0.15,
        true_mu_c=mu,
        true_tau_c=tau,
        true_beta_gc=beta,
        true_alpha_g={g: 3.0 for g in geos},
        geo_hierarchy={
            "dma_dense_a": {"state": "s1", "region": "r1"},
            "dma_dense_b": {"state": "s1", "region": "r1"},
            "dma_sparse": {"state": "s2", "region": "r1"},
        },
        sparse_geos=("dma_sparse",),
        expected_diagnostic_behavior={
            "shrinkage_expected": True,
            "sparse_geo": "dma_sparse",
        },
    )


def _conflicting_evidence_world() -> RecoveryWorldSpec:
    geos = ("dma_a", "dma_b")
    mu = {"tv": 0.40, "search": 0.20}
    tau = {"tv": 0.10, "search": 0.08}
    beta = {
        "dma_a": {"tv": 0.55, "search": 0.18},
        "dma_b": {"tv": 0.38, "search": 0.22},
    }
    signals = (
        {
            "signal_id": "h4-conflict-tv-dma-a",
            "scope_type": "dma",
            "scope_id": "dma_a",
            "channel": "tv",
            "claimed_direction": "negative",
            "claimed_lift": -0.30,
            "note": "Conflicts with positive generative beta_gc for tv on dma_a",
        },
    )
    return RecoveryWorldSpec(
        world_id=WORLD_BAYES_H4_CONFLICTING_EVIDENCE,
        geo_order=geos,
        channels=H4_CHANNELS,
        weeks_by_geo={g: H4_WEEKS_FULL for g in geos},
        noise_sigma=0.10,
        true_mu_c=mu,
        true_tau_c=tau,
        true_beta_gc=beta,
        true_alpha_g={g: 3.0 for g in geos},
        geo_hierarchy={g: {"state": "s1", "region": "r1"} for g in geos},
        calibration_signals=signals,
        expected_diagnostic_behavior={
            "conflict_warning_expected": True,
        },
    )


_WORLDS: dict[str, RecoveryWorldSpec] = {
    WORLD_BAYES_H4_SIMPLE_POOLING: _simple_pooling_world(),
    WORLD_BAYES_H4_SPARSE_GEO: _sparse_geo_world(),
    WORLD_BAYES_H4_CONFLICTING_EVIDENCE: _conflicting_evidence_world(),
}


def get_recovery_world(world_id: str) -> RecoveryWorldSpec:
    if world_id not in _WORLDS:
        raise KeyError(f"unknown Bayes-H4 recovery world: {world_id!r}; known: {sorted(_WORLDS)}")
    return _WORLDS[world_id]


def list_recovery_worlds() -> tuple[RecoveryWorldSpec, ...]:
    return tuple(_WORLDS[w] for w in H4_WORLD_IDS)
