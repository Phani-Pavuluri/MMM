"""Bayes-H4c extended recovery worlds (research only — reliability map, not promotion)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from mmm.research.bayes_h3_sandbox.recovery_worlds import (
    H4_CHANNELS,
    H4_SEED,
    H4_WEEKS_FULL,
    RecoveryWorldSpec,
)

WORLD_BAYES_H4C_CLEAN_RECOVERY = "WORLD-BAYES-H4C-CLEAN-RECOVERY"
WORLD_BAYES_H4C_CORRELATED_CHANNELS = "WORLD-BAYES-H4C-CORRELATED-CHANNELS"
WORLD_BAYES_H4C_ADSTOCKED_MEDIA = "WORLD-BAYES-H4C-ADSTOCKED-MEDIA"
WORLD_BAYES_H4C_SATURATION = "WORLD-BAYES-H4C-SATURATION"
WORLD_BAYES_H4C_WEAK_SIGNAL = "WORLD-BAYES-H4C-WEAK-SIGNAL"
WORLD_BAYES_H4C_SPARSE_RECOVERY = "WORLD-BAYES-H4C-SPARSE-RECOVERY"

H4C_WORLD_IDS: tuple[str, ...] = (
    WORLD_BAYES_H4C_CLEAN_RECOVERY,
    WORLD_BAYES_H4C_CORRELATED_CHANNELS,
    WORLD_BAYES_H4C_ADSTOCKED_MEDIA,
    WORLD_BAYES_H4C_SATURATION,
    WORLD_BAYES_H4C_WEAK_SIGNAL,
    WORLD_BAYES_H4C_SPARSE_RECOVERY,
)

H4C_WEEKS_LONG = 20


def _standardized(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / (x.std() + 1e-6)


def _adstock(x: np.ndarray, decay: float) -> np.ndarray:
    out = np.zeros_like(x, dtype=float)
    carry = 0.0
    for i, v in enumerate(x):
        carry = float(v) + decay * carry
        out[i] = carry
    return _standardized(out)


def _hill(x: np.ndarray, *, half: float = 2.0, slope: float = 1.5) -> np.ndarray:
    x_pos = np.maximum(x, 0.0)
    return (x_pos**slope) / (half**slope + x_pos**slope + 1e-6)


def materialize_h4c_panel(spec: RecoveryWorldSpec, *, panel_seed: int | None = None) -> pd.DataFrame:
    """Build H4c panel from generative_kind (deterministic given spec + seed)."""
    rng = np.random.default_rng(panel_seed if panel_seed is not None else spec.mcmc_seed)
    kind = str(spec.expected_diagnostic_behavior.get("generative_kind", "linear"))
    params = dict(spec.expected_diagnostic_behavior.get("generative_params") or {})
    rows: list[dict[str, Any]] = []

    for geo in spec.geo_order:
        n_weeks = spec.weeks_by_geo[geo]
        alpha = spec.true_alpha_g[geo]
        betas = np.array([spec.true_beta_gc[geo][c] for c in spec.channels])

        if kind == "correlated":
            rho = float(params.get("rho", 0.92))
            tv_raw = rng.uniform(1.0, 5.0, size=n_weeks)
            search_raw = rho * tv_raw + rng.normal(0.0, 0.15, size=n_weeks)
            xs = {"tv": _standardized(tv_raw), "search": _standardized(search_raw)}
        elif kind == "weak_signal":
            scale = float(params.get("media_scale", 0.2))
            xs = {ch: _standardized(rng.normal(0.0, scale, size=n_weeks)) for ch in spec.channels}
        else:
            xs = {ch: _standardized(rng.uniform(1.0, 5.0, size=n_weeks)) for ch in spec.channels}

        if kind == "adstock":
            decay = float(params.get("decay", 0.7))
            xs = {ch: _adstock(xs[ch], decay) for ch in spec.channels}
        elif kind == "saturation":
            xs = {
                ch: _hill(xs[ch], half=float(params.get("half", 2.0)), slope=float(params.get("slope", 1.5)))
                for ch in spec.channels
            }

        x_mat = np.column_stack([xs[ch] for ch in spec.channels])
        noise = float(spec.noise_sigma)
        if kind == "weak_signal":
            noise = float(params.get("noise_sigma", noise))
        log_y = alpha + x_mat @ betas + rng.normal(0.0, noise, size=n_weeks)
        y = np.exp(log_y)
        for t in range(n_weeks):
            row: dict[str, Any] = {"geo_id": geo, "week": t, "y": float(max(y[t], 1e-3))}
            for ch in spec.channels:
                row[ch] = float(abs(xs[ch][t]) + 0.5)
            rows.append(row)
    return pd.DataFrame(rows)


def _clean_recovery() -> RecoveryWorldSpec:
    geos = ("dma_a", "dma_b", "dma_c")
    mu = {"tv": 0.32, "search": 0.20}
    tau = {"tv": 0.05, "search": 0.04}
    beta = {g: {"tv": mu["tv"] + 0.02, "search": mu["search"] - 0.01} for g in geos}
    return RecoveryWorldSpec(
        world_id=WORLD_BAYES_H4C_CLEAN_RECOVERY,
        geo_order=geos,
        channels=H4_CHANNELS,
        weeks_by_geo={g: H4C_WEEKS_LONG for g in geos},
        noise_sigma=0.08,
        true_mu_c=mu,
        true_tau_c=tau,
        true_beta_gc=beta,
        true_alpha_g={g: 3.0 for g in geos},
        geo_hierarchy={g: {"state": "s1", "region": "r1"} for g in geos},
        expected_diagnostic_behavior={
            "generative_kind": "linear",
            "h4c_classification": "recovery_candidate",
            "recovery_expectation": "good_mu_and_beta_recovery",
        },
    )


def _correlated_channels() -> RecoveryWorldSpec:
    geos = ("dma_a", "dma_b")
    mu = {"tv": 0.30, "search": 0.18}
    tau = {"tv": 0.10, "search": 0.08}
    beta = {"dma_a": {"tv": 0.32, "search": 0.17}, "dma_b": {"tv": 0.28, "search": 0.19}}
    return RecoveryWorldSpec(
        world_id=WORLD_BAYES_H4C_CORRELATED_CHANNELS,
        geo_order=geos,
        channels=H4_CHANNELS,
        weeks_by_geo={g: H4C_WEEKS_LONG for g in geos},
        noise_sigma=0.10,
        true_mu_c=mu,
        true_tau_c=tau,
        true_beta_gc=beta,
        true_alpha_g={g: 3.0 for g in geos},
        geo_hierarchy={g: {"state": "s1", "region": "r1"} for g in geos},
        expected_diagnostic_behavior={
            "generative_kind": "correlated",
            "generative_params": {"rho": 0.92},
            "h4c_classification": "weak_identification",
            "collinearity_warning_expected": True,
        },
    )


def _adstocked_media() -> RecoveryWorldSpec:
    geos = ("dma_a", "dma_b")
    mu = {"tv": 0.28, "search": 0.16}
    tau = {"tv": 0.08, "search": 0.06}
    beta = {"dma_a": {"tv": 0.30, "search": 0.15}, "dma_b": {"tv": 0.26, "search": 0.17}}
    return RecoveryWorldSpec(
        world_id=WORLD_BAYES_H4C_ADSTOCKED_MEDIA,
        geo_order=geos,
        channels=H4_CHANNELS,
        weeks_by_geo={g: H4C_WEEKS_LONG for g in geos},
        noise_sigma=0.10,
        true_mu_c=mu,
        true_tau_c=tau,
        true_beta_gc=beta,
        true_alpha_g={g: 3.0 for g in geos},
        geo_hierarchy={g: {"state": "s1", "region": "r1"} for g in geos},
        expected_diagnostic_behavior={
            "generative_kind": "adstock",
            "generative_params": {"decay": 0.7},
            "h4c_classification": "transform_mismatch",
            "model_mismatch": "MVP fits raw media; outcome uses adstock",
            "transform_mismatch_warning_expected": True,
        },
    )


def _saturation() -> RecoveryWorldSpec:
    geos = ("dma_a", "dma_b")
    mu = {"tv": 0.26, "search": 0.14}
    tau = {"tv": 0.07, "search": 0.05}
    beta = {"dma_a": {"tv": 0.28, "search": 0.13}, "dma_b": {"tv": 0.24, "search": 0.15}}
    return RecoveryWorldSpec(
        world_id=WORLD_BAYES_H4C_SATURATION,
        geo_order=geos,
        channels=H4_CHANNELS,
        weeks_by_geo={g: H4C_WEEKS_LONG for g in geos},
        noise_sigma=0.10,
        true_mu_c=mu,
        true_tau_c=tau,
        true_beta_gc=beta,
        true_alpha_g={g: 3.0 for g in geos},
        geo_hierarchy={g: {"state": "s1", "region": "r1"} for g in geos},
        expected_diagnostic_behavior={
            "generative_kind": "saturation",
            "generative_params": {"half": 2.0, "slope": 1.5},
            "h4c_classification": "transform_mismatch",
            "model_mismatch": "MVP semi_log linear in standardized media; outcome uses saturation",
            "transform_mismatch_warning_expected": True,
        },
    )


def _weak_signal() -> RecoveryWorldSpec:
    geos = ("dma_a", "dma_b")
    mu = {"tv": 0.25, "search": 0.15}
    tau = {"tv": 0.12, "search": 0.10}
    beta = {"dma_a": {"tv": 0.27, "search": 0.14}, "dma_b": {"tv": 0.23, "search": 0.16}}
    return RecoveryWorldSpec(
        world_id=WORLD_BAYES_H4C_WEAK_SIGNAL,
        geo_order=geos,
        channels=H4_CHANNELS,
        weeks_by_geo={g: H4_WEEKS_FULL for g in geos},
        noise_sigma=0.40,
        true_mu_c=mu,
        true_tau_c=tau,
        true_beta_gc=beta,
        true_alpha_g={g: 3.0 for g in geos},
        geo_hierarchy={g: {"state": "s1", "region": "r1"} for g in geos},
        expected_diagnostic_behavior={
            "generative_kind": "weak_signal",
            "generative_params": {"media_scale": 0.2, "noise_sigma": 0.45},
            "h4c_classification": "weak_identification",
            "recovery_expectation": "poor_beta_recovery",
        },
    )


def _sparse_recovery() -> RecoveryWorldSpec:
    geos = ("dma_dense_a", "dma_dense_b", "dma_sparse")
    mu = {"tv": 0.30, "search": 0.18}
    tau = {"tv": 0.12, "search": 0.10}
    beta = {
        "dma_dense_a": {"tv": 0.31, "search": 0.17},
        "dma_dense_b": {"tv": 0.29, "search": 0.19},
        "dma_sparse": {"tv": 0.33, "search": 0.17},
    }
    return RecoveryWorldSpec(
        world_id=WORLD_BAYES_H4C_SPARSE_RECOVERY,
        geo_order=geos,
        channels=H4_CHANNELS,
        weeks_by_geo={"dma_dense_a": H4C_WEEKS_LONG, "dma_dense_b": H4C_WEEKS_LONG, "dma_sparse": H4C_WEEKS_LONG},
        noise_sigma=0.12,
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
            "generative_kind": "linear",
            "h4c_classification": "recovery_candidate",
            "sparse_role": "recovery_not_stress",
            "note": "Contrast with WORLD-BAYES-H4-SPARSE-GEO stress world (report-only)",
        },
        mcmc_seed=H4_SEED,
    )


H4C_WORLDS: dict[str, RecoveryWorldSpec] = {
    WORLD_BAYES_H4C_CLEAN_RECOVERY: _clean_recovery(),
    WORLD_BAYES_H4C_CORRELATED_CHANNELS: _correlated_channels(),
    WORLD_BAYES_H4C_ADSTOCKED_MEDIA: _adstocked_media(),
    WORLD_BAYES_H4C_SATURATION: _saturation(),
    WORLD_BAYES_H4C_WEAK_SIGNAL: _weak_signal(),
    WORLD_BAYES_H4C_SPARSE_RECOVERY: _sparse_recovery(),
}


def list_h4c_recovery_worlds() -> tuple[RecoveryWorldSpec, ...]:
    return tuple(H4C_WORLDS[w] for w in H4C_WORLD_IDS)


def get_h4c_recovery_world(world_id: str) -> RecoveryWorldSpec:
    if world_id not in H4C_WORLDS:
        raise KeyError(f"unknown H4c world: {world_id!r}; known: {sorted(H4C_WORLDS)}")
    return H4C_WORLDS[world_id]
