"""Bayes-H5 validation worlds (research only — transform alignment / mismatch probes)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from mmm.research.bayes_h3_sandbox.h5_transforms import apply_channel_transform
from mmm.research.bayes_h3_sandbox.recovery_worlds import (
    H4_CHANNELS,
    H4_SEED,
    H4_WEEKS_FULL,
    RecoveryWorldSpec,
)

WORLD_BAYES_H5_ADSTOCK_ALIGNED = "WORLD-BAYES-H5-ADSTOCK-ALIGNED"
WORLD_BAYES_H5_SATURATION_ALIGNED = "WORLD-BAYES-H5-SATURATION-ALIGNED"
WORLD_BAYES_H5_ADSTOCK_MISMATCH = "WORLD-BAYES-H5-ADSTOCK-MISMATCH"
WORLD_BAYES_H5_SATURATION_MISMATCH = "WORLD-BAYES-H5-SATURATION-MISMATCH"
WORLD_BAYES_H5_CORRELATED_CHANNELS = "WORLD-BAYES-H5-CORRELATED-CHANNELS"
WORLD_BAYES_H5_WEAK_SIGNAL = "WORLD-BAYES-H5-WEAK-SIGNAL"
WORLD_BAYES_H5_SPARSE_RECOVERY = "WORLD-BAYES-H5-SPARSE-RECOVERY"

H5_WORLD_IDS: tuple[str, ...] = (
    WORLD_BAYES_H5_ADSTOCK_ALIGNED,
    WORLD_BAYES_H5_SATURATION_ALIGNED,
    WORLD_BAYES_H5_ADSTOCK_MISMATCH,
    WORLD_BAYES_H5_SATURATION_MISMATCH,
    WORLD_BAYES_H5_CORRELATED_CHANNELS,
    WORLD_BAYES_H5_WEAK_SIGNAL,
    WORLD_BAYES_H5_SPARSE_RECOVERY,
)

H5_WEEKS_LONG = 20


def h5_world_production_flags() -> dict[str, bool]:
    """All H5 validation worlds remain research-only (never production decisioning)."""
    return {
        "research_only": True,
        "approved_for_prod": False,
        "prod_decisioning_allowed": False,
        "production_promotion": False,
        "hard_gate": False,
        "decision_grade": False,
        "production_decision_surface": False,
    }


def _standardized(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / (x.std() + 1e-6)


def _generative_media(
    rng: np.random.Generator,
    spec: RecoveryWorldSpec,
    *,
    n_weeks: int,
) -> dict[str, np.ndarray]:
    exp = spec.expected_diagnostic_behavior
    gen = str(exp.get("generative_transform", exp.get("generative_kind", "linear")))
    params = dict(exp.get("generative_params") or {})

    if gen == "correlated":
        rho = float(params.get("rho", 0.92))
        tv_raw = rng.uniform(1.0, 5.0, size=n_weeks)
        search_raw = rho * tv_raw + rng.normal(0.0, 0.15, size=n_weeks)
        return {"tv": _standardized(tv_raw), "search": _standardized(search_raw)}
    if gen == "weak_signal":
        scale = float(params.get("media_scale", 0.2))
        return {ch: _standardized(rng.normal(0.0, scale, size=n_weeks)) for ch in spec.channels}

    xs = {ch: _standardized(rng.uniform(1.0, 5.0, size=n_weeks)) for ch in spec.channels}

    if gen in ("geometric_adstock", "adstock"):
        decay = float(params.get("decay", 0.7))
        xs = {ch: apply_channel_transform(xs[ch], "geometric_adstock", params={"decay": decay}) for ch in spec.channels}
    elif gen in ("hill_saturation", "saturation"):
        xs = {
            ch: apply_channel_transform(xs[ch], "hill_saturation", params=params)
            for ch in spec.channels
        }
    elif gen == "adstock_then_saturation":
        xs = {
            ch: apply_channel_transform(xs[ch], "adstock_then_saturation", params=params)
            for ch in spec.channels
        }
    return xs


def materialize_h5_panel(spec: RecoveryWorldSpec, *, panel_seed: int | None = None) -> pd.DataFrame:
    """Build H5 panel from generative_transform (deterministic given spec + seed)."""
    rng = np.random.default_rng(panel_seed if panel_seed is not None else spec.mcmc_seed)
    exp = spec.expected_diagnostic_behavior
    gen = str(exp.get("generative_transform", "linear"))
    params = dict(exp.get("generative_params") or {})
    rows: list[dict[str, Any]] = []

    for geo in spec.geo_order:
        n_weeks = spec.weeks_by_geo[geo]
        alpha = spec.true_alpha_g[geo]
        betas = np.array([spec.true_beta_gc[geo][c] for c in spec.channels])
        xs = _generative_media(rng, spec, n_weeks=n_weeks)

        x_mat = np.column_stack([xs[ch] for ch in spec.channels])
        noise = float(spec.noise_sigma)
        if gen == "weak_signal":
            noise = float(params.get("noise_sigma", noise))
        log_y = alpha + x_mat @ betas + rng.normal(0.0, noise, size=n_weeks)
        y = np.exp(log_y)
        for t in range(n_weeks):
            row: dict[str, Any] = {"geo_id": geo, "week": t, "y": float(max(y[t], 1e-3))}
            for ch in spec.channels:
                row[ch] = float(abs(xs[ch][t]) + 0.5)
            rows.append(row)
    return pd.DataFrame(rows)


def fitted_transforms_for_world(spec: RecoveryWorldSpec) -> dict[str, str]:
    exp = spec.expected_diagnostic_behavior
    per_ch = exp.get("media_transforms_by_channel")
    if per_ch:
        return {str(k): str(v) for k, v in per_ch.items()}
    fitted = str(exp.get("fitted_transform_id", "identity"))
    return {ch: fitted for ch in spec.channels}


def sandbox_overrides_for_h5_world(spec: RecoveryWorldSpec) -> dict[str, Any]:
    exp = spec.expected_diagnostic_behavior
    overrides = dict(spec.sandbox_model_overrides)
    overrides["media_transforms_by_channel"] = fitted_transforms_for_world(spec)
    overrides["transform_params_by_channel"] = dict(exp.get("transform_params_by_channel") or {})
    overrides["h5_generative_transform"] = str(exp.get("generative_transform", "linear"))
    overrides["h5_transform_mismatch_mode"] = str(exp.get("transform_mismatch_mode", "aligned"))
    return overrides


def _adstock_aligned() -> RecoveryWorldSpec:
    geos = ("dma_a", "dma_b")
    mu = {"tv": 0.28, "search": 0.16}
    tau = {"tv": 0.08, "search": 0.06}
    beta = {"dma_a": {"tv": 0.30, "search": 0.15}, "dma_b": {"tv": 0.26, "search": 0.17}}
    return RecoveryWorldSpec(
        world_id=WORLD_BAYES_H5_ADSTOCK_ALIGNED,
        geo_order=geos,
        channels=H4_CHANNELS,
        weeks_by_geo={g: H5_WEEKS_LONG for g in geos},
        noise_sigma=0.10,
        true_mu_c=mu,
        true_tau_c=tau,
        true_beta_gc=beta,
        true_alpha_g={g: 3.0 for g in geos},
        geo_hierarchy={g: {"state": "s1", "region": "r1"} for g in geos},
        expected_diagnostic_behavior={
            "role": "recovery_candidate",
            "h5_classification": "recovery_candidate",
            "generative_transform": "geometric_adstock",
            "generative_params": {"decay": 0.7},
            "fitted_transform_id": "geometric_adstock",
            "fitted_transform_expectation": "geometric_adstock",
            "transform_mismatch_mode": "aligned",
            "transform_mismatch_warning_expected": False,
            "recovery_expectation": "improved_beta_recovery_vs_h4c_mismatch",
        },
        sandbox_model_overrides={"tau_channel_prior_sigma": 0.5},
    )


def _saturation_aligned() -> RecoveryWorldSpec:
    geos = ("dma_a", "dma_b")
    mu = {"tv": 0.26, "search": 0.14}
    tau = {"tv": 0.07, "search": 0.05}
    beta = {"dma_a": {"tv": 0.28, "search": 0.13}, "dma_b": {"tv": 0.24, "search": 0.15}}
    return RecoveryWorldSpec(
        world_id=WORLD_BAYES_H5_SATURATION_ALIGNED,
        geo_order=geos,
        channels=H4_CHANNELS,
        weeks_by_geo={g: H5_WEEKS_LONG for g in geos},
        noise_sigma=0.10,
        true_mu_c=mu,
        true_tau_c=tau,
        true_beta_gc=beta,
        true_alpha_g={g: 3.0 for g in geos},
        geo_hierarchy={g: {"state": "s1", "region": "r1"} for g in geos},
        expected_diagnostic_behavior={
            "role": "recovery_candidate",
            "h5_classification": "recovery_candidate",
            "generative_transform": "hill_saturation",
            "generative_params": {"half": 2.0, "slope": 1.5},
            "fitted_transform_id": "hill_saturation",
            "fitted_transform_expectation": "hill_saturation",
            "transform_mismatch_mode": "aligned",
            "transform_mismatch_warning_expected": False,
            "recovery_expectation": "improved_beta_recovery_vs_h4c_mismatch",
        },
    )


def _adstock_mismatch() -> RecoveryWorldSpec:
    geos = ("dma_a", "dma_b")
    mu = {"tv": 0.28, "search": 0.16}
    tau = {"tv": 0.08, "search": 0.06}
    beta = {"dma_a": {"tv": 0.30, "search": 0.15}, "dma_b": {"tv": 0.26, "search": 0.17}}
    return RecoveryWorldSpec(
        world_id=WORLD_BAYES_H5_ADSTOCK_MISMATCH,
        geo_order=geos,
        channels=H4_CHANNELS,
        weeks_by_geo={g: H5_WEEKS_LONG for g in geos},
        noise_sigma=0.10,
        true_mu_c=mu,
        true_tau_c=tau,
        true_beta_gc=beta,
        true_alpha_g={g: 3.0 for g in geos},
        geo_hierarchy={g: {"state": "s1", "region": "r1"} for g in geos},
        expected_diagnostic_behavior={
            "role": "transform_mismatch",
            "h5_classification": "transform_mismatch",
            "generative_transform": "geometric_adstock",
            "generative_params": {"decay": 0.7},
            "fitted_transform_id": "identity",
            "fitted_transform_expectation": "identity",
            "transform_mismatch_mode": "intentional_mismatch",
            "transform_mismatch_warning_expected": True,
            "model_mismatch": "Outcome adstocked; H5 fit uses identity on raw media",
        },
    )


def _saturation_mismatch() -> RecoveryWorldSpec:
    geos = ("dma_a", "dma_b")
    mu = {"tv": 0.26, "search": 0.14}
    tau = {"tv": 0.07, "search": 0.05}
    beta = {"dma_a": {"tv": 0.28, "search": 0.13}, "dma_b": {"tv": 0.24, "search": 0.15}}
    return RecoveryWorldSpec(
        world_id=WORLD_BAYES_H5_SATURATION_MISMATCH,
        geo_order=geos,
        channels=H4_CHANNELS,
        weeks_by_geo={g: H5_WEEKS_LONG for g in geos},
        noise_sigma=0.10,
        true_mu_c=mu,
        true_tau_c=tau,
        true_beta_gc=beta,
        true_alpha_g={g: 3.0 for g in geos},
        geo_hierarchy={g: {"state": "s1", "region": "r1"} for g in geos},
        expected_diagnostic_behavior={
            "role": "transform_mismatch",
            "h5_classification": "transform_mismatch",
            "generative_transform": "hill_saturation",
            "generative_params": {"half": 2.0, "slope": 1.5},
            "fitted_transform_id": "identity",
            "fitted_transform_expectation": "identity",
            "transform_mismatch_mode": "intentional_mismatch",
            "transform_mismatch_warning_expected": True,
            "model_mismatch": "Outcome saturated; H5 fit uses identity on raw media",
        },
    )


def _correlated_channels() -> RecoveryWorldSpec:
    geos = ("dma_a", "dma_b")
    mu = {"tv": 0.30, "search": 0.18}
    tau = {"tv": 0.10, "search": 0.08}
    beta = {"dma_a": {"tv": 0.32, "search": 0.17}, "dma_b": {"tv": 0.28, "search": 0.19}}
    return RecoveryWorldSpec(
        world_id=WORLD_BAYES_H5_CORRELATED_CHANNELS,
        geo_order=geos,
        channels=H4_CHANNELS,
        weeks_by_geo={g: H5_WEEKS_LONG for g in geos},
        noise_sigma=0.10,
        true_mu_c=mu,
        true_tau_c=tau,
        true_beta_gc=beta,
        true_alpha_g={g: 3.0 for g in geos},
        geo_hierarchy={g: {"state": "s1", "region": "r1"} for g in geos},
        expected_diagnostic_behavior={
            "role": "weak_identification",
            "h5_classification": "weak_identification",
            "generative_transform": "correlated",
            "generative_params": {"rho": 0.92},
            "fitted_transform_id": "identity",
            "fitted_transform_expectation": "identity",
            "transform_mismatch_mode": "aligned",
            "collinearity_warning_expected": True,
        },
    )


def _weak_signal() -> RecoveryWorldSpec:
    geos = ("dma_a", "dma_b")
    mu = {"tv": 0.25, "search": 0.15}
    tau = {"tv": 0.12, "search": 0.10}
    beta = {"dma_a": {"tv": 0.27, "search": 0.14}, "dma_b": {"tv": 0.23, "search": 0.16}}
    return RecoveryWorldSpec(
        world_id=WORLD_BAYES_H5_WEAK_SIGNAL,
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
            "role": "weak_identification",
            "h5_classification": "weak_identification",
            "generative_transform": "weak_signal",
            "generative_params": {"media_scale": 0.2, "noise_sigma": 0.45},
            "fitted_transform_id": "identity",
            "fitted_transform_expectation": "identity",
            "transform_mismatch_mode": "aligned",
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
        world_id=WORLD_BAYES_H5_SPARSE_RECOVERY,
        geo_order=geos,
        channels=H4_CHANNELS,
        weeks_by_geo={"dma_dense_a": H5_WEEKS_LONG, "dma_dense_b": H5_WEEKS_LONG, "dma_sparse": H5_WEEKS_LONG},
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
            "role": "recovery_candidate",
            "h5_classification": "recovery_candidate",
            "generative_transform": "linear",
            "fitted_transform_id": "identity",
            "fitted_transform_expectation": "identity",
            "transform_mismatch_mode": "aligned",
            "sparse_role": "recovery_not_stress",
        },
        mcmc_seed=H4_SEED,
    )


H5_WORLDS: dict[str, RecoveryWorldSpec] = {
    WORLD_BAYES_H5_ADSTOCK_ALIGNED: _adstock_aligned(),
    WORLD_BAYES_H5_SATURATION_ALIGNED: _saturation_aligned(),
    WORLD_BAYES_H5_ADSTOCK_MISMATCH: _adstock_mismatch(),
    WORLD_BAYES_H5_SATURATION_MISMATCH: _saturation_mismatch(),
    WORLD_BAYES_H5_CORRELATED_CHANNELS: _correlated_channels(),
    WORLD_BAYES_H5_WEAK_SIGNAL: _weak_signal(),
    WORLD_BAYES_H5_SPARSE_RECOVERY: _sparse_recovery(),
}


def list_h5_validation_worlds() -> tuple[RecoveryWorldSpec, ...]:
    return tuple(H5_WORLDS[w] for w in H5_WORLD_IDS)


def get_h5_validation_world(world_id: str) -> RecoveryWorldSpec:
    if world_id not in H5_WORLDS:
        raise KeyError(f"unknown H5 world: {world_id!r}; known: {sorted(H5_WORLDS)}")
    return H5_WORLDS[world_id]


def h5_world_catalog_metadata() -> list[dict[str, Any]]:
    """Serializable catalog for pilot artifacts (research only)."""
    rows: list[dict[str, Any]] = []
    for wid in H5_WORLD_IDS:
        spec = H5_WORLDS[wid]
        exp = dict(spec.expected_diagnostic_behavior)
        rows.append(
            {
                "world_id": wid,
                "role": exp.get("role"),
                "h5_classification": exp.get("h5_classification"),
                "generative_transform": exp.get("generative_transform"),
                "fitted_transform_expectation": exp.get("fitted_transform_expectation"),
                "transform_mismatch_mode": exp.get("transform_mismatch_mode"),
                "expected_diagnostic_behavior": exp,
                **h5_world_production_flags(),
            }
        )
    return rows
