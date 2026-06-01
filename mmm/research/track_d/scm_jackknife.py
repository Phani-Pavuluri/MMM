"""Research-only SCM + unit jackknife readout (characterization; not GeoX production)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ScmJackknifeSpec:
    """Synthetic unit panel layout for D5-POW."""

    n_control_units: int = 10
    n_pre_periods: int = 20
    n_post_periods: int = 10
    noise_sigma: float = 0.08
    pre_trend: float = 0.02


@dataclass(frozen=True)
class ScmJackknifeReadout:
    """Point effect, jackknife interval, and detection flags."""

    point_effect: float
    jk_std: float
    ci_low: float
    ci_high: float
    excludes_zero: bool
    detected_interval: bool
    n_jk_replicates: int
    jk_effects: tuple[float, ...]


def _simple_scm_weights(
    control_pre: np.ndarray,
    treated_pre: np.ndarray,
) -> np.ndarray:
    """Non-negative weights summing to 1 (pre-period L2 match on levels)."""
    n_control, _ = control_pre.shape
    treated_level = float(np.mean(treated_pre))
    control_levels = np.mean(control_pre, axis=1)
    dist = np.abs(control_levels - treated_level) + 1e-6
    inv = 1.0 / dist
    w = inv / inv.sum()
    return w


def scm_point_effect(
    control_pre: np.ndarray,
    control_post: np.ndarray,
    treated_pre: np.ndarray,
    treated_post: np.ndarray,
    *,
    leave_out: int | None = None,
) -> float:
    """
    SCM-style lift: treated post mean minus synthetic control post mean.

    Optional leave_out drops one control unit index for jackknife.
    """
    idx = [i for i in range(control_pre.shape[0]) if leave_out is None or i != leave_out]
    c_pre = control_pre[idx]
    c_post = control_post[idx]
    w = _simple_scm_weights(c_pre, treated_pre)
    synth_post = float(np.sum(w * np.mean(c_post, axis=1)))
    treated_post_mean = float(np.mean(treated_post))
    return treated_post_mean - synth_post


def scm_unit_jackknife_readout(
    control_pre: np.ndarray,
    control_post: np.ndarray,
    treated_pre: np.ndarray,
    treated_post: np.ndarray,
    *,
    z_crit: float = 1.96,
) -> ScmJackknifeReadout:
    """
    Unit jackknife on control units (post-fix target: unit-level SCM lift).

    CI uses normal approximation on leave-one-out effects; research-only.
    """
    n_control = control_pre.shape[0]
    point = scm_point_effect(control_pre, control_post, treated_pre, treated_post)
    jk: list[float] = []
    for leave in range(n_control):
        jk.append(
            scm_point_effect(
                control_pre,
                control_post,
                treated_pre,
                treated_post,
                leave_out=leave,
            )
        )
    jk_arr = np.asarray(jk, dtype=float)
    jk_mean = float(np.mean(jk_arr))
    # Standard jackknife variance multiplier (n-1)
    if n_control > 1:
        jk_var = float((n_control - 1) * np.mean((jk_arr - jk_mean) ** 2))
        jk_std = float(np.sqrt(max(jk_var, 0.0)))
    else:
        jk_std = float("nan")

    if jk_std == jk_std and jk_std > 0:
        ci_low = point - z_crit * jk_std
        ci_high = point + z_crit * jk_std
    else:
        ci_low = ci_high = point

    excludes_zero = bool(ci_low > 0.0 or ci_high < 0.0)
    return ScmJackknifeReadout(
        point_effect=float(point),
        jk_std=jk_std,
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        excludes_zero=excludes_zero,
        detected_interval=excludes_zero,
        n_jk_replicates=n_control,
        jk_effects=tuple(float(x) for x in jk),
    )


def simulate_unit_panel(
    spec: ScmJackknifeSpec,
    *,
    injected_lift: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return control_pre, control_post, treated_pre, treated_post."""
    n_c = spec.n_control_units
    n_pre = spec.n_pre_periods
    n_post = spec.n_post_periods

    def _series(base: float) -> np.ndarray:
        t = np.arange(n_pre + n_post, dtype=float)
        noise = rng.normal(0.0, spec.noise_sigma, size=t.shape)
        trend = spec.pre_trend * t
        return base + trend + noise

    control_pre = np.stack([_series(1.0 + 0.05 * i)[:n_pre] for i in range(n_c)])
    control_post = np.stack([_series(1.0 + 0.05 * i)[n_pre:] for i in range(n_c)])
    treated_pre = _series(1.2)[:n_pre]
    treated_post = _series(1.2 + injected_lift)[n_pre:]
    return control_pre, control_post, treated_pre, treated_post
