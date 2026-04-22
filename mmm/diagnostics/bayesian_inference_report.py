"""Bayesian convergence + PPC checks for **decision-grade** claims (not sigma-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from mmm.config.schema import PoolingMode
from mmm.diagnostics.bayesian_ppc import build_bayesian_predictive_artifact


def _var_names_for_diagnostics(idata: Any, pooling: PoolingMode) -> list[str]:
    """Select posterior variables that must pass R-hat / ESS (exclude huge latent sheets like ``z``)."""
    try:
        names = list(idata.posterior.data_vars)
    except Exception:
        return ["sigma"]
    out: list[str] = []
    for v in ("sigma", "tau", "beta_mu", "beta_media", "beta_mu_ctrl", "beta_ctrl", "beta", "alpha_geo"):
        if v in names:
            out.append(v)
    if not out and "sigma" in names:
        out = ["sigma"]
    if pooling == PoolingMode.NONE and "beta" in names:
        # High-dimensional per-geo beta: still require sigma + tau if present
        out = [x for x in out if x != "beta"]
        if "tau" in names and "tau" not in out:
            out.append("tau")
        if "sigma" not in out and "sigma" in names:
            out.insert(0, "sigma")
    return out or ["sigma"]


def compute_bayesian_decision_diagnostics(
    idata: Any,
    *,
    y_obs: np.ndarray,
    pooling: PoolingMode,
    max_rhat: float = 1.01,
    min_ess: float = 200.0,
    max_divergences: int = 0,
    max_mean_abs_ppc_gap: float | None = None,
) -> dict[str, Any]:
    """
    Summarize **all decision-relevant** scalar/vector parameters (not ``sigma`` alone).

    Sets ``posterior_diagnostics_ok`` / ``posterior_predictive_ok`` booleans used by governance.
    """
    try:
        import arviz as az  # type: ignore
    except ImportError:  # pragma: no cover
        return {
            "posterior_diagnostics_ok": False,
            "posterior_predictive_ok": False,
            "notes": ["arviz not installed"],
        }

    notes: list[str] = []
    var_names = _var_names_for_diagnostics(idata, pooling)
    summ = az.summary(idata, var_names=var_names, round_to=None)
    r_hat = summ["r_hat"].to_numpy(dtype=float) if "r_hat" in summ.columns else np.array([np.nan])
    ess = summ["ess_bulk"].to_numpy(dtype=float) if "ess_bulk" in summ.columns else np.array([np.nan])
    rhat_max = float(np.nanmax(r_hat))
    ess_min = float(np.nanmin(ess))
    try:
        div = int(idata.sample_stats["diverging"].sum())
    except Exception:
        div = -1

    r_ok = np.isfinite(rhat_max) and rhat_max <= max_rhat
    e_ok = np.isfinite(ess_min) and ess_min >= min_ess
    d_ok = div >= 0 and div <= max_divergences
    posterior_diagnostics_ok = bool(r_ok and e_ok and d_ok)
    decision_vars = [v for v in var_names if v != "sigma"]
    if posterior_diagnostics_ok and not decision_vars:
        posterior_diagnostics_ok = False
        notes.append(
            "diagnostics_sigma_only: insufficient_parameter_coverage_for_decision_grade "
            "(require media/control/hierarchical summaries, not sigma alone)"
        )

    ppc = build_bayesian_predictive_artifact(idata, config=None, y_obs=y_obs)
    gap = None
    if ppc.get("posterior_predictive_check"):
        gap = ppc["posterior_predictive_check"].get("mean_abs_gap")
    thr = max_mean_abs_ppc_gap if max_mean_abs_ppc_gap is not None else float("inf")
    posterior_predictive_ok = False
    if gap is not None and np.isfinite(gap):
        posterior_predictive_ok = bool(gap <= thr)
    elif "posterior_predictive" not in (set(idata.groups()) if hasattr(idata, "groups") else set()):
        posterior_predictive_ok = False
        ppc.setdefault("notes", []).append("No posterior_predictive group; PPC not run or empty")
    else:
        posterior_predictive_ok = False

    if notes:
        ppc.setdefault("notes", []).extend(notes)

    return {
        "var_names_summarized": var_names,
        "decision_var_names": decision_vars,
        "rhat_max": rhat_max,
        "ess_bulk_min": ess_min,
        "divergences": div,
        "thresholds": {"max_rhat": max_rhat, "min_ess": min_ess, "max_divergences": max_divergences},
        "posterior_diagnostics_ok": posterior_diagnostics_ok,
        "posterior_predictive_ok": posterior_predictive_ok,
        "arviz_summary_rows": int(len(summ)),
        "ppc_artifact": ppc,
        "notes": notes,
    }
