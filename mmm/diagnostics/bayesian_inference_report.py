"""Bayesian convergence + PPC checks for **decision-grade** claims (not sigma-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from mmm.config.schema import MMMConfig, PoolingMode
from mmm.diagnostics.bayesian_ppc import build_bayesian_predictive_artifact


def _idata_groups(idata: Any) -> set[str]:
    try:
        return set(idata.groups())
    except Exception:
        return set()


def _var_names_for_diagnostics(idata: Any, pooling: PoolingMode) -> list[str]:
    """
    Posterior variables summarized for R-hat / ESS — decision-relevant structure, not ``sigma`` alone.

    High-dimensional latent sheets (``z``) are excluded; global / geo intercepts, scale, and **media /
    control coefficient nodes** are included when present on ``idata``.
    """
    try:
        names = list(idata.posterior.data_vars)
    except Exception:
        return ["sigma"]
    ordered_candidates = (
        "sigma",
        "tau",
        "alpha_geo",
        "beta_mu_media",
        "beta_mu_ctrl",
        "beta_mu",
        "beta_media",
        "beta_ctrl",
        "beta",
    )
    out: list[str] = []
    for v in ordered_candidates:
        if v in names:
            out.append(v)
    if pooling == PoolingMode.NONE and "beta" in out:
        # Per-geo beta is high-dimensional; still require it for NONE pooling decisions,
        # but drop from az.summary if it would duplicate alpha_geo path — keep beta for NONE.
        pass
    if pooling == PoolingMode.PARTIAL and "z" in names:
        # z is latent; gate on tau, beta_mu*, sigma, alpha_geo instead
        out = [x for x in out if x != "z"]
    if not out and "sigma" in names:
        out = ["sigma"]
    return out or ["sigma"]


def compute_bayesian_decision_diagnostics(
    idata: Any,
    *,
    y_obs: np.ndarray,
    pooling: PoolingMode,
    config: MMMConfig | None = None,
    max_rhat: float = 1.01,
    min_ess: float = 200.0,
    max_divergences: int = 0,
    max_mean_abs_ppc_gap: float | None = None,
    min_ppc_empirical_coverage: float | None = None,
) -> dict[str, Any]:
    """
    Summarize decision-relevant parameters and PPC artifacts.

    Sets ``posterior_diagnostics_ok`` / ``posterior_predictive_ok`` booleans used by governance.

    ``config`` is forwarded to :func:`build_bayesian_predictive_artifact` when provided; PPC still
    runs when ``config`` is ``None`` (v2 artifact builder does not stub out PP groups).
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

    ppc = build_bayesian_predictive_artifact(idata, config=config, y_obs=y_obs)
    chk = ppc.get("posterior_predictive_check") or {}
    gap = chk.get("mean_abs_gap")
    cov = chk.get("empirical_coverage_p90")
    groups = _idata_groups(idata)
    has_pp = "posterior_predictive" in groups

    thr = max_mean_abs_ppc_gap if max_mean_abs_ppc_gap is not None else None
    posterior_predictive_ok = False
    if not has_pp:
        posterior_predictive_ok = False
        ppc.setdefault("notes", []).append("No posterior_predictive group; PPC not run or empty")
    elif thr is not None:
        if gap is not None and np.isfinite(gap):
            posterior_predictive_ok = bool(float(gap) <= float(thr))
        else:
            posterior_predictive_ok = False
            ppc.setdefault("notes", []).append("posterior_predictive present but mean_abs_gap missing")
    else:
        # No strict PPC gap threshold configured: require a substantive numeric PPC summary.
        if gap is not None and np.isfinite(gap):
            posterior_predictive_ok = True
        elif cov is not None and np.isfinite(cov):
            posterior_predictive_ok = True
        else:
            posterior_predictive_ok = False
            ppc.setdefault("notes", []).append("PPC group present but neither mean_abs_gap nor coverage computed")

    if (
        posterior_predictive_ok
        and min_ppc_empirical_coverage is not None
        and cov is not None
        and np.isfinite(cov)
        and float(cov) < float(min_ppc_empirical_coverage)
    ):
        posterior_predictive_ok = False
        ppc.setdefault("notes", []).append(
            f"empirical_coverage_p90 {float(cov):.4f} below governance minimum {float(min_ppc_empirical_coverage)}"
        )

    if notes:
        ppc.setdefault("notes", []).extend(notes)

    out = {
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
    if config is not None:
        try:
            from mmm.config.transform_policy import build_transform_policy_manifest

            out["transform_policy"] = build_transform_policy_manifest(config)
        except Exception as e:  # pragma: no cover
            out.setdefault("notes", []).append(f"transform_policy_manifest_failed: {e}")
        gv = config.extensions.governance
        mc = getattr(gv, "bayesian_min_ppc_empirical_coverage", None)
        if mc is not None and cov is None and out["posterior_predictive_ok"]:
            out["posterior_predictive_ok"] = False
            ppc.setdefault("notes", []).append(
                "governance.bayesian_min_ppc_empirical_coverage set but empirical_coverage_p90 not computed"
            )
    return out
