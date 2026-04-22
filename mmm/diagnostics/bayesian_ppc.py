"""Prior / posterior predictive packaging (Sprint 8)."""

from __future__ import annotations

from typing import Any


def build_bayesian_predictive_artifact(
    idata: Any | None,
    *,
    config: Any | None = None,
    y_obs: Any | None = None,
) -> dict[str, Any]:
    """
    Summarize optional ``prior_predictive`` / ``posterior_predictive`` groups on ``idata``.

    ``y_obs`` is optional observed vector for a crude PPC mean check vs posterior predictive mean.
    """
    out: dict[str, Any] = {
        "prior_predictive": None,
        "posterior_predictive_check": None,
        "idata_present": idata is not None,
        "notes": [],
    }
    if idata is None or config is None:
        out["notes"].append("No idata or config; PPC artifact minimal.")
        return out

    try:
        import numpy as np
    except ImportError:  # pragma: no cover
        return out

    # Prior predictive summary (if trainer merged prior groups)
    try:
        groups = set(idata.groups())
    except Exception:
        groups = set()
    if "prior_predictive" in groups:
        try:
            pp = idata["prior_predictive"]
            if "obs" in pp:
                arr = np.asarray(pp["obs"].values)
                out["prior_predictive"] = {
                    "obs_draws_shape": list(arr.shape),
                    "obs_prior_mean": float(np.mean(arr)),
                    "obs_prior_std": float(np.std(arr)),
                }
        except Exception as e:  # pragma: no cover
            out["notes"].append(f"prior_predictive summary failed: {e}")

    if "posterior_predictive" in groups:
        try:
            ppp = idata["posterior_predictive"]
            if "obs" in ppp:
                arr = np.asarray(ppp["obs"].values)
                sim_mean = float(np.mean(arr))
                chk: dict[str, Any] = {"posterior_predictive_obs_mean": sim_mean}
                if y_obs is not None:
                    yo = np.asarray(y_obs, dtype=float)
                    chk["actual_obs_mean"] = float(np.mean(yo))
                    chk["mean_abs_gap"] = float(abs(sim_mean - np.mean(yo)))
                out["posterior_predictive_check"] = chk
        except Exception as e:  # pragma: no cover
            out["notes"].append(f"posterior_predictive_check failed: {e}")

    return out


def ppc_artifact_stub(idata: Any | None, *, config: Any | None = None, y_obs: Any | None = None) -> dict[str, Any]:
    """Backward-compatible name for :func:`build_bayesian_predictive_artifact`."""
    return build_bayesian_predictive_artifact(idata, config=config, y_obs=y_obs)
