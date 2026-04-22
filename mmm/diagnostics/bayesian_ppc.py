"""Prior / posterior predictive packaging — substantive PPC summaries for governance."""

from __future__ import annotations

from typing import Any


def build_bayesian_predictive_artifact(
    idata: Any | None,
    *,
    config: Any | None = None,
    y_obs: Any | None = None,
) -> dict[str, Any]:
    """
    Summarize ``prior_predictive`` / ``posterior_predictive`` groups on ``idata``.

    ``config`` is optional: PPC reads use ``idata`` + ``y_obs`` only. When ``config`` is omitted,
    governance still receives mean/coverage checks (no silent stub).

    ``y_obs`` is the **modeling-scale** target vector used in training (e.g. log revenue for semi_log).
    When its length matches the trailing dimension of ``posterior_predictive/obs``, we add an
    empirical coverage check (obs inside PP 5–95% interval per time/row).
    """
    out: dict[str, Any] = {
        "prior_predictive": None,
        "posterior_predictive_check": None,
        "idata_present": idata is not None,
        "notes": [],
        "artifact_version": "mmm_bayesian_ppc_artifact_v2",
    }
    if idata is None:
        out["notes"].append("idata is None; no PPC summary.")
        return out
    if config is None:
        out["notes"].append("config omitted: PPC checks use idata + y_obs only (no config-specific gates).")

    try:
        import numpy as np
    except ImportError:  # pragma: no cover
        return out

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
                arr = np.asarray(ppp["obs"].values, dtype=float)
                out["posterior_predictive_draw_shape"] = list(arr.shape)
                sim_mean = float(np.mean(arr))
                chk: dict[str, Any] = {
                    "posterior_predictive_obs_mean": sim_mean,
                    "posterior_predictive_obs_std": float(np.std(arr)),
                }
                if y_obs is not None:
                    yo = np.asarray(y_obs, dtype=float).ravel()
                    chk["actual_obs_mean"] = float(np.mean(yo))
                    chk["mean_abs_gap"] = float(abs(sim_mean - float(np.mean(yo))))
                    chk["std_ratio_pp_over_obs"] = float(
                        (float(np.std(arr)) + 1e-12) / (float(np.std(yo)) + 1e-12)
                    )
                    flat = arr
                    if flat.ndim >= 3:
                        flat = flat.reshape(-1, flat.shape[-1])
                    if flat.ndim == 2 and flat.shape[1] == yo.size:
                        lo = np.percentile(flat, 5.0, axis=0)
                        hi = np.percentile(flat, 95.0, axis=0)
                        inside = (yo >= lo) & (yo <= hi)
                        chk["empirical_coverage_p90"] = float(np.mean(inside.astype(float)))
                        chk["n_obs_matched_for_coverage"] = int(yo.size)
                        med = np.median(flat, axis=0)
                        chk["median_abs_residual_vs_obs"] = float(np.median(np.abs(med - yo)))
                    elif flat.size == yo.size and yo.size > 0:
                        flat2 = flat.reshape(-1, yo.size)
                        lo = np.percentile(flat2, 5.0, axis=0)
                        hi = np.percentile(flat2, 95.0, axis=0)
                        inside = (yo >= lo) & (yo <= hi)
                        chk["empirical_coverage_p90"] = float(np.mean(inside.astype(float)))
                        chk["n_obs_matched_for_coverage"] = int(yo.size)
                        med2 = np.median(flat2, axis=0)
                        chk["median_abs_residual_vs_obs"] = float(np.median(np.abs(med2 - yo)))
                out["posterior_predictive_check"] = chk
        except Exception as e:  # pragma: no cover
            out["notes"].append(f"posterior_predictive_check failed: {e}")
    else:
        out["notes"].append("No posterior_predictive group on idata (PPC not run or not merged).")

    return out


def ppc_artifact_stub(idata: Any | None, *, config: Any | None = None, y_obs: Any | None = None) -> dict[str, Any]:
    """Backward-compatible name for :func:`build_bayesian_predictive_artifact`."""
    return build_bayesian_predictive_artifact(idata, config=config, y_obs=y_obs)
