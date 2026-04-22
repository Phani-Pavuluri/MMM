"""Map a fitted PyMC ``idata`` object into Ridge-shaped coefficient draws for posterior planning."""

from __future__ import annotations

from typing import Any

import numpy as np

from mmm.config.schema import PoolingMode


def linear_coef_draws_from_pymc_idata(
    idata: Any,
    *,
    pooling: PoolingMode,
    n_media: int,
    n_controls: int,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """
    Export ``(n_draws, n_coef)`` on the **global** linear coefficient vector (channels then controls).

    **Supported:** ``PoolingMode.FULL`` where ``idata.posterior`` contains ``beta`` with last axis
    ``n_media + n_controls``.

    **Not supported here:** ``PARTIAL`` / ``NONE`` — use
    :func:`hierarchical_coefficient_draws_from_pymc_idata` + hierarchical μ in posterior planning.
    """
    meta: dict[str, Any] = {
        "export_status": "unsupported",
        "pooling": pooling.value if hasattr(pooling, "value") else str(pooling),
        "notes": [],
    }
    p = int(n_media + n_controls)
    if pooling != PoolingMode.FULL:
        meta["notes"].append(
            "linear_coef_draws export is only for pooling=full; use hierarchical_coefficient_draws_from_pymc_idata "
            "for partial/none pooling."
        )
        return None, meta
    try:
        post = idata.posterior
    except Exception as e:  # pragma: no cover
        meta["notes"].append(f"no_posterior_group: {e}")
        return None, meta
    if "beta" not in post.data_vars:
        meta["notes"].append("posterior_missing_beta")
        return None, meta
    b = post["beta"]
    try:
        stacked = b.stack(sample=("chain", "draw"))
        other_dims = tuple(d for d in stacked.dims if d != "sample")
        ordered = stacked.transpose("sample", *other_dims)
        mat = np.asarray(ordered.values, dtype=float)
        draws = mat.reshape(stacked.sizes["sample"], -1)
    except Exception as e:  # pragma: no cover
        meta["notes"].append(f"stack_failed: {e}")
        return None, meta
    if draws.shape[1] != p:
        meta["notes"].append(f"coef_dim_mismatch draws.shape={draws.shape} expected_p={p}")
        return None, meta
    meta["export_status"] = "ok"
    meta["n_draws"] = int(draws.shape[0])
    meta["n_coef"] = int(draws.shape[1])
    return draws, meta


def hierarchical_coefficient_draws_from_pymc_idata(
    idata: Any,
    *,
    pooling: PoolingMode,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """
    Export per-geo intercept and coefficient draws for hierarchical μ simulation.

    **Supported:** ``PoolingMode.PARTIAL`` and ``PoolingMode.NONE`` when ``idata.posterior`` contains
    ``alpha_geo`` (``chain``, ``draw``, ``geo``) and ``beta`` (``chain``, ``draw``, ``geo``, ``coef``).

    Returns a **pack** dict suitable for :func:`mmm.planning.posterior_planning.delta_mu_draws_hierarchical`
    (keys ``alpha_draws``, ``beta_draws``, ``pooling``, …) or ``None`` if shapes cannot be resolved.
    """
    meta: dict[str, Any] = {"export_status": "unsupported", "pooling": str(pooling), "notes": []}
    if pooling == PoolingMode.FULL:
        meta["notes"].append("full_pooling: use linear_coef_draws_from_pymc_idata")
        return None, meta
    if pooling not in (PoolingMode.PARTIAL, PoolingMode.NONE):
        meta["notes"].append("pooling_mode_not_supported_for_hierarchical_export")
        return None, meta
    try:
        post = idata.posterior
    except Exception as e:  # pragma: no cover
        meta["notes"].append(f"no_posterior: {e}")
        return None, meta
    if "alpha_geo" not in post.data_vars or "beta" not in post.data_vars:
        meta["notes"].append("missing_alpha_geo_or_beta_on_posterior")
        return None, meta

    def _stack_sample_first(var: Any) -> np.ndarray:
        s = var.stack(sample=("chain", "draw"))
        oth = tuple(d for d in s.dims if d != "sample")
        t = s.transpose("sample", *oth)
        return np.asarray(t.values, dtype=float)

    try:
        ag = _stack_sample_first(post["alpha_geo"])
        bg = _stack_sample_first(post["beta"])
    except Exception as e:  # pragma: no cover
        meta["notes"].append(f"stack_failed: {e}")
        return None, meta

    if ag.ndim > 2:
        ag = ag.reshape(ag.shape[0], -1)
    if bg.ndim != 3:
        meta["notes"].append(f"beta_expected_ndim_3_got_{bg.ndim}_shape_{bg.shape}")
        return None, meta
    s_draw, n_geo, p = bg.shape
    if ag.shape != (s_draw, n_geo):
        meta["notes"].append(f"alpha_shape_{ag.shape}_vs_beta_{bg.shape}")
        return None, meta

    pooling_val = pooling.value if hasattr(pooling, "value") else str(pooling)
    pack: dict[str, Any] = {
        "kind": "hierarchical_geo_linear",
        "pooling": pooling_val,
        "alpha_draws": ag,
        "beta_draws": bg,
        "n_draws": int(s_draw),
        "n_geo": int(n_geo),
        "n_coef": int(p),
    }
    meta["export_status"] = "ok"
    meta["n_draws"] = int(s_draw)
    meta["n_geo"] = int(n_geo)
    meta["n_coef"] = int(p)
    return pack, meta
