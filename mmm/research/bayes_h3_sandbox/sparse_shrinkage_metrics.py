"""Sparse partial-pooling shrinkage metrics for Bayes-H4 recovery (research only)."""

from __future__ import annotations

from typing import Any

from mmm.research.bayes_h3_sandbox.recovery_worlds import RecoveryWorldSpec

_EPS = 1e-6


def shrinkage_ratio_vs_center(
    true_beta: float,
    post_beta: float,
    pool_center: float,
) -> float | None:
    """
    Ratio of posterior distance to pool center vs generative outlier distance.

    Values < 1 indicate posterior beta moved closer to pool_center than true beta was.
  """
    if post_beta != post_beta or pool_center != pool_center:
        return None
    d_true = abs(true_beta - pool_center)
    d_post = abs(post_beta - pool_center)
    if d_true <= _EPS:
        return None
    return float(d_post / d_true)


def beta_geo_index_order(artifact: dict[str, Any], spec: RecoveryWorldSpec) -> list[str]:
    """Geo labels in beta posterior row order (from fit artifact)."""
    hier = artifact.get("hierarchy_evidence_diagnostics") or {}
    order = hier.get("beta_geo_index_order")
    if isinstance(order, list) and order:
        return [str(g) for g in order]
    return list(spec.geo_order)


def channel_index_order(artifact: dict[str, Any], spec: RecoveryWorldSpec) -> list[str]:
    hier = artifact.get("hierarchy_evidence_diagnostics") or {}
    order = hier.get("channel_index_order")
    if isinstance(order, list) and order:
        return [str(c) for c in order]
    return list(spec.channels)


def posterior_beta_means_by_geo(artifact: dict[str, Any], spec: RecoveryWorldSpec) -> dict[str, dict[str, float]]:
    """Map geo_id → channel → posterior mean beta using fit index order."""
    hier = artifact.get("hierarchy_evidence_diagnostics") or {}
    raw = hier.get("beta_geo_channel_mean") or {}
    geo_order = beta_geo_index_order(artifact, spec)
    ch_order = channel_index_order(artifact, spec)
    out: dict[str, dict[str, float]] = {}
    for gi, geo in enumerate(geo_order):
        ch_map = raw.get(str(gi), raw.get(geo, {}))
        if not isinstance(ch_map, dict):
            continue
        out[geo] = {}
        for ch in ch_order:
            val = ch_map.get(ch, ch_map.get(str(ch)))
            if val is not None:
                out[geo][ch] = float(val)
    return out


def posterior_mu_tau(artifact: dict[str, Any], spec: RecoveryWorldSpec) -> tuple[dict[str, float], dict[str, float]]:
    ps = artifact.get("posterior_summary") or {}
    mu = {c: float(ps.get("mu_channel_mean", {}).get(c, float("nan"))) for c in spec.channels}
    tau = {c: float(ps.get("tau_channel_mean", {}).get(c, float("nan"))) for c in spec.channels}
    pool = artifact.get("pooling_diagnostics") or {}
    if pool.get("tau_channel_mean"):
        tau = {c: float(pool["tau_channel_mean"].get(c, tau[c])) for c in spec.channels}
    if pool.get("mu_channel_mean"):
        mu = {c: float(pool["mu_channel_mean"].get(c, mu[c])) for c in spec.channels}
    return mu, tau


def beta_posterior_interval_width(
    artifact: dict[str, Any],
    spec: RecoveryWorldSpec,
    geo: str,
    channel: str,
    *,
    q_low: float = 0.05,
    q_high: float = 0.95,
) -> float | None:
    import numpy as np

    idata = artifact.get("idata")
    if idata is None:
        return None
    geo_order = beta_geo_index_order(artifact, spec)
    ch_order = channel_index_order(artifact, spec)
    if geo not in geo_order or channel not in ch_order:
        return None
    gi = geo_order.index(geo)
    ci = ch_order.index(channel)
    try:
        beta_da = idata.posterior["beta"].stack(sample=("chain", "draw"))
        beta = np.asarray(beta_da.values).reshape(-1, beta_da.shape[-2], beta_da.shape[-1])
        if gi >= beta.shape[1] or ci >= beta.shape[2]:
            return None
        draws = beta[:, gi, ci]
        return float(np.quantile(draws, q_high) - np.quantile(draws, q_low))
    except Exception:
        return None


def compute_sparse_shrinkage_decomposition(
    artifact: dict[str, Any],
    spec: RecoveryWorldSpec,
) -> dict[str, Any]:
    """Per (sparse geo, channel) shrinkage decomposition for INV-H4-001."""
    beta_post = posterior_beta_means_by_geo(artifact, spec)
    mu_post, tau_post = posterior_mu_tau(artifact, spec)
    entries: list[dict[str, Any]] = []
    ratios_posterior_mu: list[float] = []
    ratios_true_mu: list[float] = []

    geo_order = beta_geo_index_order(artifact, spec)
    ch_order = channel_index_order(artifact, spec)
    for geo in spec.sparse_geos:
        for ch in spec.channels:
            true_b = float(spec.true_beta_gc[geo][ch])
            true_mu = float(spec.true_mu_c[ch])
            post_b = beta_post.get(geo, {}).get(ch, float("nan"))
            post_mu = float(mu_post.get(ch, float("nan")))
            post_tau = float(tau_post.get(ch, float("nan")))
            interval_w = beta_posterior_interval_width(artifact, spec, geo, ch)

            d_true_to_true_mu = abs(true_b - true_mu)
            d_true_to_post_mu = abs(true_b - post_mu) if post_mu == post_mu else None
            d_post_to_true_mu = abs(post_b - true_mu) if post_b == post_b else None
            d_post_to_post_mu = abs(post_b - post_mu) if post_b == post_b and post_mu == post_mu else None

            ratio_post_mu = shrinkage_ratio_vs_center(true_b, post_b, post_mu)
            ratio_true_mu = shrinkage_ratio_vs_center(true_b, post_b, true_mu)
            if ratio_post_mu is not None:
                ratios_posterior_mu.append(ratio_post_mu)
            if ratio_true_mu is not None:
                ratios_true_mu.append(ratio_true_mu)

            geo_idx = geo_order.index(geo) if geo in geo_order else None
            ch_idx = ch_order.index(ch) if ch in ch_order else None
            entries.append(
                {
                    "geo": geo,
                    "channel": ch,
                    "geo_index": geo_idx,
                    "channel_index": ch_idx,
                    "true_beta_gc": true_b,
                    "true_mu_c": true_mu,
                    "posterior_beta_gc_mean": post_b if post_b == post_b else None,
                    "posterior_mu_c_mean": post_mu if post_mu == post_mu else None,
                    "posterior_tau_c_mean": post_tau if post_tau == post_tau else None,
                    "distance_true_sparse_to_true_mu": d_true_to_true_mu,
                    "distance_true_sparse_to_posterior_mu": d_true_to_post_mu,
                    "distance_posterior_sparse_to_true_mu": d_post_to_true_mu,
                    "distance_posterior_sparse_to_posterior_mu": d_post_to_post_mu,
                    "shrinkage_ratio_vs_posterior_mu": ratio_post_mu,
                    "shrinkage_ratio_vs_true_mu": ratio_true_mu,
                    "posterior_beta_interval_width_90": interval_w,
                    "shrinkage_expected": d_true_to_true_mu > _EPS,
                }
            )

    def _mean(vals: list[float]) -> float | None:
        return float(sum(vals) / len(vals)) if vals else None

    return {
        "sparse_geos": list(spec.sparse_geos),
        "beta_geo_index_order": beta_geo_index_order(artifact, spec),
        "channel_index_order": channel_index_order(artifact, spec),
        "by_geo_channel": entries,
        "shrinkage_ratio_sparse": _mean(ratios_posterior_mu),
        "shrinkage_ratio_sparse_vs_true_mu": _mean(ratios_true_mu),
        "metric_definition": {
            "primary": "mean(|posterior_beta - posterior_mu| / |true_beta - posterior_mu|) over sparse geos",
            "legacy_vs_true_mu": "mean(|posterior_beta - true_mu| / |true_beta - true_mu|) — H4a/H4b pilot definition",
            "interpret_lt_1": "posterior beta closer to pool center than generative outlier was",
            "pool_center_primary": "posterior_mu_c",
        },
    }
