"""Bayes-H3 research sandbox hierarchical Bayesian MMM prototype (H2d-aligned, diagnostic only).

Inference backend: PyMC (see docs/05_validation/bayes_h3_research_sandbox_backend_adr.md).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from mmm.config.schema import MMMConfig, ModelForm
from mmm.data.panel_order import sort_panel_for_modeling
from mmm.data.schema import PanelSchema, validate_panel
from mmm.hierarchy.pooling import partial_pooling_indices
from mmm.utils.math import safe_log

MODEL_KIND = "bayes_h3_hierarchical_mvp_v1"


def fit_h3_sandbox_hierarchical(
    config: MMMConfig,
    schema: PanelSchema,
    df: pd.DataFrame,
    *,
    geo_hierarchy_mapping: dict[str, Any] | None = None,
    calibration_signals_stub: list[dict[str, Any]] | None = None,
    sandbox_model_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Run a small partial-pooling hierarchical Gaussian MMM (research sandbox only).

    Generative sketch (H2d MVP):
      beta_{g,c} ~ Normal(mu_c, tau_c^2)
      log(y_{g,t}) ~ Normal(alpha_g + sum_c beta_{g,c} * x_{g,t,c}, sigma)
    """
    try:
        import pymc as pm  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise ImportError("Bayes-H3 sandbox requires pymc: pip install mmm[bayesian]") from e

    df = validate_panel(df, schema, integrity_qa=False, calendar_strict=False)
    df = sort_panel_for_modeling(df, schema)
    beta_geo_index_order = df[schema.geo_column].unique().tolist()
    geo_idx = partial_pooling_indices(df, schema)
    n_geo = int(geo_idx.max() + 1)
    channels = list(schema.channel_columns)
    n_c = len(channels)
    x = df[list(channels)].to_numpy(dtype=float)
    x = (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-6)
    y = df[schema.target_column].to_numpy(dtype=float)
    y_obs = safe_log(y) if config.model_form == ModelForm.SEMI_LOG else y.astype(float)

    draws = int(config.bayesian.draws)
    tune = int(config.bayesian.tune)
    chains = int(config.bayesian.chains)
    nuts_seed = int(config.bayesian.nuts_seed)
    overrides = sandbox_model_overrides or {}
    tau_prior_sigma = float(overrides.get("tau_channel_prior_sigma", 0.5))

    with pm.Model() as model:
        alpha_geo = pm.Normal("alpha_geo", mu=0.0, sigma=1.0, shape=n_geo)
        sigma = pm.HalfNormal("sigma", sigma=1.0)
        mu_c = pm.Normal("mu_channel", mu=0.0, sigma=0.5, shape=n_c)
        tau_c = pm.HalfNormal("tau_channel", sigma=tau_prior_sigma, shape=n_c)
        z = pm.Normal("z_beta", mu=0.0, sigma=1.0, shape=(n_geo, n_c))
        beta = pm.Deterministic("beta", mu_c + z * tau_c)
        mu = alpha_geo[geo_idx] + (x * beta[geo_idx]).sum(axis=-1)
        pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y_obs)

        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=float(config.bayesian.target_accept),
            random_seed=nuts_seed,
            progressbar=False,
            return_inferencedata=True,
        )

    return _package_mvp_fit(
        idata,
        model=model,
        channels=channels,
        geo_idx=geo_idx,
        n_geo=n_geo,
        n_obs=len(df),
        geo_hierarchy_mapping=geo_hierarchy_mapping or {},
        calibration_signals_stub=calibration_signals_stub or [],
        beta_geo_index_order=beta_geo_index_order,
        sandbox_model_overrides=overrides,
    )


def _package_mvp_fit(
    idata: Any,
    *,
    model: Any,
    channels: list[str],
    geo_idx: np.ndarray,
    n_geo: int,
    n_obs: int,
    geo_hierarchy_mapping: dict[str, Any],
    calibration_signals_stub: list[dict[str, Any]],
    beta_geo_index_order: list[str],
    sandbox_model_overrides: dict[str, Any],
) -> dict[str, Any]:
    import arviz as az  # type: ignore

    summary = az.summary(idata, var_names=["mu_channel", "tau_channel", "sigma", "alpha_geo"])
    rhat_max = float(summary["r_hat"].max()) if "r_hat" in summary.columns and len(summary) else float("nan")
    ess_min = float(summary["ess_bulk"].min()) if "ess_bulk" in summary.columns and len(summary) else float("nan")

    mu_post = idata.posterior["mu_channel"].mean(dim=("chain", "draw")).values
    tau_post = idata.posterior["tau_channel"].mean(dim=("chain", "draw")).values
    beta_post = idata.posterior["beta"].mean(dim=("chain", "draw")).values

    posterior_summary: dict[str, Any] = {
        "model_kind": MODEL_KIND,
        "n_obs": n_obs,
        "n_geo": n_geo,
        "channels": channels,
        "mu_channel_mean": {ch: float(mu_post[i]) for i, ch in enumerate(channels)},
        "tau_channel_mean": {ch: float(tau_post[i]) for i, ch in enumerate(channels)},
        "sigma_mean": float(idata.posterior["sigma"].mean(dim=("chain", "draw")).values),
        "outputs_are_diagnostic_only": True,
    }

    convergence_diagnostics: dict[str, Any] = {
        "rhat_max": rhat_max,
        "ess_bulk_min": ess_min,
        "chains": int(idata.posterior.sizes.get("chain", 0)),
        "draws_per_chain": int(idata.posterior.sizes.get("draw", 0)),
    }

    hierarchy_evidence_diagnostics: dict[str, Any] = {
        "h2d_alignment": "partial_pooling_beta_gc",
        "geo_hierarchy_mapping": geo_hierarchy_mapping,
        "calibration_signal_slots": calibration_signals_stub,
        "beta_geo_index_order": list(beta_geo_index_order),
        "channel_index_order": list(channels),
        "beta_geo_channel_mean": {
            str(g): {channels[c]: float(beta_post[g, c]) for c in range(len(channels))} for g in range(n_geo)
        },
        "beta_posterior_coord": "beta[geo_idx, channel_idx] aligned to beta_geo_index_order × channel_index_order",
    }

    pooling_diagnostics: dict[str, Any] = {
        "pooling_mode": "partial",
        "tau_channel_mean": {ch: float(tau_post[i]) for i, ch in enumerate(channels)},
        "mu_channel_mean": {ch: float(mu_post[i]) for i, ch in enumerate(channels)},
    }

    return {
        "model_kind": MODEL_KIND,
        "idata": idata,
        "pymc_model": model,
        "posterior_summary": posterior_summary,
        "convergence_diagnostics": convergence_diagnostics,
        "hierarchy_evidence_diagnostics": hierarchy_evidence_diagnostics,
        "pooling_diagnostics": pooling_diagnostics,
        "calibration_signal_slots": {
            "reserved": True,
            "signals": calibration_signals_stub,
            "likelihood_integrated": False,
        },
        "outputs_are_diagnostic_only": True,
        "production_decision_surface": False,
        "production_recommendation": False,
        "decision_surface": None,
        "optimizer_ready_curves": None,
        "budget_recommendation": None,
    }


def build_diagnostic_trust_from_fit(raw: dict[str, Any]) -> dict[str, Any]:
    from mmm.research.bayes_h3_sandbox.diagnostic_trust import build_diagnostic_trust_stub

    return build_diagnostic_trust_stub(
        posterior_summary=raw.get("posterior_summary"),
        convergence_diagnostics=raw.get("convergence_diagnostics"),
        hierarchy_evidence=raw.get("hierarchy_evidence_diagnostics"),
        pooling_diagnostics=raw.get("pooling_diagnostics"),
        extra={"model_kind": raw.get("model_kind", MODEL_KIND)},
    )
