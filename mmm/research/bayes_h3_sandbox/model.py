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
H5_MODEL_SPEC_VERSION = "bayes_h5_sandbox_spec_v1"
H5_MODEL_KIND = "bayes_h5_hierarchical_sandbox_v1"


def _convergence_parameter_family(parameter: str) -> str:
    name = str(parameter)
    if name.startswith("alpha_geo"):
        return "intercept"
    if name.startswith("tau_channel"):
        return "tau"
    if name.startswith("mu_channel"):
        return "mu_channel"
    if name.startswith("sigma"):
        return "sigma"
    if name.startswith("z_beta"):
        return "beta_offset"
    if name.startswith("beta_channel"):
        return "beta_channel"
    if name.startswith("beta"):
        return "beta"
    return "other"


def _summary_var_names_for_geometry(geometry: dict[str, Any]) -> list[str]:
    from mmm.research.bayes_h3_sandbox.h5_geometry_config import (
        HIERARCHY_FIXED_TAU,
        HIERARCHY_FULL_GEO_CHANNEL,
        HIERARCHY_POOLED_CHANNEL,
        PARAMETERIZATION_NON_CENTERED,
        TAU_PARAM_LOG_TAU,
        TAU_PARAM_NONCENTERED_LOG_TAU,
    )

    hier = geometry.get("hierarchy_policy", HIERARCHY_FULL_GEO_CHANNEL)
    param = geometry.get("parameterization", PARAMETERIZATION_NON_CENTERED)
    tau_param = geometry.get("tau_parameterization", "current")
    names = ["alpha_geo", "sigma"]
    if hier == HIERARCHY_POOLED_CHANNEL:
        names.append("beta_channel")
        return names
    names.append("mu_channel")
    if hier == HIERARCHY_FIXED_TAU:
        names.append("tau_channel")
    elif tau_param == TAU_PARAM_LOG_TAU:
        names.append("log_tau_channel")
    elif tau_param == TAU_PARAM_NONCENTERED_LOG_TAU:
        names.extend(["mu_log_tau", "z_log_tau"])
    else:
        names.append("tau_channel")
    if param == PARAMETERIZATION_NON_CENTERED:
        names.append("z_beta")
    else:
        names.append("beta")
    return names


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
    geometry: dict[str, Any] | None = None,
) -> dict[str, Any]:
    import arviz as az  # type: ignore

    from mmm.research.bayes_h3_sandbox.h5_geometry_config import (
        HIERARCHY_POOLED_CHANNEL,
        resolve_geometry_config,
    )

    geom = geometry or resolve_geometry_config(sandbox_model_overrides)
    var_names = _summary_var_names_for_geometry(geom)
    summary = az.summary(idata, var_names=var_names)
    rhat_max = float(summary["r_hat"].max()) if "r_hat" in summary.columns and len(summary) else float("nan")
    ess_min = float(summary["ess_bulk"].min()) if "ess_bulk" in summary.columns and len(summary) else float("nan")
    per_parameter: list[dict[str, Any]] = []
    for param_name, row in summary.iterrows():
        per_parameter.append(
            {
                "parameter": str(param_name),
                "family": _convergence_parameter_family(str(param_name)),
                "r_hat": float(row["r_hat"]) if "r_hat" in row and pd.notna(row["r_hat"]) else None,
                "ess_bulk": float(row["ess_bulk"]) if "ess_bulk" in row and pd.notna(row["ess_bulk"]) else None,
            }
        )
    worst_rhat = sorted(
        [p for p in per_parameter if p.get("r_hat") is not None],
        key=lambda p: float(p["r_hat"]),
        reverse=True,
    )[:8]

    sigma_mean = float(idata.posterior["sigma"].mean(dim=("chain", "draw")).values)
    posterior_summary: dict[str, Any] = {
        "model_kind": MODEL_KIND,
        "n_obs": n_obs,
        "n_geo": n_geo,
        "channels": channels,
        "sigma_mean": sigma_mean,
        "outputs_are_diagnostic_only": True,
    }
    if geom.get("hierarchy_policy") == HIERARCHY_POOLED_CHANNEL:
        beta_ch = idata.posterior["beta_channel"].mean(dim=("chain", "draw")).values
        posterior_summary["beta_channel_mean"] = {ch: float(beta_ch[i]) for i, ch in enumerate(channels)}
        beta_post = np.broadcast_to(beta_ch, (n_geo, len(channels)))
    else:
        mu_post = idata.posterior["mu_channel"].mean(dim=("chain", "draw")).values
        tau_post = idata.posterior["tau_channel"].mean(dim=("chain", "draw")).values
        beta_post = idata.posterior["beta"].mean(dim=("chain", "draw")).values
        posterior_summary["mu_channel_mean"] = {ch: float(mu_post[i]) for i, ch in enumerate(channels)}
        posterior_summary["tau_channel_mean"] = {ch: float(tau_post[i]) for i, ch in enumerate(channels)}

    divergence_count = 0
    if hasattr(idata, "sample_stats") and "diverging" in getattr(idata, "sample_stats", {}):
        try:
            divergence_count = int(idata.sample_stats["diverging"].sum().values)
        except (TypeError, ValueError, AttributeError):
            divergence_count = 0

    convergence_diagnostics: dict[str, Any] = {
        "rhat_max": rhat_max,
        "ess_bulk_min": ess_min,
        "divergence_count": divergence_count,
        "chains": int(idata.posterior.sizes.get("chain", 0)),
        "draws_per_chain": int(idata.posterior.sizes.get("draw", 0)),
        "per_parameter": per_parameter,
        "worst_rhat_parameters": worst_rhat,
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

    if geom.get("hierarchy_policy") == HIERARCHY_POOLED_CHANNEL:
        pooling_diagnostics = {
            "pooling_mode": "pooled_channel_ablation",
            "beta_channel_mean": posterior_summary.get("beta_channel_mean", {}),
        }
    else:
        mu_post = idata.posterior["mu_channel"].mean(dim=("chain", "draw")).values
        tau_post = idata.posterior["tau_channel"].mean(dim=("chain", "draw")).values
        pooling_diagnostics = {
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
        "h5_geometry_diagnostics": None,
    }


def fit_h5_sandbox_hierarchical(
    config: MMMConfig,
    schema: PanelSchema,
    df: pd.DataFrame,
    *,
    geo_hierarchy_mapping: dict[str, Any] | None = None,
    calibration_signals_stub: list[dict[str, Any]] | None = None,
    sandbox_model_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Bayes-H5 sandbox fit: H3 partial-pooling structure with per-channel media transforms.

    Research only — requires explicit sandbox gating at entrypoint.
    """
    try:
        import pymc as pm  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise ImportError("Bayes-H5 sandbox requires pymc: pip install mmm[bayesian]") from e

    from mmm.research.bayes_h3_sandbox.h5_transforms import (
        TRANSFORM_REGISTRY_ID,
        apply_media_transforms_matrix,
        compute_transform_mismatch_detected,
        transforms_aligned,
    )

    df = validate_panel(df, schema, integrity_qa=False, calendar_strict=False)
    df = sort_panel_for_modeling(df, schema)
    beta_geo_index_order = df[schema.geo_column].unique().tolist()
    geo_idx = partial_pooling_indices(df, schema)
    n_geo = int(geo_idx.max() + 1)
    channels = list(schema.channel_columns)
    n_c = len(channels)
    from mmm.research.bayes_h3_sandbox.h5_geometry_config import (
        BETA_PRIOR_CHANNEL_SCALED,
        HIERARCHY_FIXED_TAU,
        HIERARCHY_FULL_GEO_CHANNEL,
        HIERARCHY_POOLED_CHANNEL,
        HIERARCHY_STRENGTH_FIXED_TAU,
        PARAMETERIZATION_CENTERED,
        PARAMETERIZATION_NON_CENTERED,
        TAU_PARAM_LOG_TAU,
        TAU_PARAM_NONCENTERED_LOG_TAU,
        apply_geometry_priors,
        geometry_record_for_artifact,
        resolve_geometry_config,
    )

    overrides = dict(sandbox_model_overrides or {})
    geometry = resolve_geometry_config(overrides)
    transforms_by_channel: dict[str, str] = dict(overrides.get("media_transforms_by_channel") or {})
    if not transforms_by_channel:
        transforms_by_channel = {ch: "identity" for ch in channels}
    transform_params_by_channel: dict[str, dict[str, Any]] = dict(
        overrides.get("transform_params_by_channel") or {}
    )

    work = df.copy()
    if overrides.get("media_prescale") == "zscore_panel":
        for ch in channels:
            col = work[ch].to_numpy(dtype=float)
            work[ch] = (col - col.mean()) / (col.std() + 1e-6)

    raw_x = work[list(channels)].to_numpy(dtype=float)
    x = apply_media_transforms_matrix(
        raw_x,
        channels,
        transforms_by_channel,
        transform_params_by_channel=transform_params_by_channel,
    )
    channel_scales = (np.std(x, axis=0) + 1e-6).tolist()
    overrides = apply_geometry_priors(overrides, geometry, channel_scales=channel_scales)

    y = work[schema.target_column].to_numpy(dtype=float)
    if config.model_form == ModelForm.SEMI_LOG:
        y_obs = safe_log(y)
    else:
        y_obs = y.astype(float)
    if overrides.get("outcome_prescale") == "zscore_log":
        y_obs = (y_obs - float(np.mean(y_obs))) / (float(np.std(y_obs)) + 1e-6)

    draws = int(config.bayesian.draws)
    tune = int(config.bayesian.tune)
    chains = int(config.bayesian.chains)
    nuts_seed = int(config.bayesian.nuts_seed)
    tau_prior_sigma = float(overrides.get("tau_channel_prior_sigma", 0.5))
    mu_prior_sigma = float(overrides.get("mu_channel_prior_sigma", 0.5))
    z_beta_sigma = float(overrides.get("z_beta_prior_sigma", 1.0))
    sigma_prior_sigma = float(overrides.get("sigma_prior_sigma", 1.0))
    beta_channel_scales = np.asarray(overrides.get("beta_channel_scales") or [1.0] * n_c, dtype=float)
    if len(beta_channel_scales) != n_c:
        beta_channel_scales = np.ones(n_c, dtype=float)

    from mmm.research.bayes_h3_sandbox.h5_transforms import is_real_panel_generative

    panel_context = str(overrides.get("h5_panel_context", ""))
    gen_transform = str(overrides.get("h5_generative_transform", "linear"))
    if panel_context == "real_panel" or overrides.get("h5_real_panel"):
        gen_transform = "real_panel"
    mismatch_mode = str(overrides.get("h5_transform_mismatch_mode", "aligned"))
    fitted_uniform = {ch: transforms_by_channel.get(ch, "identity") for ch in channels}
    fitted_id = next(iter(fitted_uniform.values()), "identity")
    if is_real_panel_generative(gen_transform):
        aligned = None
        generative_reported = "unknown"
    else:
        aligned = transforms_aligned(gen_transform, fitted_id)
        generative_reported = gen_transform
    transform_mismatch_detected = compute_transform_mismatch_detected(
        gen_transform,
        fitted_id,
        transform_mismatch_mode=mismatch_mode,
    )

    hier_policy = geometry.get("hierarchy_policy", HIERARCHY_FULL_GEO_CHANNEL)
    param = geometry.get("parameterization", PARAMETERIZATION_NON_CENTERED)
    tau_param = geometry.get("tau_parameterization", "current")
    strength = geometry.get("hierarchy_strength_policy", "learned_tau")
    sigma_floor = float(overrides.get("sigma_floor", 0.0))
    use_fixed_tau = hier_policy == HIERARCHY_FIXED_TAU or strength == HIERARCHY_STRENGTH_FIXED_TAU

    with pm.Model() as model:
        alpha_geo = pm.Normal("alpha_geo", mu=0.0, sigma=1.0, shape=n_geo)
        sigma_raw = pm.HalfNormal("sigma", sigma=sigma_prior_sigma)
        if sigma_floor > 0:
            sigma_like = pm.Deterministic("sigma_eff", pm.math.maximum(sigma_raw, sigma_floor))
        else:
            sigma_like = sigma_raw

        if hier_policy == HIERARCHY_POOLED_CHANNEL:
            beta_ch = pm.Normal("beta_channel", mu=0.0, sigma=0.5, shape=n_c)
            mu = alpha_geo[geo_idx] + (x * beta_ch).sum(axis=-1)
        else:
            mu_c = pm.Normal("mu_channel", mu=0.0, sigma=mu_prior_sigma, shape=n_c)
            if use_fixed_tau:
                tau_fixed = float(geometry.get("fixed_tau_value", 0.2))
                tau_c = pm.Deterministic("tau_channel", pm.math.ones(n_c) * tau_fixed)
            elif tau_param == TAU_PARAM_LOG_TAU:
                log_tau = pm.Normal(
                    "log_tau_channel",
                    mu=float(np.log(max(tau_prior_sigma, 0.05))),
                    sigma=0.5,
                    shape=n_c,
                )
                tau_c = pm.Deterministic("tau_channel", pm.math.exp(log_tau))
            elif tau_param == TAU_PARAM_NONCENTERED_LOG_TAU:
                mu_log = pm.Normal("mu_log_tau", mu=float(np.log(max(tau_prior_sigma, 0.05))), sigma=0.3)
                z_log = pm.Normal("z_log_tau", mu=0.0, sigma=1.0, shape=n_c)
                log_tau = mu_log + 0.5 * z_log
                tau_c = pm.Deterministic("tau_channel", pm.math.exp(log_tau))
            else:
                tau_c = pm.HalfNormal("tau_channel", sigma=tau_prior_sigma, shape=n_c)
            if geometry.get("beta_prior_policy") == BETA_PRIOR_CHANNEL_SCALED:
                scale_vec = pm.math.constant(beta_channel_scales)
            else:
                scale_vec = 1.0
            if param == PARAMETERIZATION_CENTERED:
                beta = pm.Normal("beta", mu=mu_c, sigma=tau_c / scale_vec, shape=(n_geo, n_c))
            else:
                z = pm.Normal("z_beta", mu=0.0, sigma=z_beta_sigma, shape=(n_geo, n_c))
                beta = pm.Deterministic("beta", mu_c + (z * tau_c) / scale_vec)
            mu = alpha_geo[geo_idx] + (x * beta[geo_idx]).sum(axis=-1)

        pm.Normal("y_obs", mu=mu, sigma=sigma_like, observed=y_obs)

        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=float(config.bayesian.target_accept),
            random_seed=nuts_seed,
            progressbar=False,
            return_inferencedata=True,
        )

    out = _package_mvp_fit(
        idata,
        model=model,
        channels=channels,
        geo_idx=geo_idx,
        n_geo=n_geo,
        n_obs=len(work),
        geo_hierarchy_mapping=geo_hierarchy_mapping or {},
        calibration_signals_stub=calibration_signals_stub or [],
        beta_geo_index_order=beta_geo_index_order,
        sandbox_model_overrides=overrides,
        geometry=geometry,
    )
    out["h5_geometry_diagnostics"] = geometry_record_for_artifact(geometry)
    out["model_kind"] = H5_MODEL_KIND
    out["model_spec_version"] = H5_MODEL_SPEC_VERSION
    out["posterior_summary"]["model_kind"] = H5_MODEL_KIND
    out["posterior_summary"]["model_spec_version"] = H5_MODEL_SPEC_VERSION
    out["h5_transform_diagnostics"] = {
        "transform_registry_id": TRANSFORM_REGISTRY_ID,
        "media_transforms_by_channel": dict(transforms_by_channel),
        "generative_transform_expected": generative_reported,
        "panel_context": panel_context or ("real_panel" if is_real_panel_generative(gen_transform) else "synthetic_world"),
        "transform_mismatch_mode": mismatch_mode,
        "transform_mismatch_detected": transform_mismatch_detected,
        "transforms_aligned": aligned,
        "research_only": True,
    }
    return out


def build_diagnostic_trust_from_fit(raw: dict[str, Any]) -> dict[str, Any]:
    from mmm.research.bayes_h3_sandbox.diagnostic_trust import build_diagnostic_trust_stub

    return build_diagnostic_trust_stub(
        posterior_summary=raw.get("posterior_summary"),
        convergence_diagnostics=raw.get("convergence_diagnostics"),
        hierarchy_evidence=raw.get("hierarchy_evidence_diagnostics"),
        pooling_diagnostics=raw.get("pooling_diagnostics"),
        extra={"model_kind": raw.get("model_kind", MODEL_KIND)},
    )
