"""Bayesian MMM with PyMC — partial pooling on channel coefficients per geo."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from mmm.config.schema import BayesianBackend, Framework, MMMConfig, ModelForm, PoolingMode
from mmm.data.schema import PanelSchema, validate_panel
from mmm.diagnostics.bayesian_inference_report import compute_bayesian_decision_diagnostics
from mmm.diagnostics.bayesian_ppc import build_bayesian_predictive_artifact
from mmm.hierarchy.pooling import partial_pooling_indices
from mmm.models.base import BayesianMMMBase
from mmm.transforms.stack import build_channel_features_from_params
from mmm.utils.math import safe_log


@dataclass
class BayesianPosteriorSummary:
    beta_global_mean: np.ndarray
    beta_global_q: dict[str, np.ndarray]
    diagnostics: dict[str, Any]


class BayesianMMMTrainer(BayesianMMMBase):
    def __init__(self, config: MMMConfig, schema: PanelSchema) -> None:
        if config.framework != Framework.BAYESIAN:
            raise ValueError("BayesianMMMTrainer requires framework=bayesian")
        if config.bayesian.backend != BayesianBackend.PYMC:
            raise ValueError("Only pymc backend implemented in this trainer")
        self.config = config
        self.schema = schema
        self._idata: Any = None
        self._summary: BayesianPosteriorSummary | None = None
        self._X_cols: int = 0

    def fit(self, df: pd.DataFrame) -> dict[str, Any]:
        try:
            import pymc as pm  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise ImportError("Install pymc and arviz extras: pip install mmm[bayesian]") from e

        df = validate_panel(df, self.schema)
        decay = float(self.config.transforms.adstock_params.get("decay", 0.5))
        hill_half = float(self.config.transforms.saturation_params.get("half_max", 1.0))
        hill_slope = float(self.config.transforms.saturation_params.get("slope", 2.0))
        X = build_channel_features_from_params(
            df,
            self.schema,
            self.config.transforms,
            decay=decay,
            hill_half=hill_half,
            hill_slope=hill_slope,
        )
        y = df[self.schema.target_column].to_numpy(dtype=float)
        if self.config.model_form == ModelForm.SEMI_LOG:
            y_obs = safe_log(y)
        else:
            X = safe_log(np.maximum(X, 1e-9))
            y_obs = safe_log(y)
        if self.schema.control_columns:
            X = np.column_stack([X] + [df[c].to_numpy(dtype=float) for c in self.schema.control_columns])
        self._X_cols = X.shape[1]
        geo_idx = partial_pooling_indices(df, self.schema)
        n_geo = int(geo_idx.max() + 1)
        p = X.shape[1]
        n_m = len(self.schema.channel_columns)
        n_c = len(self.schema.control_columns)
        assert n_m + n_c == p
        s_m = float(self.config.bayesian.media_coef_sigma)
        s_c = float(self.config.bayesian.control_coef_sigma)

        n_pr = int(getattr(self.config.bayesian, "prior_predictive_draws", 0) or 0)
        n_pp = int(getattr(self.config.bayesian, "posterior_predictive_draws", 0) or 0)
        idata_prior = None
        with pm.Model() as model:
            alpha_geo = pm.Normal("alpha_geo", mu=0.0, sigma=1.0, shape=n_geo)
            sigma = pm.HalfNormal("sigma", sigma=1.0)
            if self.config.pooling == PoolingMode.PARTIAL:
                parts_mu = []
                if n_m > 0:
                    parts_mu.append(pm.HalfNormal("beta_mu_media", sigma=s_m, shape=n_m))
                if n_c > 0:
                    parts_mu.append(pm.Normal("beta_mu_ctrl", mu=0.0, sigma=s_c, shape=n_c))
                beta_mu = pm.Deterministic("beta_mu", pm.math.concatenate(parts_mu))
                tau = pm.HalfNormal("tau", sigma=0.5)
                z = pm.Normal("z", mu=0.0, sigma=1.0, shape=(n_geo, p))
                beta = pm.Deterministic("beta", beta_mu + z * tau)
                mu = alpha_geo[geo_idx] + (X * beta[geo_idx]).sum(axis=-1)
            elif self.config.pooling == PoolingMode.FULL:
                parts_b = []
                if n_m > 0:
                    parts_b.append(pm.HalfNormal("beta_media", sigma=s_m, shape=n_m))
                if n_c > 0:
                    parts_b.append(pm.Normal("beta_ctrl", mu=0.0, sigma=s_c, shape=n_c))
                beta = pm.Deterministic("beta", pm.math.concatenate(parts_b))
                mu = alpha_geo[geo_idx] + pm.math.dot(X, beta)
            else:
                beta = pm.HalfNormal("beta", sigma=s_m, shape=(n_geo, p))
                mu = alpha_geo[geo_idx] + (X * beta[geo_idx]).sum(axis=-1)
            pm.Normal("obs", mu=mu, sigma=sigma, observed=y_obs)
            if n_pr > 0:
                idata_prior = pm.sample_prior_predictive(
                    draws=min(n_pr, 512),
                    random_seed=self.config.bayesian.nuts_seed,
                )
            trace = pm.sample(
                draws=self.config.bayesian.draws,
                tune=self.config.bayesian.tune,
                chains=self.config.bayesian.chains,
                target_accept=self.config.bayesian.target_accept,
                random_seed=self.config.bayesian.nuts_seed,
                progressbar=False,
            )
        idata_out = trace
        if idata_prior is not None:
            try:
                idata_out = idata_prior + trace
            except Exception:
                idata_out = trace
        if n_pp > 0:
            with model:
                pm.sample_posterior_predictive(
                    idata_out,
                    random_seed=self.config.bayesian.nuts_seed,
                    progressbar=False,
                    extend_inferencedata=True,
                )
        self._idata = idata_out
        gv = self.config.extensions.governance
        diag_pack = compute_bayesian_decision_diagnostics(
            idata_out,
            y_obs=y_obs,
            pooling=self.config.pooling,
            max_rhat=gv.bayesian_max_rhat,
            min_ess=gv.bayesian_min_ess_bulk,
            max_divergences=gv.bayesian_max_divergences,
            max_mean_abs_ppc_gap=gv.bayesian_max_mean_abs_ppc_gap,
        )
        post = idata_out.posterior
        if self.config.pooling == PoolingMode.PARTIAL:
            b = np.asarray(post["beta_mu"].stack(sample=("chain", "draw")))
        elif self.config.pooling == PoolingMode.FULL:
            b = np.asarray(post["beta"].stack(sample=("chain", "draw")))
        else:
            bt = np.asarray(post["beta"].stack(sample=("chain", "draw")))
            b = bt.mean(axis=1).T
        self._summary = BayesianPosteriorSummary(
            beta_global_mean=np.mean(b, axis=-1),
            beta_global_q={
                "p10": np.quantile(b, 0.1, axis=-1),
                "p50": np.quantile(b, 0.5, axis=-1),
                "p90": np.quantile(b, 0.9, axis=-1),
            },
            diagnostics={
                "rhat_max": diag_pack.get("rhat_max"),
                "ess_bulk_min": diag_pack.get("ess_bulk_min"),
                "divergences": diag_pack.get("divergences"),
                "posterior_diagnostics_ok": diag_pack.get("posterior_diagnostics_ok"),
                "posterior_predictive_ok": diag_pack.get("posterior_predictive_ok"),
                "var_names_summarized": diag_pack.get("var_names_summarized"),
                "thresholds": diag_pack.get("thresholds"),
            },
        )
        ppc = build_bayesian_predictive_artifact(idata_out, config=self.config, y_obs=y_obs)
        ppc["decision_inference"] = diag_pack
        return {
            "idata": idata_out,
            "summary": self._summary,
            "ppc": ppc,
        }

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self._idata is None:
            raise RuntimeError("fit first")
        df = validate_panel(df, self.schema)
        decay = float(self.config.transforms.adstock_params.get("decay", 0.5))
        hill_half = float(self.config.transforms.saturation_params.get("half_max", 1.0))
        hill_slope = float(self.config.transforms.saturation_params.get("slope", 2.0))
        X = build_channel_features_from_params(
            df,
            self.schema,
            self.config.transforms,
            decay=decay,
            hill_half=hill_half,
            hill_slope=hill_slope,
        )
        if self.config.model_form == ModelForm.LOG_LOG:
            X = safe_log(np.maximum(X, 1e-9))
        if self.schema.control_columns:
            X = np.column_stack([X] + [df[c].to_numpy(dtype=float) for c in self.schema.control_columns])
        geo_idx = partial_pooling_indices(df, self.schema)
        post = self._idata.posterior
        alpha = post["alpha_geo"].mean(dim=("chain", "draw")).to_numpy()
        if self.config.pooling == PoolingMode.FULL:
            beta_vec = post["beta"].mean(dim=("chain", "draw")).to_numpy()
            sigma = float(post["sigma"].mean())
            mu = alpha[geo_idx] + X @ beta_vec
        else:
            beta = post["beta"].mean(dim=("chain", "draw")).to_numpy()
            sigma = float(post["sigma"].mean())
            mu = alpha[geo_idx] + (X * beta[geo_idx]).sum(axis=-1)
        return np.exp(mu + 0.5 * sigma**2)
