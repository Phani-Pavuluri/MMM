"""Decision diagnostics on a minimal ArviZ object (no PyMC fit)."""

from __future__ import annotations

import numpy as np
import pytest

from mmm.config.schema import PoolingMode
from mmm.diagnostics.bayesian_inference_report import compute_bayesian_decision_diagnostics


def test_compute_bayesian_decision_diagnostics_includes_media_controls():
    az = pytest.importorskip("arviz")
    rng = np.random.default_rng(7)
    chains, draws, n_obs = 2, 30, 12
    idata = az.from_dict(
        posterior={
            "sigma": np.abs(rng.normal(size=(chains, draws))),
            "beta_media": rng.normal(size=(chains, draws, 2)),
            "beta_ctrl": rng.normal(size=(chains, draws, 1)),
            "alpha_geo": rng.normal(size=(chains, draws, 2)),
        },
        posterior_predictive={"obs": rng.normal(size=(chains, draws, n_obs))},
        sample_stats={"diverging": np.zeros((chains, draws), dtype=int)},
    )
    y_obs = rng.normal(size=(n_obs,)).astype(float)
    out = compute_bayesian_decision_diagnostics(
        idata,
        y_obs=y_obs,
        pooling=PoolingMode.FULL,
        config=None,
        max_rhat=2.0,
        min_ess=5.0,
        max_divergences=10,
        max_mean_abs_ppc_gap=None,
    )
    assert "beta_media" in out["var_names_summarized"]
    assert "beta_ctrl" in out["var_names_summarized"]
    assert out["decision_var_names"]
    assert "ppc_artifact" in out
    chk = out["ppc_artifact"].get("posterior_predictive_check") or {}
    assert "mean_abs_gap" in chk or "empirical_coverage_p90" in chk
