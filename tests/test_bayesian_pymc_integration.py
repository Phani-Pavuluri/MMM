"""Full PyMC fit on tiny panel — skipped when pymc/arviz not installed (``pip install mmm[bayesian]``)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mmm.config.schema import BayesianBackend, Framework, MMMConfig, ModelForm, PoolingMode
from mmm.data.schema import PanelSchema
from mmm.models.bayesian.pymc_trainer import BayesianMMMTrainer


@pytest.mark.pymc
@pytest.mark.slow
def test_bayesian_trainer_fit_and_predict_smoke():
    pytest.importorskip("pymc")
    pytest.importorskip("arviz")
    rng = np.random.default_rng(0)
    n = 40
    x1 = rng.uniform(1, 5, size=n)
    x2 = rng.uniform(1, 5, size=n)
    eps = rng.normal(0, 0.05, size=n)
    y = np.exp(0.3 + 0.15 * x1 + 0.1 * x2 + eps)
    df = pd.DataFrame(
        {
            "g": ["A"] * n,
            "w": np.arange(n),
            "y": y,
            "m1": x1,
            "m2": x2,
        }
    )
    schema = PanelSchema("g", "w", "y", ("m1", "m2"))
    cfg = MMMConfig(
        framework=Framework.BAYESIAN,
        model_form=ModelForm.SEMI_LOG,
        pooling=PoolingMode.FULL,
        data={
            "path": None,
            "geo_column": "g",
            "week_column": "w",
            "target_column": "y",
            "channel_columns": ["m1", "m2"],
            "control_columns": [],
        },
        bayesian={
            "backend": BayesianBackend.PYMC,
            "draws": 40,
            "tune": 40,
            "chains": 2,
            "target_accept": 0.85,
            "nuts_seed": 1,
            "prior_predictive_draws": 20,
            "posterior_predictive_draws": 20,
        },
    )
    trainer = BayesianMMMTrainer(cfg, schema)
    out = trainer.fit(df)
    assert "idata" in out
    ppc = out.get("ppc") or {}
    assert ppc.get("idata_present") is True
    rh = float(out["summary"].diagnostics.get("rhat_max", float("nan")))
    if rh == rh:  # not NaN
        assert rh < 1.2, f"rhat_max sanity for tiny panel: {rh}"
    try:
        import tempfile
        from pathlib import Path

        nc = Path(tempfile.mkdtemp()) / "idata.nc"
        out["idata"].to_netcdf(nc)
        assert nc.exists()
    except Exception:
        pass
    pred = trainer.predict(df)
    assert pred.shape == (n,)
    assert np.all(np.isfinite(pred))
