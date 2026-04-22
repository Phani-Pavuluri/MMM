"""Sprint 8: PPC artifact builder (no PyMC fit required)."""

from types import SimpleNamespace

import numpy as np

from mmm.diagnostics.bayesian_ppc import build_bayesian_predictive_artifact


class FakeInferenceData:
    def groups(self) -> list[str]:
        return ["prior_predictive", "posterior_predictive"]

    def __getitem__(self, key: str):
        arr = np.ones((2, 2, 5)) * (0.5 if key == "prior_predictive" else 1.0)
        return {"obs": SimpleNamespace(values=arr)}


def test_build_bayesian_predictive_artifact_reads_pp_groups():
    idata = FakeInferenceData()
    cfg = SimpleNamespace(bayesian=SimpleNamespace())
    y = np.ones(5)
    out = build_bayesian_predictive_artifact(idata, config=cfg, y_obs=y)
    assert out["prior_predictive"] is not None
    assert out["posterior_predictive_check"] is not None
    assert "mean_abs_gap" in out["posterior_predictive_check"]


def test_build_bayesian_predictive_artifact_config_none_still_runs_pp():
    """Regression: diagnostics must not pass config=None and silently skip PPC."""
    idata = FakeInferenceData()
    y = np.ones(5)
    out = build_bayesian_predictive_artifact(idata, config=None, y_obs=y)
    assert out["posterior_predictive_check"] is not None
    assert "mean_abs_gap" in out["posterior_predictive_check"]
