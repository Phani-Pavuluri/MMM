import numpy as np

from mmm.identifiability.engine import IdentifiabilityEngine


def test_identifiability_engine_runs():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    y_lin = (X @ np.array([0.1, 0.2, 0.3]) + rng.normal(0, 0.1, size=200)).astype(float)
    y_log = np.log(np.maximum(y_lin, 1e-6))
    eng = IdentifiabilityEngine()
    r = eng.analyze(X, ["a", "b", "c"], y_log, rng)
    assert "identifiability_score" in r.to_json()
    assert r.max_vif >= 1.0
