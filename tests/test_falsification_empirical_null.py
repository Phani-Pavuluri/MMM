"""Falsification empirical-null mode (permuted-y null distribution)."""

import numpy as np

from mmm.config.extensions import FalsificationConfig
from mmm.data.schema import PanelSchema
from mmm.falsification.engine import FalsificationEngine


def test_empirical_y_permutation_null_reports_metadata() -> None:
    schema = PanelSchema("g", "w", "y", ("c1", "c2"))
    cfg = FalsificationConfig(
        null_calibration_method="empirical_y_permutation",
        empirical_null_n_permutations=16,
        placebo_draws=4,
    )
    eng = FalsificationEngine(schema, cfg)
    rng = np.random.default_rng(42)
    n, p = 80, 2
    x = rng.normal(0, 1.0, size=(n, p))
    y = 0.35 * x[:, 0] + 0.05 * rng.normal(size=(n,))
    y_log = np.log(np.maximum(y, 1e-6))
    rep = eng.run(x, y_log, rng, ridge_log_alpha=0.0, geo_ids=None)
    js = rep.to_json()
    assert js["null_calibration_method"] == "empirical_y_permutation"
    assert "empirical_y_permutation_null" in js["tests"]
    assert int(js["tests"]["empirical_y_permutation_null"]["n_permutations"]) >= 4


def test_fixed_scale_floor_still_populates_method() -> None:
    schema = PanelSchema("g", "w", "y", ("c1",))
    cfg = FalsificationConfig(null_calibration_method="fixed_scale_floor", placebo_draws=3)
    eng = FalsificationEngine(schema, cfg)
    rng = np.random.default_rng(1)
    x = rng.normal(size=(40, 1))
    y = rng.normal(size=(40,))
    rep = eng.run(x, y, rng, ridge_log_alpha=0.0)
    assert rep.to_json()["null_calibration_method"] == "fixed_scale_floor"
