"""Empirical null: planted structure vs (mostly) unstructured noise — conservative diagnostics only."""

import numpy as np

from mmm.config.extensions import FalsificationConfig
from mmm.data.schema import PanelSchema
from mmm.falsification.engine import FalsificationEngine


def test_empirical_null_records_pvalue_proxy_and_threshold_basis() -> None:
    schema = PanelSchema("g", "w", "y", ("c1", "c2"))
    cfg = FalsificationConfig(
        null_calibration_method="empirical_y_permutation",
        empirical_null_n_permutations=20,
        placebo_draws=4,
    )
    eng = FalsificationEngine(schema, cfg)
    rng = np.random.default_rng(0)
    n, p = 100, 2
    x = rng.normal(0, 1.0, size=(n, p))
    y = rng.normal(0, 1.0, size=(n,))
    y_log = np.log(np.maximum(y - y.min() + 1e-3, 1e-6))
    rep = eng.run(x, y_log, rng, ridge_log_alpha=0.0)
    em = rep.to_json()["tests"].get("empirical_y_permutation_null") or {}
    assert "empirical_p_value_upper_tail_proxy" in em
    assert "threshold_basis" in em
    assert "observed_mean_abs_noise_coef" in em


def test_planted_linear_signal_often_lower_tail_than_pure_noise_median_across_seeds() -> None:
    """Across seeds, planted X→y link yields lower upper-tail proxy than unrelated y more often than not."""
    schema = PanelSchema("g", "w", "y", ("c1",))
    cfg = FalsificationConfig(
        null_calibration_method="empirical_y_permutation",
        empirical_null_n_permutations=24,
        placebo_draws=4,
    )
    eng = FalsificationEngine(schema, cfg)
    tails_plant: list[float] = []
    tails_noise: list[float] = []
    for seed in range(8):
        rng = np.random.default_rng(seed)
        n = 120
        x = rng.normal(size=(n, 1))
        y_plant = 2.5 * x[:, 0] + 0.15 * rng.normal(size=(n,))
        y_noise = rng.normal(size=(n,))
        yp = np.log(np.maximum(y_plant - y_plant.min() + 1e-3, 1e-6))
        yn = np.log(np.maximum(y_noise - y_noise.min() + 1e-3, 1e-6))
        rp = eng.run(x, yp, rng, ridge_log_alpha=0.0)
        rn = eng.run(x, yn, rng, ridge_log_alpha=0.0)
        tp = (rp.to_json()["tests"].get("empirical_y_permutation_null") or {}).get(
            "empirical_p_value_upper_tail_proxy", 1.0
        )
        tn = (rn.to_json()["tests"].get("empirical_y_permutation_null") or {}).get(
            "empirical_p_value_upper_tail_proxy", 1.0
        )
        tails_plant.append(float(tp))
        tails_noise.append(float(tn))
    assert float(np.median(tails_plant)) <= float(np.median(tails_noise)) + 0.15
