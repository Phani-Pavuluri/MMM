"""Tier 2: correlated channel spends; Ridge still assigns positive mass to both."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mmm.config.schema import CVConfig, Framework, MMMConfig, ModelForm
from mmm.data.schema import PanelSchema
from mmm.models.ridge_bo.trainer import RidgeBOMMMTrainer


def test_ridge_positive_coefs_under_correlated_media():
    rng = np.random.default_rng(4)
    n_geos, n_weeks = 4, 90
    rows = []
    base = rng.lognormal(0, 0.2, size=n_weeks)
    for g in range(n_geos):
        noise = rng.normal(0, 0.05, size=n_weeks)
        for t in range(n_weeks):
            c1 = float(base[t] * (1.0 + 0.1 * g) + noise[t])
            c2 = float(0.85 * base[t] + 0.02 * rng.standard_normal())  # correlated with c1
            mu = 2.5 + 0.04 * c1 + 0.035 * c2
            y = float(np.exp(mu + rng.normal(0, 0.05)))
            rows.append({"geo": f"G{g}", "week": t, "rev": y, "m1": c1, "m2": c2})
    df = pd.DataFrame(rows)
    schema = PanelSchema("geo", "week", "rev", ("m1", "m2"))
    cfg = MMMConfig(
        framework=Framework.RIDGE_BO,
        model_form=ModelForm.SEMI_LOG,
        data={
            "path": None,
            "geo_column": "geo",
            "week_column": "week",
            "target_column": "rev",
            "channel_columns": ["m1", "m2"],
        },
        cv=CVConfig(mode="rolling", n_splits=3, min_train_weeks=24, horizon_weeks=4),
        ridge_bo={"n_trials": 12},
    )
    tr = RidgeBOMMMTrainer(cfg, schema)
    fit = tr.fit(df)
    coef = np.asarray(fit["artifacts"].coef, dtype=float).ravel()[:2]
    assert coef[0] > 0 and coef[1] > 0
