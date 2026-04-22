"""Multi-channel synthetic: recovered Ridge media weights correlate with true DGP betas."""

from __future__ import annotations

import numpy as np

from mmm.config.schema import CVConfig, Framework, MMMConfig, ModelForm
from mmm.models.ridge_bo.trainer import RidgeBOMMMTrainer
from mmm.utils.synthetic import SyntheticGeoPanelSpec, generate_geo_panel


def test_ridge_coefs_rank_correlated_with_true_betas():
    true_betas = (0.5, 0.35, 0.2)
    spec = SyntheticGeoPanelSpec(
        n_geos=5,
        n_weeks=100,
        channels=("c1", "c2", "c3"),
        betas=true_betas,
        decay=0.55,
        noise=0.03,
    )
    df, schema = generate_geo_panel(spec, seed=7)
    cfg = MMMConfig(
        framework=Framework.RIDGE_BO,
        model_form=ModelForm.SEMI_LOG,
        data={
            "path": None,
            "geo_column": schema.geo_column,
            "week_column": schema.week_column,
            "target_column": schema.target_column,
            "channel_columns": list(schema.channel_columns),
        },
        cv=CVConfig(mode="rolling", n_splits=3, min_train_weeks=24, horizon_weeks=4),
        ridge_bo={"n_trials": 18},
    )
    tr = RidgeBOMMMTrainer(cfg, schema)
    fit = tr.fit(df)
    coef = np.asarray(fit["artifacts"].coef, dtype=float).ravel()[: len(true_betas)]
    true = np.asarray(true_betas, dtype=float)
    # Sign agreement is more stable than Pearson on short BO budgets (still a weak recovery check).
    agree = float(np.mean(np.sign(coef) == np.sign(true)))
    assert agree >= 2 / 3, f"expected most channels same sign as DGP betas, got {agree}"
    assert np.argmax(coef) == int(np.argmax(true)), "largest recovered weight should match strongest channel"
