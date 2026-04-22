"""Sprint 7/12: Ridge recovers positive media effect on synthetic DGP with known positive betas."""

import numpy as np

from mmm.config.schema import CVConfig, Framework, MMMConfig, ModelForm
from mmm.models.ridge_bo.trainer import RidgeBOMMMTrainer
from mmm.utils.synthetic import SyntheticGeoPanelSpec, generate_geo_panel


def test_ridge_first_channel_coef_positive_on_synthetic():
    spec = SyntheticGeoPanelSpec(
        n_geos=4,
        n_weeks=80,
        channels=("search",),
        betas=(0.8,),
        decay=0.55,
        noise=0.02,
    )
    df, schema = generate_geo_panel(spec, seed=2)
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
        cv=CVConfig(mode="rolling", n_splits=2, min_train_weeks=20, horizon_weeks=4),
        ridge_bo={"n_trials": 6},
    )
    tr = RidgeBOMMMTrainer(cfg, schema)
    fit = tr.fit(df)
    coef = np.asarray(fit["artifacts"].coef, dtype=float).ravel()
    assert coef[0] > 0.0, "expected positive recovered weight for sole channel with positive DGP beta"
