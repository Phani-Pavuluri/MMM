"""Ridge BO applies per-fold train masks; validation rows excluded from fit loss."""

from __future__ import annotations

import numpy as np

from mmm.config.schema import CVConfig, Framework, MMMConfig, ModelForm
from mmm.features.design_matrix import apply_masks_for_fit, build_design_matrix
from mmm.models.ridge_bo.ridge import fit_ridge, predict_ridge
from mmm.utils.synthetic import SyntheticGeoPanelSpec, generate_geo_panel
from mmm.validation.cv import auto_cv_mode


def test_validation_rows_not_in_fold_fit() -> None:
    spec = SyntheticGeoPanelSpec(n_geos=2, n_weeks=50, channels=("search",), betas=(0.5,))
    df, schema = generate_geo_panel(spec, seed=0)
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
        cv=CVConfig(mode="rolling", n_splits=2, min_train_weeks=18, horizon_weeks=4),
    )
    cv = auto_cv_mode(df, schema, cfg.cv)
    splits = cv.split(df, schema)
    bundle = build_design_matrix(df, schema, cfg, decay=0.5, hill_half=1.0, hill_slope=2.0)
    train_mask, val_mask = splits[0]
    assert not np.any(train_mask & val_mask)
    X_tr, y_tr = apply_masks_for_fit(bundle, train_mask)
    coef, intercept = fit_ridge(X_tr, y_tr, 1.0)
    assert X_tr.shape[0] == int(train_mask.sum())
    assert X_tr.shape[0] < bundle.X.shape[0]
    yhat_va = predict_ridge(bundle.X[val_mask], coef, intercept)
    assert len(yhat_va) == int(val_mask.sum())
