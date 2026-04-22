"""Sprint 1: same design-matrix path for fit vs extensions."""

import numpy as np

from mmm.config.schema import CVConfig, Framework, MMMConfig, ModelForm
from mmm.data.panel_order import sort_panel_for_modeling
from mmm.evaluation.feature_pipeline import build_extension_design_bundle
from mmm.features.design_matrix import build_design_matrix, media_design_matrix
from mmm.models.ridge_bo.trainer import RidgeBOMMMTrainer
from mmm.utils.synthetic import SyntheticGeoPanelSpec, generate_geo_panel


def test_extension_x_media_matches_trainer_bundle():
    df, schema = generate_geo_panel(SyntheticGeoPanelSpec(n_geos=3, n_weeks=50))
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
        cv=CVConfig(mode="rolling", n_splits=2, min_train_weeks=12, horizon_weeks=3),
        ridge_bo={"n_trials": 1},
    )
    tr = RidgeBOMMMTrainer(cfg, schema)
    fit = tr.fit(df)
    params = fit["artifacts"].best_params
    panel_s = sort_panel_for_modeling(df, schema)
    _, x_ext = build_extension_design_bundle(panel_s, schema, cfg, fit)
    bundle_fit = build_design_matrix(
        panel_s,
        schema,
        cfg,
        decay=params["decay"],
        hill_half=params["hill_half"],
        hill_slope=params["hill_slope"],
    )
    x_fit = media_design_matrix(bundle_fit, schema)
    assert x_ext.shape == x_fit.shape
    np.testing.assert_allclose(x_ext, x_fit, rtol=0, atol=0)


def test_unsorted_input_same_x_after_sort():
    df, schema = generate_geo_panel(SyntheticGeoPanelSpec(n_geos=2, n_weeks=20))
    df_shuffled = df.sample(frac=1, random_state=99).reset_index(drop=True)
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
        ridge_bo={"n_trials": 1},
    )
    p1 = sort_panel_for_modeling(df, schema)
    p2 = sort_panel_for_modeling(df_shuffled, schema)
    b1, _ = build_extension_design_bundle(p1, schema, cfg, None)
    b2, _ = build_extension_design_bundle(p2, schema, cfg, None)
    np.testing.assert_array_equal(b1.X, b2.X)
