
from mmm.config.schema import CVConfig, Framework, MMMConfig, ModelForm
from mmm.models.ridge_bo.trainer import RidgeBOMMMTrainer
from mmm.utils.synthetic import SyntheticGeoPanelSpec, generate_geo_panel


def test_ridge_bo_end_to_end():
    df, schema = generate_geo_panel(SyntheticGeoPanelSpec(n_geos=3, n_weeks=60))
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
        cv=CVConfig(mode="rolling", n_splits=3, min_train_weeks=15, horizon_weeks=3),
        ridge_bo={"n_trials": 4},
    )
    trainer = RidgeBOMMMTrainer(cfg, schema)
    out = trainer.fit(df)
    assert "artifacts" in out
    yhat = trainer.predict(df)
    assert len(yhat) == len(df)
    assert isinstance(out["artifacts"].best_params, dict)
