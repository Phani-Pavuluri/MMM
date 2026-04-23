from mmm.config.schema import CVConfig, Framework, MMMConfig, ModelForm
from mmm.evaluation.extension_runner import run_post_fit_extensions
from mmm.models.ridge_bo.trainer import RidgeBOMMMTrainer
from mmm.utils.synthetic import SyntheticGeoPanelSpec, generate_geo_panel


def test_extension_runner_after_ridge():
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
        ridge_bo={"n_trials": 2},
    )
    tr = RidgeBOMMMTrainer(cfg, schema)
    fit = tr.fit(df)
    yhat = tr.predict(df)
    rep = run_post_fit_extensions(
        panel=df,
        schema=schema,
        config=cfg,
        fit_out=fit,
        yhat=yhat,
        store=None,
    )
    assert "identifiability" in rep
    assert "governance" in rep
    assert "baselines" in rep
    assert "ridge_fit_summary" in rep
    assert rep["ridge_fit_summary"]["best_params"]
    assert "decision_policy" in rep
    assert rep["decision_policy"]["planner_mode"] == "full_model"
    em = rep["economics_output_metadata"]
    assert em["economics_version"]
    assert em["computation_mode"] in ("exact", "approximate", "unknown")
    assert em["baseline_type"] == "extension_train_reference"
    assert "decision_bundle" in rep
    assert rep["decision_bundle"]["economics_output_metadata"]["baseline_type"] == "extension_train_reference"
    assert "panel_qa" in rep
    assert rep["panel_qa"]["panel_qa_version"] == "mmm_panel_qa_v1"
    assert "model_release" in rep
    assert rep["model_release"]["state"]
    assert "run_manifest" in rep
    assert rep["run_manifest"]["manifest_version"]
    assert "post_fit_validation" in rep
    assert "operational_health" in rep
    assert rep["operational_health"]["status"] in ("healthy", "warning", "blocked")
