"""Full-panel decision simulate() vs BAU baseline."""

from __future__ import annotations

from mmm.config.schema import CVConfig, DataConfig, Framework, MMMConfig, ModelForm
from mmm.planning import bau_baseline_from_panel, simulate
from mmm.planning.context import ridge_context_from_fit
from mmm.models.ridge_bo.trainer import RidgeBOMMMTrainer
from mmm.utils.synthetic import SyntheticGeoPanelSpec, generate_geo_panel


def test_simulate_delta_mu_matches_bau_when_plan_equals_bau() -> None:
    df, schema = generate_geo_panel(
        SyntheticGeoPanelSpec(n_geos=2, n_weeks=30, channels=("a", "b"), betas=(0.5, 0.5))
    )
    cfg = MMMConfig(
        framework=Framework.RIDGE_BO,
        model_form=ModelForm.SEMI_LOG,
        data=DataConfig(
            path=None,
            geo_column=schema.geo_column,
            week_column=schema.week_column,
            target_column=schema.target_column,
            channel_columns=list(schema.channel_columns),
            control_columns=[],
        ),
        cv=CVConfig(mode="rolling", n_splits=2, min_train_weeks=10, horizon_weeks=2),
        ridge_bo={"n_trials": 2},
    )
    tr = RidgeBOMMMTrainer(cfg, schema)
    fit = tr.fit(df)
    ctx = ridge_context_from_fit(df, schema, cfg, fit)
    bau = bau_baseline_from_panel(df, schema)
    sim = simulate(dict(bau.spend_by_channel), ctx, baseline_plan=bau, uncertainty_mode="point")
    assert abs(sim.delta_mu) < 1e-6
    assert sim.decision_safe
    assert sim.baseline_type == "bau"
    j = sim.to_json()
    assert j["planner_mode"] == "full_model"
    assert j["economics_version"]
    assert j["counterfactual_construction_method"]
    assert j["spend_path_assumption"]
    assert j["aggregation_semantics"]
    assert j["candidate_plan_type"] == "constant_channel_levels"
    assert j["kpi_column"] == schema.target_column
    assert "economics_output_metadata" in j
    em = j["economics_output_metadata"]
    assert em["target_kpi_column"] == schema.target_column
    assert em["economics_version"]
    assert em["computation_mode"] in ("exact", "approximate", "unknown")
    assert em["baseline_type"] == "bau"
    assert isinstance(em["decision_safe"], bool)
