"""Decision stress diagnostics."""

from __future__ import annotations

from mmm.config.schema import CVConfig, Framework, MMMConfig, ModelForm
from mmm.governance.decision_stress import build_decision_stress_report
from mmm.models.ridge_bo.trainer import RidgeBOMMMTrainer
from mmm.utils.synthetic import SyntheticGeoPanelSpec, generate_geo_panel


def _cfg() -> MMMConfig:
    return MMMConfig(framework=Framework.RIDGE_BO, data={"channel_columns": ["tv", "search"]})


def test_behavioral_stress_with_panel() -> None:
    panel, schema = generate_geo_panel(SyntheticGeoPanelSpec(n_geos=2, n_weeks=30), seed=1)
    cfg = MMMConfig(
        framework=Framework.RIDGE_BO,
        model_form=ModelForm.SEMI_LOG,
        data={
            "geo_column": schema.geo_column,
            "week_column": schema.week_column,
            "target_column": schema.target_column,
            "channel_columns": list(schema.channel_columns),
        },
        cv=CVConfig(mode="rolling", n_splits=2, min_train_weeks=10, horizon_weeks=3),
        ridge_bo={"n_trials": 2},
        budget={"enabled": True, "total_budget": 200.0},
    )
    fit = RidgeBOMMMTrainer(cfg, schema).fit(panel)
    er = {
        "ridge_fit_summary": {
            "coef": list(fit["artifacts"].coef),
            "intercept": list(fit["artifacts"].intercept),
            "model_form": "semi_log",
            "best_params": dict(fit["artifacts"].best_params),
        },
        "governance": {"approved_for_optimization": True},
        "calibration_readiness_report": {"stale_calibration_warning": None},
        "calibration_summary": {"replay_generalization_gap_severity": "low"},
    }
    rep = build_decision_stress_report(cfg, er, panel=panel, schema=schema)
    assert rep["stress_mode"] == "behavioral"
    assert rep["stress_scope"] == "train_time"
    assert "decision_instability_index" in rep
    assert rep["auto_budget_change"] is False


def test_signal_only_without_panel() -> None:
    er = {
        "ridge_fit_summary": {"coef": [0.2, 0.1]},
        "calibration_readiness_report": {"stale_calibration_warning": None},
        "calibration_summary": {"replay_generalization_gap_severity": "severe"},
        "governance": {"approved_for_optimization": True},
    }
    rep = build_decision_stress_report(_cfg(), er)
    assert rep["stress_mode"] == "signal_only"
    assert rep["stress_scope"] == "train_time_signal_only"
    assert rep["stress_severity"] == "critical"
