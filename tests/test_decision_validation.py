"""Decision validation: prior recommendations vs subsequent experiment evidence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mmm.config.schema import CalibrationConfig, Framework, MMMConfig
from mmm.validation.decision_validation import build_decision_validation_report


def _evidence(**kwargs: object) -> dict:
    defaults = dict(
        experiment_id="exp-1",
        experiment_type="geox",
        channel="tv",
        kpi="revenue",
        estimand="ATT",
        lift_estimate=0.4,
        standard_error=0.1,
        time_window={"start": "2024-01-01", "end": "2024-06-01"},
        geo_scope=["US"],
        geo_granularity="national",
        source_system="test",
        freshness_date="2024-06-15",
        approval_status="accepted",
    )
    defaults.update(kwargs)
    return defaults


def _cfg(
    tmp_path: Path,
    *,
    enabled: bool = True,
    decision_dir: str | None = None,
) -> MMMConfig:
    ev_path = str(tmp_path / "evidence.json")
    return MMMConfig(
        framework=Framework.RIDGE_BO,
        data={"channel_columns": ["tv", "search"]},
        calibration=CalibrationConfig(evidence_registry_path=ev_path),
        extensions={
            "decision_validation": {
                "enabled": enabled,
                "decision_registry_dir": decision_dir,
                "experiment_registry_path": ev_path,
            },
        },
    )


def test_disabled_by_default() -> None:
    rep = build_decision_validation_report(
        MMMConfig(framework=Framework.RIDGE_BO, data={"channel_columns": ["tv"]})
    )
    assert rep["enabled"] is False
    assert rep.get("skipped") is True
    assert rep["diagnostic_only"] is True
    assert rep["auto_budget_change"] is False
    assert rep["auto_optimizer_change"] is False
    assert rep["decision_safe"] is False


def test_no_decision_registry_not_evaluable(tmp_path: Path) -> None:
    (tmp_path / "evidence.json").write_text(json.dumps([_evidence()]), encoding="utf-8")
    cfg = _cfg(tmp_path, decision_dir=None)
    rep = build_decision_validation_report(cfg)
    assert rep.get("skipped") is True
    assert rep.get("reason") == "decision_registry_dir_not_set"


def test_predicted_vs_measured_lift(tmp_path: Path) -> None:
    (tmp_path / "evidence.json").write_text(
        json.dumps([_evidence(lift_estimate=0.4, freshness_date="2024-06-15")]),
        encoding="utf-8",
    )
    dec_dir = tmp_path / "decisions"
    dec_dir.mkdir()
    (dec_dir / "decisions.json").write_text(
        json.dumps(
            {
                "registry_version": "mmm_decision_validation_registry_v1",
                "decisions": [
                    {
                        "decision_id": "d1",
                        "decided_at": "2024-03-01",
                        "recommended_allocation": {"tv": 100.0, "search": 50.0},
                        "predicted_lifts_by_channel": {"tv": 0.5},
                        "channel_ranking": ["tv", "search"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    cfg = _cfg(tmp_path, decision_dir=str(dec_dir))
    rep = build_decision_validation_report(cfg)
    assert rep.get("skipped") is False
    assert rep["n_decision_experiment_pairs"] >= 1
    row = next(r for r in rep["per_decision_results"] if r.get("classification") == "evaluated")
    assert row["prediction_error"] == pytest.approx(0.1)
    assert row["realized_experimental_lift"] == pytest.approx(0.4)


def test_ranking_stability_calculation(tmp_path: Path) -> None:
    (tmp_path / "evidence.json").write_text(
        json.dumps(
            [
                _evidence(channel="tv", lift_estimate=0.9, experiment_id="e1"),
                _evidence(
                    channel="search",
                    lift_estimate=0.1,
                    experiment_id="e2",
                    freshness_date="2024-06-20",
                ),
            ]
        ),
        encoding="utf-8",
    )
    dec_dir = tmp_path / "decisions"
    dec_dir.mkdir()
    (dec_dir / "decisions.json").write_text(
        json.dumps(
            {
                "decisions": [
                    {
                        "decision_id": "d1",
                        "decided_at": "2024-03-01",
                        "recommended_allocation": {"tv": 80.0, "search": 20.0},
                        "predicted_lifts_by_channel": {"tv": 0.5, "search": 0.5},
                        "channel_ranking": ["tv", "search"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    cfg = _cfg(tmp_path, decision_dir=str(dec_dir))
    rep = build_decision_validation_report(cfg)
    assert rep["ranking_stability"]["mean_top3_overlap"] is not None


def test_allocation_regret_proxy(tmp_path: Path) -> None:
    (tmp_path / "evidence.json").write_text(
        json.dumps([_evidence(channel="tv", lift_estimate=1.0)]),
        encoding="utf-8",
    )
    dec_dir = tmp_path / "decisions"
    dec_dir.mkdir()
    (dec_dir / "decisions.json").write_text(
        json.dumps(
            {
                "decisions": [
                    {
                        "decision_id": "d1",
                        "decided_at": "2024-03-01",
                        "recommended_allocation": {"tv": 10.0, "search": 90.0},
                        "predicted_lifts_by_channel": {"tv": 0.5},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    cfg = _cfg(tmp_path, decision_dir=str(dec_dir))
    rep = build_decision_validation_report(cfg)
    assert rep["allocation_regret_proxy"]["mean"] is not None
    assert rep["allocation_regret_proxy"]["mean"] > 0


def test_observational_evidence_not_evaluable(tmp_path: Path) -> None:
    (tmp_path / "evidence.json").write_text(
        json.dumps(
            [
                _evidence(
                    metadata={"validation_design": "observational"},
                )
            ]
        ),
        encoding="utf-8",
    )
    dec_dir = tmp_path / "decisions"
    dec_dir.mkdir()
    (dec_dir / "decisions.json").write_text(
        json.dumps(
            {
                "decisions": [
                    {
                        "decision_id": "d1",
                        "decided_at": "2024-03-01",
                        "recommended_allocation": {"tv": 100.0},
                        "predicted_lifts_by_channel": {"tv": 0.5},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    cfg = _cfg(tmp_path, decision_dir=str(dec_dir))
    rep = build_decision_validation_report(cfg)
    row = rep["per_decision_results"][0]
    assert row["classification"] == "not_evaluable"
    assert "observational" in row["not_evaluable_reason"]


def test_diagnostic_only_flags_always_true(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, enabled=True, decision_dir=str(tmp_path / "none"))
    rep = build_decision_validation_report(cfg)
    assert rep["diagnostic_only"] is True
    assert rep["research_only"] is True
    assert rep["prod_decisioning_allowed"] is False
    assert rep["decision_safe"] is False


def test_extension_runner_key_only_when_enabled() -> None:
    from mmm.config.schema import CVConfig, ModelForm
    from mmm.evaluation.extension_runner import run_post_fit_extensions
    from mmm.models.ridge_bo.trainer import RidgeBOMMMTrainer
    from mmm.utils.synthetic import SyntheticGeoPanelSpec, generate_geo_panel

    df, schema = generate_geo_panel(SyntheticGeoPanelSpec(n_geos=2, n_weeks=40))
    cfg = MMMConfig(
        framework=Framework.RIDGE_BO,
        model_form=ModelForm.SEMI_LOG,
        data={
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
    rep = run_post_fit_extensions(
        panel=df, schema=schema, config=cfg, fit_out=fit, yhat=tr.predict(df), store=None
    )
    assert "decision_validation_report" not in rep
