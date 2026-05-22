"""Continuous validation: prior model predictions vs experiment evidence (diagnostic only)."""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import pytest

from mmm.config.schema import CalibrationConfig, Framework, MMMConfig
from mmm.validation.continuous_validation import build_continuous_validation_report


def _evidence(**kwargs: object) -> dict:
    defaults = dict(
        experiment_id="exp-1",
        experiment_type="geox",
        channel="tv",
        kpi="revenue",
        estimand="ATT",
        lift_estimate=0.5,
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
    registry_dir: str | None = None,
    evidence_path: str | None = None,
    require_se: bool = False,
) -> MMMConfig:
    ev_file = evidence_path or str(tmp_path / "evidence.json")
    return MMMConfig(
        framework=Framework.RIDGE_BO,
        data={"channel_columns": ["tv"]},
        calibration=CalibrationConfig(evidence_registry_path=ev_file),
        extensions={
            "continuous_validation": {
                "enabled": enabled,
                "registry_dir": registry_dir,
                "require_experiment_se": require_se,
            },
        },
    )


def test_disabled_by_default() -> None:
    cfg = MMMConfig(framework=Framework.RIDGE_BO, data={"channel_columns": ["tv"]})
    rep = build_continuous_validation_report(cfg)
    assert rep["enabled"] is False
    assert rep.get("skipped") is True
    assert rep["diagnostic_only"] is True
    assert rep["auto_retrain"] is False
    assert rep["auto_budget_change"] is False


def test_no_registry_skipped(tmp_path: Path) -> None:
    (tmp_path / "evidence.json").write_text(
        json.dumps([_evidence()]),
        encoding="utf-8",
    )
    cfg = _cfg(tmp_path, registry_dir=None)
    rep = build_continuous_validation_report(cfg)
    assert rep.get("skipped") is True
    assert rep.get("reason") == "no_accepted_run_registry"


def test_aligned_prediction(tmp_path: Path) -> None:
    (tmp_path / "evidence.json").write_text(
        json.dumps([_evidence(lift_estimate=0.5, standard_error=0.1)]),
        encoding="utf-8",
    )
    reg = tmp_path / "runs"
    reg.mkdir()
    (reg / "accepted_runs.json").write_text(
        json.dumps(
            {
                "registry_version": "mmm_accepted_run_registry_v1",
                "runs": [
                    {
                        "run_id": "r1",
                        "completed_at": "2024-01-15",
                        "predicted_lifts": [{"experiment_id": "exp-1", "predicted_lift": 0.5}],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    cfg = _cfg(tmp_path, registry_dir=str(reg))
    rep = build_continuous_validation_report(cfg)
    assert rep.get("skipped") is False
    assert rep["n_aligned"] == 1
    assert rep["n_severe_miss"] == 0
    assert rep["model_trust_score"] == pytest.approx(1.0)


def test_severe_miss_classification(tmp_path: Path) -> None:
    (tmp_path / "evidence.json").write_text(
        json.dumps([_evidence(lift_estimate=0.0, standard_error=0.05)]),
        encoding="utf-8",
    )
    reg = tmp_path / "runs"
    reg.mkdir()
    (reg / "accepted_runs.json").write_text(
        json.dumps(
            {
                "runs": [
                    {
                        "run_id": "r1",
                        "completed_at": "2024-02-01",
                        "predicted_lifts": [{"experiment_id": "exp-1", "predicted_lift": 1.0}],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    cfg = _cfg(tmp_path, registry_dir=str(reg))
    rep = build_continuous_validation_report(cfg)
    assert rep["n_severe_miss"] == 1
    assert rep["model_trust_score"] < 1.0
    assert rep["recommended_action"] in (
        "model_review_required",
        "recalibrate_recommended",
        "monitor",
    )


def test_missing_se_not_evaluable(tmp_path: Path) -> None:
    (tmp_path / "evidence.json").write_text(
        json.dumps([_evidence(standard_error=None)]),
        encoding="utf-8",
    )
    reg = tmp_path / "runs"
    reg.mkdir()
    (reg / "accepted_runs.json").write_text(
        json.dumps(
            {
                "runs": [
                    {
                        "run_id": "r1",
                        "completed_at": "2024-02-01",
                        "predicted_lifts": [{"experiment_id": "exp-1", "predicted_lift": 0.5}],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    cfg = _cfg(tmp_path, registry_dir=str(reg), require_se=True)
    rep = build_continuous_validation_report(cfg)
    assert rep["n_not_evaluable"] >= 1
    pe = rep["per_experiment_results"][0]
    assert pe["classification"] == "not_evaluable"
    assert pe["not_evaluable_reason"] == "missing_se"


def test_freshness_stale_evidence(tmp_path: Path) -> None:
    stale = (date.today() - timedelta(days=400)).isoformat()
    (tmp_path / "evidence.json").write_text(
        json.dumps([_evidence(freshness_date=stale, lift_estimate=0.5)]),
        encoding="utf-8",
    )
    reg = tmp_path / "runs"
    reg.mkdir()
    (reg / "accepted_runs.json").write_text(
        json.dumps(
            {
                "runs": [
                    {
                        "run_id": "r1",
                        "completed_at": "2023-01-01",
                        "predicted_lifts": [{"experiment_id": "exp-1", "predicted_lift": 0.5}],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    cfg = _cfg(tmp_path, registry_dir=str(reg))
    rep = build_continuous_validation_report(cfg)
    fr = rep["evidence_freshness_report"]
    assert fr["stale_count"] >= 1


def test_trust_decreases_with_misses(tmp_path: Path) -> None:
    experiments = [
        _evidence(experiment_id="e1", lift_estimate=0.0, standard_error=0.05),
        _evidence(experiment_id="e2", lift_estimate=0.0, standard_error=0.05),
    ]
    (tmp_path / "evidence.json").write_text(json.dumps(experiments), encoding="utf-8")
    reg = tmp_path / "runs"
    reg.mkdir()
    (reg / "accepted_runs.json").write_text(
        json.dumps(
            {
                "runs": [
                    {
                        "run_id": "r1",
                        "completed_at": "2024-02-01",
                        "predicted_lifts": [
                            {"experiment_id": "e1", "predicted_lift": 1.0},
                            {"experiment_id": "e2", "predicted_lift": 1.0},
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    cfg = _cfg(tmp_path, registry_dir=str(reg))
    miss_rep = build_continuous_validation_report(cfg)
    (tmp_path / "evidence.json").write_text(
        json.dumps([_evidence(lift_estimate=0.5, standard_error=0.1)]),
        encoding="utf-8",
    )
    (reg / "accepted_runs.json").write_text(
        json.dumps(
            {
                "runs": [
                    {
                        "run_id": "r1",
                        "completed_at": "2024-02-01",
                        "predicted_lifts": [{"experiment_id": "exp-1", "predicted_lift": 0.5}],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    align_rep = build_continuous_validation_report(cfg)
    assert miss_rep["model_trust_score"] < align_rep["model_trust_score"]


def test_governance_no_auto_actions(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, enabled=True, registry_dir=str(tmp_path / "missing"))
    rep = build_continuous_validation_report(cfg)
    assert rep["prod_decisioning_allowed"] is False
    assert rep["auto_retrain"] is False
    assert rep["auto_registry_promotion"] is False
    assert rep["auto_budget_change"] is False


def test_extension_runner_key_only_when_enabled() -> None:
    from mmm.config.schema import CVConfig, ModelForm
    from mmm.evaluation.extension_runner import run_post_fit_extensions
    from mmm.models.ridge_bo.trainer import RidgeBOMMMTrainer
    from mmm.utils.synthetic import SyntheticGeoPanelSpec, generate_geo_panel

    df, schema = generate_geo_panel(SyntheticGeoPanelSpec(n_geos=2, n_weeks=50))
    cfg_off = MMMConfig(
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
    tr = RidgeBOMMMTrainer(cfg_off, schema)
    fit = tr.fit(df)
    rep_off = run_post_fit_extensions(
        panel=df, schema=schema, config=cfg_off, fit_out=fit, yhat=tr.predict(df), store=None
    )
    assert "continuous_validation_report" not in rep_off
