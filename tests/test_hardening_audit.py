"""Regression tests for audit hardening (governance, transforms, posterior prod, matching)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mmm.calibration.matching import MatchedExperiment, compute_experiment_weight_audit
from mmm.calibration.schema import ExperimentObservation
from mmm.config.schema import BayesianConfig, Framework, MMMConfig, RunEnvironment
from mmm.data.schema import PanelSchema
from mmm.evaluation.baselines import BaselineComparisonReport
from mmm.planning.posterior_planning import posterior_planning_gate
from mmm.services.governance_service import build_governance_bundle


def test_prod_bayesian_requires_bayesian_max_mean_abs_ppc_gap() -> None:
    with pytest.raises(ValueError, match="bayesian_max_mean_abs_ppc_gap"):
        MMMConfig(
            framework=Framework.BAYESIAN,
            run_environment=RunEnvironment.PROD,
            bayesian=BayesianConfig(posterior_predictive_draws=100),
            cv={"mode": "rolling"},
            objective={"normalization_profile": "strict_prod"},
            data={
                "path": None,
                "geo_column": "g",
                "week_column": "w",
                "target_column": "y",
                "channel_columns": ["c1"],
                "control_columns": [],
            },
        )


def test_prod_posterior_planning_gate_always_blocks() -> None:
    cfg = MMMConfig(
        data={"channel_columns": ["c1"], "control_columns": []},
        run_environment=RunEnvironment.PROD,
        prod_canonical_modeling_contract_id="ridge_bo_semi_log_calendar_cv_v1",
        cv={"mode": "rolling"},
        objective={
            "normalization_profile": "strict_prod",
            "named_profile": "ridge_bo_standard_v1",
        },
    )
    meta = {"posterior_diagnostics_ok": True, "posterior_predictive_ok": True}
    g = posterior_planning_gate(cfg, meta, linear_coef_draws=np.ones((3, 1)))
    assert g["allowed"] is False
    assert "prod_posterior_planning_blocked_experimental_only_policy" in g["reasons"]


def test_prod_governance_rejects_non_replay_calibration_loss() -> None:
    cfg = MMMConfig(
        data={
            "path": None,
            "geo_column": "g",
            "week_column": "w",
            "target_column": "y",
            "channel_columns": ["c1"],
            "control_columns": [],
        },
        run_environment=RunEnvironment.PROD,
        prod_canonical_modeling_contract_id="ridge_bo_semi_log_calendar_cv_v1",
        cv={"mode": "rolling"},
        objective={
            "normalization_profile": "strict_prod",
            "named_profile": "ridge_bo_standard_v1",
        },
    )
    schema = PanelSchema("g", "w", "y", ("c1",))
    panel = pd.DataFrame({"g": ["a"], "w": [0], "y": [1.0], "c1": [1.0]})
    bl = BaselineComparisonReport(0.1, 1.0, 1.0, 1.0, True, {})
    js = build_governance_bundle(
        config=cfg,
        panel=panel,
        schema=schema,
        yhat=panel["y"].to_numpy(),
        baselines=bl,
        identifiability_json={"identifiability_score": 0.0},
        falsification_flags=[],
        calibration_loss=0.5,
        calibration_is_replay=False,
    )
    assert js["approved_for_optimization"] is False
    assert "prod_rejects_non_replay_calibration_path_replay_only_for_decision_grade" in js["notes"]


def test_experiment_weight_audit_detects_dominance() -> None:
    obs = ExperimentObservation(
        experiment_id="a",
        geo_id="G",
        channel="c",
        start_week=None,
        end_week=None,
        lift=0.1,
        lift_se=0.001,
    )
    obs2 = ExperimentObservation(
        experiment_id="b",
        geo_id="G",
        channel="c",
        start_week=None,
        end_week=None,
        lift=0.2,
        lift_se=1.0,
    )
    matched = [
        MatchedExperiment(obs=obs, weight=1000.0, quality_score=1.0),
        MatchedExperiment(obs=obs2, weight=1.0, quality_score=1.0),
    ]
    audit = compute_experiment_weight_audit(matched)
    assert audit["max_inverse_se_share"] > 0.65
    assert any("0_65" in w for w in audit["dominance_warnings"])
