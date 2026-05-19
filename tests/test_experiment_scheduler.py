"""Experiment scheduler — prioritization layer above feature separability."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mmm.config.extensions import ExperimentSchedulerConfig, ExtensionSuiteConfig, FeatureSeparabilityConfig
from mmm.config.schema import CVConfig, Framework, MMMConfig, ModelForm
from mmm.data.schema import PanelSchema
from mmm.evaluation.experiment_scheduler import (
    calibration_staleness_score,
    deterministic_request_id,
    recommend_scheduler_action,
    score_feature_group_unit,
)
from mmm.evaluation.extension_runner import run_post_fit_extensions
from mmm.models.ridge_bo.trainer import RidgeBOMMMTrainer


def _meta_group_row(
    *,
    spend_share: float = 0.12,
    separability: float = 0.3,
    cal: str = "absent",
    sep_cls: str = "low",
    experiment_eligible: bool = True,
) -> dict:
    return {
        "feature_group": "Meta",
        "member_columns": ["Meta_prospecting", "Meta_retargeting"],
        "separability_score": separability,
        "separability_classification": sep_cls,
        "recommended_action": "experiment_recommended",
        "correlation_band": "high",
        "unstable_coefficient_members": ["Meta_prospecting"],
        "contribution_stability": {"unstable": True},
        "calibration_evidence": {"classification": cal, "channels_with_experiments": []},
        "business_importance": {
            "group_spend_share_of_panel": spend_share,
            "material_spend_share": spend_share >= 0.08,
            "material_contribution_share": True,
            "experiment_eligible": experiment_eligible,
            "used_in_optimization": True,
            "contribution_share_by_member": {"Meta_prospecting": 0.12, "Meta_retargeting": 0.08},
        },
    }


def test_tiny_spend_groups_do_not_trigger_experiments():
    sched = ExperimentSchedulerConfig()
    sep = FeatureSeparabilityConfig(experiment_min_group_spend_share=0.03)
    row = score_feature_group_unit(
        _meta_group_row(spend_share=0.01, experiment_eligible=False),
        identifiability_json={"vif_by_channel": {"Meta_prospecting": 20.0}, "identifiability_score": 0.8},
        governance_json={"approved_for_optimization": True},
        curve_sensitivity={"Meta_prospecting": 0.9, "Meta_retargeting": 0.9},
        experiment_matching_json={"evidence_strength": "weak"},
        sched_cfg=sched,
        sep_cfg=sep,
        optimization_gate_allowed=True,
        target_column_hint="y",
    )
    assert row["experiment_eligible"] is False
    assert row["recommended_action"] in ("rollup_recommended", "monitor", "no_action")
    assert row["experiment_request"] is None


def test_stale_experiment_evidence_increases_priority():
    sched = ExperimentSchedulerConfig()
    sep = FeatureSeparabilityConfig()
    stale = score_feature_group_unit(
        _meta_group_row(cal="absent"),
        identifiability_json={"vif_by_channel": {}, "identifiability_score": 0.5},
        governance_json={},
        curve_sensitivity={},
        experiment_matching_json={"evidence_strength": "weak"},
        sched_cfg=sched,
        sep_cfg=sep,
        optimization_gate_allowed=True,
        target_column_hint="y",
    )
    fresh = score_feature_group_unit(
        _meta_group_row(cal="strong"),
        identifiability_json={"vif_by_channel": {}, "identifiability_score": 0.5},
        governance_json={},
        curve_sensitivity={},
        experiment_matching_json={"evidence_strength": "strong"},
        sched_cfg=sched,
        sep_cfg=sep,
        optimization_gate_allowed=True,
        target_column_hint="y",
    )
    assert stale["experiment_staleness_score"] > fresh["experiment_staleness_score"]
    assert stale["experiment_priority_score"] >= fresh["experiment_priority_score"]


def test_strong_calibration_lowers_priority():
    sched = ExperimentSchedulerConfig()
    sep = FeatureSeparabilityConfig()
    weak = score_feature_group_unit(
        _meta_group_row(cal="absent"),
        identifiability_json={},
        governance_json={},
        curve_sensitivity={},
        experiment_matching_json={},
        sched_cfg=sched,
        sep_cfg=sep,
        optimization_gate_allowed=True,
        target_column_hint="y",
    )
    strong = score_feature_group_unit(
        _meta_group_row(cal="strong", separability=0.85, sep_cls="high"),
        identifiability_json={},
        governance_json={},
        curve_sensitivity={},
        experiment_matching_json={},
        sched_cfg=sched,
        sep_cfg=sep,
        optimization_gate_allowed=True,
        target_column_hint="y",
    )
    assert strong["experiment_priority_score"] < weak["experiment_priority_score"]


def test_optimizer_sensitive_channels_increase_priority():
    sched = ExperimentSchedulerConfig()
    sep = FeatureSeparabilityConfig()
    base = _meta_group_row()
    sensitive = score_feature_group_unit(
        base,
        identifiability_json={},
        governance_json={"approved_for_optimization": True},
        curve_sensitivity={"Meta_prospecting": 0.95, "Meta_retargeting": 0.9},
        experiment_matching_json={},
        sched_cfg=sched,
        sep_cfg=sep,
        optimization_gate_allowed=True,
        target_column_hint="y",
    )
    stable = score_feature_group_unit(
        base,
        identifiability_json={},
        governance_json={"approved_for_optimization": True},
        curve_sensitivity={"Meta_prospecting": 0.0, "Meta_retargeting": 0.0},
        experiment_matching_json={},
        sched_cfg=sched,
        sep_cfg=sep,
        optimization_gate_allowed=True,
        target_column_hint="y",
    )
    assert sensitive["decision_impact_score"] > stable["decision_impact_score"]
    assert sensitive["experiment_priority_score"] >= stable["experiment_priority_score"]


def test_low_separability_low_spend_recommends_rollup_not_experiment():
    action = recommend_scheduler_action(
        priority_tier="high",
        priority_score=0.9,
        uncertainty=0.9,
        business_importance=0.1,
        separability_classification="low",
        calibration_classification="absent",
        experiment_eligible=False,
        optimizer_sensitive=True,
        separability_recommended_action="experiment_recommended",
    )
    assert action == "rollup_recommended"


def test_high_separability_strong_evidence_recommends_no_action():
    action = recommend_scheduler_action(
        priority_tier="low",
        priority_score=0.2,
        uncertainty=0.2,
        business_importance=0.5,
        separability_classification="high",
        calibration_classification="strong",
        experiment_eligible=True,
        optimizer_sensitive=False,
        separability_recommended_action="keep_split",
    )
    assert action == "no_action"


def test_deterministic_request_ids_for_identical_inputs():
    a = deterministic_request_id(
        channel_or_group="Meta",
        reason="test reason",
        uncertainty_source="low_separability",
    )
    b = deterministic_request_id(
        channel_or_group="Meta",
        reason="test reason",
        uncertainty_source="low_separability",
    )
    c = deterministic_request_id(
        channel_or_group="Meta",
        reason="other",
        uncertainty_source="low_separability",
    )
    assert a == b
    assert a != c


def test_staleness_score_ordering():
    sched = ExperimentSchedulerConfig()
    stale = calibration_staleness_score("absent", sched_cfg=sched, global_evidence_strength="weak")
    fresh = calibration_staleness_score("strong", sched_cfg=sched, global_evidence_strength="strong")
    assert stale > fresh


def _panel_meta_split(*, correlated: bool) -> tuple[pd.DataFrame, PanelSchema]:
    n = 80
    rng = np.random.default_rng(7)
    weeks = np.arange(1, n // 2 + 1)
    geos = ["G1", "G2"]
    rows = []
    for g in geos:
        for w in weeks:
            base = rng.uniform(5, 20)
            if correlated:
                p = base + rng.normal(0, 0.5)
                r = base + rng.normal(0, 0.5)
            else:
                p = rng.uniform(5, 25)
                r = rng.uniform(4, 22)
            rows.append(
                {
                    "geo": g,
                    "week": w,
                    "Meta_prospecting": max(p, 0.0),
                    "Meta_retargeting": max(r, 0.0),
                    "Search": rng.uniform(8, 30),
                    "y": 100.0 + 0.4 * p + 0.2 * r + rng.normal(0, 2),
                }
            )
    df = pd.DataFrame(rows)
    schema = PanelSchema("geo", "week", "y", ("Meta_prospecting", "Meta_retargeting", "Search"))
    return df, schema


def test_extension_runner_includes_scheduler_report():
    df, schema = _panel_meta_split(correlated=True)
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
        extensions=ExtensionSuiteConfig(
            feature_separability=FeatureSeparabilityConfig(
                feature_groups={"Meta": ["Meta_prospecting", "Meta_retargeting"]},
                experiment_min_group_spend_share=0.03,
            ),
        ),
    )
    tr = RidgeBOMMMTrainer(cfg, schema)
    fit = tr.fit(df)
    yhat = tr.predict(df)
    rep = run_post_fit_extensions(
        panel=df, schema=schema, config=cfg, fit_out=fit, yhat=yhat, store=None
    )
    sched = rep["experiment_scheduler_report"]
    assert sched["policy_version"] == "experiment_scheduler_v1"
    assert sched["diagnostic_only"] is True
    assert sched["auto_experiment_execution_forbidden"] is True
    assert "experiment_priority_summary" in sched
    assert "high_priority_requests" in sched
