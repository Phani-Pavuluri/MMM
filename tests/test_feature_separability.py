"""Feature separability governance — diagnostic-only guidance."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mmm.config.extensions import ExtensionSuiteConfig, FeatureSeparabilityConfig
from mmm.config.schema import CVConfig, Framework, MMMConfig, ModelForm
from mmm.data.schema import PanelSchema
from mmm.evaluation.extension_runner import run_post_fit_extensions
from mmm.evaluation.feature_separability import (
    analyze_feature_group,
    bootstrap_channel_coefs,
    build_governance_artifacts,
    business_importance_for_group,
    classify_separability,
    coefficient_stability_metrics,
    compute_feature_separability_report,
    contribution_stability_metrics,
    infer_feature_groups,
    pairwise_correlations,
    recommend_action_and_text,
    separability_score_from_signals,
)
from mmm.models.ridge_bo.trainer import RidgeBOMMMTrainer


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


def test_infer_feature_groups_prefix():
    groups = infer_feature_groups(
        ["Meta_prospecting", "Meta_retargeting", "Search"],
        explicit_groups=None,
        auto_group_prefix=True,
    )
    assert groups == {"Meta": ["Meta_prospecting", "Meta_retargeting"]}


def test_clean_separable_group_recommends_keep_split():
    df, schema = _panel_meta_split(correlated=False)
    pairs, max_corr = pairwise_correlations(df, ["Meta_prospecting", "Meta_retargeting"])
    assert max_corr < 0.5
    boot = {
        "Meta_prospecting": [0.11, 0.12, 0.115, 0.118],
        "Meta_retargeting": [0.08, 0.082, 0.079, 0.081],
    }
    scales = {"Meta_prospecting": 1.0, "Meta_retargeting": 1.0}
    coef_m, unstable, coef_bad = coefficient_stability_metrics(
        boot,
        scale_by_channel=scales,
        sign_flip_threshold=0.2,
        coef_cv_threshold=0.5,
    )
    assert "standardized_effect_cv" in coef_m["Meta_prospecting"]
    assert not coef_bad
    assert not unstable
    shares = [
        {"Meta_prospecting": 0.6, "Meta_retargeting": 0.4},
        {"Meta_prospecting": 0.58, "Meta_retargeting": 0.42},
        {"Meta_prospecting": 0.61, "Meta_retargeting": 0.39},
    ]
    contrib = contribution_stability_metrics(
        shares, ["Meta_prospecting", "Meta_retargeting"], variance_threshold=0.08
    )
    assert contrib["unstable"] is False
    score = separability_score_from_signals(
        correlation_band="low",
        vif_band="healthy",
        standardized_effect_unstable=False,
        contribution_unstable=False,
        calibration="strong",
    )
    assert classify_separability(score) == "high"
    action, _ = recommend_action_and_text(
        classification="high",
        importance_band="high",
        experiment_eligible=True,
        feature_group="Meta",
        member_columns=["Meta_prospecting", "Meta_retargeting"],
        correlation_band="low",
        standardized_effect_unstable=False,
        contribution_unstable=False,
        calibration="strong",
    )
    assert action == "keep_split"


def test_high_correlation_low_separability_recommends_rollup():
    df, _ = _panel_meta_split(correlated=True)
    _, max_corr = pairwise_correlations(df, ["Meta_prospecting", "Meta_retargeting"])
    assert max_corr >= 0.5
    score = separability_score_from_signals(
        correlation_band="high",
        vif_band="high",
        standardized_effect_unstable=True,
        contribution_unstable=True,
        calibration="absent",
    )
    assert classify_separability(score) == "low"
    action, text = recommend_action_and_text(
        classification="low",
        importance_band="low",
        experiment_eligible=False,
        feature_group="Meta",
        member_columns=["Meta_prospecting", "Meta_retargeting"],
        correlation_band="high",
        standardized_effect_unstable=False,
        contribution_unstable=False,
        calibration="absent",
    )
    assert action == "rollup_recommended"
    assert "Roll up" in text


def test_coefficient_sign_instability_flags_unstable():
    boot = {
        "Meta_prospecting": [0.2, -0.15, 0.18, -0.12, 0.1],
        "Meta_retargeting": [0.05, 0.04, 0.06, 0.05, 0.04],
    }
    _, unstable, coef_bad = coefficient_stability_metrics(
        boot,
        scale_by_channel={"Meta_prospecting": 1.0, "Meta_retargeting": 1.0},
        sign_flip_threshold=0.2,
        coef_cv_threshold=0.5,
    )
    assert coef_bad
    assert "Meta_prospecting" in unstable


def test_raw_coef_variance_not_flagged_when_standardized_effects_stable():
    """High raw coef CV with stable sign/magnitude on standardized effects stays stable."""
    boot = {"Tiny": [10.0, 12.0, 8.0, 11.0]}
    per, unstable, bad = coefficient_stability_metrics(
        boot,
        scale_by_channel={"Tiny": 0.01},
        sign_flip_threshold=0.2,
        coef_cv_threshold=0.5,
    )
    assert per["Tiny"]["raw_coefficient_std"] > 0.5
    assert not bad
    assert not unstable


def test_unstable_contribution_detected():
    shares = [
        {"Meta_prospecting": 0.9, "Meta_retargeting": 0.1},
        {"Meta_prospecting": 0.1, "Meta_retargeting": 0.9},
        {"Meta_prospecting": 0.85, "Meta_retargeting": 0.15},
    ]
    contrib = contribution_stability_metrics(
        shares, ["Meta_prospecting", "Meta_retargeting"], variance_threshold=0.05
    )
    assert contrib["unstable"] is True
    assert contrib["rank_change_rate"] > 0.0


def test_high_spend_low_separability_recommends_experiment():
    action, text = recommend_action_and_text(
        classification="low",
        importance_band="high",
        experiment_eligible=True,
        feature_group="Meta",
        member_columns=["Meta_prospecting", "Meta_retargeting"],
        correlation_band="high",
        standardized_effect_unstable=True,
        contribution_unstable=True,
        calibration="absent",
    )
    assert action == "experiment_recommended"
    assert "experiment" in text.lower()


def test_tiny_spend_low_separability_caution_not_experiment():
    df, schema = _panel_meta_split(correlated=True)
    spend = {"Meta_prospecting": 0.001, "Meta_retargeting": 0.001}
    biz = business_importance_for_group(
        ["Meta_prospecting", "Meta_retargeting"],
        panel=df,
        schema=schema,
        spend_by_member=spend,
        contribution_shares={"Meta_prospecting": 0.02, "Meta_retargeting": 0.02},
        cfg=FeatureSeparabilityConfig(
            business_importance_high_spend_share=0.08,
            business_importance_high_contribution_share=0.10,
            experiment_min_group_spend_share=0.03,
        ),
        governance_approved_for_optimization=True,
        planner_mode="full_model",
    )
    assert biz["importance_band"] == "low"
    assert biz["experiment_eligible"] is False
    action, text = recommend_action_and_text(
        classification="low",
        importance_band="low",
        experiment_eligible=False,
        feature_group="Meta",
        member_columns=["Meta_prospecting", "Meta_retargeting"],
        correlation_band="high",
        standardized_effect_unstable=True,
        contribution_unstable=True,
        calibration="absent",
    )
    assert action == "rollup_recommended"
    assert "experiment" not in text.lower()


def test_optimization_approval_alone_does_not_make_high_importance():
    df, schema = _panel_meta_split(correlated=False)
    biz = business_importance_for_group(
        ["Meta_prospecting", "Meta_retargeting"],
        panel=df,
        schema=schema,
        spend_by_member={"Meta_prospecting": 0.001, "Meta_retargeting": 0.001},
        contribution_shares={"Meta_prospecting": 0.01, "Meta_retargeting": 0.01},
        cfg=FeatureSeparabilityConfig(),
        governance_approved_for_optimization=True,
        planner_mode="full_model",
    )
    assert biz["used_in_optimization"] is True
    assert biz["importance_band"] == "low"
    assert biz["experiment_eligible"] is False


def test_governance_artifacts_experiment_and_unsupported_claims():
    groups = [
        {
            "feature_group": "Meta",
            "member_columns": ["Meta_prospecting", "Meta_retargeting"],
            "recommended_action": "experiment_recommended",
            "separability_classification": "low",
            "recommendation": "Run experiment",
            "business_importance": {"used_in_optimization": True, "importance_band": "high"},
        }
    ]
    rollup, experiment, unsupported, summary = build_governance_artifacts(groups)
    assert not rollup
    assert len(experiment) == 1
    assert len(unsupported) == 1
    assert summary["optimization_use_warnings"]


def test_extension_report_includes_feature_separability():
    df, schema = _panel_meta_split(correlated=False)
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
            )
        ),
    )
    tr = RidgeBOMMMTrainer(cfg, schema)
    fit = tr.fit(df)
    yhat = tr.predict(df)
    rep = run_post_fit_extensions(
        panel=df, schema=schema, config=cfg, fit_out=fit, yhat=yhat, store=None
    )
    fs = rep["feature_separability_report"]
    assert fs["policy_version"] == "feature_separability_v1"
    assert fs["diagnostic_only"] is True
    assert len(fs["feature_groups"]) == 1
    assert fs["feature_groups"][0]["feature_group"] == "Meta"
    assert fs["feature_groups"][0]["recommended_action"] in (
        "keep_split",
        "keep_with_caution",
        "rollup_recommended",
        "experiment_recommended",
    )


def test_training_channel_columns_unchanged_by_report():
    """Separability must not mutate config channel list (no automatic merge)."""
    df, schema = _panel_meta_split(correlated=True)
    channels_before = list(schema.channel_columns)
    cfg = MMMConfig(
        framework=Framework.RIDGE_BO,
        model_form=ModelForm.SEMI_LOG,
        data={
            "path": None,
            "geo_column": schema.geo_column,
            "week_column": schema.week_column,
            "target_column": schema.target_column,
            "channel_columns": channels_before,
        },
        cv=CVConfig(mode="rolling", n_splits=2, min_train_weeks=12, horizon_weeks=3),
        ridge_bo={"n_trials": 2},
    )
    tr = RidgeBOMMMTrainer(cfg, schema)
    fit = tr.fit(df)
    yhat = tr.predict(df)
    run_post_fit_extensions(panel=df, schema=schema, config=cfg, fit_out=fit, yhat=yhat, store=None)
    assert list(schema.channel_columns) == channels_before
    assert list(cfg.data.channel_columns) == channels_before


def test_analyze_feature_group_uses_vif_from_identifiability():
    df, schema = _panel_meta_split(correlated=False)
    X = np.column_stack(
        [
            df["Meta_prospecting"].to_numpy(),
            df["Meta_retargeting"].to_numpy(),
            df["Search"].to_numpy(),
        ]
    )
    y_log = np.log(np.clip(df["y"].to_numpy(), 1e-9, None))
    boot = bootstrap_channel_coefs(
        X,
        y_log,
        list(schema.channel_columns),
        ridge_alpha=1.0,
        rng=np.random.default_rng(0),
        rounds=4,
        bootstrap_frac=0.85,
    )
    result = analyze_feature_group(
        "Meta",
        ["Meta_prospecting", "Meta_retargeting"],
        panel=df,
        schema=schema,
        X_media=X,
        channel_names=list(schema.channel_columns),
        y_log=y_log,
        vif_by_channel={"Meta_prospecting": 2.0, "Meta_retargeting": 2.1},
        boot_coefs=boot,
        matched_channels=set(),
        cfg=FeatureSeparabilityConfig(),
        governance_approved_for_optimization=False,
        planner_mode="full_model",
    )
    assert result["vif_metrics"]["Meta_prospecting"] == 2.0
    assert "separability_score" in result


def test_compute_report_skipped_when_disabled():
    df, schema = _panel_meta_split(correlated=False)
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
        extensions=ExtensionSuiteConfig(
            feature_separability=FeatureSeparabilityConfig(enabled=False),
        ),
    )
    rep = compute_feature_separability_report(
        panel=df,
        schema=schema,
        config=cfg,
        fit_out={},
        X_media=np.zeros((len(df), 3)),
        identifiability_json=None,
        experiment_matching_json=None,
        rng=np.random.default_rng(0),
    )
    assert rep["skipped"] is True
