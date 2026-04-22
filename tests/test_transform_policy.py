"""Transform lineage and cross-framework comparability."""

from __future__ import annotations

from mmm.config.schema import DataConfig, Framework, MMMConfig, ModelForm
from mmm.config.transform_policy import (
    build_transform_policy_manifest,
    cross_framework_transform_drift,
)


def test_manifest_tags_framework_family() -> None:
    cfg = MMMConfig(
        framework=Framework.RIDGE_BO,
        model_form=ModelForm.SEMI_LOG,
        data=DataConfig(channel_columns=["c1"], target_column="y"),
    )
    m = build_transform_policy_manifest(cfg)
    assert m["mode_family"] == "ridge_bo_joint_hyperparams"
    assert m["policy_version"]


def test_cross_framework_drift_high_when_frameworks_differ() -> None:
    a = MMMConfig(
        framework=Framework.RIDGE_BO,
        model_form=ModelForm.SEMI_LOG,
        data=DataConfig(channel_columns=["c1"], target_column="y"),
    )
    b = MMMConfig(
        framework=Framework.BAYESIAN,
        model_form=ModelForm.SEMI_LOG,
        data=DataConfig(channel_columns=["c1"], target_column="y"),
    )
    d = cross_framework_transform_drift(a, b)
    assert d["comparability_risk"] == "high"
