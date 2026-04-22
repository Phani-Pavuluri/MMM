"""Explicit transform + framework lineage so Ridge vs Bayesian runs are not silently conflated."""

from __future__ import annotations

from typing import Any, Literal

from mmm.config.schema import Framework, MMMConfig

TRANSFORM_POLICY_VERSION = "mmm_transform_policy_v1"

ModeFamily = Literal["ridge_bo_joint_hyperparams", "bayesian_fixed_yaml_hyperparams"]


def framework_transform_mode_family(framework: Framework) -> ModeFamily:
    if framework == Framework.RIDGE_BO:
        return "ridge_bo_joint_hyperparams"
    return "bayesian_fixed_yaml_hyperparams"


def build_transform_policy_manifest(config: MMMConfig) -> dict[str, Any]:
    """
    Serializable transform stack for lineage, economics alignment, and comparability notes.

    **Comparability:** Ridge+BO fits *decay*, *hill_half*, *hill_slope* inside CV trials (see design matrix).
    The PyMC trainer uses **fixed** ``transforms.*_params`` from YAML for feature construction — not the
    same estimation problem. Cross-framework leaderboards are **diagnostic only** unless you manually
    align hyperparameters and freeze the Ridge path to the same values.
    """
    t = config.transforms
    return {
        "policy_version": TRANSFORM_POLICY_VERSION,
        "framework": config.framework.value,
        "mode_family": framework_transform_mode_family(config.framework),
        "model_form": config.model_form.value,
        "adstock": t.adstock,
        "saturation": t.saturation,
        "adstock_params": dict(t.adstock_params),
        "saturation_params": dict(t.saturation_params),
        "fast_path_feature_builder": "mmm.transforms.stack.build_channel_features_from_params",
        "ridge_bo_trainer_note": (
            "Ridge path: geometric adstock + Hill saturation only; decay/hill learned per Optuna/grid trial."
        ),
        "bayesian_trainer_note": (
            "Bayesian PyMC path: same geometric+Hill feature builder; hyperparameters taken from "
            "YAML transforms.*_params (not jointly re-estimated inside NUTS in this trainer)."
        ),
        "cross_framework_comparability": "not_automatic",
        "cross_framework_comparability_detail": (
            "Do not treat Ridge vs Bayesian CV objectives as directly comparable unless transforms and "
            "media hyperparameters are explicitly harmonized (same decay/hill, same feature definition)."
        ),
        "bayesian_decision_transform_stance": config.extensions.product.bayesian_decision_transform_stance,
    }


def cross_framework_transform_drift(a: MMMConfig, b: MMMConfig) -> dict[str, Any]:
    """Summarize transform / mode differences between two configs (e.g. CLI compare)."""
    ma, mb = build_transform_policy_manifest(a), build_transform_policy_manifest(b)
    same_stack = (
        ma["adstock"] == mb["adstock"]
        and ma["saturation"] == mb["saturation"]
        and ma["adstock_params"] == mb["adstock_params"]
        and ma["saturation_params"] == mb["saturation_params"]
    )
    return {
        "same_yaml_transform_stack": same_stack,
        "framework_a": ma["framework"],
        "framework_b": mb["framework"],
        "mode_family_a": ma["mode_family"],
        "mode_family_b": mb["mode_family"],
        "comparability_risk": (
            "high"
            if ma["framework"] != mb["framework"] or not same_stack
            else ("medium" if ma["mode_family"] != mb["mode_family"] else "low")
        ),
    }
