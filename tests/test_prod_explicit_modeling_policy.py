"""Phase 1: explicit prod modeling policy (CV, objective, geo–Δμ alignment)."""

from __future__ import annotations

import pytest

from mmm.config.extensions import ExtensionSuiteConfig, ProductScopeConfig
from mmm.config.schema import Framework, MMMConfig, RunEnvironment
from mmm.governance.policy import PolicyError


def test_prod_rejects_cv_auto() -> None:
    with pytest.raises(PolicyError, match="cv.mode=auto"):
        MMMConfig(
            run_environment=RunEnvironment.PROD,
            prod_canonical_modeling_contract_id="ridge_bo_semi_log_calendar_cv_v1",
            data={"channel_columns": ["c1"], "control_columns": []},
            cv={"mode": "auto", "split_axis": "calendar_week"},
            objective={
                "normalization_profile": "strict_prod",
                "named_profile": "ridge_bo_standard_v1",
            },
        )


def test_prod_rejects_implicit_research_normalization() -> None:
    with pytest.raises(PolicyError, match="strict_prod"):
        MMMConfig(
            run_environment=RunEnvironment.PROD,
            prod_canonical_modeling_contract_id="ridge_bo_semi_log_calendar_cv_v1",
            data={"channel_columns": ["c1"], "control_columns": []},
            cv={"mode": "rolling"},
            objective={"normalization_profile": "research"},
        )


def test_prod_ridge_bo_requires_named_objective_profile() -> None:
    with pytest.raises(PolicyError, match="objective.named_profile"):
        MMMConfig(
            run_environment=RunEnvironment.PROD,
            prod_canonical_modeling_contract_id="ridge_bo_semi_log_calendar_cv_v1",
            framework=Framework.RIDGE_BO,
            data={"channel_columns": ["c1"], "control_columns": []},
            cv={"mode": "rolling", "split_axis": "calendar_week"},
            objective={"normalization_profile": "strict_prod"},
        )


def test_geo_budget_requires_geo_mean_delta_mu_aggregation() -> None:
    with pytest.raises(PolicyError, match="planning_delta_mu_aggregation"):
        MMMConfig(
            data={"channel_columns": ["c1"], "control_columns": []},
            budget={"geo_budget_enabled": True},
        )

    MMMConfig(
        data={"channel_columns": ["c1"], "control_columns": []},
        budget={"geo_budget_enabled": True},
        extensions=ExtensionSuiteConfig(
            product=ProductScopeConfig(planning_delta_mu_aggregation="geo_mean_then_global_mean")
        ),
    )
