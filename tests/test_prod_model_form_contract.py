"""Prod Ridge+BO requires explicit prod_canonical_modeling_contract_id aligned to model_form."""

from __future__ import annotations

import pytest

from mmm.config.schema import Framework, MMMConfig, ModelForm, RunEnvironment
from mmm.governance.policy import PolicyError


def test_prod_ridge_missing_model_contract_fails() -> None:
    with pytest.raises(PolicyError, match="prod_canonical_modeling_contract_id"):
        MMMConfig(
            run_environment=RunEnvironment.PROD,
            framework=Framework.RIDGE_BO,
            model_form=ModelForm.SEMI_LOG,
            data={"channel_columns": ["c1"], "control_columns": []},
            cv={"mode": "rolling"},
            objective={
                "normalization_profile": "strict_prod",
                "named_profile": "ridge_bo_standard_v1",
            },
        )


def test_prod_ridge_wrong_contract_for_model_form_fails() -> None:
    with pytest.raises(PolicyError, match="ridge_bo_log_log_calendar_cv_v1"):
        MMMConfig(
            run_environment=RunEnvironment.PROD,
            framework=Framework.RIDGE_BO,
            model_form=ModelForm.LOG_LOG,
            prod_canonical_modeling_contract_id="ridge_bo_semi_log_calendar_cv_v1",
            data={"channel_columns": ["c1"], "control_columns": []},
            cv={"mode": "rolling"},
            objective={
                "normalization_profile": "strict_prod",
                "named_profile": "ridge_bo_standard_v1",
            },
        )


def test_prod_bayesian_does_not_require_ridge_contract() -> None:
    MMMConfig(
        run_environment=RunEnvironment.PROD,
        framework=Framework.BAYESIAN,
        data={"channel_columns": ["c1"], "control_columns": []},
        cv={"mode": "rolling"},
        objective={"normalization_profile": "strict_prod"},
        bayesian={"posterior_predictive_draws": 100},
        extensions={
            "governance": {"bayesian_max_mean_abs_ppc_gap": 0.5},
            "optimization_gates": {"enabled": True},
        },
    )
