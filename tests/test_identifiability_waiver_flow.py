"""Identifiability waiver artifact: prod policy + validation."""

from datetime import timedelta, timezone

import pytest

from mmm.config.extensions import ExtensionSuiteConfig, GovernanceConfig, OptimizationGateConfig
from mmm.config.schema import MMMConfig, RunEnvironment
from mmm.governance.identifiability_waiver import IdentifiabilityWaiverArtifact
from mmm.governance.policy import PolicyError, require_identifiability_for_prod_decision, runtime_policy_from_config


def _waiver_dict(*, score_cap: float = 0.99) -> dict:
    from datetime import datetime

    now = datetime.now(timezone.utc)
    exp = (now + timedelta(days=2)).replace(microsecond=0).isoformat()
    return {
        "waiver_id": "test-waiver-ident-1",
        "created_at": now.replace(microsecond=0).isoformat(),
        "reason": "integration test waiver for identifiability exceedance",
        "max_identifiability_score_waived": score_cap,
        "expires_at": exp,
    }


def test_prod_identifiability_block_without_waiver() -> None:
    cfg = MMMConfig(
        run_environment=RunEnvironment.PROD,
        prod_canonical_modeling_contract_id="ridge_bo_semi_log_calendar_cv_v1",
        data={"channel_columns": ["c1"], "control_columns": []},
        cv={"mode": "rolling"},
        objective={
            "normalization_profile": "strict_prod",
            "named_profile": "ridge_bo_standard_v1",
        },
        extensions=ExtensionSuiteConfig(
            governance=GovernanceConfig(
                max_identifiability_risk=0.65,
                identifiability_decision_safety_margin=0.85,
                allow_identifiability_waiver=False,
            ),
            optimization_gates=OptimizationGateConfig(enabled=True),
        ),
    )
    er = {"identifiability": {"identifiability_score": 0.99}}
    pol = runtime_policy_from_config(cfg)
    with pytest.raises(PolicyError, match="identifiability_score"):
        require_identifiability_for_prod_decision(cfg, er, pol)


def test_prod_identifiability_waiver_allows_when_policy_enabled() -> None:
    cfg = MMMConfig(
        run_environment=RunEnvironment.PROD,
        prod_canonical_modeling_contract_id="ridge_bo_semi_log_calendar_cv_v1",
        data={"channel_columns": ["c1"], "control_columns": []},
        cv={"mode": "rolling"},
        objective={
            "normalization_profile": "strict_prod",
            "named_profile": "ridge_bo_standard_v1",
        },
        extensions=ExtensionSuiteConfig(
            governance=GovernanceConfig(
                max_identifiability_risk=0.65,
                identifiability_decision_safety_margin=0.85,
                allow_identifiability_waiver=True,
            ),
            optimization_gates=OptimizationGateConfig(enabled=True),
        ),
    )
    er = {
        "identifiability": {"identifiability_score": 0.9},
        "identifiability_waiver": _waiver_dict(score_cap=0.95),
        "config_fingerprint_sha256": "ab" * 32,
        "model_release": {"state": "planning_allowed", "release_id": "mr-42"},
    }
    pol = runtime_policy_from_config(cfg)
    require_identifiability_for_prod_decision(cfg, er, pol)
    assert "_identifiability_waiver_applied" in er


def test_validate_waiver_rejects_past_expiry() -> None:
    from datetime import datetime

    from mmm.governance.identifiability_waiver import validate_waiver_for_run

    cre = datetime(2020, 1, 1, tzinfo=timezone.utc).isoformat()
    exp = datetime(2020, 1, 2, tzinfo=timezone.utc).isoformat()
    w = IdentifiabilityWaiverArtifact(
        waiver_id="w-expired",
        created_at=cre,
        reason="integration test expired waiver path",
        max_identifiability_score_waived=1.0,
        expires_at=exp,
    )
    with pytest.raises(PolicyError, match="expired"):
        validate_waiver_for_run(
            w,
            identifiability_score=0.9,
            model_release_id=None,
            config_fingerprint_sha256=None,
        )
