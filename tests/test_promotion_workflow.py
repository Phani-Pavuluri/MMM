"""Immutable promotion registry and prod decision gate."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pytest

from mmm.config.schema import Framework, GovernanceWorkflowConfig, MMMConfig, RunEnvironment
from mmm.governance.model_release import ModelReleaseState
from mmm.governance.policy import PolicyError
from mmm.governance.promotion import build_promotion_record, promotion_signature_hash
from mmm.governance.promotion_registry import (
    PromotionRegistryError,
    append_promotion_record,
    get_promotion_by_id,
    load_promotion_registry,
    promote_run,
    rollback_promotion,
)

_MIN_DATA = {"channel_columns": ["c1"]}


def _extension_report(*, planning: bool = True) -> dict:
    return {
        "governance": {
            "approved_for_optimization": planning,
            "approved_for_reporting": True,
        },
        "model_release": {"state": ModelReleaseState.PLANNING_ALLOWED.value},
        "calibration_summary": {"replay_calibration_active": True, "n_units": 1},
        "experiment_matching": {"n_matched": 1},
    }


def test_valid_promotion_record_created(tmp_path: Path) -> None:
    reg = tmp_path / "promotions.jsonl"
    cfg = MMMConfig(
        framework=Framework.RIDGE_BO,
        run_environment=RunEnvironment.RESEARCH,
        data=_MIN_DATA,
        governance=GovernanceWorkflowConfig(promotion_registry_path=str(reg)),
    )
    fp = {"fingerprint_version": "fingerprint_v2", "combined_hash": "abc", "hash": "abc"}
    rec = promote_run(
        registry_path=reg,
        config=cfg,
        extension_report=_extension_report(),
        artifact_uri="/runs/1",
        data_fingerprint=fp,
        config_fingerprint="cfg123",
        model_fingerprint="model123",
        seed_resolution={"master_seed": 1},
        promoted_by="reviewer",
    )
    assert rec.promotion_id
    loaded = get_promotion_by_id(reg, rec.promotion_id)
    assert loaded is not None
    assert loaded.signature_hash == promotion_signature_hash(loaded.to_dict())


def test_immutable_record_cannot_overwrite(tmp_path: Path) -> None:
    reg = tmp_path / "promotions.jsonl"
    rec = build_promotion_record(
        promotion_id="pid-1",
        run_id="r1",
        model_id="m1",
        artifact_uri="/a",
        data_fingerprint={"hash": "h"},
        config_fingerprint="c",
        model_fingerprint="m",
        seed_resolution={},
        promoted_by="u",
        governance_summary={},
        calibration_summary={},
        unsupported_questions=[],
    )
    append_promotion_record(reg, rec)
    with pytest.raises(PromotionRegistryError, match="already exists"):
        append_promotion_record(reg, rec)


def test_rollback_creates_new_record(tmp_path: Path) -> None:
    reg = tmp_path / "promotions.jsonl"
    cfg = MMMConfig(
        data=_MIN_DATA,
        governance=GovernanceWorkflowConfig(promotion_registry_path=str(reg)),
    )
    fp = {"fingerprint_version": "fingerprint_v2", "combined_hash": "h1", "hash": "h1"}
    first = promote_run(
        registry_path=reg,
        config=cfg,
        extension_report=_extension_report(),
        artifact_uri="/r1",
        data_fingerprint=fp,
        config_fingerprint="c1",
        model_fingerprint="m1",
        seed_resolution={},
        promoted_by="a",
    )
    second = rollback_promotion(
        registry_path=reg,
        prior_promotion_id=first.promotion_id,
        config=cfg,
        extension_report=_extension_report(),
        artifact_uri="/r2",
        data_fingerprint=fp,
        config_fingerprint="c1",
        model_fingerprint="m2",
        seed_resolution={},
        promoted_by="b",
    )
    assert second.rollback_of == first.promotion_id
    assert len(load_promotion_registry(reg)) == 2


def test_prod_decide_requires_promotion_when_enabled(tmp_path: Path) -> None:
    from mmm.governance.policy import require_promoted_model_for_prod_decision

    cfg = MMMConfig(
        run_environment=RunEnvironment.PROD,
        data=_MIN_DATA,
        cv={"mode": "rolling"},
        prod_canonical_modeling_contract_id="ridge_bo_semi_log_calendar_cv_v1",
        objective={
            "normalization_profile": "strict_prod",
            "named_profile": "ridge_bo_standard_v1",
        },
        governance=GovernanceWorkflowConfig(
            require_promoted_model_for_prod_decision=True,
            promotion_registry_path=str(tmp_path / "p.jsonl"),
        ),
    )
    with pytest.raises(PolicyError, match="promoted_model"):
        require_promoted_model_for_prod_decision(cfg, promotion_record=None, surface="simulate")


def test_expired_promotion_fails(tmp_path: Path) -> None:
    from mmm.governance.promotion import assert_promotion_valid_for_decision

    yesterday = (date.today() - timedelta(days=1)).isoformat()
    rec = build_promotion_record(
        promotion_id="exp",
        run_id="r",
        model_id="m",
        artifact_uri="/a",
        data_fingerprint={"hash": "h"},
        config_fingerprint="c",
        model_fingerprint="m",
        seed_resolution={},
        promoted_by="u",
        governance_summary={},
        calibration_summary={},
        unsupported_questions=[],
        expiration_date=yesterday,
    )
    with pytest.raises(PolicyError, match="expired"):
        assert_promotion_valid_for_decision(
            rec,
            surface="simulate",
            data_fingerprint={"hash": "h"},
            config_fingerprint="c",
        )


def test_fingerprint_mismatch_fails() -> None:
    from mmm.governance.promotion import assert_promotion_valid_for_decision

    rec = build_promotion_record(
        promotion_id="x",
        run_id="r",
        model_id="m",
        artifact_uri="/a",
        data_fingerprint={"combined_hash": "a"},
        config_fingerprint="cfg_a",
        model_fingerprint="m",
        seed_resolution={},
        promoted_by="u",
        governance_summary={},
        calibration_summary={},
        unsupported_questions=[],
        allowed_surfaces=["simulate"],
    )
    with pytest.raises(PolicyError, match="fingerprint"):
        assert_promotion_valid_for_decision(
            rec,
            surface="simulate",
            data_fingerprint={"combined_hash": "b"},
            config_fingerprint="cfg_a",
        )


def test_default_behavior_unchanged_when_requirement_disabled() -> None:
    from mmm.governance.policy import require_promoted_model_for_prod_decision

    cfg = MMMConfig(
        run_environment=RunEnvironment.PROD,
        data=_MIN_DATA,
        cv={"mode": "rolling"},
        prod_canonical_modeling_contract_id="ridge_bo_semi_log_calendar_cv_v1",
        objective={
            "normalization_profile": "strict_prod",
            "named_profile": "ridge_bo_standard_v1",
        },
    )
    require_promoted_model_for_prod_decision(cfg, promotion_record=None, surface="simulate")
