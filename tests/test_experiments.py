"""Experiment registry and payload signing."""

from __future__ import annotations

import pytest

from mmm.experiments import (
    ApprovalState,
    ExperimentRecord,
    ExperimentRegistry,
    experiment_readiness,
    new_experiment_id,
    sign_payload,
    verify_payload,
)


def test_new_experiment_id_is_uuid_like() -> None:
    eid = new_experiment_id()
    assert len(eid) == 36
    assert eid != new_experiment_id()


def test_registry_register_and_require_approved() -> None:
    reg = ExperimentRegistry()
    eid = new_experiment_id()
    rec = ExperimentRecord(
        experiment_id=eid,
        approval=ApprovalState.DRAFT,
        calibration_artifact_ref="s3://bucket/calib.json",
        payload_signature="abc",
    )
    reg.register(rec)
    with pytest.raises(PermissionError):
        reg.require_approved(eid)
    reg.set_approval(eid, ApprovalState.APPROVED)
    out = reg.require_approved(eid)
    assert out.experiment_id == eid


def test_register_duplicate_raises() -> None:
    reg = ExperimentRegistry()
    eid = new_experiment_id()
    reg.register(ExperimentRecord(experiment_id=eid))
    with pytest.raises(ValueError):
        reg.register(ExperimentRecord(experiment_id=eid))


def test_experiment_readiness() -> None:
    ok = ExperimentRecord(
        experiment_id=new_experiment_id(),
        approval=ApprovalState.APPROVED,
        calibration_artifact_ref="ref",
        payload_signature="sig",
    )
    assert experiment_readiness(ok)["ready"] is True
    bad = ExperimentRecord(experiment_id=new_experiment_id(), approval=ApprovalState.DRAFT)
    r = experiment_readiness(bad)
    assert r["ready"] is False
    assert r["reasons"]


def test_sign_verify_roundtrip() -> None:
    p = {"experiment_id": "e1", "channels": ["a", "b"]}
    s = sign_payload(p, "secret")
    assert verify_payload(p, s, "secret")
    assert not verify_payload({**p, "channels": ["a"]}, s, "secret")
