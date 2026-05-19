"""Prod replay evidence gate semantics."""

from __future__ import annotations

import pytest

from mmm.governance.policy import PolicyError, RuntimePolicy, require_replay_calibration
from mmm.governance.replay_evidence import (
    experiment_matching_satisfies_prod,
    prod_replay_evidence_ok,
    replay_calibration_active,
)


def test_skipped_experiment_matching_fails_prod_gate() -> None:
    policy = RuntimePolicy(prod=True, require_replay_calibration=True)
    with pytest.raises(PolicyError, match="skipped"):
        require_replay_calibration(
            None,
            {"skipped": True, "reason": "panel_and_schema_required_for_matching_trace"},
            policy,
        )


def test_nonempty_experiment_matching_without_n_matched_fails() -> None:
    ok, code = prod_replay_evidence_ok(None, {"replay_ok": True})
    assert not ok
    assert code == "missing_replay_or_matched_experiments"


def test_n_matched_passes() -> None:
    assert experiment_matching_satisfies_prod({"n_matched": 3, "evidence_strength": "moderate"})
    ok, code = prod_replay_evidence_ok(None, {"n_matched": 1})
    assert ok
    assert code == "experiment_matching_n_matched"


def test_replay_calibration_active_passes() -> None:
    assert replay_calibration_active({"replay_calibration_active": True, "replay_loss": 1.2})
    ok, code = prod_replay_evidence_ok({"replay_calibration_active": True}, None)
    assert ok
    assert code == "replay_calibration_active"


def test_research_skips_gate() -> None:
    policy = RuntimePolicy(prod=False, require_replay_calibration=True)
    require_replay_calibration(None, {"skipped": True}, policy)
