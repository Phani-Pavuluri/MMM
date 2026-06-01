"""INV-071 Bayes-H4 recovery threshold policy (report-only, claim-specific)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mmm.research.bayes_h3_sandbox.h4_recovery_threshold_policy import (
    DEFAULT_POLICY_PATH,
    POLICY_ID,
    RECOVERY_CANDIDATE_WORLDS,
    STRESS_DIAGNOSTIC_WORLDS,
    TRANSFORM_MISMATCH_WORLDS,
    WEAK_IDENTIFICATION_WORLDS,
    WORLD_BAYES_H4_SPARSE_GEO,
    build_threshold_policy,
    evaluate_world_against_policy,
    gate_behavior_for_role,
    load_threshold_policy,
    world_policy_role,
)
from mmm.research.bayes_h3_sandbox.h4c_recovery_worlds import (
    WORLD_BAYES_H4C_ADSTOCKED_MEDIA,
    WORLD_BAYES_H4C_CLEAN_RECOVERY,
    WORLD_BAYES_H4C_WEAK_SIGNAL,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_policy_build_deterministic() -> None:
    a = build_threshold_policy()
    b = build_threshold_policy()
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_committed_policy_artifact_loads() -> None:
    path = REPO_ROOT / DEFAULT_POLICY_PATH
    if not path.exists():
        pytest.skip("threshold policy artifact not materialized")
    pol = load_threshold_policy(path)
    assert pol["policy_id"] == POLICY_ID
    assert pol["hard_gate"] is False
    assert pol["approved_for_prod"] is False
    assert pol["production_promotion"] is False
    assert pol["prod_decisioning_allowed"] is False


def test_recovery_candidates_separate_from_stress_and_mismatch() -> None:
    assert WORLD_BAYES_H4C_CLEAN_RECOVERY in RECOVERY_CANDIDATE_WORLDS
    assert WORLD_BAYES_H4_SPARSE_GEO in STRESS_DIAGNOSTIC_WORLDS
    assert WORLD_BAYES_H4C_ADSTOCKED_MEDIA in TRANSFORM_MISMATCH_WORLDS
    assert WORLD_BAYES_H4C_WEAK_SIGNAL in WEAK_IDENTIFICATION_WORLDS
    assert WORLD_BAYES_H4_SPARSE_GEO not in RECOVERY_CANDIDATE_WORLDS
    assert WORLD_BAYES_H4C_ADSTOCKED_MEDIA not in RECOVERY_CANDIDATE_WORLDS


def test_pooling_metric_not_true_effect_gate() -> None:
    pol = build_threshold_policy()
    pooling = pol["metric_definitions"]["pooling_mechanics"]["shrinkage_ratio_sparse"]
    assert pooling["true_effect_recovery_gate"] is False
    assert pol["metric_definitions"]["true_effect_recovery"]["beta_gc_mae"]["role"] == "point_recovery"
    assert WORLD_BAYES_H4C_CLEAN_RECOVERY in RECOVERY_CANDIDATE_WORLDS


def test_transform_mismatch_not_global_failure() -> None:
    beh = gate_behavior_for_role(world_policy_role(WORLD_BAYES_H4C_ADSTOCKED_MEDIA))
    assert beh["global_model_failure"] is False
    ev = evaluate_world_against_policy(
        WORLD_BAYES_H4C_ADSTOCKED_MEDIA,
        {"beta_gc_mae": 0.5},
    )
    assert ev["global_model_failure"] is False
    assert ev["fail_for_claim"] is False
    assert ev["outcome"] != "fail_for_claim"


def test_weak_identification_not_global_failure() -> None:
    beh = gate_behavior_for_role(world_policy_role(WORLD_BAYES_H4C_WEAK_SIGNAL))
    assert beh["global_model_failure"] is False


def test_stress_world_report_only() -> None:
    ev = evaluate_world_against_policy(WORLD_BAYES_H4_SPARSE_GEO, {"beta_gc_mae": 99.0})
    assert ev["policy_role"] == "stress_diagnostic"
    assert ev["outcome"] == "report_only"
    assert ev["global_model_failure"] is False


def test_recovery_candidate_can_pass() -> None:
    ev = evaluate_world_against_policy(
        WORLD_BAYES_H4C_CLEAN_RECOVERY,
        {"beta_gc_mae": 0.15, "mu_c_mae": 0.10},
    )
    assert ev["policy_role"] == "recovery_candidate"
    assert ev["outcome"] == "pass"
    assert ev["fail_for_claim"] is False
