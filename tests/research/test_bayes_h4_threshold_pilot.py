"""Bayes-H4a recovery threshold pilot (research-only, report-only)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mmm.research.bayes_h3_sandbox.h4_threshold_pilot import (
    DEFAULT_ARTIFACT_PATH,
    PILOT_ID,
    build_pilot_summary,
    build_provisional_threshold_bands,
    load_h4_threshold_pilot_artifact,
)
from mmm.research.bayes_h3_sandbox.recovery_worlds import H4_WORLD_IDS

REPO_ROOT = Path(__file__).resolve().parents[2]


def _fixture_world_rows() -> list[dict]:
    return [
        {
            "world_id": "WORLD-BAYES-H4-SIMPLE-POOLING",
            "seed": 4400,
            "beta_gc_mae": 0.18,
            "mu_c_mae": 0.09,
            "beta_gc_coverage_90": 0.5,
            "shrinkage_ratio_sparse": None,
            "conflict_warnings": [],
            "convergence": {"rhat_max": 1.05, "converged_sanity": True},
            "known_truth_ref": {"channels": ["tv", "search"]},
            "expected_diagnostic_behavior": {},
            "research_only": True,
            "approved_for_prod": False,
            "prod_decisioning_allowed": False,
            "production_promotion": False,
            "decision_grade": False,
            "production_decision_surface": False,
            "has_decision_surface": False,
            "has_budget_recommendation": False,
            "has_optimizer_ready_curves": False,
        },
        {
            "world_id": "WORLD-BAYES-H4-SPARSE-GEO",
            "seed": 4400,
            "beta_gc_mae": 0.22,
            "mu_c_mae": 0.11,
            "beta_gc_coverage_90": 0.4,
            "shrinkage_ratio_sparse": 0.55,
            "conflict_warnings": [],
            "convergence": {"rhat_max": 1.08, "converged_sanity": True},
            "known_truth_ref": {"channels": ["tv", "search"]},
            "expected_diagnostic_behavior": {"shrinkage_expected": True},
            "research_only": True,
            "approved_for_prod": False,
            "prod_decisioning_allowed": False,
            "production_promotion": False,
            "decision_grade": False,
            "production_decision_surface": False,
            "has_decision_surface": False,
            "has_budget_recommendation": False,
            "has_optimizer_ready_curves": False,
        },
        {
            "world_id": "WORLD-BAYES-H4-CONFLICTING-EVIDENCE",
            "seed": 4400,
            "beta_gc_mae": 0.2,
            "mu_c_mae": 0.1,
            "beta_gc_coverage_90": 0.45,
            "shrinkage_ratio_sparse": None,
            "conflict_warnings": ["conflict:h4-conflict-tv-dma-a: claimed negative"],
            "convergence": {"rhat_max": 1.06, "converged_sanity": True},
            "known_truth_ref": {"channels": ["tv", "search"]},
            "expected_diagnostic_behavior": {"conflict_warning_expected": True},
            "research_only": True,
            "approved_for_prod": False,
            "prod_decisioning_allowed": False,
            "production_promotion": False,
            "decision_grade": False,
            "production_decision_surface": False,
            "has_decision_surface": False,
            "has_budget_recommendation": False,
            "has_optimizer_ready_curves": False,
        },
    ]


def test_pilot_summary_schema_deterministic() -> None:
    rows = _fixture_world_rows()
    a = build_pilot_summary(rows)
    b = build_pilot_summary(rows)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_pilot_summary_includes_all_worlds_and_research_flags() -> None:
    summary = build_pilot_summary(_fixture_world_rows())
    assert summary["pilot_id"] == PILOT_ID
    assert set(summary["world_ids"]) == set(H4_WORLD_IDS)
    assert summary["research_only"] is True
    assert summary["approved_for_prod"] is False
    assert summary["production_promotion"] is False
    assert summary["prod_decisioning_allowed"] is False


def test_pilot_summary_no_prod_decision_paths() -> None:
    summary = build_pilot_summary(_fixture_world_rows())
    blob = json.dumps(summary)
    assert "optimizer_ready" not in blob or summary["worlds"][0]["has_optimizer_ready_curves"] is False
    for w in summary["worlds"]:
        assert w["has_decision_surface"] is False
        assert w["has_budget_recommendation"] is False
        assert w["production_decision_surface"] is False


def test_provisional_thresholds_are_report_only() -> None:
    bands = build_provisional_threshold_bands(_fixture_world_rows())
    assert bands["report_only"] is True
    assert bands["hard_gate"] is False
    assert bands["production_promotion"] is False
    assert bands["beta_gc_mae"]["mode"] == "report"
    assert bands["conflict_warnings"]["mode"] == "required_non_empty"
    assert bands["shrinkage_ratio_sparse"]["expect_lt"] == 1.0


def test_committed_pilot_artifact_loads() -> None:
    path = REPO_ROOT / DEFAULT_ARTIFACT_PATH
    if not path.exists():
        pytest.skip("pilot artifact not materialized in workspace")
    art = load_h4_threshold_pilot_artifact(path)
    assert art["pilot_id"] == PILOT_ID
    assert len(art["worlds"]) == len(H4_WORLD_IDS)
    assert art["research_only"] is True
    bands = art["provisional_thresholds"]
    assert bands["report_only"] is True


@pytest.mark.pymc
@pytest.mark.slow
def test_live_pilot_simple_pooling_world() -> None:
    pytest.importorskip("pymc")
    from mmm.research.bayes_h3_sandbox.h4_threshold_pilot import run_h4_threshold_pilot
    from mmm.research.bayes_h3_sandbox.recovery_worlds import WORLD_BAYES_H4_SIMPLE_POOLING

    summary = run_h4_threshold_pilot((WORLD_BAYES_H4_SIMPLE_POOLING,), fast_mcmc=True)
    assert summary["research_only"] is True
    row = summary["worlds"][0]
    assert row["beta_gc_mae"] is not None
    assert row["approved_for_prod"] is False


@pytest.mark.pymc
@pytest.mark.slow
def test_live_pilot_full_catalog() -> None:
    pytest.importorskip("pymc")
    from mmm.research.bayes_h3_sandbox.h4_threshold_pilot import run_h4_threshold_pilot

    summary = run_h4_threshold_pilot(fast_mcmc=True)
    assert len(summary["worlds"]) == len(H4_WORLD_IDS)
    conflict = next(w for w in summary["worlds"] if "CONFLICTING" in w["world_id"])
    assert conflict["conflict_warnings"]
    sparse = next(w for w in summary["worlds"] if "SPARSE" in w["world_id"])
    ratio = sparse.get("shrinkage_ratio_sparse")
    if ratio is not None:
        assert ratio < 1.0
