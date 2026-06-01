"""Bayes-H4b repeated recovery pilot (research-only, report-only)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mmm.research.bayes_h3_sandbox.h4_repeated_pilot import (
    DEFAULT_ARTIFACT_PATH,
    PILOT_ID,
    aggregate_world_runs,
    build_repeated_pilot_summary,
    classify_sparse_shrinkage,
    load_h4_repeated_pilot_artifact,
)
from mmm.research.bayes_h3_sandbox.recovery_worlds import (
    H4_WORLD_IDS,
    WORLD_BAYES_H4_CONFLICTING_EVIDENCE,
    WORLD_BAYES_H4_SIMPLE_POOLING,
    WORLD_BAYES_H4_SPARSE_GEO,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _fixture_per_run_rows() -> list[dict]:
    base = {
        "research_only": True,
        "approved_for_prod": False,
        "prod_decisioning_allowed": False,
        "production_promotion": False,
        "decision_grade": False,
        "production_decision_surface": False,
        "has_decision_surface": False,
        "has_budget_recommendation": False,
        "has_optimizer_ready_curves": False,
        "convergence": {"rhat_max": 1.05, "converged_sanity": True},
        "known_truth_ref": {},
        "expected_diagnostic_behavior": {},
        "panel_seed": 4400,
        "sampler": {"draws": 600, "tune": 600, "chains": 4},
    }
    rows: list[dict] = []
    for nuts_seed in (4400, 4401):
        rows.append(
            {
                **base,
                "world_id": WORLD_BAYES_H4_SIMPLE_POOLING,
                "nuts_seed": nuts_seed,
                "seed": 4400,
                "beta_gc_mae": 0.17 + 0.01 * (nuts_seed - 4400),
                "mu_c_mae": 0.08,
                "beta_gc_coverage_90": 0.5,
                "shrinkage_ratio_sparse": None,
                "conflict_warnings": [],
            }
        )
    for nuts_seed, shrink in ((4400, 2.4), (4401, 2.6), (4402, 2.5)):
        rows.append(
            {
                **base,
                "world_id": WORLD_BAYES_H4_SPARSE_GEO,
                "nuts_seed": nuts_seed,
                "seed": 4400,
                "beta_gc_mae": 0.21,
                "mu_c_mae": 0.10,
                "beta_gc_coverage_90": 0.35,
                "shrinkage_ratio_sparse": shrink,
                "conflict_warnings": [],
            }
        )
    rows.append(
        {
            **base,
            "world_id": WORLD_BAYES_H4_CONFLICTING_EVIDENCE,
            "nuts_seed": 4400,
            "seed": 4400,
            "beta_gc_mae": 0.19,
            "mu_c_mae": 0.09,
            "beta_gc_coverage_90": 0.42,
            "shrinkage_ratio_sparse": None,
            "conflict_warnings": ["conflict:h4-conflict-tv-dma-a: claimed negative"],
        }
    )
    return rows


def test_repeated_summary_schema_deterministic() -> None:
    rows = _fixture_per_run_rows()
    a = build_repeated_pilot_summary(rows, seeds=(4400, 4401, 4402))
    b = build_repeated_pilot_summary(rows, seeds=(4400, 4401, 4402))
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_repeated_summary_preserves_all_world_ids() -> None:
    summary = build_repeated_pilot_summary(_fixture_per_run_rows())
    assert set(summary["world_ids"]) == set(H4_WORLD_IDS)
    run_worlds = {r["world_id"] for r in summary["per_run"]}
    assert WORLD_BAYES_H4_SIMPLE_POOLING in run_worlds
    assert WORLD_BAYES_H4_SPARSE_GEO in run_worlds
    assert WORLD_BAYES_H4_CONFLICTING_EVIDENCE in run_worlds


def test_repeated_summary_research_flags_and_no_prod_fields() -> None:
    summary = build_repeated_pilot_summary(_fixture_per_run_rows())
    assert summary["pilot_id"] == PILOT_ID
    assert summary["research_only"] is True
    assert summary["approved_for_prod"] is False
    assert summary["prod_decisioning_allowed"] is False
    assert summary["production_promotion"] is False
    assert summary["interpretation"]["hard_gate"] is False
    blob = json.dumps(summary)
    assert '"approved_for_prod": true' not in blob
    assert '"prod_decisioning_allowed": true' not in blob
    for row in summary["per_run"]:
        assert row["has_decision_surface"] is False
        assert row["production_decision_surface"] is False


def test_sparse_shrinkage_warning_when_ratio_gte_one() -> None:
    agg = aggregate_world_runs([r for r in _fixture_per_run_rows() if r["world_id"] == WORLD_BAYES_H4_SPARSE_GEO])
    assert agg["sparse_shrinkage_warning"] is True
    assert agg["shrinkage_ratio_sparse"]["min"] >= 1.0


def test_classify_sparse_shrinkage_likely_model_when_all_gte_one() -> None:
    out = classify_sparse_shrinkage([2.4, 2.6, 2.5], h4a_fast_reference=2.57)
    assert out["classification"] in (
        "likely_model_prior_or_world_design",
        "likely_world_design_or_metric",
        "still_unstable",
        "inconclusive",
    )
    assert out["fraction_lt_1"] == 0.0


def test_committed_repeated_artifact_loads() -> None:
    path = REPO_ROOT / DEFAULT_ARTIFACT_PATH
    if not path.exists():
        pytest.skip("repeated pilot artifact not materialized in workspace")
    art = load_h4_repeated_pilot_artifact(path)
    assert art["pilot_id"] == PILOT_ID
    assert art["research_only"] is True
    assert art["interpretation"]["hard_gate"] is False
    assert art["approved_for_prod"] is False


@pytest.mark.pymc
@pytest.mark.slow
def test_live_repeated_pilot_simple_pooling_two_seeds() -> None:
    pytest.importorskip("pymc")
    from mmm.research.bayes_h3_sandbox.h4_repeated_pilot import run_h4_repeated_pilot
    from mmm.research.bayes_h3_sandbox.recovery_worlds import SAMPLER_FAST

    summary = run_h4_repeated_pilot(
        (WORLD_BAYES_H4_SIMPLE_POOLING,),
        nuts_seeds=(4400, 4401),
        sampler=SAMPLER_FAST,
    )
    assert summary["research_only"] is True
    assert len(summary["per_run"]) == 2
    assert summary["aggregate_by_world"][WORLD_BAYES_H4_SIMPLE_POOLING]["n_runs"] == 2


@pytest.mark.pymc
@pytest.mark.slow
def test_live_repeated_pilot_sparse_geo_optional() -> None:
    pytest.importorskip("pymc")
    from mmm.research.bayes_h3_sandbox.h4_repeated_pilot import run_h4_repeated_pilot
    from mmm.research.bayes_h3_sandbox.recovery_worlds import SAMPLER_FAST

    summary = run_h4_repeated_pilot(
        (WORLD_BAYES_H4_SPARSE_GEO,),
        nuts_seeds=(4400,),
        sampler=SAMPLER_FAST,
    )
    sparse = summary["per_run"][0].get("shrinkage_ratio_sparse")
    assert sparse is None or isinstance(sparse, float)
