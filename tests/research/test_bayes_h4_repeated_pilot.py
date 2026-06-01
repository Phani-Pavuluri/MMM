"""Bayes-H4b repeated recovery pilot (research-only, report-only)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mmm.research.bayes_h3_sandbox.h4_repeated_pilot import (
    PILOT_ID_PRIMARY,
    PRIMARY_ARTIFACT_PATH,
    aggregate_world_runs,
    build_repeated_pilot_summary,
    classify_sparse_shrinkage_legacy,
    classify_sparse_shrinkage_primary,
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
        "beta_geo_index_order": ["dma_dense_a", "dma_dense_b", "dma_sparse"],
        "channel_index_order": ["tv", "search"],
        "sparse_shrinkage_decomposition": {"by_geo_channel": []},
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
                "shrinkage_ratio_sparse_vs_true_mu": None,
                "conflict_warnings": [],
            }
        )
    for nuts_seed, primary, legacy in ((4400, 0.52, 2.57), (4401, 0.58, 2.65), (4402, 0.54, 2.73)):
        rows.append(
            {
                **base,
                "world_id": WORLD_BAYES_H4_SPARSE_GEO,
                "nuts_seed": nuts_seed,
                "seed": 4400,
                "beta_gc_mae": 0.21,
                "mu_c_mae": 0.10,
                "beta_gc_coverage_90": 0.35,
                "shrinkage_ratio_sparse": primary,
                "shrinkage_ratio_sparse_vs_true_mu": legacy,
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
            "shrinkage_ratio_sparse_vs_true_mu": None,
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
    assert summary["pilot_id"] == PILOT_ID_PRIMARY
    assert summary["research_only"] is True
    assert summary["approved_for_prod"] is False
    assert summary["prod_decisioning_allowed"] is False
    assert summary["production_promotion"] is False
    assert summary["interpretation"]["hard_gate"] is False
    assert summary["interpretation"]["primary_shrinkage_role"] == "pooling_mechanics_only"
    blob = json.dumps(summary)
    assert '"approved_for_prod": true' not in blob
    for row in summary["per_run"]:
        assert row["has_decision_surface"] is False
        assert row.get("beta_geo_index_order")
        assert row.get("channel_index_order")


def test_sparse_primary_pooling_stable_fixture() -> None:
    agg = aggregate_world_runs([r for r in _fixture_per_run_rows() if r["world_id"] == WORLD_BAYES_H4_SPARSE_GEO])
    assert agg["sparse_shrinkage_warning_primary"] is False
    assert agg["shrinkage_ratio_sparse"]["max"] < 1.0
    assert agg["sparse_shrinkage_warning_legacy"] is True


def test_classify_primary_pooling_stable() -> None:
    out = classify_sparse_shrinkage_primary([0.52, 0.58, 0.54])
    assert out["classification"] == "pooling_toward_posterior_mu_stable"
    assert out["fraction_lt_1"] == 1.0


def test_classify_legacy_weak_recovery() -> None:
    out = classify_sparse_shrinkage_legacy([2.57, 2.65, 2.73], h4a_fast_reference=2.57)
    assert out["classification"] == "weak_recovery_vs_true_mu"
    assert out.get("not_a_pooling_gate") is True


def test_committed_primary_repeated_artifact_loads() -> None:
    path = REPO_ROOT / PRIMARY_ARTIFACT_PATH
    if not path.exists():
        pytest.skip("primary-metric repeated pilot artifact not materialized in workspace")
    art = load_h4_repeated_pilot_artifact(path)
    assert art["pilot_id"] == PILOT_ID_PRIMARY
    assert art["research_only"] is True
    assert art["interpretation"]["hard_gate"] is False
    assert art["metric_definitions"]["shrinkage_ratio_sparse"]["role"] == "pooling_mechanics_primary"
    sparse_runs = [r for r in art["per_run"] if r["world_id"] == WORLD_BAYES_H4_SPARSE_GEO]
    assert sparse_runs[0].get("sparse_shrinkage_decomposition") is not None
    assert sparse_runs[0].get("shrinkage_ratio_sparse_vs_true_mu") is not None


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
    row = summary["per_run"][0]
    assert row.get("shrinkage_ratio_sparse") is None or isinstance(row["shrinkage_ratio_sparse"], float)
    assert row.get("shrinkage_ratio_sparse_vs_true_mu") is None or isinstance(
        row["shrinkage_ratio_sparse_vs_true_mu"], float
    )
