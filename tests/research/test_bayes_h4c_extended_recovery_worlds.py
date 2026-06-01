"""Bayes-H4c extended recovery worlds (research-only reliability map)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mmm.research.bayes_h3_sandbox.h4c_extended_recovery_pilot import (
    DEFAULT_ARTIFACT_PATH,
    PILOT_ID,
    build_h4c_pilot_summary,
    classify_h4c_world,
    load_h4c_extended_recovery_pilot_artifact,
)
from mmm.research.bayes_h3_sandbox.h4c_recovery_worlds import H4C_WORLD_IDS, list_h4c_recovery_worlds
from mmm.research.bayes_h3_sandbox.recovery_runner import validate_h4c_world_catalog
from mmm.research.bayes_h3_sandbox.recovery_worlds import WORLD_BAYES_H4_SPARSE_GEO, get_recovery_world

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("world_id", H4C_WORLD_IDS)
def test_h4c_world_panel_deterministic(world_id: str) -> None:
    spec = get_recovery_world(world_id)
    from mmm.research.bayes_h3_sandbox.recovery_worlds import materialize_recovery_bundle

    df1 = materialize_recovery_bundle(spec)[2]
    df2 = materialize_recovery_bundle(spec)[2]
    assert df1.equals(df2)


@pytest.mark.parametrize("world_id", H4C_WORLD_IDS)
def test_h4c_world_has_known_truth_and_classification(world_id: str) -> None:
    spec = get_recovery_world(world_id)
    truth = spec.known_truth
    assert truth["true_mu_c"]
    assert truth["true_beta_gc"]
    assert spec.expected_diagnostic_behavior.get("h4c_classification")


def test_h4c_catalog_validation_passes() -> None:
    summary = validate_h4c_world_catalog()
    assert summary["status"] == "pass"
    assert len(summary["worlds"]) == len(H4C_WORLD_IDS)


def test_list_h4c_worlds_covers_ids() -> None:
    ids = {w.world_id for w in list_h4c_recovery_worlds()}
    assert ids == set(H4C_WORLD_IDS)


def _mock_world_row(world_id: str, *, classification: str) -> dict:
    return {
        "world_id": world_id,
        "beta_gc_mae": 0.2,
        "mu_c_mae": 0.1,
        "beta_gc_coverage_90": 0.4,
        "shrinkage_ratio_sparse": 0.55,
        "shrinkage_ratio_sparse_vs_true_mu": 2.5,
        "h4c_diagnostic_warnings": [],
        "beta_interval_width_90_mean": 0.8,
        "sparse_shrinkage_decomposition": {"by_geo_channel": []},
        "beta_geo_index_order": ["dma_a"],
        "channel_index_order": ["tv", "search"],
        "reliability_map": {"classification": classification, "hard_gate": False},
        "approved_for_prod": False,
        "prod_decisioning_allowed": False,
        "research_only": True,
    }


def test_pilot_summary_schema_and_flags() -> None:
    rows = [_mock_world_row(wid, classification="recovery_candidate") for wid in H4C_WORLD_IDS]
    summary = build_h4c_pilot_summary(rows)
    assert summary["pilot_id"] == PILOT_ID
    assert summary["research_only"] is True
    assert summary["approved_for_prod"] is False
    assert summary["interpretation"]["hard_gate"] is False
    assert len(summary["worlds"]) == len(H4C_WORLD_IDS)
    assert json.dumps(summary, sort_keys=True)


def test_transform_mismatch_classification() -> None:
    spec = get_recovery_world("WORLD-BAYES-H4C-ADSTOCKED-MEDIA")
    row = _mock_world_row(spec.world_id, classification="transform_mismatch")
    row["h4c_diagnostic_warnings"] = [f"h4c:transform_mismatch:{spec.world_id}: generative=adstock"]
    rm = classify_h4c_world(row, spec)
    assert rm["classification"] == "transform_mismatch"
    assert rm["hard_gate"] is False


def test_stress_world_not_in_h4c_catalog() -> None:
    assert WORLD_BAYES_H4_SPARSE_GEO not in H4C_WORLD_IDS


def test_committed_h4c_artifact_loads() -> None:
    path = REPO_ROOT / DEFAULT_ARTIFACT_PATH
    if not path.exists():
        pytest.skip("H4c pilot artifact not materialized")
    art = load_h4c_extended_recovery_pilot_artifact(path)
    assert art["pilot_id"] == PILOT_ID
    assert art["production_promotion"] is False
    for w in art["worlds"]:
        assert w["approved_for_prod"] is False


@pytest.mark.pymc
@pytest.mark.slow
def test_live_h4c_clean_recovery() -> None:
    pytest.importorskip("pymc")
    from mmm.research.bayes_h3_sandbox.h4c_extended_recovery_pilot import run_h4c_extended_recovery_pilot
    from mmm.research.bayes_h3_sandbox.h4c_recovery_worlds import WORLD_BAYES_H4C_CLEAN_RECOVERY

    summary = run_h4c_extended_recovery_pilot((WORLD_BAYES_H4C_CLEAN_RECOVERY,), fast_mcmc=True)
    row = summary["worlds"][0]
    assert row["beta_gc_mae"] is not None
    assert row["reliability_map"]["hard_gate"] is False


@pytest.mark.pymc
@pytest.mark.slow
def test_live_h4c_weak_signal_optional() -> None:
    pytest.importorskip("pymc")
    from mmm.research.bayes_h3_sandbox.h4c_extended_recovery_pilot import run_h4c_extended_recovery_pilot
    from mmm.research.bayes_h3_sandbox.h4c_recovery_worlds import WORLD_BAYES_H4C_WEAK_SIGNAL

    summary = run_h4c_extended_recovery_pilot((WORLD_BAYES_H4C_WEAK_SIGNAL,), fast_mcmc=True)
    rm = summary["worlds"][0]["reliability_map"]
    assert rm["classification"] in ("weak_identification", "recovery_degraded", "recovery_moderate")
