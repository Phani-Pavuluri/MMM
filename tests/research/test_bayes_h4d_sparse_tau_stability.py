"""Bayes-H4d sparse/τ stability pilot (research-only, report-only)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mmm.research.bayes_h3_sandbox.h4_recovery_threshold_policy import (
    RECOVERY_CANDIDATE_WORLDS,
    STRESS_DIAGNOSTIC_WORLDS,
)
from mmm.research.bayes_h3_sandbox.h4c_recovery_worlds import (
    WORLD_BAYES_H4C_CLEAN_RECOVERY,
    WORLD_BAYES_H4C_SPARSE_RECOVERY,
)
from mmm.research.bayes_h3_sandbox.h4d_sparse_tau_stability import (
    DEFAULT_ARTIFACT_PATH,
    EXTENDED_ARTIFACT_PATH,
    H4D_WORLD_IDS,
    PILOT_ID,
    PILOT_ID_EXTENDED,
    TAU_GRID,
    build_h4d_summary,
    classify_metric_stability,
    compare_to_fast_pilot,
    h4d_tau_grid,
    h4d_world_ids,
    load_h4d_stability_artifact,
    recommend_disposition,
)
from mmm.research.bayes_h3_sandbox.recovery_worlds import WORLD_BAYES_H4_SPARSE_GEO

CLEAN = WORLD_BAYES_H4C_CLEAN_RECOVERY

REPO_ROOT = Path(__file__).resolve().parents[2]


def _fixture_run(
    world_id: str,
    *,
    nuts_seed: int,
    tau_label: str = "default",
    tau_sigma: float | None = None,
    beta_mae: float = 0.25,
) -> dict:
    from mmm.research.bayes_h3_sandbox.h4_recovery_threshold_policy import evaluate_world_against_policy

    metrics = {"beta_gc_mae": beta_mae, "mu_c_mae": 0.20}
    return {
        "world_id": world_id,
        "nuts_seed": nuts_seed,
        "panel_seed": 4400,
        "tau_label": tau_label,
        "tau_channel_prior_sigma": tau_sigma,
        "sparse_variant": None,
        "beta_gc_mae": beta_mae,
        "mu_c_mae": 0.20,
        "beta_gc_coverage_90": 0.4,
        "beta_interval_width_90_mean": 1.2,
        "shrinkage_ratio_sparse": 0.65,
        "shrinkage_ratio_sparse_vs_true_mu": 2.6,
        "world_role": "recovery_candidate" if world_id in RECOVERY_CANDIDATE_WORLDS else "stress_diagnostic",
        "h4c_classification": "recovery_candidate",
        "h4c_diagnostic_warnings": [],
        "warning_summary": {"count": 0, "types": [], "messages": []},
        "policy_evaluation": evaluate_world_against_policy(world_id, metrics),
        "approved_for_prod": False,
        "prod_decisioning_allowed": False,
        "production_promotion": False,
        "hard_gate": False,
        "research_only": True,
    }


def _fixture_rows_stable_clean() -> list[dict]:
    rows = []
    for seed in (4400, 4401, 4402):
        rows.append(_fixture_run(CLEAN, nuts_seed=seed, beta_mae=0.22 + 0.01 * (seed - 4400)))
    for seed in (4400, 4401, 4402):
        rows.append(
            _fixture_run(
                WORLD_BAYES_H4C_SPARSE_RECOVERY,
                nuts_seed=seed,
                beta_mae=0.35 + 0.08 * (seed - 4400),
            )
        )
    rows.append(_fixture_run(WORLD_BAYES_H4_SPARSE_GEO, nuts_seed=4400, beta_mae=0.55))
    return rows


def test_tau_grid_deterministic() -> None:
    assert h4d_tau_grid() == TAU_GRID
    assert len(TAU_GRID) == 4
    labels = [t["label"] for t in TAU_GRID]
    assert labels == ["default", "tau_0.30", "tau_0.20", "tau_0.15"]


def test_h4d_world_ids_separate_recovery_from_stress() -> None:
    ids = set(h4d_world_ids())
    assert CLEAN in ids
    assert WORLD_BAYES_H4_SPARSE_GEO in ids
    assert WORLD_BAYES_H4_SPARSE_GEO in STRESS_DIAGNOSTIC_WORLDS
    assert WORLD_BAYES_H4_SPARSE_GEO not in RECOVERY_CANDIDATE_WORLDS
    assert WORLD_BAYES_H4C_SPARSE_RECOVERY in RECOVERY_CANDIDATE_WORLDS


def test_build_summary_production_flags_false() -> None:
    summary = build_h4d_summary(_fixture_rows_stable_clean())
    assert summary["hard_gate"] is False
    assert summary["approved_for_prod"] is False
    assert summary["production_promotion"] is False
    assert summary["prod_decisioning_allowed"] is False
    assert summary["interpretation"]["does_not_claim_global_recovery"] is True


def test_build_summary_deterministic() -> None:
    rows = _fixture_rows_stable_clean()
    a = build_h4d_summary(rows)
    b = build_h4d_summary(rows)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_pooling_not_true_effect_gate_in_interpretation() -> None:
    summary = build_h4d_summary(_fixture_rows_stable_clean())
    assert summary["interpretation"]["pooling_not_true_effect_gate"] is True


def test_policy_evaluation_report_only() -> None:
    summary = build_h4d_summary(_fixture_rows_stable_clean())
    for row in summary["per_run"]:
        pe = row["policy_evaluation"]
        assert pe["hard_gate"] is False
        assert pe["production_promotion"] is False
        assert pe["global_model_failure"] is False


def test_recommend_disposition_unstable_clean() -> None:
    agg = {
        f"{CLEAN}|default|None": {
            "beta_gc_mae": {"mean": 0.3},
            "stability": {"beta_gc_mae": {"stable": False}},
        },
    }
    rec = recommend_disposition(agg)
    assert rec["disposition"] == "model_spec_work_needed_before_thresholds"
    assert rec["approved_for_prod"] is False


def test_recommend_disposition_tau_does_not_imply_production() -> None:
    agg = {
        f"{CLEAN}|default|None": {
            "beta_gc_mae": {"mean": 0.25},
            "stability": {"beta_gc_mae": {"stable": True}},
        },
        f"{CLEAN}|tau_0.15|None": {"beta_gc_mae": {"mean": 0.24}},
        f"{WORLD_BAYES_H4C_SPARSE_RECOVERY}|default|None": {
            "beta_gc_mae": {"mean": 0.40},
            "stability": {"beta_gc_mae": {"stable": True}},
        },
        f"{WORLD_BAYES_H4C_SPARSE_RECOVERY}|tau_0.15|None": {"beta_gc_mae": {"mean": 0.30}},
    }
    rec = recommend_disposition(agg)
    assert rec["disposition"] == "recommend_tau_prior_sandbox_research_update"
    assert rec["does_not_authorize_production"] is True
    assert rec["production_promotion"] is False


def test_classify_metric_stability_stable() -> None:
    s = classify_metric_stability([0.25, 0.26, 0.24])
    assert s["stable"] is True


def test_committed_artifact_if_present() -> None:
    path = REPO_ROOT / DEFAULT_ARTIFACT_PATH
    if not path.exists():
        pytest.skip("H4d artifact not materialized")
    art = load_h4d_stability_artifact(path)
    assert art["pilot_id"] == PILOT_ID
    assert art["hard_gate"] is False
    assert set(art["world_ids"]) == set(H4D_WORLD_IDS)


def test_extended_artifact_schema_if_present() -> None:
    path = REPO_ROOT / EXTENDED_ARTIFACT_PATH
    if not path.exists():
        pytest.skip("H4d extended artifact not materialized")
    art = load_h4d_stability_artifact(path)
    assert art["pilot_id"] == PILOT_ID_EXTENDED
    assert art["mcmc_profile"] == "extended"
    assert art["hard_gate"] is False
    assert art["approved_for_prod"] is False
    assert art["production_promotion"] is False
    assert art["prod_decisioning_allowed"] is False
    assert len(art["tau_grid"]) == 4
    assert art.get("comparison_to_fast_pilot")
    cmp = art["comparison_to_fast_pilot"]
    assert cmp["fast_pilot_available"] is True
    for row in art["per_run"]:
        assert row["policy_evaluation"]["hard_gate"] is False


def test_compare_to_fast_pilot_from_fixture() -> None:
    rows = _fixture_rows_stable_clean()
    extended_rows = []
    for r in rows:
        extended_rows.append({**r, "beta_gc_mae": float(r["beta_gc_mae"]) + 0.001})
    ext = build_h4d_summary(extended_rows, fast_mcmc=False, pilot_id=PILOT_ID_EXTENDED)
    # inject fast comparison without file by building partial
    cmp = compare_to_fast_pilot(
        {**ext, "aggregated_by_world_tau": ext["aggregated_by_world_tau"]},
        fast_artifact_path=REPO_ROOT / DEFAULT_ARTIFACT_PATH,
    )
    if (REPO_ROOT / DEFAULT_ARTIFACT_PATH).exists():
        assert "conclusions_hold" in cmp


@pytest.mark.slow
@pytest.mark.pymc
def test_h4d_single_recovery_and_sparse_run() -> None:
    pytest.importorskip("pymc")
    from mmm.research.bayes_h3_sandbox.h4d_sparse_tau_stability import _run_h4d_row
    from mmm.research.bayes_h3_sandbox.recovery_worlds import SAMPLER_FAST

    tau = {"label": "default", "tau_channel_prior_sigma": None}
    sampler = dict(SAMPLER_FAST)
    clean = _run_h4d_row(
        CLEAN,
        nuts_seed=4400,
        panel_seed=4400,
        tau_entry=tau,
        sampler=sampler,
        fast_mcmc=True,
    )
    sparse = _run_h4d_row(
        WORLD_BAYES_H4_SPARSE_GEO,
        nuts_seed=4400,
        panel_seed=4400,
        tau_entry=tau,
        sampler=sampler,
        fast_mcmc=True,
    )
    assert clean["approved_for_prod"] is False
    assert clean["world_role"] == "recovery_candidate"
    assert sparse["world_role"] == "stress_diagnostic"
    assert clean["beta_gc_mae"] is not None
