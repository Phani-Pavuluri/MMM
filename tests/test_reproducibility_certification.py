"""Operational trust: reproducibility certification."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np

from mmm.governance.reproducibility_certification import (
    build_reproducibility_certification_report,
    compare_reproducibility_snapshots,
    extract_reproducibility_snapshot,
)


def _base_snap() -> dict[str, Any]:
    fit_out = {
        "artifacts": type(
            "A",
            (),
            {
                "coef": np.array([0.1, 0.2]),
                "intercept": np.array([1.0]),
                "best_params": {"decay": 0.5, "hill_half": 1.0, "hill_slope": 2.0},
            },
        )()
    }
    er = {
        "ridge_fit_summary": {"coef": [0.1, 0.2], "intercept": [1.0]},
        "calibration_summary": {"replay_train_loss": 0.1, "n_units": 2},
        "data_fingerprint": {"sha256_combined": "abc"},
        "seed_resolution": {"master_seed": 7},
    }
    bundle = {
        "bundle_version": "mmm_decision_bundle_v2",
        "config_fingerprint_sha256": "cfg1",
        "decision_safe": True,
        "panel_fingerprint": {"sha256_combined": "abc"},
    }
    return extract_reproducibility_snapshot(
        fit_out=fit_out,
        extension_report=er,
        decision_bundle=bundle,
        optimizer_result={"allocation": {"tv": 100.0}, "delta_mu": 5.0},
        promotion_lineage={"promotion_id": "p1"},
        simulation_json={"decision_safe": True},
    )


def test_identical_rerun_exact_match() -> None:
    ref = _base_snap()
    rep = build_reproducibility_certification_report(reference=ref)
    assert rep["identical_output"] is True
    assert rep["reproducibility_status"] == "certified"
    assert rep["mismatched_components"] == []


def test_changed_seed_mismatch() -> None:
    ref = _base_snap()
    cand = copy.deepcopy(ref)
    cand["seed_resolution"] = {"master_seed": 99}
    cand["design_matrix_fingerprint"] = "different"
    rep = compare_reproducibility_snapshots(ref, cand)
    assert rep["identical_output"] is False
    assert "design_matrix_fingerprint" in rep["mismatched_components"] or rep["mismatched_components"]


def test_changed_transform_params_mismatch() -> None:
    ref = _base_snap()
    cand = copy.deepcopy(ref)
    cand["raw"]["transform_parameters"] = {"decay": 0.9}
    cand["transform_parameters"] = "other"
    rep = compare_reproducibility_snapshots(ref, cand)
    assert rep["identical_output"] is False


def test_changed_promotion_mismatch() -> None:
    ref = _base_snap()
    cand = copy.deepcopy(ref)
    cand["raw"]["promotion_lineage"] = {"promotion_id": "p2"}
    cand["promotion_lineage"] = "other_hash"
    rep = compare_reproducibility_snapshots(ref, cand)
    assert rep["identical_output"] is False


def test_changed_config_mismatch() -> None:
    ref = _base_snap()
    cand = copy.deepcopy(ref)
    cand["decision_bundle"] = "other_bundle_hash"
    rep = compare_reproducibility_snapshots(ref, cand)
    assert rep["identical_output"] is False
    assert "decision_bundle" in rep["mismatched_components"]
