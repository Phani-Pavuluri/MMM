"""Operational trust: reproducibility certification."""

from __future__ import annotations

import copy
import json
from pathlib import Path
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


def test_self_certification_does_not_self_pass() -> None:
    ref = _base_snap()
    rep = build_reproducibility_certification_report(reference=ref)
    assert rep["self_certification"] is True
    assert rep["identical_output"] is None
    assert rep["reproducibility_evidence"] is False
    assert rep["certification_status"] == "incomplete"


def test_independent_snapshots_match() -> None:
    ref = _base_snap()
    rep = build_reproducibility_certification_report(reference=ref, candidate=ref)
    assert rep["self_certification"] is False
    assert rep["identical_output"] is True
    assert rep["certification_status"] == "pass"
    assert rep["reproducibility_evidence"] is True


def test_changed_seed_mismatch() -> None:
    ref = _base_snap()
    cand = copy.deepcopy(ref)
    cand["seed_resolution"] = {"master_seed": 99}
    cand["design_matrix_fingerprint"] = "different"
    rep = compare_reproducibility_snapshots(ref, cand)
    assert rep["identical_output"] is False
    assert rep["design_matrix_match"] is False


def test_reference_run_path_comparison(tmp_path: Path) -> None:
    ref_er = {
        "ridge_fit_summary": {"coef": [0.1, 0.2], "intercept": [1.0]},
        "data_fingerprint": {"sha256_combined": "abc"},
        "calibration_summary": {"replay_train_loss": 0.1},
    }
    (tmp_path / "extension_report.json").write_text(json.dumps(ref_er), encoding="utf-8")
    cand = _base_snap()
    rep = build_reproducibility_certification_report(reference=cand, reference_run_path=tmp_path)
    assert rep["self_certification"] is False
    assert rep["reference_run_path"] == str(tmp_path)
    assert rep["coefficients_match"] is True


def test_reference_run_path_mismatch(tmp_path: Path) -> None:
    ref_er = {
        "ridge_fit_summary": {"coef": [0.9, 0.1], "intercept": [1.0]},
        "data_fingerprint": {"sha256_combined": "other"},
    }
    (tmp_path / "extension_report.json").write_text(json.dumps(ref_er), encoding="utf-8")
    cand = _base_snap()
    rep = build_reproducibility_certification_report(reference=cand, reference_run_path=tmp_path)
    assert rep["certification_status"] == "fail"
    assert rep["fingerprint_match"] is False
