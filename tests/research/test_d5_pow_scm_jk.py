"""D5-POW SCM+UnitJackknife research validation (Track D lane)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mmm.research.track_d.d5_pow import (
    D5_POW_ID,
    DEFAULT_RESULTS_PATH,
    build_d5_pow_results,
    load_d5_pow_results,
)
from mmm.research.track_d.scm_jackknife import ScmJackknifeSpec, scm_unit_jackknife_readout, simulate_unit_panel

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_build_results_governance_flags() -> None:
    res = build_d5_pow_results(
        injection_grid=(0.0, 0.1),
        replicate_seeds=(101, 102),
    )
    assert res["investigation_id"] == D5_POW_ID
    assert res["hard_gate"] is False
    assert res["approved_for_prod"] is False
    assert res["production_promotion"] is False
    assert res["prod_decisioning_allowed"] is False


def test_point_recovery_tracks_injection() -> None:
    res = build_d5_pow_results(
        injection_grid=(0.0, 0.05, 0.15),
        replicate_seeds=(200, 201, 202),
    )
    assert res["point_recovery"]["tracks_injection_grid"] is True
    corr = res["point_recovery"]["injection_vs_point_mean_correlation"]
    assert corr is not None and corr > 0.9


def test_stop_condition_null_monitor_only() -> None:
    res = build_d5_pow_results(injection_grid=(0.0,), replicate_seeds=(1,))
    disp = res["recommended_disposition"]
    assert disp["scm_jk_supports_null_monitor_only"] is True
    assert disp["scm_jk_supports_power_mde_interpretation"] is False
    assert disp["requires_different_readout_aligned_power_metric"] is True


def test_interval_detection_not_validated_as_power_gate() -> None:
    res = build_d5_pow_results()
    assert res["interval_degeneracy_diagnosis"]["interval_excludes_zero_valid_as_detection_criterion"] is False


def test_committed_results_artifact_if_present() -> None:
    path = REPO_ROOT / DEFAULT_RESULTS_PATH
    if not path.exists():
        pytest.skip("D5_POW_results.json not materialized")
    art = load_d5_pow_results(path)
    assert art["results_id"] == "D5_POW_results"
    assert len(art["simulation_spec"]["injection_grid"]) >= 4
    assert art["measurement_instrument"] == "SCM+UnitJackKnife"


def test_jackknife_readout_finite() -> None:
    import numpy as np

    rng = np.random.default_rng(0)
    spec = ScmJackknifeSpec(n_control_units=5)
    panels = simulate_unit_panel(spec, injected_lift=0.1, rng=rng)
    readout = scm_unit_jackknife_readout(*panels)
    assert readout.point_effect == readout.point_effect
    assert readout.ci_low <= readout.ci_high


def test_build_deterministic() -> None:
    a = build_d5_pow_results(injection_grid=(0.0, 0.1), replicate_seeds=(1, 2))
    b = build_d5_pow_results(injection_grid=(0.0, 0.1), replicate_seeds=(1, 2))
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
