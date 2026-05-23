"""Exact optimizer certification on deterministic synthetic surfaces."""

from __future__ import annotations

from mmm.optimization.optimizer_certification import build_optimizer_certification_report


def test_optimizer_certification_passes_default_scenarios() -> None:
    rep = build_optimizer_certification_report(seed=42)
    assert rep["certification_status"] == "pass"
    assert rep["n_pass"] == rep["n_scenarios"]
    assert rep["certification_mode"] in ("analytic_tolerance", "directional_fallback")
    scenario_a = next(s for s in rep["scenarios"] if s["scenario"] == "A_log_elasticity")
    assert scenario_a["certification_status"] == "pass"
    assert scenario_a["certification_mode"] in ("analytic_tolerance", "directional_fallback")
    assert scenario_a["feasibility"] is True
    if scenario_a["certification_mode"] == "analytic_tolerance":
        assert scenario_a["optimum_distance"] <= rep["optimum_l1_tolerance"]


def test_optimizer_repeatability_scenario_present() -> None:
    rep = build_optimizer_certification_report()
    repeat = next(s for s in rep["scenarios"] if s["scenario"] == "repeatability")
    assert "allocation_l1_std" in repeat
    assert repeat["certification_status"] == "pass"
