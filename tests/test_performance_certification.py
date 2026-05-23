"""Operational trust: performance certification report."""

from __future__ import annotations

from mmm.config.schema import Framework, MMMConfig
from mmm.evaluation.performance_certification import SCENARIOS, build_performance_certification_report


def test_report_disabled_by_default() -> None:
    cfg = MMMConfig(framework=Framework.RIDGE_BO, data={"channel_columns": ["tv"]})
    rep = build_performance_certification_report(cfg)
    assert rep.get("enabled") is False


def test_scaling_scenario_creation() -> None:
    assert len(SCENARIOS) == 3
    names = {s.name for s in SCENARIOS}
    assert names == {"small", "medium", "large"}


def test_report_generation_small_only() -> None:
    cfg = MMMConfig(
        framework=Framework.RIDGE_BO,
        data={"channel_columns": ["tv"]},
        extensions={
            "performance_certification": {
                "enabled": True,
                "include_large_scenario": False,
                "n_trials_per_scenario": 1,
            },
        },
    )
    rep = build_performance_certification_report(cfg)
    assert rep["enabled"] is True
    assert rep["scenarios"]
    assert all(s.get("all_stages_success") for s in rep["scenarios"])
    assert "runtime_by_stage" in rep
    assert rep["scaling_summary"]["largest_rows"] > 0
