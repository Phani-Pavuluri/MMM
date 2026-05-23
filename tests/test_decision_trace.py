"""Operational trust: decision observability trace."""

from __future__ import annotations

from mmm.config.schema import Framework, GovernanceWorkflowConfig, MMMConfig
from mmm.governance.decision_trace import build_decision_trace, write_decision_trace_json


def test_decision_trace_generated() -> None:
    cfg = MMMConfig(
        framework=Framework.RIDGE_BO,
        data={"channel_columns": ["tv"]},
        governance=GovernanceWorkflowConfig(promotion_registry_path="/tmp/p.jsonl"),
    )
    er = {
        "calibration_summary": {"replay_train_loss": 0.2, "replay_generalization_gap": 0.05},
        "governance": {"approved_for_optimization": True},
        "model_release": {"state": "planning_allowed"},
        "data_fingerprint": {"sha256_combined": "fp1"},
        "seed_resolution": {"master_seed": 1},
    }
    sim = {"delta_mu": 10.0, "decision_safe": True, "planning_assumptions": {"controls_assumption": "observed"}}
    trace = build_decision_trace(
        config=cfg,
        extension_report=er,
        simulation_json=sim,
        decision_bundle={"decision_safe": True, "config_fingerprint_sha256": "c1"},
        promotion_lineage={"promotion_id": "pr1", "promoted_model_id": "m1"},
        surface="simulate",
        decision_id="dec-test-1",
    )
    assert trace["identity"]["decision_id"] == "dec-test-1"
    assert trace["lineage"]["data_fingerprint"]["sha256_combined"] == "fp1"
    assert trace["calibration"]["replay_summary"]
    assert trace["decision"]["expected_delta_mu"] == 10.0


def test_missing_optional_artifacts_handled() -> None:
    cfg = MMMConfig(framework=Framework.RIDGE_BO, data={"channel_columns": ["tv"]})
    trace = build_decision_trace(config=cfg, extension_report={}, surface="simulate")
    assert trace["trace_version"]
    assert trace["governance"]["unsupported_questions"] is None or isinstance(
        trace["governance"]["unsupported_questions"], list
    )


def test_write_trace_json(tmp_path) -> None:
    trace = build_decision_trace(
        config=MMMConfig(framework=Framework.RIDGE_BO, data={"channel_columns": ["tv"]}),
        extension_report={"data_fingerprint": {"sha256_combined": "x"}},
        surface="simulate",
    )
    write_decision_trace_json(trace, str(tmp_path / "out.json"))
    assert (tmp_path / "decision_trace.json").is_file()
    text = (tmp_path / "decision_trace.json").read_text(encoding="utf-8")
    assert "trace_version" in text
    assert "x" in text
