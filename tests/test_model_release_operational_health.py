from mmm.config.schema import MMMConfig, RunEnvironment
from mmm.governance.model_release import ModelReleaseState, infer_model_release_state


def test_operational_health_blocked_invalidates_release() -> None:
    cfg = MMMConfig(
        run_environment=RunEnvironment.RESEARCH,
        data={"channel_columns": ["c1"], "control_columns": []},
    )
    oh = {"status": "blocked", "block_reasons": ["synthetic_block"]}
    mr = infer_model_release_state(
        config=cfg,
        panel_qa_max_severity="info",
        governance_approved_for_optimization=True,
        governance_approved_for_reporting=True,
        ridge_fit_summary_present=True,
        operational_health=oh,
    )
    assert mr["state"] == ModelReleaseState.INVALIDATED.value
    assert any("operational_health" in r for r in mr["reasons"])
