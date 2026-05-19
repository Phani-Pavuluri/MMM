"""Auto-generated model card."""

from __future__ import annotations

from mmm.governance.decision_uncertainty import build_decision_uncertainty
from mmm.reporting.model_card import generate_model_card, required_sections_present


def test_model_card_generation_and_sections() -> None:
    er = {
        "governance": {"approved_for_optimization": False, "approved_for_reporting": True},
        "calibration_summary": {"replay_calibration_active": False},
        "decision_uncertainty": build_decision_uncertainty(
            __import__("mmm.config.schema", fromlist=["MMMConfig"]).MMMConfig(data={"channel_columns": ["c1"]})
        ),
        "decision_bundle": {
            "framework": "ridge_bo",
            "run_environment": "research",
            "unsupported_questions": ["Curve-based ROI as budget truth."],
            "resolved_config_snapshot": {
                "data": {
                    "target_column": "revenue",
                    "channel_columns": ["search"],
                    "control_columns": [],
                }
            },
        },
    }
    md = generate_model_card(extension_report=er, decision_bundle=er["decision_bundle"])
    assert "# Model card" in md
    missing = required_sections_present(md)
    assert not missing, f"missing sections: {missing}"
    assert "Point estimates only" in md
