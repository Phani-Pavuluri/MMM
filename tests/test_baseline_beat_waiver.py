"""Baseline-beat waiver must surface in governance, model card, and governance_summary."""

from __future__ import annotations

import pandas as pd

from mmm.config.schema import MMMConfig
from mmm.data.schema import PanelSchema
from mmm.evaluation.baselines import BaselineComparisonReport
from mmm.governance.baseline_beat_waiver import BASELINE_BEAT_WAIVER_MESSAGE
from mmm.reporting.model_card import generate_model_card
from mmm.services.governance_service import build_governance_bundle


def test_waiver_in_governance_model_card_and_summary() -> None:
    base = MMMConfig(
        data={
            "path": None,
            "geo_column": "g",
            "week_column": "w",
            "target_column": "y",
            "channel_columns": ["c1"],
            "control_columns": [],
        },
    )
    cfg = base.model_copy(
        update={
            "extensions": base.extensions.model_copy(
                update={
                    "governance": base.extensions.governance.model_copy(
                        update={"require_beats_baselines_for_approval": False}
                    )
                }
            )
        }
    )
    schema = PanelSchema("g", "w", "y", ("c1",))
    panel = pd.DataFrame({"g": ["a"], "w": [0], "y": [1.0], "c1": [1.0]})
    bl = BaselineComparisonReport(0.1, 1.0, 1.0, 1.0, False, {})
    gov = build_governance_bundle(
        config=cfg,
        panel=panel,
        schema=schema,
        yhat=panel["y"].to_numpy(),
        baselines=bl,
        identifiability_json={"identifiability_score": 0.0},
        falsification_flags=[],
        calibration_loss=None,
    )
    assert gov.get("baseline_beat_waiver_active") is True
    assert any(BASELINE_BEAT_WAIVER_MESSAGE in n for n in gov.get("notes", []))
    summary = gov.get("governance_summary") or {}
    assert BASELINE_BEAT_WAIVER_MESSAGE in (summary.get("release_review_warnings") or [])

    card = generate_model_card(extension_report={"governance": gov})
    assert "baseline beat check waived" in card.lower()
