"""Ridge uncertainty remains disclosure-only (no fabricated intervals)."""

from __future__ import annotations

from mmm.config.schema import Framework, MMMConfig
from mmm.governance.decision_uncertainty import build_decision_uncertainty


def test_ridge_uncertainty_not_available_with_investigation_notes() -> None:
    cfg = MMMConfig(framework=Framework.RIDGE_BO, data={"channel_columns": ["c1"]})
    block = build_decision_uncertainty(cfg)
    assert block["uncertainty_available"] is False
    assert block["uncertainty_unavailable"] is True
    assert block["methods_investigated"]["bootstrap_intervals"] == "not_implemented"
    assert "confidence interval" in block["disclosure_text"].lower() or "intervals" in block["disclosure_text"].lower()
