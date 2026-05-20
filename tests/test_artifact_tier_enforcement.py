"""Decision consumers reject wrong artifact tiers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mmm.artifacts.decision_inputs import load_decision_tier_artifact, load_training_extension_report
from mmm.governance.policy import PolicyError


def test_training_loader_rejects_decision_bundle_without_ridge_summary(tmp_path: Path) -> None:
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"artifact_tier": "decision", "delta_mu": 1.0}), encoding="utf-8")
    with pytest.raises(PolicyError, match="ridge_fit_summary"):
        load_training_extension_report(p)


def test_decision_loader_rejects_research_tier(tmp_path: Path) -> None:
    p = tmp_path / "research.json"
    p.write_text(
        json.dumps(
            {
                "artifact_tier": "research",
                "decision_safe": False,
                "baseline_mu": 1.0,
                "plan_mu": 1.1,
                "delta_mu": 0.1,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(PolicyError, match="tier"):
        load_decision_tier_artifact(p)


def test_training_loader_accepts_extension_report_shape(tmp_path: Path) -> None:
    p = tmp_path / "ext.json"
    p.write_text(
        json.dumps(
            {
                "artifact_tier": "research",
                "ridge_fit_summary": {"coef": [0.1], "intercept": [0.0], "best_params": {}},
            }
        ),
        encoding="utf-8",
    )
    data = load_training_extension_report(p)
    assert data["ridge_fit_summary"]["coef"]
