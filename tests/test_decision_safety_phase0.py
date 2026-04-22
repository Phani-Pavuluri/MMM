"""Phase 0: analysis-only freeze for decision-facing APIs."""

import pandas as pd
from typer.testing import CliRunner

from mmm.config.extensions import GovernanceConfig
from mmm.config.schema import MMMConfig
from mmm.data.schema import PanelSchema
from mmm.evaluation.baselines import BaselineComparisonReport
from mmm.governance.decision_safety import decision_safety_artifact
from mmm.governance.scorecard import build_scorecard
from mmm.services.governance_service import build_governance_bundle


def test_scorecard_blocks_optimization_when_freeze():
    sc = build_scorecard(
        cfg=GovernanceConfig(),
        fit_mae=0.1,
        baseline_mae=1.0,
        identifiability_score=0.1,
        calibration_loss=None,
        falsification_flags=[],
        beats_baselines=True,
        decision_api_freeze=True,
    )
    assert sc.approved_for_reporting is True
    assert sc.approved_for_optimization is False
    assert any("decision_safety_freeze" in n for n in sc.notes)


def test_scorecard_allows_optimization_when_unfrozen_and_passing():
    sc = build_scorecard(
        cfg=GovernanceConfig(),
        fit_mae=0.1,
        baseline_mae=1.0,
        identifiability_score=0.1,
        calibration_loss=None,
        falsification_flags=[],
        beats_baselines=True,
        decision_api_freeze=False,
    )
    assert sc.approved_for_optimization is True


def test_governance_bundle_includes_decision_safety_labels():
    cfg = MMMConfig(
        data={
            "path": None,
            "geo_column": "g",
            "week_column": "w",
            "target_column": "y",
            "channel_columns": ["c1"],
            "control_columns": [],
        },
        allow_unsafe_decision_apis=False,
    )
    schema = PanelSchema("g", "w", "y", ("c1",))
    panel = pd.DataFrame({"g": ["a"], "w": [0], "y": [1.0], "c1": [1.0]})
    bl = BaselineComparisonReport(0.1, 1.0, 1.0, 1.0, True, {})
    js = build_governance_bundle(
        config=cfg,
        panel=panel,
        schema=schema,
        yhat=panel["y"].to_numpy(),
        baselines=bl,
        identifiability_json={"identifiability_score": 0.0},
        falsification_flags=[],
        calibration_loss=None,
    )
    assert js["approved_for_optimization"] is False
    assert "decision_safety" in js
    assert js["decision_safety"]["labels"]["calibration"].startswith("not_decision_safe_yet")


def test_cli_optimize_budget_blocked_without_flags(tmp_path):
    from mmm.cli import main as cli_main

    yaml = tmp_path / "c.yaml"
    yaml.write_text(
        """
data:
  channel_columns: [c1]
  control_columns: []
budget:
  total_budget: 100
""",
        encoding="utf-8",
    )
    runner = CliRunner()
    result = runner.invoke(cli_main.app, ["optimize-budget", str(yaml)])
    assert result.exit_code == 2
    combined = (result.stderr or "") + (result.stdout or "")
    assert "blocked" in combined.lower() or "safety" in combined.lower()


def test_decision_safety_artifact_never_claims_coefficient_lift():
    art = decision_safety_artifact(allow_unsafe_decision_apis=False)
    assert art["coefficient_aligned_experiment_lift"] is False
    art2 = decision_safety_artifact(allow_unsafe_decision_apis=True)
    assert art2["coefficient_aligned_experiment_lift"] is False
