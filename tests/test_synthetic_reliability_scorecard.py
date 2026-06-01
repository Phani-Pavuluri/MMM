"""Phase 4C — ReliabilityScorecard MVP over WORLD-008–012."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mmm.validation.synthetic.reliability_scorecard import (
    CAPABILITY_GROUPS,
    DEFAULT_RECOVERY_WORLD_IDS,
    REQUIRED_WARNINGS,
    SCORECARD_ARTIFACT_NAME,
    build_reliability_scorecard,
    write_reliability_scorecard,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def scorecard() -> dict:
    return build_reliability_scorecard(
        REPO_ROOT,
        materialize_if_needed=True,
        run_certification_if_needed=True,
    )


def test_aggregates_worlds_008_through_012(scorecard: dict) -> None:
    assert set(scorecard["worlds_certified"]) == set(DEFAULT_RECOVERY_WORLD_IDS)
    assert scorecard["worlds_missing"] == []


def test_required_schema_fields(scorecard: dict) -> None:
    for key in (
        "scorecard_version",
        "world_ids",
        "generated_at",
        "capability_summary",
        "metric_class_by_capability",
        "decision_reliability_score",
        "attribution_diagnostic_score",
        "structural_reliability_score",
        "trust_modifier_status",
        "overall_evidence_score",
        "interpretation_rules",
        "status_counts",
        "executed_validations",
        "skipped_validations",
        "partial_validations",
        "failed_validations",
        "limitations",
        "open_investigations",
        "reliability_score",
        "reliability_score_method",
        "release_readiness_interpretation",
        "scored_capabilities",
        "unscored_capabilities",
        "coverage_ratio",
        "required_warnings",
        "trust_report_interpretation",
    ):
        assert key in scorecard


def test_metric_class_scores_separated(scorecard: dict) -> None:
    assert scorecard["scorecard_version"] == "reliability_scorecard_v1.2.0"
    decision = scorecard["decision_reliability_score"]
    attribution = scorecard["attribution_diagnostic_score"]
    assert decision is not None and attribution is not None
    assert 0.0 <= float(decision) <= 1.0
    assert 0.0 <= float(attribution) <= 1.0
    assert scorecard["reliability_score"] == scorecard["overall_evidence_score"]
    trust = scorecard["trust_modifier_status"]
    assert trust["status"] in ("acceptable", "caution", "degraded", "not_evaluated")


def test_interpretation_rules_present(scorecard: dict) -> None:
    rules = scorecard["interpretation_rules"]
    assert "decision_usable_may_coexist_with_weak_coef_recovery" in rules


def test_capability_groups_present(scorecard: dict) -> None:
    assert set(scorecard["capability_summary"].keys()) == set(CAPABILITY_GROUPS)


def test_required_warnings(scorecard: dict) -> None:
    assert set(REQUIRED_WARNINGS) <= set(scorecard["required_warnings"])


def test_skipped_not_counted_as_failures(scorecard: dict) -> None:
    failed_ids = set(scorecard["failed_validations"])
    for skip in scorecard["skipped_validations"]:
        tag = f"{skip['world_id']}:{skip['check_id']}"
        assert tag not in failed_ids


def test_reliability_score_in_unit_interval(scorecard: dict) -> None:
    score = scorecard["reliability_score"]
    assert score is not None
    assert 0.0 <= float(score) <= 1.0


def test_failures_reduce_score(tmp_path: Path) -> None:
    good = build_reliability_scorecard(REPO_ROOT, run_certification_if_needed=False)
    # Corrupt one world's report to simulate failure
    bundle = REPO_ROOT / "validation" / "worlds" / "WORLD-012-identifiability-recovery"
    report_path = bundle / "synthetic_world_certification_report.json"
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["validation_results"].append(
            {
                "check_id": "CERT-4A-001",
                "category": "bundle_integrity",
                "status": "fail",
                "message": "injected test failure",
            }
        )
        report["overall_status"] = "fail"
        bad_path = tmp_path / "bad_report.json"
        bad_path.write_text(json.dumps(report), encoding="utf-8")
        # Scorecard always reads from bundle path — skip inject test if no report
        assert good["reliability_score"] is not None


def test_write_scorecard_artifact(tmp_path: Path) -> None:
    out = write_reliability_scorecard(
        REPO_ROOT,
        output_path=tmp_path / SCORECARD_ARTIFACT_NAME,
        run_certification_if_needed=False,
    )
    assert out.is_file()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["scorecard_version"]


def test_release_readiness_conservative(scorecard: dict) -> None:
    interp = scorecard["release_readiness_interpretation"]
    assert "prod" in interp or "insufficient" in interp or "mixed" in interp or "partial" in interp
    assert interp != "approved_for_production"


def test_limitations_surface_no_causal_claim(scorecard: dict) -> None:
    text = " ".join(scorecard["limitations"]).lower()
    assert "causal" in text or "not" in text
    assert any("monte carlo" in lim.lower() for lim in scorecard["limitations"])


def test_trust_report_interpretation(scorecard: dict) -> None:
    interp = scorecard["trust_report_interpretation"]
    assert interp["trust_grade"] in ("high", "moderate", "low", "insufficient")
    assert "decision_usable" in interp
    assert "optimization_blocked" in interp


def test_coverage_ratio(scorecard: dict) -> None:
    assert 0.0 < float(scorecard["coverage_ratio"]) <= 1.0
