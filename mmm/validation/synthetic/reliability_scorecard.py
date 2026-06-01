"""Phase 4C — aggregate synthetic world certification into a reliability scorecard."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from mmm.validation.synthetic.certification_registry import REPORT_ARTIFACT_NAME
from mmm.validation.synthetic.certification_runner import run_world_certification
from mmm.validation.synthetic.dgp_materializer import materialize_dgp_world

SCORECARD_VERSION = "reliability_scorecard_v1.2.0"
RELIABILITY_SCORE_METHOD = "mean_capability_score_v1"
OVERALL_EVIDENCE_SCORE_METHOD = "mean_capability_score_v1"

# Phase 5D — metric class semantics (INV-056 / reliability_threshold_governance.md).
METRIC_CLASS_BY_CAPABILITY: dict[str, str] = {
    "structural_integrity": "structural",
    "transform_consistency": "diagnostic_attribution",
    "coefficient_recovery": "diagnostic_attribution",
    "delta_mu_recovery": "decision_grade",
    "optimizer_recovery": "decision_grade",
    "replay_recovery": "decision_grade",
    "drift_behavior": "trust_modifier",
    "identifiability_behavior": "trust_modifier",
    "platform_contract_compatibility": "structural",
    "artifact_integrity": "structural",
    "governance_reaction": "trust_modifier",
}

DECISION_GRADE_CAPABILITIES: frozenset[str] = frozenset(
    c for c, m in METRIC_CLASS_BY_CAPABILITY.items() if m == "decision_grade"
)
ATTRIBUTION_DIAGNOSTIC_CAPABILITIES: frozenset[str] = frozenset(
    c for c, m in METRIC_CLASS_BY_CAPABILITY.items() if m == "diagnostic_attribution"
)
STRUCTURAL_CAPABILITIES: frozenset[str] = frozenset(
    c for c, m in METRIC_CLASS_BY_CAPABILITY.items() if m == "structural"
)
TRUST_MODIFIER_CAPABILITIES: frozenset[str] = frozenset(
    c for c, m in METRIC_CLASS_BY_CAPABILITY.items() if m == "trust_modifier"
)

INTERPRETATION_RULES: tuple[str, ...] = (
    "decision_usable_may_coexist_with_weak_coef_recovery",
    "attribution_unsafe_when_diagnostic_score_low_even_if_decision_score_high",
    "severe_trust_modifiers_may_block_despite_delta_mu_pass",
    "curves_and_decomposition_remain_diagnostic_unless_attribution_profile_certified",
    "do_not_use_overall_evidence_score_alone_for_release",
)

OPEN_INVESTIGATIONS_MVP: tuple[str, ...] = ("INV-058", "INV-060")
SCORECARD_ARTIFACT_NAME = "synthetic_reliability_scorecard.json"

DEFAULT_RECOVERY_WORLD_IDS: tuple[str, ...] = (
    "WORLD-008-exact-recovery",
    "WORLD-009-optimizer-recovery",
    "WORLD-010-replay-recovery",
    "WORLD-011-drift-recovery",
    "WORLD-012-identifiability-recovery",
)

CAPABILITY_GROUPS: tuple[str, ...] = (
    "structural_integrity",
    "transform_consistency",
    "coefficient_recovery",
    "delta_mu_recovery",
    "optimizer_recovery",
    "replay_recovery",
    "drift_behavior",
    "identifiability_behavior",
    "platform_contract_compatibility",
    "artifact_integrity",
    "governance_reaction",
)

ScoreValue = float | None
OutcomeLabel = Literal["pass", "partial", "fail", "skipped", "unsupported"]

# Check IDs mapped to capability groups (MVP; first match wins for listing).
CHECK_CAPABILITY_MAP: dict[str, str] = {
    "CERT-4A-001": "structural_integrity",
    "CERT-4A-002": "structural_integrity",
    "CERT-4A-003": "replay_recovery",
    "CERT-4A-004": "transform_consistency",
    "CERT-4A-005": "structural_integrity",
    "CERT-4A-006": "governance_reaction",
    "CERT-4A-007": "artifact_integrity",
    "CERT-4A-008": "artifact_integrity",
    "CERT-4A-009": "platform_contract_compatibility",
    "CERT-4A-010": "platform_contract_compatibility",
    "CERT-4A-011": "platform_contract_compatibility",
    "CERT-4A-012": "platform_contract_compatibility",
    "CERT-4A-013": "platform_contract_compatibility",
    "VAL-001": "coefficient_recovery",
    "VAL-002": "transform_consistency",
    "VAL-003": "transform_consistency",
    "VAL-004": "delta_mu_recovery",
    "VAL-005": "optimizer_recovery",
    "VAL-006": "replay_recovery",
    "VAL-007": "artifact_integrity",
    "VAL-008": "governance_reaction",
    "VAL-009": "artifact_integrity",
    "VAL-010": "artifact_integrity",
    "VAL-011": "governance_reaction",
    "VAL-012": "drift_behavior",
    "VAL-013": "governance_reaction",
    "VAL-014": "governance_reaction",
    "REC-4B2-001": "coefficient_recovery",
    "REC-4B2-002": "transform_consistency",
    "REC-4B2-003": "transform_consistency",
    "REC-4B2-004": "transform_consistency",
    "REC-4B2-005": "delta_mu_recovery",
    "REC-4B2-006": "optimizer_recovery",
    "REC-4B2-007": "artifact_integrity",
    "REC-4B2-008": "artifact_integrity",
    "REC-4B3-OPT": "optimizer_recovery",
    "REC-4B3-OPT-PATH": "optimizer_recovery",
    "REC-4B4-REPLAY": "replay_recovery",
    "REC-4B4-LOAD": "replay_recovery",
    "REC-4B4-TRAIN": "replay_recovery",
    "REC-4B5-DRIFT": "drift_behavior",
    "REC-4B5-DRIFT-COEF": "coefficient_recovery",
    "REC-4B5-ID": "identifiability_behavior",
    "REC-4B5-ID-COEF": "coefficient_recovery",
}

# Capabilities expected to be exercised (scored if checks run) per recovery world.
WORLD_CAPABILITY_SCOPE: dict[str, frozenset[str]] = {
    "WORLD-008-exact-recovery": frozenset(
        {
            "structural_integrity",
            "transform_consistency",
            "coefficient_recovery",
            "delta_mu_recovery",
            "platform_contract_compatibility",
            "artifact_integrity",
        }
    ),
    "WORLD-009-optimizer-recovery": frozenset(
        {
            "structural_integrity",
            "optimizer_recovery",
            "platform_contract_compatibility",
        }
    ),
    "WORLD-010-replay-recovery": frozenset(
        {
            "structural_integrity",
            "replay_recovery",
            "platform_contract_compatibility",
            "artifact_integrity",
        }
    ),
    "WORLD-011-drift-recovery": frozenset(
        {
            "structural_integrity",
            "drift_behavior",
            "governance_reaction",
            "platform_contract_compatibility",
        }
    ),
    "WORLD-012-identifiability-recovery": frozenset(
        {
            "structural_integrity",
            "identifiability_behavior",
            "governance_reaction",
            "platform_contract_compatibility",
        }
    ),
}

# Skips that must not reduce reliability score when in scope for the world.
EXPECTED_SKIP_CHECKS: dict[str, frozenset[str]] = {
    "WORLD-008-exact-recovery": frozenset(
        {
            "VAL-005",
            "VAL-006",
            "VAL-007",
            "REC-4B2-006",
        }
    ),
    "WORLD-009-optimizer-recovery": frozenset(
        {
            "VAL-001",
            "VAL-002",
            "VAL-003",
            "VAL-004",
            "VAL-006",
        }
    ),
    "WORLD-010-replay-recovery": frozenset(
        {
            "VAL-001",
            "VAL-005",
        }
    ),
    "WORLD-011-drift-recovery": frozenset(
        {
            "VAL-001",
            "VAL-004",
            "VAL-005",
            "REC-4B5-DRIFT-COEF",
        }
    ),
    "WORLD-012-identifiability-recovery": frozenset(
        {
            "VAL-001",
            "VAL-004",
            "VAL-005",
            "REC-4B5-ID-COEF",
        }
    ),
}

PARTIAL_VALIDATION_IDS: frozenset[str] = frozenset()

REQUIRED_WARNINGS: tuple[str, ...] = (
    "synthetic_only",
    "no_causal_claim",
    "limited_world_count",
    "thresholds_TBD_v1_runtime",
    "no_monte_carlo",
    "no_real_experiment_validation",
)

def _mean_capability_subset(
    capability_scores: dict[str, ScoreValue],
    caps: frozenset[str],
) -> ScoreValue:
    vals = [float(capability_scores[c]) for c in caps if capability_scores.get(c) is not None]
    return float(sum(vals) / len(vals)) if vals else None


def _trust_modifier_status(
    capability_scores: dict[str, ScoreValue],
    capability_summary: dict[str, Any],
    *,
    world_reports: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    scores = [
        float(capability_scores[c])
        for c in TRUST_MODIFIER_CAPABILITIES
        if capability_scores.get(c) is not None
    ]
    failures: list[str] = []
    partials: list[str] = []
    for cap in TRUST_MODIFIER_CAPABILITIES:
        summary = capability_summary.get(cap) or {}
        failures.extend(summary.get("failures") or [])
        partials.extend(summary.get("partials") or [])

    if not scores and not failures and not partials:
        return {"status": "not_evaluated", "min_score": None, "warnings": []}

    min_score = min(scores) if scores else None
    warnings: list[str] = []
    if failures:
        warnings.append("trust_modifier_check_failures")
    if partials:
        warnings.append("trust_modifier_partial_execution")
    if min_score is not None and min_score < 0.5:
        warnings.append("trust_modifier_score_below_half")

    drift_severity = "none"
    if world_reports:
        from mmm.validation.synthetic.trust_report_semantics import _max_drift_severity

        drift_severity = _max_drift_severity(world_reports)
        if drift_severity == "severe":
            warnings.append("drift_severity_severe")
        elif drift_severity == "moderate":
            warnings.append("drift_severity_moderate")

    if (
        failures
        or (min_score is not None and min_score < 0.5)
        or drift_severity == "severe"
    ):
        status = "degraded"
    elif partials or (min_score is not None and min_score < 1.0) or drift_severity == "moderate":
        status = "caution"
    else:
        status = "acceptable"

    return {
        "status": status,
        "min_score": min_score,
        "drift_severity_max": drift_severity,
        "warnings": sorted(set(warnings)),
        "failures": failures,
        "partials": partials,
    }


def _release_readiness_interpretation(
    *,
    structural_score: ScoreValue,
    decision_score: ScoreValue,
    trust_status: dict[str, Any],
    failed_validations: list[str],
    overall_score: ScoreValue,
) -> str:
    if structural_score is not None and structural_score < 0.75:
        return "synthetic_evidence_structural_review_required"
    if trust_status.get("status") == "degraded":
        return "synthetic_evidence_trust_degraded_review_required"
    n_fail = len(failed_validations)
    if n_fail > 0:
        return "synthetic_evidence_mixed_review_required"
    if decision_score is None and overall_score is None:
        return "insufficient_evidence"
    primary = decision_score if decision_score is not None else overall_score
    assert primary is not None
    if primary < 0.75:
        return "synthetic_evidence_mixed_review_required"
    if primary < 0.9 or trust_status.get("status") == "caution":
        return "synthetic_evidence_partial_not_prod_ready"
    return "synthetic_evidence_favorable_still_not_prod_ready"


def _score_for_outcome(label: OutcomeLabel) -> ScoreValue:
    if label == "pass":
        return 1.0
    if label == "partial":
        return 0.5
    if label == "fail":
        return 0.0
    return None


def _classify_val_012_row(
    row: dict[str, Any],
    status: str,
) -> tuple[OutcomeLabel, ScoreValue, str]:
    details = row.get("details") if isinstance(row.get("details"), dict) else {}
    outcome = str(details.get("val_012_outcome", ""))
    if outcome == "warning":
        return "partial", 0.5, "val_012_warning"
    if outcome == "severe" or status == "fail":
        return "fail", 0.0, "val_012_severe"
    if outcome == "pass" or status == "pass":
        return "pass", 1.0, ""
    return "fail", 0.0, "val_012_fail"


def _classify_validation_row(
    row: dict[str, Any],
    *,
    world_id: str,
    mode: str = "recovery",
) -> tuple[OutcomeLabel, ScoreValue, str]:
    check_id = str(row.get("check_id", ""))
    status = str(row.get("status", ""))
    skip_reason = str(row.get("skip_reason") or "")

    if mode == "lattice_structural" and check_id.startswith("VAL-") and status == "skipped":
        return "skipped", None, "expected_lattice_deferred"

    details = row.get("details") if isinstance(row.get("details"), dict) else {}
    if check_id == "REC-4B5-DRIFT" or str(details.get("registry_validation_id", "")) == "VAL-012":
        return _classify_val_012_row(row, status)

    if check_id in PARTIAL_VALIDATION_IDS and status == "pass":
        return "partial", 0.5, "partial_execution"

    if status == "pass" and str(details.get("registry_validation_id", "")) in PARTIAL_VALIDATION_IDS:
        return "partial", 0.5, "partial_registry_execution"

    if status == "pass":
        return "pass", 1.0, ""
    if status == "fail":
        return "fail", 0.0, ""
    if status == "skipped":
        if check_id in EXPECTED_SKIP_CHECKS.get(world_id, frozenset()):
            return "skipped", None, "expected_skip"
        if skip_reason in ("not_applicable", "recovery_marked_unstable", "unsupported"):
            return "unsupported", None, skip_reason or "unsupported"
        return "skipped", None, skip_reason or "skipped"
    return "unsupported", None, status


def _capability_for_check(check_id: str) -> str | None:
    if check_id in CHECK_CAPABILITY_MAP:
        return CHECK_CAPABILITY_MAP[check_id]
    if check_id.startswith("CERT-4A-"):
        return "structural_integrity"
    if check_id.startswith("REC-4B2-00") and check_id not in CHECK_CAPABILITY_MAP:
        return "artifact_integrity"
    return None


def _load_or_run_report(
    bundle_dir: Path,
    *,
    materialize: bool,
    run_certification: bool,
) -> dict[str, Any]:
    report_path = bundle_dir / REPORT_ARTIFACT_NAME
    if not report_path.is_file():
        if materialize and (bundle_dir / "world_truth.json").is_file():
            materialize_dgp_world(bundle_dir, overwrite=True)
        if run_certification:
            run_world_certification(bundle_dir, write_report=True)
    if not report_path.is_file():
        raise FileNotFoundError(
            f"missing {REPORT_ARTIFACT_NAME} under {bundle_dir}; run certification first"
        )
    return json.loads(report_path.read_text(encoding="utf-8"))


def _contract_capability_score(report: dict[str, Any]) -> ScoreValue:
    contract = report.get("contract_compatibility") or {}
    if contract.get("passed") is True:
        return 1.0
    if contract.get("passed") is False:
        return 0.0
    return None


def build_scorecard_from_reports(
    reports: dict[str, dict[str, Any]],
    *,
    mode: Literal["recovery", "lattice_structural"] = "recovery",
    world_scope_overrides: dict[str, frozenset[str]] | None = None,
    expected_skip_overrides: dict[str, frozenset[str]] | None = None,
) -> dict[str, Any]:
    """
    Aggregate pre-loaded certification reports (e.g. lattice sweep worlds).

    ``lattice_structural`` scores CERT-4A / contract only; defers VAL-* skips without penalty.
    """
    world_reports = dict(reports)
    missing: list[str] = []

    status_counts: dict[str, int] = {
        "pass": 0,
        "partial": 0,
        "fail": 0,
        "skipped": 0,
        "unsupported": 0,
    }
    executed: list[str] = []
    skipped: list[dict[str, Any]] = []
    partial: list[str] = []
    failed: list[str] = []

    capability_buckets: dict[str, list[dict[str, Any]]] = {c: [] for c in CAPABILITY_GROUPS}
    all_scored: list[float] = []

    structural_caps = frozenset(
        {
            "structural_integrity",
            "transform_consistency",
            "platform_contract_compatibility",
            "artifact_integrity",
            "governance_reaction",
            "replay_recovery",
        }
    )

    for world_id, report in world_reports.items():
        if mode == "lattice_structural":
            in_scope = structural_caps
        elif world_scope_overrides is not None and world_id in world_scope_overrides:
            in_scope = world_scope_overrides[world_id]
        else:
            in_scope = WORLD_CAPABILITY_SCOPE.get(world_id, frozenset())

        contract_score = _contract_capability_score(report)
        if contract_score is not None and (
            mode == "lattice_structural" or "platform_contract_compatibility" in in_scope
        ):
            capability_buckets["platform_contract_compatibility"].append(
                {
                    "world_id": world_id,
                    "check_id": "contract_compatibility",
                    "outcome": "pass" if contract_score == 1.0 else "fail",
                    "score": contract_score,
                }
            )
            if mode == "lattice_structural" or "platform_contract_compatibility" in in_scope:
                all_scored.append(float(contract_score))
                status_counts["pass" if contract_score == 1.0 else "fail"] += 1

        recovery_rows = {
            str(r.get("check_id", "")): r for r in (report.get("recovery_validation_results") or [])
        }
        seen_checks: set[str] = set()

        for row in report.get("validation_results") or []:
            check_id = str(row.get("check_id", ""))
            if check_id in seen_checks:
                continue
            seen_checks.add(check_id)
            merged = dict(row)
            if check_id in recovery_rows and isinstance(recovery_rows[check_id].get("details"), dict):
                merged["details"] = recovery_rows[check_id]["details"]
            row = merged
            cap = _capability_for_check(check_id)
            if cap is None:
                continue
            if mode == "lattice_structural" and cap not in structural_caps:
                continue
            expected_skips = (
                expected_skip_overrides.get(world_id, frozenset())
                if expected_skip_overrides
                else EXPECTED_SKIP_CHECKS.get(world_id, frozenset())
            )
            if check_id in expected_skips and row.get("status") == "skipped":
                label, score, note = "skipped", None, "expected_skip"
            else:
                label, score, note = _classify_validation_row(row, world_id=world_id, mode=mode)
            status_counts[label] = status_counts.get(label, 0) + 1

            if row.get("status") != "skipped":
                executed.append(check_id)
            else:
                skipped.append(
                    {
                        "world_id": world_id,
                        "check_id": check_id,
                        "skip_reason": row.get("skip_reason"),
                        "message": row.get("message"),
                    }
                )

            if label == "partial":
                partial.append(f"{world_id}:{check_id}")
                details = row.get("details") if isinstance(row.get("details"), dict) else {}
                reg_val = str(details.get("registry_validation_id", ""))
                if reg_val:
                    partial.append(f"{world_id}:{reg_val}")
            if label == "fail":
                failed.append(f"{world_id}:{check_id}")

            scoped = mode == "lattice_structural" or cap in in_scope
            entry = {
                "world_id": world_id,
                "check_id": check_id,
                "outcome": label,
                "score": score,
                "in_scope": scoped,
                "note": note,
            }
            capability_buckets[cap].append(entry)
            if score is not None and scoped:
                all_scored.append(float(score))

    capability_summary: dict[str, Any] = {}
    capability_scores: dict[str, ScoreValue] = {}
    scored_caps: list[str] = []
    unscored_caps: list[str] = []

    for cap in CAPABILITY_GROUPS:
        entries = capability_buckets[cap]
        scoped_scores = [
            float(e["score"])
            for e in entries
            if e.get("in_scope") and e.get("score") is not None
        ]
        if scoped_scores:
            capability_scores[cap] = float(sum(scoped_scores) / len(scoped_scores))
            scored_caps.append(cap)
        else:
            capability_scores[cap] = None
            unscored_caps.append(cap)
        capability_summary[cap] = {
            "n_entries": len(entries),
            "n_scored": len(scoped_scores),
            "mean_score": capability_scores[cap],
            "worlds": sorted({str(e["world_id"]) for e in entries}),
            "failures": [f"{e['world_id']}:{e['check_id']}" for e in entries if e["outcome"] == "fail"],
            "partials": [f"{e['world_id']}:{e['check_id']}" for e in entries if e["outcome"] == "partial"],
        }

    overall_evidence_score = (
        float(sum(capability_scores[c] for c in scored_caps) / len(scored_caps))
        if scored_caps
        else None
    )
    decision_reliability_score = _mean_capability_subset(
        capability_scores, DECISION_GRADE_CAPABILITIES
    )
    attribution_diagnostic_score = _mean_capability_subset(
        capability_scores, ATTRIBUTION_DIAGNOSTIC_CAPABILITIES
    )
    structural_reliability_score = _mean_capability_subset(
        capability_scores, STRUCTURAL_CAPABILITIES
    )
    trust_modifier_status = _trust_modifier_status(
        capability_scores, capability_summary, world_reports=world_reports
    )
    total_in_scope = sum(
        len([e for e in capability_buckets[c] if e.get("in_scope")]) for c in CAPABILITY_GROUPS
    )
    coverage_ratio = float(len(all_scored) / total_in_scope) if total_in_scope else 0.0

    limitations = [
        "Synthetic certification evidence only — not causal validity or incrementality",
        "thresholds_TBD_v1_runtime provisional — not production gates (DR-04)",
        "metric_classes_separated — see reliability_threshold_governance.md",
        "no_monte_carlo",
        "no_real_experiment_validation",
    ]
    if mode == "lattice_structural":
        limitations.extend(
            [
                "Lattice structural sweep — behavioral VAL rows deferred (not scored)",
                "Smoke materializer panels — axes declared in truth, not full DGP simulation",
                "limited_world_count",
            ]
        )
    else:
        limitations.extend(
            [
                "Limited to recovery worlds (WORLD-008–012); not a Monte Carlo sweep",
                "VAL-012 uses drift_detection_runner on drift worlds",
            ]
        )
    if missing:
        limitations.append(f"Missing reports: {', '.join(missing)}")

    readiness = _release_readiness_interpretation(
        structural_score=structural_reliability_score,
        decision_score=decision_reliability_score,
        trust_status=trust_modifier_status,
        failed_validations=sorted(set(failed)),
        overall_score=overall_evidence_score,
    )

    metric_class_scores = {
        "decision_grade": decision_reliability_score,
        "diagnostic_attribution": attribution_diagnostic_score,
        "structural": structural_reliability_score,
        "trust_modifier_min": trust_modifier_status.get("min_score"),
    }

    from mmm.validation.synthetic.trust_report_semantics import enrich_scorecard_with_trust_report

    payload = {
        "scorecard_version": SCORECARD_VERSION,
        "world_ids": sorted(world_reports.keys()),
        "worlds_certified": sorted(world_reports.keys()),
        "worlds_missing": missing,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "capability_summary": capability_summary,
        "capability_scores": capability_scores,
        "metric_class_by_capability": dict(METRIC_CLASS_BY_CAPABILITY),
        "metric_class_scores": metric_class_scores,
        "decision_reliability_score": decision_reliability_score,
        "attribution_diagnostic_score": attribution_diagnostic_score,
        "structural_reliability_score": structural_reliability_score,
        "trust_modifier_status": trust_modifier_status,
        "overall_evidence_score": overall_evidence_score,
        "overall_evidence_score_method": OVERALL_EVIDENCE_SCORE_METHOD,
        "interpretation_rules": list(INTERPRETATION_RULES),
        "status_counts": status_counts,
        "executed_validations": sorted(set(executed)),
        "skipped_validations": skipped,
        "partial_validations": sorted(set(partial)),
        "failed_validations": sorted(set(failed)),
        "limitations": limitations,
        "open_investigations": list(OPEN_INVESTIGATIONS_MVP) if mode == "recovery" else [],
        "required_warnings": list(REQUIRED_WARNINGS),
        "reliability_score": overall_evidence_score,
        "reliability_score_method": RELIABILITY_SCORE_METHOD,
        "reliability_score_note": "deprecated_alias_of_overall_evidence_score",
        "scored_capabilities": scored_caps,
        "unscored_capabilities": unscored_caps,
        "coverage_ratio": coverage_ratio,
        "release_readiness_interpretation": readiness,
        "scorecard_mode": mode,
    }
    return enrich_scorecard_with_trust_report(payload, world_reports=world_reports)


def build_reliability_scorecard(
    repo_root: str | Path,
    *,
    world_ids: tuple[str, ...] | None = None,
    materialize_if_needed: bool = True,
    run_certification_if_needed: bool = True,
) -> dict[str, Any]:
    """
    Aggregate ``synthetic_world_certification_report.json`` artifacts for recovery worlds.

    Does not train models or claim production readiness — summarizes synthetic evidence only.
    """
    root = Path(repo_root)
    worlds = world_ids or DEFAULT_RECOVERY_WORLD_IDS
    world_reports: dict[str, dict[str, Any]] = {}
    missing: list[str] = []

    for wid in worlds:
        bundle = root / "validation" / "worlds" / wid
        if not bundle.is_dir():
            missing.append(wid)
            continue
        try:
            world_reports[wid] = _load_or_run_report(
                bundle,
                materialize=materialize_if_needed,
                run_certification=run_certification_if_needed,
            )
        except FileNotFoundError:
            missing.append(wid)

    partial_report = build_scorecard_from_reports(world_reports, mode="recovery")
    partial_report["worlds_missing"] = missing
    if missing:
        partial_report["limitations"] = list(partial_report.get("limitations", [])) + [
            f"Missing or uncertified worlds: {', '.join(missing)}"
        ]
    return partial_report


def write_reliability_scorecard(
    repo_root: str | Path,
    output_path: str | Path | None = None,
    **kwargs: Any,
) -> Path:
    """Build scorecard and write ``synthetic_reliability_scorecard.json``."""
    root = Path(repo_root)
    scorecard = build_reliability_scorecard(root, **kwargs)
    out = Path(output_path) if output_path is not None else root / "validation" / SCORECARD_ARTIFACT_NAME
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(scorecard, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out
