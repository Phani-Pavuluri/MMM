"""Single-file decision observability trace (audit; no auto-actions)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

TRACE_VERSION = "mmm_decision_trace_v1"


def build_decision_trace(
    *,
    config: Any,
    extension_report: dict[str, Any] | None,
    simulation_json: dict[str, Any] | None = None,
    decision_bundle: dict[str, Any] | None = None,
    optimizer_result: dict[str, Any] | None = None,
    promotion_lineage: dict[str, Any] | None = None,
    policy_checks: list[dict[str, Any]] | None = None,
    surface: str = "simulate",
    decision_id: str | None = None,
) -> dict[str, Any]:
    """
    Assemble an end-to-end trace for a prod or research decision path.

    Missing optional artifacts are omitted or marked unavailable — never invented.
    """
    er = extension_report if isinstance(extension_report, dict) else {}
    cal = er.get("calibration_summary") if isinstance(er.get("calibration_summary"), dict) else {}
    gov = er.get("governance") if isinstance(er.get("governance"), dict) else {}
    mr = er.get("model_release") if isinstance(er.get("model_release"), dict) else {}
    fp = er.get("data_fingerprint") or er.get("panel_fingerprint")
    if decision_bundle and not fp:
        fp = decision_bundle.get("data_fingerprint") or decision_bundle.get("panel_fingerprint")
    readiness = er.get("calibration_readiness_report")
    cont_val = er.get("continuous_validation_report")
    repro = er.get("reproducibility_certification_report")

    promo = promotion_lineage or {}
    if not promo and isinstance(decision_bundle, dict):
        promo = {
            k: decision_bundle[k]
            for k in (
                "promoted_model_id",
                "promotion_id",
                "promotion_registry_ref",
                "promotion_fingerprint_match",
                "promotion_expiration_date",
                "rollback_lineage",
            )
            if k in decision_bundle
        }

    sim = simulation_json or {}
    opt = optimizer_result or {}
    sim_at_raw = opt.get("simulation_at_recommendation")
    sim_at = sim_at_raw if isinstance(sim_at_raw, dict) else {}

    replay_gap = {
        "replay_generalization_gap": cal.get("replay_generalization_gap"),
        "replay_generalization_gap_severity": cal.get("replay_generalization_gap_severity"),
        "replay_holdout_available": cal.get("replay_holdout_available"),
        "replay_overfit_warning": cal.get("replay_overfit_warning"),
    }

    evidence_summary = er.get("evidence_weighted_replay_summary")
    if not evidence_summary and isinstance(cal.get("replay_meta"), dict):
        evidence_summary = cal["replay_meta"].get("evidence_weighted_replay_summary")

    freshness: dict[str, Any] = {}
    if isinstance(readiness, dict):
        freshness = {
            "stale_calibration_warning": readiness.get("stale_calibration_warning"),
            "coefficient_shift_score": readiness.get("coefficient_shift_score"),
            "replay_miss_rate": readiness.get("replay_miss_rate"),
            "recommended_action": readiness.get("recommended_action"),
        }
    elif isinstance(cont_val, dict):
        ef = cont_val.get("evidence_freshness_report") or {}
        freshness = {"evidence_freshness": ef, "model_trust_score": cont_val.get("model_trust_score")}

    decision_block: dict[str, Any] = {
        "surface": surface,
        "baseline_assumptions": sim.get("planning_assumptions") or sim.get("baseline_definition"),
        "constraints": {
            "total_budget": getattr(getattr(config, "budget", None), "total_budget", None),
            "channel_min": getattr(getattr(config, "budget", None), "channel_min", None),
            "channel_max": getattr(getattr(config, "budget", None), "channel_max", None),
        },
        "optimizer_settings": {
            "total_budget": opt.get("total_budget") or getattr(getattr(config, "budget", None), "total_budget", None),
            "optimizer_success": opt.get("optimizer_success", opt.get("success")),
        },
        "expected_delta_mu": sim.get("delta_mu") or sim_at.get("delta_mu"),
        "roi_summaries": sim.get("roi") or sim_at.get("roi"),
        "decision_safe": (
            sim.get("decision_safe")
            if sim
            else (decision_bundle.get("decision_safe") if decision_bundle else None)
        ),
        "decision_safe_reasons": sim.get("decision_safe_reasons") or sim.get("unsupported_questions"),
        "policy_checks": policy_checks or [],
    }

    return {
        "trace_version": TRACE_VERSION,
        "decision_id": decision_id or str(uuid4()),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "identity": {
            "decision_id": decision_id or None,
            "promoted_model_id": promo.get("promoted_model_id"),
            "promotion_id": promo.get("promotion_id"),
            "run_id": getattr(config, "run_id", None),
            "surface": surface,
        },
        "lineage": {
            "data_fingerprint": fp,
            "config_fingerprint_sha256": (
                decision_bundle.get("config_fingerprint_sha256") if decision_bundle else None
            ),
            "seed_resolution": er.get("seed_resolution") or (decision_bundle or {}).get("seed_resolution"),
            "artifact_references": {
                "extension_report_keys": sorted(er.keys())[:40],
                "bundle_version": decision_bundle.get("bundle_version") if decision_bundle else None,
            },
            "reproducibility_status": repro.get("reproducibility_status") if isinstance(repro, dict) else None,
        },
        "calibration": {
            "replay_summary": {
                k: cal[k]
                for k in cal
                if str(k).startswith("replay_") or k in ("n_units", "calibration_score_source")
            },
            "replay_gap_summary": replay_gap,
            "evidence_summary": evidence_summary,
            "calibration_freshness": freshness,
        },
        "decision": decision_block,
        "governance": {
            "unsupported_questions": (
                (decision_bundle or {}).get("unsupported_questions")
                or sim.get("unsupported_questions")
                or er.get("unsupported_questions")
            ),
            "warnings": list(er.get("legacy_replay_warnings") or [])[:20],
            "approval_metadata": {
                "approved_for_optimization": gov.get("approved_for_optimization"),
                "approved_for_reporting": gov.get("approved_for_reporting"),
                "model_release_state": mr.get("state"),
                "model_release_reasons": mr.get("reasons"),
            },
        },
    }


def write_decision_trace_json(trace: dict[str, Any], path: str) -> None:
    """Persist trace as ``decision_trace.json`` (stable formatting)."""
    import json
    from pathlib import Path

    p = Path(path)
    if p.name == "decision_trace.json":
        target = p
    elif p.suffix == ".json":
        target = p.parent / "decision_trace.json"
    else:
        target = p / "decision_trace.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(trace, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
