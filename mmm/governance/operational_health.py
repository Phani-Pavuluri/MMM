"""Operational health contract: aggregate extension signals into healthy / warning / blocked (Phase 9)."""

from __future__ import annotations

from typing import Any

from mmm.config.schema import MMMConfig, RunEnvironment

OPERATIONAL_HEALTH_VERSION = "operational_health_v1"


def compute_operational_health(
    *,
    config: MMMConfig,
    extension_report: dict[str, Any],
    optimization_gate_allowed: bool | None = None,
) -> dict[str, Any]:
    """
    Machine-readable run health for CI / promotion hooks (not a full monitoring platform).

    ``status``:
    - ``blocked``: hard failures (post-fit gate block, panel_qa block, prod optimization gate deny).
    - ``warning``: elevated risk (warn-level QA, falsification flags, identifiability stress, replay weak).
    - ``healthy``: no blocking or warning triggers in this policy map.
    """
    reasons: list[str] = []
    warns: list[str] = []
    pq = extension_report.get("panel_qa") if isinstance(extension_report.get("panel_qa"), dict) else {}
    pq_sev = str(pq.get("max_severity", "info")).lower()
    if pq_sev == "block":
        reasons.append("panel_qa_max_severity_block")

    pf = extension_report.get("post_fit_validation")
    if isinstance(pf, dict):
        psev = str(pf.get("release_gate_severity", "ok")).lower()
        if psev == "block":
            reasons.append("post_fit_validation_release_gate_block")
        elif psev == "warn":
            warns.append("post_fit_validation_release_gate_warn")

    fals = extension_report.get("falsification") if isinstance(extension_report.get("falsification"), dict) else {}
    flags = fals.get("flags") or []
    n_flags = len(flags) if isinstance(flags, list) else 0
    gv = config.extensions.governance
    cap = gv.falsification_max_allowed_flags_for_optimization
    if cap is not None and n_flags > int(cap):
        if config.run_environment == RunEnvironment.PROD:
            reasons.append(f"falsification_flags_{n_flags}_exceed_prod_cap_{cap}")
        else:
            warns.append(f"falsification_flags_{n_flags}_exceed_soft_cap_{cap}")

    min_pf = gv.falsification_prod_min_reported_placebo_families
    if config.run_environment == RunEnvironment.PROD and min_pf is not None:
        pfams = fals.get("placebo_families_run")
        n_fam = len(pfams) if isinstance(pfams, list) else 0
        if n_fam < int(min_pf):
            reasons.append(f"falsification_placebo_families_reported_{n_fam}_below_prod_min_{int(min_pf)}")

    ident = extension_report.get("identifiability") if isinstance(extension_report.get("identifiability"), dict) else {}
    score = float(ident.get("identifiability_score", 0.0))
    lim = float(gv.max_identifiability_risk)
    margin = float(gv.identifiability_decision_safety_margin)
    if score > lim * margin + 1e-9:
        warns.append(f"identifiability_score_{score:.3f}_above_scaled_limit_{lim * margin:.3f}")

    if pq_sev == "warn":
        warns.append("panel_qa_max_severity_warn")

    replay = extension_report.get("calibration_summary") if isinstance(extension_report.get("calibration_summary"), dict) else {}
    if replay and not replay.get("replay_calibration_active"):
        if config.run_environment == RunEnvironment.PROD:
            warns.append("prod_decision_paths_prefer_replay_calibration_evidence_missing_or_inactive")

    if optimization_gate_allowed is False:
        if config.run_environment == RunEnvironment.PROD:
            reasons.append("optimization_safety_gate_denied")
        else:
            warns.append("optimization_safety_gate_denied")

    if reasons:
        status = "blocked"
    elif warns:
        status = "warning"
    else:
        status = "healthy"

    return {
        "operational_health_version": OPERATIONAL_HEALTH_VERSION,
        "status": status,
        "block_reasons": reasons,
        "warning_reasons": warns,
        "inputs": {
            "panel_qa_max_severity": pq_sev,
            "post_fit_gate": (pf or {}).get("release_gate_severity") if isinstance(pf, dict) else None,
            "n_falsification_flags": n_flags,
            "n_falsification_placebo_families": len(fals.get("placebo_families_run"))
            if isinstance(fals.get("placebo_families_run"), list)
            else 0,
            "identifiability_score": score,
            "optimization_gate_allowed": optimization_gate_allowed,
        },
    }
