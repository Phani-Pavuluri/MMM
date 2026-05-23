"""Aggregate production readiness certification for Ridge semi-log decision paths."""

from __future__ import annotations

from typing import Any

from mmm.config.schema import MMMConfig, RunEnvironment
from mmm.governance.synthetic_certification import run_synthetic_certification_suite

REPORT_VERSION = "mmm_production_readiness_v2"

_PLANNING_ALLOWED_STATES = frozenset({"planning_allowed", "approved_for_planning"})

GOVERNANCE_WARNINGS: tuple[str, ...] = (
    "Production readiness is a certification rollup — not proof of causal incrementality.",
    "When governance.require_production_certification=true, missing certification evidence blocks approval.",
    "Prod decide always surfaces a severe warning when approved_for_prod=false; strict mode also fails closed.",
    "optimizer_certification_mode=directional_fallback is advisory unless strict gate is enabled.",
    "Research extensions lower readiness_score but may remain enabled in non-prod environments.",
)


def _research_features_enabled(config: MMMConfig, extension_report: dict[str, Any]) -> list[str]:
    enabled: list[str] = []
    ext = config.extensions
    if ext.robust_optimization_research.enabled:
        enabled.append("robust_optimization_research")
    if ext.continuous_validation.enabled:
        enabled.append("continuous_validation")
    if ext.decision_validation.enabled:
        enabled.append("decision_validation")
    if ext.ridge_uncertainty_research.enabled:
        enabled.append("ridge_uncertainty_research")
    if config.framework.value == "bayesian":
        enabled.append("bayesian_framework")
    if str(config.model_form.value) == "log_log":
        enabled.append("log_log_model_form")
    if extension_report.get("robust_optimization_research") and "robust_optimization_research" not in enabled:
        enabled.append("robust_optimization_research")
    return enabled


def _decision_contract_valid(extension_report: dict[str, Any]) -> tuple[bool, list[str]]:
    missing: list[str] = []
    rfs = extension_report.get("ridge_fit_summary")
    if not isinstance(rfs, dict):
        missing.append("ridge_fit_summary")
    else:
        for key in ("coef", "intercept", "best_params", "model_form"):
            if rfs.get(key) is None:
                missing.append(f"ridge_fit_summary.{key}")
        bp = rfs.get("best_params")
        if isinstance(bp, dict):
            for k in ("decay", "hill_half", "hill_slope"):
                if k not in bp:
                    missing.append(f"ridge_fit_summary.best_params.{k}")
    if not extension_report.get("transform_policy"):
        missing.append("transform_policy")
    fp = extension_report.get("data_fingerprint") or extension_report.get("panel_fingerprint")
    if not isinstance(fp, dict) or not fp.get("sha256_combined"):
        missing.append("data_fingerprint.sha256_combined")
    return len(missing) == 0, missing


def _reproducibility_has_evidence(repro: dict[str, Any] | None) -> bool:
    if not isinstance(repro, dict):
        return False
    if repro.get("self_certification"):
        return bool(repro.get("reproducibility_evidence"))
    return repro.get("certification_status") == "pass" or repro.get("identical_output") is True


def build_production_readiness_report(
    config: MMMConfig,
    extension_report: dict[str, Any],
    *,
    synthetic_certification: dict[str, Any] | None = None,
    optimizer_certification: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Roll up certification artifacts into a single production readiness view.

    Failed **exact** synthetic certification always blocks ``approved_for_prod``.
    Stricter evidence requirements apply when ``governance.require_production_certification=true``.
    """
    er = extension_report
    blocked: list[str] = []
    warnings: list[str] = list(GOVERNANCE_WARNINGS)
    missing_requirements: list[str] = []
    strict = bool(config.governance.require_production_certification)

    contract_ok, contract_missing = _decision_contract_valid(er)
    if not contract_ok:
        missing_requirements.extend(contract_missing)
        blocked.append("decision_contract_incomplete")

    synth = synthetic_certification or er.get("synthetic_certification_report")
    if synth is None:
        synth = run_synthetic_certification_suite(mode="exact")
    synth_level = str(synth.get("certification_level", "incomplete"))
    if synth_level != "exact" or str(synth.get("certification_status")) != "pass":
        blocked.append("synthetic_certification_not_exact")
        missing_requirements.append("synthetic_certification_exact")
    if synth_level == "incomplete":
        warnings.append("synthetic_certification_level=incomplete")

    opt = optimizer_certification or er.get("optimizer_certification_report")
    if opt is None:
        if strict:
            blocked.append("optimizer_certification_missing")
            missing_requirements.append("optimizer_certification")
        else:
            warnings.append("optimizer_certification_report not present")
    elif str(opt.get("certification_status")) != "pass":
        blocked.append("optimizer_certification_failed")
        missing_requirements.append("optimizer_certification")
    else:
        opt_mode = str(opt.get("certification_mode", ""))
        if opt_mode == "directional_fallback":
            warnings.append(
                "optimizer_certification_proven_directional_fallback_only_not_analytic_tolerance"
            )
        elif opt_mode and opt_mode not in ("analytic_tolerance",):
            warnings.append(f"optimizer_certification_mode={opt_mode}")

    repro = er.get("reproducibility_certification_report")
    if config.extensions.reproducibility_certification.enabled or strict:
        if not isinstance(repro, dict):
            if strict:
                blocked.append("reproducibility_certification_missing")
                missing_requirements.append("reproducibility_certification")
        elif repro.get("self_certification") and not _reproducibility_has_evidence(repro):
            blocked.append("reproducibility_self_certification_only")
            missing_requirements.append("reproducibility_independent_run")
        elif not repro.get("self_certification") and repro.get("certification_status") != "pass":
            blocked.append("reproducibility_certification_failed")
    repro_status = "not_required"
    if isinstance(repro, dict):
        if repro.get("self_certification"):
            repro_status = "snapshot_only"
        else:
            repro_status = "pass" if _reproducibility_has_evidence(repro) else "fail"

    readiness = er.get("calibration_readiness_report") or {}
    calibration_current = not bool(readiness.get("stale_calibration_warning"))
    if not calibration_current:
        warnings.append(readiness.get("stale_calibration_warning") or "stale_calibration")
    if readiness.get("blocks_planning_allowed"):
        blocked.append("calibration_readiness_blocks_planning")

    cal = er.get("calibration_summary") or {}
    if str(cal.get("replay_generalization_gap_severity")) == "severe":
        blocked.append("severe_replay_generalization_gap")

    gov = er.get("governance") or {}
    if not gov.get("approved_for_optimization"):
        blocked.append("governance_not_approved_for_optimization")

    mr = er.get("model_release") or {}
    mr_state = str(mr.get("state", ""))
    if mr_state not in _PLANNING_ALLOWED_STATES:
        if strict:
            blocked.append("model_release_not_planning_allowed")
        else:
            warnings.append(f"model_release.state={mr_state!r}")

    if strict and not gov.get("approved_for_optimization"):
        blocked.append("planning_not_allowed")

    perf = er.get("performance_certification_report")
    if config.extensions.performance_certification.enabled:
        if not isinstance(perf, dict):
            missing_requirements.append("performance_certification_report")
        elif perf.get("any_failure"):
            warnings.append("performance_certification reported failures")

    if config.governance.require_promoted_model_for_prod_decision:
        warnings.append("promotion_required: verify promotion_record at decide time")

    research = _research_features_enabled(config, er)
    if research:
        warnings.append(f"research_features_enabled={research}")

    stress = er.get("decision_stress_report") or {}
    if str(stress.get("recommended_action")) == "block":
        warnings.append(f"decision_stress recommends block (severity={stress.get('stress_severity')!r})")

    score = 1.0
    score -= 0.15 * len(blocked)
    score -= 0.05 * len(warnings)
    score -= 0.1 * len(research)
    if synth_level == "incomplete":
        score -= 0.1
    score = max(0.0, min(1.0, score))

    approved = len(blocked) == 0

    return {
        "report_version": REPORT_VERSION,
        "approved_for_prod": approved,
        "blocked_reasons": blocked,
        "warnings": warnings,
        "research_features_enabled": research,
        "decision_contract_valid": contract_ok,
        "calibration_current": calibration_current,
        "reproducibility_status": repro_status,
        "certification_version": REPORT_VERSION,
        "readiness_score": round(score, 4),
        "missing_requirements": missing_requirements,
        "synthetic_certification_status": synth.get("certification_status"),
        "synthetic_certification_level": synth_level,
        "optimizer_certification_status": (opt or {}).get("certification_status"),
        "optimizer_certification_mode": (opt or {}).get("certification_mode"),
        "require_production_certification": strict,
        "governance_warnings": list(GOVERNANCE_WARNINGS),
    }


def production_readiness_decide_surface(
    config: MMMConfig,
    extension_report: dict[str, Any],
) -> dict[str, Any]:
    """
    Surface production readiness on prod decide payloads.

    Always emits a severe warning when ``approved_for_prod`` is false. Fail-closed blocking
    is enforced separately by ``require_production_readiness_for_prod_decide`` when strict.
    """
    if config.run_environment != RunEnvironment.PROD:
        return {
            "severe_warning": None,
            "warnings": [],
            "approved_for_prod": None,
            "blocked_reasons": [],
            "require_production_certification": False,
        }
    report = extension_report.get("production_readiness_report")
    if not isinstance(report, dict):
        report = build_production_readiness_report(config, extension_report)
    approved = bool(report.get("approved_for_prod"))
    blocked = [str(r) for r in (report.get("blocked_reasons") or [])]
    strict = bool(config.governance.require_production_certification)
    warnings: list[str] = []
    severe: str | None = None
    if not approved:
        reason_text = "; ".join(blocked) if blocked else "see production_readiness_report"
        if strict:
            gate_note = "Prod decide is blocked (governance.require_production_certification=true)."
        else:
            gate_note = (
                "Prod decide continues (advisory); set governance.require_production_certification=true "
                "to fail closed."
            )
        severe = f"PRODUCTION READINESS NOT APPROVED: {reason_text}. {gate_note}"
        warnings.append(f"production_readiness_report.approved_for_prod=false ({reason_text})")
    return {
        "severe_warning": severe,
        "warnings": warnings,
        "approved_for_prod": approved,
        "blocked_reasons": blocked,
        "require_production_certification": strict,
    }


def require_production_readiness_for_prod_decide(
    config: MMMConfig,
    extension_report: dict[str, Any],
) -> None:
    """Fail closed on prod decide when certification gate enabled and not approved."""
    from mmm.governance.policy import PolicyError

    if config.run_environment != RunEnvironment.PROD:
        return
    if not config.governance.require_production_certification:
        return
    report = extension_report.get("production_readiness_report")
    if not isinstance(report, dict):
        report = build_production_readiness_report(config, extension_report)
    if not report.get("approved_for_prod"):
        reasons = report.get("blocked_reasons") or report.get("missing_requirements") or []
        raise PolicyError(
            "production readiness certification failed (governance.require_production_certification=true): "
            + "; ".join(str(r) for r in reasons)
        )
