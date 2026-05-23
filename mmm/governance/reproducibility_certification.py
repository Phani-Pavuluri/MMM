"""Deterministic replay / decision reproducibility certification (no auto-actions)."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np

REPORT_VERSION = "mmm_reproducibility_certification_v1"

GOVERNANCE_WARNINGS: tuple[str, ...] = (
    "Reproducibility certification checks artifact equivalence only — not causal validity.",
    "Identical hashes require identical data, config, seeds, fingerprints, and promoted model lineage.",
)

_COMPONENT_KEYS = (
    "coefficients",
    "transform_parameters",
    "design_matrix_fingerprint",
    "replay_summary",
    "decision_bundle",
    "optimizer_output",
    "promotion_lineage",
    "decision_safe",
)


def _stable_hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def _coef_snapshot(fit_out: dict[str, Any] | None, extension_report: dict[str, Any] | None) -> dict[str, Any]:
    if fit_out and fit_out.get("artifacts") is not None:
        art = fit_out["artifacts"]
        coef = np.asarray(getattr(art, "coef", []), dtype=float).ravel().tolist()
        intercept = np.asarray(getattr(art, "intercept", []), dtype=float).ravel().tolist()
        return {"coef": coef, "intercept": intercept, "source": "fit_artifacts"}
    rfs = (extension_report or {}).get("ridge_fit_summary") or {}
    if isinstance(rfs, dict) and rfs.get("coef") is not None:
        return {
            "coef": list(np.asarray(rfs["coef"], dtype=float).ravel()),
            "intercept": list(np.asarray(rfs.get("intercept", []), dtype=float).ravel()),
            "source": "ridge_fit_summary",
        }
    return {}


def _transform_snapshot(fit_out: dict[str, Any] | None, extension_report: dict[str, Any] | None) -> dict[str, Any]:
    if fit_out and fit_out.get("artifacts") is not None:
        bp = getattr(fit_out["artifacts"], "best_params", None) or {}
        if isinstance(bp, dict) and bp:
            return dict(bp)
    rfs = (extension_report or {}).get("ridge_fit_summary") or {}
    if isinstance(rfs, dict):
        for key in ("decay", "hill_half", "hill_slope", "log_alpha"):
            if key in rfs:
                return {k: rfs[k] for k in ("decay", "hill_half", "hill_slope", "log_alpha") if k in rfs}
    return {}


def _design_matrix_fingerprint(
    extension_report: dict[str, Any] | None,
    transform_snapshot: dict[str, Any],
) -> str:
    fp = (extension_report or {}).get("data_fingerprint") or (extension_report or {}).get("panel_fingerprint")
    body = {"panel_fingerprint": fp, "transform_parameters": transform_snapshot}
    return _stable_hash(body)


def _replay_snapshot(extension_report: dict[str, Any] | None) -> dict[str, Any]:
    cal = (extension_report or {}).get("calibration_summary") or {}
    if not isinstance(cal, dict):
        return {}
    keys = (
        "replay_train_loss",
        "replay_holdout_loss",
        "replay_generalization_gap",
        "replay_generalization_gap_severity",
        "calibration_refit_mode",
        "replay_refit_mode",
        "replay_uses_full_panel_refit",
        "n_units",
        "weighted_replay_loss",
        "calibration_score_source",
    )
    return {k: cal[k] for k in keys if k in cal}


def _bundle_snapshot(decision_bundle: dict[str, Any] | None) -> str:
    if not isinstance(decision_bundle, dict):
        return ""
    keys = (
        "bundle_version",
        "config_fingerprint_sha256",
        "panel_fingerprint",
        "data_fingerprint",
        "decision_safe",
        "artifact_tier",
        "seed_resolution",
        "promotion_id",
        "promoted_model_id",
        "promotion_fingerprint_match",
    )
    subset = {k: decision_bundle.get(k) for k in keys if k in decision_bundle}
    return _stable_hash(subset)


def _optimizer_snapshot(optimizer_result: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(optimizer_result, dict):
        return {}
    alloc = optimizer_result.get("allocation") or optimizer_result.get("recommended_allocation")
    sim_at = optimizer_result.get("simulation_at_recommendation") or {}
    delta = None
    if isinstance(sim_at, dict):
        delta = sim_at.get("delta_mu")
    return {
        "allocation": alloc,
        "delta_mu": delta,
        "optimizer_success": optimizer_result.get("optimizer_success", optimizer_result.get("success")),
        "total_budget": optimizer_result.get("total_budget"),
    }


def _promotion_snapshot(
    promotion_lineage: dict[str, Any] | None,
    extension_report: dict[str, Any] | None,
) -> dict[str, Any]:
    if isinstance(promotion_lineage, dict) and promotion_lineage:
        return dict(promotion_lineage)
    promo = (extension_report or {}).get("promotion_record")
    return dict(promo) if isinstance(promo, dict) else {}


def _decision_safe_snapshot(
    decision_bundle: dict[str, Any] | None,
    simulation_json: dict[str, Any] | None,
) -> bool | None:
    if isinstance(decision_bundle, dict) and "decision_safe" in decision_bundle:
        return bool(decision_bundle["decision_safe"])
    if isinstance(simulation_json, dict) and "decision_safe" in simulation_json:
        return bool(simulation_json["decision_safe"])
    return None


def extract_reproducibility_snapshot(
    *,
    fit_out: dict[str, Any] | None = None,
    extension_report: dict[str, Any] | None = None,
    decision_bundle: dict[str, Any] | None = None,
    optimizer_result: dict[str, Any] | None = None,
    promotion_lineage: dict[str, Any] | None = None,
    simulation_json: dict[str, Any] | None = None,
    seed_resolution: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Canonical component hashes for reproducibility comparison."""
    coef = _coef_snapshot(fit_out, extension_report)
    transform = _transform_snapshot(fit_out, extension_report)
    er = extension_report or {}
    if seed_resolution is None and isinstance(er.get("seed_resolution"), dict):
        seed_resolution = er["seed_resolution"]
    components: dict[str, Any] = {
        "coefficients": _stable_hash(coef) if coef else None,
        "transform_parameters": _stable_hash(transform) if transform else None,
        "design_matrix_fingerprint": _design_matrix_fingerprint(er, transform),
        "replay_summary": _stable_hash(_replay_snapshot(er)) if _replay_snapshot(er) else None,
        "decision_bundle": _bundle_snapshot(decision_bundle) or None,
        "optimizer_output": _stable_hash(_optimizer_snapshot(optimizer_result))
        if _optimizer_snapshot(optimizer_result)
        else None,
        "promotion_lineage": _stable_hash(_promotion_snapshot(promotion_lineage, er))
        if _promotion_snapshot(promotion_lineage, er)
        else None,
        "decision_safe": _decision_safe_snapshot(decision_bundle, simulation_json),
        "seed_resolution": seed_resolution,
        "raw": {
            "coefficients": coef,
            "transform_parameters": transform,
            "replay_summary": _replay_snapshot(er),
            "optimizer_output": _optimizer_snapshot(optimizer_result),
            "promotion_lineage": _promotion_snapshot(promotion_lineage, er),
        },
    }
    return components


def compare_reproducibility_snapshots(
    reference: dict[str, Any],
    candidate: dict[str, Any],
    *,
    atol: float = 1e-9,
    rtol: float = 1e-6,
) -> dict[str, Any]:
    """Compare two snapshots; return certification report."""
    mismatched: list[str] = []
    coef_deltas: dict[str, Any] = {}
    optimizer_deltas: dict[str, Any] = {}
    warnings: list[str] = list(GOVERNANCE_WARNINGS)

    ref_raw = reference.get("raw") or {}
    cand_raw = candidate.get("raw") or {}

    for key in _COMPONENT_KEYS:
        ref_v = reference.get(key)
        cand_v = candidate.get(key)
        if ref_v is None and cand_v is None:
            continue
        if key == "decision_safe":
            if ref_v is not None and cand_v is not None and bool(ref_v) != bool(cand_v):
                mismatched.append(key)
            continue
        if ref_v != cand_v:
            mismatched.append(key)

    ref_coef = ref_raw.get("coefficients") or {}
    cand_coef = cand_raw.get("coefficients") or {}
    if ref_coef and cand_coef:
        rc = np.asarray(ref_coef.get("coef", []), dtype=float)
        cc = np.asarray(cand_coef.get("coef", []), dtype=float)
        if rc.shape == cc.shape:
            delta = cc - rc
            if not np.allclose(rc, cc, atol=atol, rtol=rtol):
                mismatched.append("coefficients_numeric")
                coef_deltas = {
                    "max_abs_delta": float(np.max(np.abs(delta))) if delta.size else 0.0,
                    "mean_abs_delta": float(np.mean(np.abs(delta))) if delta.size else 0.0,
                }
        else:
            mismatched.append("coefficients_shape")
            coef_deltas = {"reference_len": int(rc.size), "candidate_len": int(cc.size)}

    ref_opt = ref_raw.get("optimizer_output") or {}
    cand_opt = cand_raw.get("optimizer_output") or {}
    if ref_opt and cand_opt:
        rd = ref_opt.get("delta_mu")
        cd = cand_opt.get("delta_mu")
        if rd is not None and cd is not None and not np.isclose(float(rd), float(cd), atol=atol, rtol=rtol):
            optimizer_deltas["delta_mu_delta"] = float(cd) - float(rd)
            if "optimizer_output" not in mismatched:
                mismatched.append("optimizer_output")

    bundle_match = (
        reference.get("decision_bundle") == candidate.get("decision_bundle")
        if reference.get("decision_bundle") and candidate.get("decision_bundle")
        else None
    )
    identical = len(mismatched) == 0
    return {
        "report_version": REPORT_VERSION,
        "diagnostic_only": True,
        "reproducibility_status": "certified" if identical else "mismatch",
        "identical_output": identical,
        "mismatched_components": mismatched,
        "coefficient_deltas": coef_deltas,
        "optimizer_output_deltas": optimizer_deltas,
        "bundle_hash_match": bundle_match,
        "certification_warnings": warnings,
        "reference_component_hashes": {k: reference.get(k) for k in _COMPONENT_KEYS},
        "candidate_component_hashes": {k: candidate.get(k) for k in _COMPONENT_KEYS},
    }


def build_reproducibility_certification_report(
    *,
    reference: dict[str, Any],
    candidate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build report comparing reference vs candidate (self-compare when candidate omitted)."""
    cand = candidate if candidate is not None else reference
    report = compare_reproducibility_snapshots(reference, cand)
    report["self_certification"] = candidate is None
    return report
