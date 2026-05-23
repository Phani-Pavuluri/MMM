"""Deterministic replay / decision reproducibility certification (no self-pass)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

REPORT_VERSION = "mmm_reproducibility_certification_v2"

GOVERNANCE_WARNINGS: tuple[str, ...] = (
    "Reproducibility certification checks artifact equivalence only — not causal validity.",
    "Self-certification records a snapshot only; identical_output requires an independent reference run.",
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
        bp = rfs.get("best_params")
        if isinstance(bp, dict):
            return dict(bp)
    tp = (extension_report or {}).get("transform_policy")
    if isinstance(tp, dict):
        return {k: tp[k] for k in ("decay", "hill_half", "hill_slope", "adstock", "saturation") if k in tp}
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
    delta = sim_at.get("delta_mu") if isinstance(sim_at, dict) else None
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


def _fingerprint_token(extension_report: dict[str, Any] | None) -> str | None:
    er = extension_report or {}
    fp = er.get("data_fingerprint") or er.get("panel_fingerprint")
    if isinstance(fp, dict):
        return str(fp.get("sha256_combined") or fp.get("sha256_panel_keycols_sorted_csv") or "") or None
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
        "panel_fingerprint_sha": _fingerprint_token(er),
        "raw": {
            "coefficients": coef,
            "transform_parameters": transform,
            "replay_summary": _replay_snapshot(er),
            "optimizer_output": _optimizer_snapshot(optimizer_result),
            "promotion_lineage": _promotion_snapshot(promotion_lineage, er),
        },
    }
    return components


def load_reference_run_snapshot(reference_run_path: str | Path) -> dict[str, Any]:
    """Load ``extension_report.json`` from an independent training run directory."""
    path = Path(reference_run_path)
    if path.is_dir():
        path = path / "extension_report.json"
    if not path.is_file():
        raise FileNotFoundError(f"reference run extension report not found: {path}")
    er = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(er, dict):
        raise ValueError("reference extension report must be a JSON object")
    return extract_reproducibility_snapshot(extension_report=er)


def compare_reproducibility_snapshots(
    reference: dict[str, Any],
    candidate: dict[str, Any],
    *,
    atol: float = 1e-9,
    rtol: float = 1e-6,
) -> dict[str, Any]:
    """Compare two snapshots; return match flags per component."""
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
    coefficients_match = True
    if ref_coef and cand_coef:
        rc = np.asarray(ref_coef.get("coef", []), dtype=float)
        cc = np.asarray(cand_coef.get("coef", []), dtype=float)
        if rc.shape == cc.shape:
            if not np.allclose(rc, cc, atol=atol, rtol=rtol):
                coefficients_match = False
                mismatched.append("coefficients_numeric")
                delta = cc - rc
                coef_deltas = {
                    "max_abs_delta": float(np.max(np.abs(delta))) if delta.size else 0.0,
                    "mean_abs_delta": float(np.mean(np.abs(delta))) if delta.size else 0.0,
                }
        else:
            coefficients_match = False
            mismatched.append("coefficients_shape")
            coef_deltas = {"reference_len": int(rc.size), "candidate_len": int(cc.size)}

    design_matrix_match = reference.get("design_matrix_fingerprint") == candidate.get("design_matrix_fingerprint")
    if not design_matrix_match and "design_matrix_fingerprint" not in mismatched:
        mismatched.append("design_matrix_fingerprint")

    ref_fp = reference.get("panel_fingerprint_sha")
    cand_fp = candidate.get("panel_fingerprint_sha")
    fingerprint_match = ref_fp == cand_fp if ref_fp and cand_fp else ref_fp == cand_fp
    if ref_fp and cand_fp and ref_fp != cand_fp:
        mismatched.append("panel_fingerprint")

    decision_output_match = True
    ref_opt = ref_raw.get("optimizer_output") or {}
    cand_opt = cand_raw.get("optimizer_output") or {}
    if ref_opt and cand_opt:
        rd = ref_opt.get("delta_mu")
        cd = cand_opt.get("delta_mu")
        if rd is not None and cd is not None and not np.isclose(float(rd), float(cd), atol=atol, rtol=rtol):
            decision_output_match = False
            optimizer_deltas["delta_mu_delta"] = float(cd) - float(rd)
            if "optimizer_output" not in mismatched:
                mismatched.append("optimizer_output")
        ref_alloc = ref_opt.get("allocation")
        cand_alloc = cand_opt.get("allocation")
        if ref_alloc != cand_alloc:
            decision_output_match = False
            if "optimizer_output" not in mismatched:
                mismatched.append("optimizer_output")

    bundle_match = (
        reference.get("decision_bundle") == candidate.get("decision_bundle")
        if reference.get("decision_bundle") and candidate.get("decision_bundle")
        else None
    )
    identical = len(mismatched) == 0
    certification_status = "pass" if identical else "fail"
    return {
        "report_version": REPORT_VERSION,
        "diagnostic_only": True,
        "reproducibility_status": "certified" if identical else "mismatch",
        "certification_status": certification_status,
        "identical_output": identical if identical else False,
        "coefficients_match": coefficients_match,
        "design_matrix_match": design_matrix_match,
        "decision_output_match": decision_output_match,
        "fingerprint_match": fingerprint_match,
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
    reference_run_path: str | Path | None = None,
) -> dict[str, Any]:
    """
    Build reproducibility certification.

    - No ``reference_run_path``: self-certification snapshot only (``identical_output=null``).
    - With ``reference_run_path``: compare current snapshot to an independent run artifact.
    """
    if reference_run_path is not None:
        ref_snap = load_reference_run_snapshot(reference_run_path)
        cand_snap = candidate if candidate is not None else reference
        report = compare_reproducibility_snapshots(ref_snap, cand_snap)
        report["self_certification"] = False
        report["reference_run_path"] = str(reference_run_path)
        report["reproducibility_evidence"] = report.get("certification_status") == "pass"
        return report

    if candidate is not None:
        report = compare_reproducibility_snapshots(reference, candidate)
        report["self_certification"] = False
        report["reproducibility_evidence"] = report.get("certification_status") == "pass"
        return report

    return {
        "report_version": REPORT_VERSION,
        "diagnostic_only": True,
        "self_certification": True,
        "identical_output": None,
        "reproducibility_status": "snapshot_only",
        "certification_status": "incomplete",
        "reproducibility_evidence": False,
        "coefficients_match": None,
        "design_matrix_match": None,
        "decision_output_match": None,
        "fingerprint_match": None,
        "mismatched_components": [],
        "certification_warnings": list(GOVERNANCE_WARNINGS)
        + ["Self-certification alone is not reproducibility evidence; supply reference_run_path."],
        "reference_component_hashes": {k: reference.get(k) for k in _COMPONENT_KEYS},
    }


# Backward-compatible alias for tests comparing two explicit snapshots
def build_independent_reproducibility_report(
    *,
    reference: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    return build_reproducibility_certification_report(reference=reference, candidate=candidate)
