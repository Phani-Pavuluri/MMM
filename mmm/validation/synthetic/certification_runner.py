"""Phase 4A — structural world-bundle certification (no train/decide execution)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mmm.calibration.replay_estimand import (
    REPLAY_TRANSFORM_MODE_FULL_PANEL,
    ReplayEstimandSpec,
)
from mmm.calibration.units_io import load_calibration_units_from_json
from mmm.validation.synthetic._io import read_json, write_json
from mmm.validation.synthetic.certification_registry import (
    CERTIFICATION_RUNNER_VERSION,
    DEFERRED_CHECKS,
    PHASE_4A_CHECKS,
    REPORT_ARTIFACT_NAME,
    SkipReason,
    ValidationStatus,
)
from mmm.validation.synthetic.dgp_materializer import DGP_MATERIALIZATION_VERSION
from mmm.validation.synthetic.materializer import (
    MATERIALIZATION_VERSION,
    WORLD_CONTRACT_VERSION,
)
from mmm.validation.synthetic.recovery_certification import (
    is_recovery_eligible,
    recovery_replaces_deferred_val_ids,
    run_recovery_certification,
)
from mmm.validation.synthetic.replay_units import lift_scale_supported
from mmm.validation.synthetic.validator import validate_bundle, verify_checksums

ALLOWED_MATERIALIZATION_VERSIONS = frozenset(
    {MATERIALIZATION_VERSION, DGP_MATERIALIZATION_VERSION}
)

CANONICAL_MODEL_FORM = "semi_log"
CANONICAL_ADSTOCK = "geometric"
CANONICAL_SATURATION = "hill"
KNOWN_RELEASE_STATES = frozenset({"planning_allowed", "research_only", "blocked"})
KNOWN_GATE_EXPECTED = frozenset({"pass", "fail", "warn"})
SUPPORTED_ESTIMANDS = frozenset({"geo_time_ATT"})


@dataclass
class CheckOutcome:
    check_id: str
    category: str
    status: ValidationStatus
    message: str
    skip_reason: SkipReason | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "check_id": self.check_id,
            "category": self.category,
            "status": self.status,
            "message": self.message,
            "skip_reason": self.skip_reason,
        }


@dataclass
class CertificationRunResult:
    bundle_dir: Path
    report: dict[str, Any]
    overall_status: str
    passed: bool

    @property
    def report_path(self) -> Path:
        return self.bundle_dir / REPORT_ARTIFACT_NAME


def run_world_certification(
    bundle_dir: str | Path,
    *,
    write_report: bool = True,
    include_deferred_registry_rows: bool = True,
    include_recovery: bool | None = None,
) -> CertificationRunResult:
    """
    Run Phase 4A structural certification; optional Phase 4B-2 recovery on eligible worlds.

    Deferred VAL-* rows are reported as skipped with explicit reasons unless recovery
    replaces deferred VAL rows per world (e.g. VAL-001/004 on WORLD-008, VAL-005 on WORLD-009).
    """
    bundle = Path(bundle_dir)
    truth_path = bundle / "world_truth.json"
    if not truth_path.is_file():
        report = _error_report(bundle, message=f"missing {truth_path}")
        if write_report:
            write_json(bundle / REPORT_ARTIFACT_NAME, report)
        return CertificationRunResult(bundle, report, "error", False)

    truth = read_json(truth_path)
    meta = truth["metadata"]
    outcomes: list[CheckOutcome] = []

    for spec in PHASE_4A_CHECKS:
        outcomes.append(_run_phase_4a_check(bundle, truth, spec.check_id, spec.category))

    recovery_result = None
    run_recovery = include_recovery if include_recovery is not None else is_recovery_eligible(truth)
    if run_recovery and is_recovery_eligible(truth):
        recovery_result = run_recovery_certification(bundle, truth)

    deferred_specs = DEFERRED_CHECKS
    if include_deferred_registry_rows:
        if recovery_result is not None:
            replaced = recovery_replaces_deferred_val_ids(truth)
            deferred_specs = tuple(s for s in DEFERRED_CHECKS if s.check_id not in replaced)
            for rc in recovery_result.checks:
                outcomes.append(
                    CheckOutcome(
                        check_id=rc.check_id,
                        category=rc.category,
                        status=rc.status,  # type: ignore[arg-type]
                        message=rc.message,
                        skip_reason=rc.skip_reason,  # type: ignore[arg-type]
                    )
                )
        for spec in deferred_specs:
            outcomes.append(
                CheckOutcome(
                    check_id=spec.check_id,
                    category=spec.category,
                    status="skipped",
                    message=spec.description,
                    skip_reason=spec.default_skip_reason or "deferred",
                )
            )

    contract = _platform_contract_summary(outcomes)
    report = _build_report(bundle, truth, meta, outcomes, contract, recovery_result=recovery_result)
    overall = report["overall_status"]
    passed = overall == "pass"

    if write_report:
        write_json(bundle / REPORT_ARTIFACT_NAME, report)

    return CertificationRunResult(bundle, report, overall, passed)


def _run_phase_4a_check(
    bundle: Path,
    truth: dict[str, Any],
    check_id: str,
    category: str,
) -> CheckOutcome:
    runners = {
        "CERT-4A-001": _check_bundle_integrity,
        "CERT-4A-002": _check_checksum_reproducibility,
        "CERT-4A-003": _check_replay_loader_compatibility,
        "CERT-4A-004": _check_transform_truth_consistency,
        "CERT-4A-005": _check_metadata_consistency,
        "CERT-4A-006": _check_governance_warning_compatibility,
        "CERT-4A-007": _check_decision_truth_structure,
        "CERT-4A-008": _check_calibration_payload_compatibility,
        "CERT-4A-009": _check_decision_surface_compatibility,
        "CERT-4A-010": _check_estimand_compatibility,
        "CERT-4A-011": _check_calibration_signal_compatibility,
        "CERT-4A-012": _check_trust_report_compatibility,
        "CERT-4A-013": _check_release_gate_compatibility,
    }
    fn = runners.get(check_id)
    if fn is None:
        return CheckOutcome(check_id, category, "fail", f"unknown check {check_id}")
    return fn(bundle, truth, check_id, category)


def _check_bundle_integrity(bundle: Path, truth: dict[str, Any], cid: str, cat: str) -> CheckOutcome:
    try:
        result = validate_bundle(bundle, max_level=3)
    except json.JSONDecodeError as exc:
        return CheckOutcome(cid, cat, "fail", f"validator error: {exc}")
    if result.passed:
        return CheckOutcome(cid, cat, "pass", f"L1–L3 validator passed (world_id={result.world_id})")
    return CheckOutcome(cid, cat, "fail", f"validator failures: {result.hard_failures[:5]}")


def _check_checksum_reproducibility(bundle: Path, truth: dict[str, Any], cid: str, cat: str) -> CheckOutcome:
    failures = verify_checksums(bundle)
    if not failures:
        return CheckOutcome(cid, cat, "pass", "checksums match on-disk artifacts")
    return CheckOutcome(cid, cat, "fail", f"checksum mismatch: {failures}")


def _check_replay_loader_compatibility(bundle: Path, truth: dict[str, Any], cid: str, cat: str) -> CheckOutcome:
    units_truth = (truth.get("experiment_truth") or {}).get("units") or []
    replay_path = bundle / "replay_units.json"
    if not units_truth:
        if replay_path.is_file():
            return CheckOutcome(cid, cat, "fail", "replay_units.json present but experiment_truth.units empty")
        return CheckOutcome(
            cid,
            cat,
            "skipped",
            "no experiment units in truth",
            skip_reason="not_applicable",
        )
    if not replay_path.is_file():
        return CheckOutcome(cid, cat, "fail", "experiment_truth.units non-empty but replay_units.json missing")
    try:
        loaded = load_calibration_units_from_json(replay_path)
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        return CheckOutcome(cid, cat, "fail", f"replay loader error: {exc}")
    if len(loaded) != len(units_truth):
        return CheckOutcome(
            cid,
            cat,
            "fail",
            f"loader count {len(loaded)} != truth units {len(units_truth)}",
        )
    for u in loaded:
        ReplayEstimandSpec.from_dict(u.replay_estimand or {})
    return CheckOutcome(cid, cat, "pass", f"loaded {len(loaded)} replay unit(s)")


def _check_transform_truth_consistency(bundle: Path, truth: dict[str, Any], cid: str, cat: str) -> CheckOutcome:
    transform = truth.get("transform_truth") or {}
    outcome = truth.get("outcome_truth") or {}
    if str(outcome.get("model_form", "")) != CANONICAL_MODEL_FORM:
        return CheckOutcome(
            cid,
            cat,
            "fail",
            f"model_form must be {CANONICAL_MODEL_FORM!r}, got {outcome.get('model_form')!r}",
        )
    if str(transform.get("adstock_family", "")) != CANONICAL_ADSTOCK:
        return CheckOutcome(cid, cat, "fail", f"adstock_family must be {CANONICAL_ADSTOCK!r}")
    if str(transform.get("saturation_family", "")) != CANONICAL_SATURATION:
        return CheckOutcome(cid, cat, "fail", f"saturation_family must be {CANONICAL_SATURATION!r}")
    channels = list((truth.get("media_truth") or {}).get("channels") or [])
    decay = transform.get("adstock_decay_by_channel") or {}
    if set(decay.keys()) != set(channels):
        return CheckOutcome(cid, cat, "fail", "adstock_decay_by_channel keys != media channels")
    for ch, d in decay.items():
        if not (0.0 < float(d) < 1.0):
            return CheckOutcome(cid, cat, "fail", f"invalid adstock decay for {ch}: {d}")
    return CheckOutcome(cid, cat, "pass", "canonical semi_log + geometric + Hill transform truth")


def _check_metadata_consistency(bundle: Path, truth: dict[str, Any], cid: str, cat: str) -> CheckOutcome:
    meta_path = bundle / "metadata.json"
    if not meta_path.is_file():
        return CheckOutcome(cid, cat, "fail", "metadata.json missing")
    bundle_meta = read_json(meta_path)
    truth_meta = truth["metadata"]
    mismatches: list[str] = []
    for key in ("world_id", "world_version", "world_contract_version", "world_generator_version"):
        if bundle_meta.get(key) != truth_meta.get(key):
            mismatches.append(key)
    if int(bundle_meta.get("seed", -1)) != int(truth_meta.get("generation_seed", -2)):
        mismatches.append("seed/generation_seed")
    if bundle_meta.get("materialization_version") not in ALLOWED_MATERIALIZATION_VERSIONS:
        mismatches.append("materialization_version")
    if mismatches:
        return CheckOutcome(cid, cat, "fail", f"metadata mismatch fields: {mismatches}")
    return CheckOutcome(cid, cat, "pass", "bundle metadata matches world_truth.metadata")


def _check_governance_warning_compatibility(bundle: Path, truth: dict[str, Any], cid: str, cat: str) -> CheckOutcome:
    warnings = (truth.get("artifact_truth") or {}).get("expected_warnings") or []
    for i, w in enumerate(warnings):
        if not isinstance(w, dict):
            return CheckOutcome(cid, cat, "fail", f"expected_warnings[{i}] not an object")
        if not str(w.get("warning_id", "")).strip():
            return CheckOutcome(cid, cat, "fail", f"expected_warnings[{i}] missing warning_id")
        sev = str(w.get("severity", ""))
        if sev not in ("low", "moderate", "high", "severe"):
            return CheckOutcome(cid, cat, "fail", f"expected_warnings[{i}] invalid severity {sev!r}")
    return CheckOutcome(cid, cat, "pass", f"{len(warnings)} expected warning(s) well-formed")


def _check_decision_truth_structure(bundle: Path, truth: dict[str, Any], cid: str, cat: str) -> CheckOutcome:
    decision = truth.get("decision_truth") or {}
    scenarios = decision.get("scenarios") or []
    channels = set((truth.get("media_truth") or {}).get("channels") or [])
    for sc in scenarios:
        if not isinstance(sc, dict):
            return CheckOutcome(cid, cat, "fail", "decision scenario row not an object")
        cand = sc.get("candidate_spend_by_channel") or {}
        base = sc.get("baseline_spend_by_channel") or {}
        if not set(cand.keys()).issubset(channels):
            return CheckOutcome(cid, cat, "fail", f"scenario {sc.get('scenario_id')} candidate channels invalid")
        if not set(base.keys()).issubset(channels):
            return CheckOutcome(cid, cat, "fail", f"scenario {sc.get('scenario_id')} baseline channels invalid")
        if "true_delta_mu" not in sc:
            return CheckOutcome(cid, cat, "fail", f"scenario {sc.get('scenario_id')} missing true_delta_mu")
    return CheckOutcome(cid, cat, "pass", f"decision_truth valid ({len(scenarios)} scenario(s))")


def _check_calibration_payload_compatibility(bundle: Path, truth: dict[str, Any], cid: str, cat: str) -> CheckOutcome:
    replay_path = bundle / "replay_units.json"
    units_truth = (truth.get("experiment_truth") or {}).get("units") or []
    if not units_truth:
        return CheckOutcome(
            cid,
            cat,
            "skipped",
            "no replay payload required",
            skip_reason="not_applicable",
        )
    if not replay_path.is_file():
        return CheckOutcome(cid, cat, "fail", "missing replay_units.json")
    try:
        raw = json.loads(replay_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return CheckOutcome(cid, cat, "fail", f"malformed replay JSON: {exc}")
    if not isinstance(raw, list):
        return CheckOutcome(cid, cat, "fail", "replay_units.json must be a list")
    for row in raw:
        if not isinstance(row, dict):
            return CheckOutcome(cid, cat, "fail", "replay row must be object")
        if not str(row.get("payload_version", "")).strip():
            return CheckOutcome(cid, cat, "fail", "replay row missing payload_version")
    return CheckOutcome(cid, cat, "pass", "replay payload structure valid")


def _check_decision_surface_compatibility(bundle: Path, truth: dict[str, Any], cid: str, cat: str) -> CheckOutcome:
    outcome = truth.get("outcome_truth") or {}
    if str(outcome.get("model_form", "")) != CANONICAL_MODEL_FORM:
        return CheckOutcome(cid, cat, "fail", "DecisionSurface requires semi_log model_form in truth")
    units = (truth.get("experiment_truth") or {}).get("units") or []
    for u in units:
        mode = str(u.get("replay_transform_mode", REPLAY_TRANSFORM_MODE_FULL_PANEL))
        if mode != REPLAY_TRANSFORM_MODE_FULL_PANEL:
            return CheckOutcome(
                cid,
                cat,
                "fail",
                f"unit {u.get('unit_id')} replay_transform_mode must be full_panel_transform_estimand_mask",
            )
    return CheckOutcome(
        cid,
        cat,
        "pass",
        "DecisionSurface: semi_log + full-panel replay transform semantics",
    )


def _check_estimand_compatibility(bundle: Path, truth: dict[str, Any], cid: str, cat: str) -> CheckOutcome:
    units = (truth.get("experiment_truth") or {}).get("units") or []
    if not units:
        return CheckOutcome(
            cid,
            cat,
            "skipped",
            "no experiment units",
            skip_reason="not_applicable",
        )
    for u in units:
        est = str(u.get("estimand", "")).strip()
        if not est:
            return CheckOutcome(cid, cat, "fail", f"unit {u.get('unit_id')} missing estimand")
        if est not in SUPPORTED_ESTIMANDS:
            return CheckOutcome(cid, cat, "fail", f"unsupported estimand {est!r} on {u.get('unit_id')}")
    replay_path = bundle / "replay_units.json"
    if replay_path.is_file():
        try:
            loaded = load_calibration_units_from_json(replay_path)
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            return CheckOutcome(cid, cat, "fail", f"replay estimand load error: {exc}")
        for u in loaded:
            if str(u.estimand or "").strip() not in SUPPORTED_ESTIMANDS:
                return CheckOutcome(cid, cat, "fail", f"replay loader estimand invalid for {u.unit_id}")
    return CheckOutcome(cid, cat, "pass", "estimand declarations compatible")


def _check_calibration_signal_compatibility(bundle: Path, truth: dict[str, Any], cid: str, cat: str) -> CheckOutcome:
    units_truth = (truth.get("experiment_truth") or {}).get("units") or []
    if not units_truth:
        return CheckOutcome(
            cid,
            cat,
            "skipped",
            "no calibration signals",
            skip_reason="not_applicable",
        )
    replay_path = bundle / "replay_units.json"
    if not replay_path.is_file():
        return CheckOutcome(cid, cat, "fail", "missing replay_units.json for CalibrationSignal")
    try:
        loaded = load_calibration_units_from_json(replay_path)
    except ValueError as exc:
        return CheckOutcome(cid, cat, "fail", f"CalibrationSignal load failed: {exc}")
    target_col = str((truth.get("outcome_truth") or {}).get("target_column", ""))
    for u in loaded:
        if not u.treated_channel_names:
            return CheckOutcome(cid, cat, "fail", f"{u.unit_id} missing treated_channel_names")
        if u.observed_lift is None:
            return CheckOutcome(cid, cat, "fail", f"{u.unit_id} missing observed_lift")
        if u.lift_se is None or float(u.lift_se) <= 0:
            return CheckOutcome(cid, cat, "fail", f"{u.unit_id} invalid lift_se")
        if not str(u.lift_scale or "").strip():
            return CheckOutcome(cid, cat, "fail", f"{u.unit_id} missing lift_scale")
        if not lift_scale_supported(str(u.lift_scale)):
            return CheckOutcome(cid, cat, "fail", f"{u.unit_id} unsupported lift_scale")
        if u.target_kpi != target_col:
            return CheckOutcome(cid, cat, "fail", f"{u.unit_id} target_kpi mismatch")
        if not u.geo_ids:
            return CheckOutcome(cid, cat, "fail", f"{u.unit_id} missing geo_ids")
        if u.replay_estimand is None:
            return CheckOutcome(cid, cat, "fail", f"{u.unit_id} missing replay_estimand")
    return CheckOutcome(cid, cat, "pass", f"CalibrationSignal fields valid ({len(loaded)} unit(s))")


def _check_trust_report_compatibility(bundle: Path, truth: dict[str, Any], cid: str, cat: str) -> CheckOutcome:
    artifact = truth.get("artifact_truth") or {}
    governance = truth.get("governance_truth") or {}
    gates = artifact.get("expected_gates") or []
    if not gates:
        return CheckOutcome(
            cid,
            cat,
            "fail",
            "artifact_truth.expected_gates empty — TrustReport gate expectations required",
        )
    for g in gates:
        if str(g.get("expected", "")) not in KNOWN_GATE_EXPECTED:
            return CheckOutcome(cid, cat, "fail", f"invalid gate expected value: {g}")
    if "expected_certification_levels" not in artifact:
        return CheckOutcome(cid, cat, "fail", "missing expected_certification_levels")
    if governance.get("model_release_state") is None:
        return CheckOutcome(cid, cat, "fail", "governance_truth.model_release_state required")
    return CheckOutcome(cid, cat, "pass", "TrustReport semantics present in truth")


def _check_release_gate_compatibility(bundle: Path, truth: dict[str, Any], cid: str, cat: str) -> CheckOutcome:
    governance = truth.get("governance_truth") or {}
    state = str(governance.get("model_release_state", ""))
    if state not in KNOWN_RELEASE_STATES:
        return CheckOutcome(cid, cat, "fail", f"unknown model_release_state {state!r}")
    if governance.get("approved_for_optimization") is None:
        return CheckOutcome(cid, cat, "fail", "approved_for_optimization must be declared")
    if governance.get("replay_calibration_active") is None:
        return CheckOutcome(cid, cat, "fail", "replay_calibration_active must be declared")
    units = (truth.get("experiment_truth") or {}).get("units") or []
    replay_active = bool(governance.get("replay_calibration_active"))
    if replay_active and not units:
        return CheckOutcome(cid, cat, "fail", "replay_calibration_active but no experiment units")
    return CheckOutcome(cid, cat, "pass", "release-gate fields consistent")


def _platform_contract_summary(outcomes: list[CheckOutcome]) -> dict[str, Any]:
    contract_ids = {
        "decision_surface_compatibility": ("CERT-4A-009",),
        "replay_compatibility": ("CERT-4A-003", "CERT-4A-008", "CERT-4A-011"),
        "trust_semantics_compatibility": ("CERT-4A-012", "CERT-4A-006"),
        "estimand_compatibility": ("CERT-4A-010",),
        "calibration_signal_compatibility": ("CERT-4A-011", "CERT-4A-008"),
        "release_gate_compatibility": ("CERT-4A-013",),
    }

    def _rollup(name: str, ids: tuple[str, ...]) -> dict[str, Any]:
        relevant = [o for o in outcomes if o.check_id in ids]
        failed = [o for o in relevant if o.status == "fail"]
        skipped = [o for o in relevant if o.status == "skipped"]
        passed = len(relevant) - len(failed) - len(skipped)
        ok = len(failed) == 0
        return {
            "passed": ok,
            "checks_considered": [o.check_id for o in relevant],
            "n_pass": passed,
            "n_fail": len(failed),
            "n_skipped": len(skipped),
            "failures": [o.check_id for o in failed],
        }

    rollups = {name: _rollup(name, ids) for name, ids in contract_ids.items()}
    all_contract_checks = [o for o in outcomes if o.check_id.startswith("CERT-4A-")]
    contract_failures = [o.check_id for o in all_contract_checks if o.status == "fail"]
    return {
        "passed": len(contract_failures) == 0,
        "failures": contract_failures,
        **rollups,
    }


def _build_report(
    bundle: Path,
    truth: dict[str, Any],
    meta: dict[str, Any],
    outcomes: list[CheckOutcome],
    contract: dict[str, Any],
    *,
    recovery_result: Any | None = None,
) -> dict[str, Any]:
    cert_outcomes = [o for o in outcomes if o.check_id.startswith("CERT-4A")]
    executed = [o for o in cert_outcomes if o.status != "skipped"]
    skipped_deferred = [o for o in outcomes if o.check_id.startswith("VAL-")]
    failed = [o for o in outcomes if o.status == "fail"]

    overall = "pass"
    if any(o.status == "fail" for o in cert_outcomes):
        overall = "fail"
    rec_outcomes = [o for o in outcomes if o.check_id.startswith("REC-4B")]
    if any(o.status == "fail" for o in rec_outcomes):
        overall = "fail"

    limitations = [
        "Phase 4A structural certification on all bundles",
        "No coefficient, adstock, Hill, Δμ, or optimizer recovery unless Phase 4B-2 runs",
        "Does not prove causal validity or production allocation correctness",
        "Skipped registry rows (VAL-*) are explicit — not passes",
    ]
    if recovery_result is not None:
        limitations = list(recovery_result.recovery_results.get("recovery_limitations", []))

    report: dict[str, Any] = {
        "report_kind": "synthetic_world_certification_report",
        "world_id": str(meta.get("world_id", bundle.name)),
        "world_version": str(meta.get("world_version", "")),
        "world_contract_version": str(meta.get("world_contract_version", WORLD_CONTRACT_VERSION)),
        "generator_version": str(meta.get("world_generator_version", "")),
        "materialization_version": str(meta.get("materialization_version", MATERIALIZATION_VERSION)),
        "certification_runner_version": CERTIFICATION_RUNNER_VERSION,
        "bundle_path": str(bundle.as_posix()),
        "overall_status": overall,
        "executed_validations": [o.check_id for o in executed],
        "skipped_validations": [
            {"check_id": o.check_id, "skip_reason": o.skip_reason, "message": o.message}
            for o in skipped_deferred
        ],
        "failed_validations": [o.check_id for o in failed],
        "validation_results": [o.to_dict() for o in outcomes],
        "warnings": _collect_warnings(truth, outcomes),
        "limitations": limitations,
        "contract_compatibility": {
            "passed": contract.get("passed", False),
            "failures": contract.get("failures", []),
        },
        "decision_surface_compatibility": contract.get("decision_surface_compatibility", {}),
        "replay_compatibility": contract.get("replay_compatibility", {}),
        "trust_semantics_compatibility": contract.get("trust_semantics_compatibility", {}),
        "estimand_compatibility": contract.get("estimand_compatibility", {}),
        "calibration_signal_compatibility": contract.get("calibration_signal_compatibility", {}),
        "release_gate_compatibility": contract.get("release_gate_compatibility", {}),
    }
    if recovery_result is not None:
        report.update(recovery_result.to_report_sections())
        rec_ids = [o.check_id for o in rec_outcomes if o.status != "skipped"]
        report["executed_validations"] = list(report["executed_validations"]) + rec_ids
    return report


def _collect_warnings(truth: dict[str, Any], outcomes: list[CheckOutcome]) -> list[str]:
    warnings: list[str] = []
    for w in (truth.get("artifact_truth") or {}).get("expected_warnings") or []:
        warnings.append(f"truth expects warning: {w.get('warning_id')} ({w.get('severity')})")
    if any(o.check_id.startswith("VAL-") and o.status == "skipped" for o in outcomes):
        warnings.append("behavioral validations VAL-001–VAL-014 deferred — not executed")
    return warnings


def _error_report(bundle: Path, *, message: str) -> dict[str, Any]:
    return {
        "report_kind": "synthetic_world_certification_report",
        "world_id": bundle.name,
        "overall_status": "error",
        "certification_runner_version": CERTIFICATION_RUNNER_VERSION,
        "executed_validations": [],
        "skipped_validations": [],
        "failed_validations": ["bundle_load"],
        "validation_results": [
            {
                "check_id": "bundle_load",
                "category": "bundle_integrity",
                "status": "fail",
                "message": message,
                "skip_reason": None,
            }
        ],
        "limitations": ["bundle could not be loaded"],
        "contract_compatibility": {"passed": False, "failures": ["bundle_load"]},
        "decision_surface_compatibility": {"passed": False},
        "replay_compatibility": {"passed": False},
        "trust_semantics_compatibility": {"passed": False},
    }
