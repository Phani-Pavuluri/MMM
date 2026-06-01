"""Phase 4A certification registry — structural checks vs deferred behavioral validations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

CERTIFICATION_RUNNER_VERSION = "cert_runner_v1.0.0"
REPORT_ARTIFACT_NAME = "synthetic_world_certification_report.json"

ValidationStatus = Literal["pass", "fail", "skipped"]
SkipReason = Literal[
    "unsupported",
    "deferred",
    "requires_rich_dgp_worlds",
    "requires_train_decide_execution",
    "requires_monte_carlo",
    "requires_thresholds",
    "not_applicable",
]

Phase4APhase = Literal["4A", "deferred"]


@dataclass(frozen=True)
class CertificationCheckSpec:
    check_id: str
    category: str
    phase: Phase4APhase
    description: str
    registry_validation_id: str | None = None
    default_skip_reason: SkipReason | None = None


# Phase 4A — structural / contract certification (executable on bundles)
PHASE_4A_CHECKS: tuple[CertificationCheckSpec, ...] = (
    CertificationCheckSpec("CERT-4A-001", "bundle_integrity", "4A", "L1–L3 bundle validator"),
    CertificationCheckSpec("CERT-4A-002", "checksum_reproducibility", "4A", "On-disk checksums match bytes"),
    CertificationCheckSpec("CERT-4A-003", "replay_loader_compatibility", "4A", "replay_units.json loads via units_io"),
    CertificationCheckSpec(
        "CERT-4A-004", "transform_truth_consistency", "4A", "Canonical prod transform families in truth"
    ),
    CertificationCheckSpec(
        "CERT-4A-005", "metadata_consistency", "4A", "Bundle metadata aligns with world_truth.metadata"
    ),
    CertificationCheckSpec(
        "CERT-4A-006", "governance_warning_compatibility", "4A", "artifact_truth warnings are well-formed"
    ),
    CertificationCheckSpec(
        "CERT-4A-007", "decision_truth_structure", "4A", "decision_truth scenarios reference valid channels"
    ),
    CertificationCheckSpec(
        "CERT-4A-008", "calibration_payload_compatibility", "4A", "Replay rows are valid CalibrationSignal payloads"
    ),
    CertificationCheckSpec(
        "CERT-4A-009", "decision_surface_compatibility", "4A", "semi_log + full-panel replay transform semantics"
    ),
    CertificationCheckSpec(
        "CERT-4A-010", "estimand_compatibility", "4A", "Declared estimands on experiment and replay rows"
    ),
    CertificationCheckSpec(
        "CERT-4A-011", "calibration_signal_compatibility", "4A", "CalibrationSignal required fields present"
    ),
    CertificationCheckSpec(
        "CERT-4A-012", "trust_report_compatibility", "4A", "TrustReport / governance truth semantics"
    ),
    CertificationCheckSpec("CERT-4A-013", "release_gate_compatibility", "4A", "Release-gate fields use known enums"),
)

# Deferred — registry VAL rows requiring train/decide, DGP, or thresholds
DEFERRED_CHECKS: tuple[CertificationCheckSpec, ...] = (
    CertificationCheckSpec(
        "VAL-001",
        "coefficient_recovery",
        "deferred",
        "Coefficient recovery vs true_beta",
        "VAL-001",
        "requires_rich_dgp_worlds",
    ),
    CertificationCheckSpec(
        "VAL-002",
        "adstock_recovery",
        "deferred",
        "Adstock parameter recovery",
        "VAL-002",
        "requires_rich_dgp_worlds",
    ),
    CertificationCheckSpec(
        "VAL-003",
        "hill_recovery",
        "deferred",
        "Hill saturation recovery",
        "VAL-003",
        "requires_rich_dgp_worlds",
    ),
    CertificationCheckSpec(
        "VAL-004",
        "delta_mu_recovery",
        "deferred",
        "Δμ recovery vs decision_truth",
        "VAL-004",
        "requires_train_decide_execution",
    ),
    CertificationCheckSpec(
        "VAL-005",
        "optimizer_recovery",
        "deferred",
        "Optimizer recovery vs true optimum",
        "VAL-005",
        "requires_train_decide_execution",
    ),
    CertificationCheckSpec(
        "VAL-006",
        "replay_consistency",
        "deferred",
        "Replay train/holdout loss vs implied y",
        "VAL-006",
        "requires_train_decide_execution",
    ),
    CertificationCheckSpec(
        "VAL-007",
        "calibration_robustness",
        "deferred",
        "Calibration weight stability across quality tiers",
        "VAL-007",
        "requires_train_decide_execution",
    ),
    CertificationCheckSpec(
        "VAL-008",
        "decision_safety",
        "deferred",
        "decision_safe and unsupported-question rate",
        "VAL-008",
        "requires_train_decide_execution",
    ),
    CertificationCheckSpec(
        "VAL-009",
        "artifact_integrity_runtime",
        "deferred",
        "Prod decide bundle tier and fingerprint at runtime",
        "VAL-009",
        "requires_train_decide_execution",
    ),
    CertificationCheckSpec(
        "VAL-010",
        "reproducibility",
        "deferred",
        "Independent-run reproducibility match",
        "VAL-010",
        "requires_train_decide_execution",
    ),
    CertificationCheckSpec(
        "VAL-011",
        "promotion_workflow",
        "deferred",
        "Promotion record validation",
        "VAL-011",
        "requires_train_decide_execution",
    ),
    CertificationCheckSpec(
        "VAL-012",
        "drift_detection",
        "deferred",
        "Drift detection accuracy vs shift_truth",
        "VAL-012",
        "requires_thresholds",
    ),
    CertificationCheckSpec(
        "VAL-013",
        "governance_behavior",
        "deferred",
        "Live governance JSON vs artifact_truth gates",
        "VAL-013",
        "requires_train_decide_execution",
    ),
    CertificationCheckSpec(
        "VAL-014",
        "certification_behavior",
        "deferred",
        "Runtime cert levels vs artifact_truth.expected_certification_levels",
        "VAL-014",
        "requires_train_decide_execution",
    ),
)

ALL_CHECK_SPECS: tuple[CertificationCheckSpec, ...] = PHASE_4A_CHECKS + DEFERRED_CHECKS

PHASE_4A_CHECK_IDS: frozenset[str] = frozenset(c.check_id for c in PHASE_4A_CHECKS)
DEFERRED_CHECK_IDS: frozenset[str] = frozenset(c.check_id for c in DEFERRED_CHECKS)
