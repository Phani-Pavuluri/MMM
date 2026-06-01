"""Phase 5F — Monte Carlo reliability pilot characterization (validation science only)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mmm.validation.synthetic.recovery_certification import (
    ALLOCATION_L1_ATOL,
    ALLOCATION_L1_RTOL,
    COEF_RECOVERY_ATOL,
    COEF_RECOVERY_RTOL,
    DELTA_MU_RECOVERY_ATOL,
    DELTA_MU_RECOVERY_RTOL,
    DRIFT_POST_PRE_MAE_RATIO_MIN,
    REPLAY_LIFT_ATOL,
    REPLAY_LIFT_RTOL,
    TRANSFORM_PARAM_RTOL,
)

PROGRAM_VERSION = "monte_carlo_reliability_v1.0.0"
CHARACTERIZATION_ARTIFACT = "monte_carlo_pilot_characterization.json"

# Structured coverage axes (not uniform random) — see monte_carlo_reliability_program.md
COVERAGE_AXES: dict[str, tuple[str, ...]] = {
    "world_family": (
        "anchor_recovery",
        "behavioral_lattice",
        "structural_lattice",
        "identifiability_mini",
        "volume_sweep",
        "negative_gate",
    ),
    "noise_level": ("zero", "low", "medium", "high"),
    "correlation_level": ("low", "moderate", "severe"),
    "n_geos": ("1", "2", "4", "8"),
    "n_periods": ("10", "14", "18", "26", "52"),
    "n_channels": ("1", "2", "3"),
    "drift": ("off", "on"),
    "replay": ("off", "on"),
    "signal_strength": ("low", "medium", "high"),
    "calibration_freshness": ("fresh", "stale", "missing"),
    "missingness": ("none", "sparse_geo", "sparse_period"),
}

TIER_TARGETS: dict[str, int] = {
    "tier_0_pilot": 25,
    "tier_1_calibration": 100,
    "tier_2_release_review": 1_000,
    "tier_3_full_monte_carlo": 10_000,
}

CURRENT_THRESHOLDS: dict[str, dict[str, Any]] = {
    "VAL-001_coef": {
        "rtol": COEF_RECOVERY_RTOL,
        "atol": COEF_RECOVERY_ATOL,
        "metric_class": "diagnostic_attribution",
    },
    "VAL-002_003_transform": {
        "rtol": TRANSFORM_PARAM_RTOL,
        "metric_class": "diagnostic_attribution",
    },
    "VAL-004_delta_mu": {
        "rtol": DELTA_MU_RECOVERY_RTOL,
        "atol": DELTA_MU_RECOVERY_ATOL,
        "metric_class": "decision_grade",
    },
    "VAL-005_optimizer": {
        "allocation_l1_rtol": ALLOCATION_L1_RTOL,
        "allocation_l1_atol": ALLOCATION_L1_ATOL,
        "metric_class": "decision_grade",
    },
    "VAL-006_replay": {
        "lift_rtol": REPLAY_LIFT_RTOL,
        "lift_atol": REPLAY_LIFT_ATOL,
        "metric_class": "decision_grade",
    },
    "VAL-012_drift": {
        "post_pre_mae_ratio_min": DRIFT_POST_PRE_MAE_RATIO_MIN,
        "metric_class": "trust_modifier",
    },
}


@dataclass
class CapabilityDistribution:
    capability: str
    metric_class: str
    n_observations: int
    pass_rate: float | None
    mean_score: float | None
    failure_regions: list[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "capability": self.capability,
            "metric_class": self.metric_class,
            "n_observations": self.n_observations,
            "pass_rate": self.pass_rate,
            "mean_score": self.mean_score,
            "failure_regions": self.failure_regions,
            "notes": self.notes,
        }


def _load_json(path: Path) -> dict[str, Any] | list[Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _pilot_from_behavioral_report(report: dict[str, Any]) -> dict[str, Any]:
    summary = report.get("scorecard_summary") or {}
    caps = report.get("capability_recovery_summary") or {}
    per_world = report.get("per_world_outcomes") or []
    return {
        "source": "behavioral_lattice_sweep_mvp_report.json",
        "n_worlds": len(per_world),
        "structural_score": summary.get("structural_score"),
        "behavioral_score": summary.get("behavioral_score"),
        "behavioral_scorecard_reliability": summary.get("behavioral_scorecard_reliability"),
        "capability_summary": {
            k: {
                "mean_score": v.get("mean_score"),
                "n_scored": v.get("n_scored"),
                "failures": v.get("failures", []),
            }
            for k, v in caps.items()
        },
        "per_world": per_world,
    }


def _pilot_from_exact_recovery(findings: dict[str, Any]) -> dict[str, Any]:
    decomp = findings.get("recovery_decomposition") or []
    vol = findings.get("data_volume_sweep", {}).get("sweep") or []
    ident_raw = findings.get("identifiability_grid") or {}
    ident = (
        ident_raw.get("variants", [])
        if isinstance(ident_raw, dict)
        else (ident_raw if isinstance(ident_raw, list) else [])
    )
    return {
        "source": "exact_recovery_findings.json",
        "recovery_decomposition": decomp,
        "data_volume_sweep": vol,
        "identifiability_grid": ident,
        "regularization_sweep": findings.get("regularization_sweep"),
        "hyperparameter_coupling": findings.get("hyperparameter_coupling"),
    }


def characterize_capability_distributions(
    behavioral: dict[str, Any],
    inv: dict[str, Any],
) -> list[CapabilityDistribution]:
    caps_meta = {
        "coefficient_recovery": "diagnostic_attribution",
        "transform_consistency": "diagnostic_attribution",
        "delta_mu_recovery": "decision_grade",
        "optimizer_recovery": "decision_grade",
        "replay_recovery": "decision_grade",
        "drift_behavior": "trust_modifier",
        "identifiability_behavior": "trust_modifier",
        "structural_integrity": "structural",
        "platform_contract_compatibility": "structural",
        "artifact_integrity": "structural",
        "governance_reaction": "trust_modifier",
    }
    out: list[CapabilityDistribution] = []
    b_caps = behavioral.get("capability_summary") or {}
    for cap, mclass in caps_meta.items():
        row = b_caps.get(cap) or {}
        n_scored = int(row.get("n_scored") or 0)
        mean_score = row.get("mean_score")
        failures = list(row.get("failures") or [])
        pass_rate = float(mean_score) if mean_score is not None and n_scored else None
        regions: list[str] = []
        if cap == "coefficient_recovery" and failures:
            regions.append("exact_recovery_world_type")
            regions.append("multi_channel_shared_transform")
        if cap == "delta_mu_recovery" and not failures:
            regions.append("passes_despite_coef_fail_on_same_worlds")
        if cap == "drift_behavior" and row.get("partials"):
            regions.append("drift_on_cells")
        out.append(
            CapabilityDistribution(
                capability=cap,
                metric_class=mclass,
                n_observations=n_scored,
                pass_rate=pass_rate,
                mean_score=float(mean_score) if mean_score is not None else None,
                failure_regions=regions,
                notes="Tier-0 pilot (n≤10 scored per capability)",
            )
        )

    decomp = inv.get("recovery_decomposition") or []
    def _snap_pass(row: dict[str, Any], key: str) -> bool:
        snap = row.get("certification_snapshot") or {}
        if snap.get(f"{key}_status") == "pass":
            return True
        block = row.get("fitted_transforms") or {}
        rec = block.get(f"{key}_recovery") or {}
        return bool(rec.get("pass"))

    coef_pass_bo = sum(1 for w in decomp if _snap_pass(w, "coef")) / max(len(decomp), 1)
    dmu_pass_bo = sum(1 for w in decomp if _snap_pass(w, "delta_mu")) / max(len(decomp), 1)
    out.append(
        CapabilityDistribution(
            capability="anchor_worlds_coef_bo",
            metric_class="diagnostic_attribution",
            n_observations=len(decomp),
            pass_rate=coef_pass_bo,
            mean_score=coef_pass_bo,
            failure_regions=["WORLD-008", "L5B exact_recovery"],
            notes="Anchor + lattice exact-recovery decomposition",
        )
    )
    out.append(
        CapabilityDistribution(
            capability="anchor_worlds_delta_mu_bo",
            metric_class="decision_grade",
            n_observations=len(decomp),
            pass_rate=dmu_pass_bo,
            mean_score=dmu_pass_bo,
            failure_regions=[],
            notes="Δμ pass rate on same worlds as coef failures",
        )
    )
    return out


def reliability_boundary_analysis(inv: dict[str, Any]) -> dict[str, Any]:
    vol = inv.get("data_volume_sweep") or []
    ident_raw = inv.get("identifiability_grid") or {}
    ident = (
        ident_raw.get("variants", [])
        if isinstance(ident_raw, dict)
        else (ident_raw if isinstance(ident_raw, list) else [])
    )
    envelopes: list[dict[str, Any]] = []
    for row in vol:
        envelopes.append(
            {
                "axis": "data_volume",
                "n_geos": row.get("n_geos"),
                "n_periods": row.get("n_periods"),
                "noise_std": row.get("noise_std"),
                "coef_pass_pinned": row.get("coef_pass_pinned"),
                "max_coef_err_pinned": row.get("max_coef_err_pinned"),
                "interpretation": (
                    "Volume alone does not fix shared-transform homogenization "
                    "on multi-channel WORLD-008 template"
                ),
            }
        )
    for row in ident:
        envelopes.append(
            {
                "axis": "identifiability",
                "variant": row.get("world_id") or row.get("variant"),
                "coef_pass_pinned": row.get("coef_pass_pinned"),
                "interpretation": (
                    "Single-channel and orthogonal 2ch pass pinned coef; "
                    "WORLD-008 fails due to collinear features not channel count alone"
                ),
            }
        )
    failure_map = {
        "expected_failures": [
            {
                "driver": "shared_transform_across_channels",
                "capabilities": ["coefficient_recovery", "transform_consistency"],
                "worlds": ["WORLD-008-exact-recovery", "L5B-exact_recovery-*"],
            },
            {
                "driver": "bo_hyperparameter_search",
                "capabilities": ["transform_consistency"],
                "worlds": ["WORLD-008-exact-recovery"],
            },
            {
                "driver": "identifiability_collinearity",
                "capabilities": ["coefficient_recovery"],
                "worlds": ["L5B-identifiability-*", "correlation_level=severe"],
            },
        ],
        "expected_passes": [
            {
                "driver": "decision_surface_invariant",
                "capabilities": ["delta_mu_recovery", "optimizer_recovery", "replay_recovery"],
                "note": "Often passes TBD_v1_runtime despite coef fail",
            },
            {
                "driver": "structural_contracts",
                "capabilities": ["structural_integrity", "platform_contract_compatibility"],
            },
        ],
    }
    coverage_table = {
        "tier_0_observed": {
            "behavioral_lattice_cells": 10,
            "anchor_worlds": 5,
            "inv_volume_points": len(vol),
            "inv_identifiability_variants": len(ident),
        },
        "tier_1_target": TIER_TARGETS["tier_1_calibration"],
        "tier_2_target": TIER_TARGETS["tier_2_release_review"],
    }
    return {
        "reliability_envelopes": envelopes,
        "failure_map": failure_map,
        "coverage_table": coverage_table,
        "boundary_hypotheses": [
            {"condition": "correlation_level=severe", "coef_recovery": "fail expected"},
            {"condition": "n_channels>=3 + shared transform", "coef_recovery": "fail expected"},
            {"condition": "noise_std<=0.02", "coef_recovery": "still fail on WORLD-008 (not noise-limited)"},
            {"condition": "n_geos>=2, n_periods>=14", "coef_recovery": "still fail on pinned WORLD-008 volume sweep"},
            {"condition": "drift=on", "drift_behavior": "pass/warning; trust_modifier caution+"},
            {"condition": "truth_pinned_transforms", "coef_recovery": "pass on 1ch/2ch-orth mini worlds"},
        ],
    }


def threshold_calibration_framework(
    distributions: list[CapabilityDistribution],
    boundaries: dict[str, Any],
) -> list[dict[str, Any]]:
    """Recommendations only — not approved thresholds."""
    recs: list[dict[str, Any]] = []
    for key, cur in CURRENT_THRESHOLDS.items():
        val_id = key.split("_")[0]
        mclass = cur["metric_class"]
        dist_note = ""
        suggested = "retain_provisional"
        confidence = "low"
        evidence = "Tier-0 pilot + INV-056"

        if "coef" in key or "transform" in key:
            dist_note = "Pilot pass rate ~0 on exact_recovery; mini-worlds pass when pinned"
            suggested = "loosen_for_diagnostic_only; do not use as release gate"
            confidence = "medium"
        elif "delta_mu" in key:
            dist_note = "Pilot pass rate ~1.0 on scored exact_recovery cells"
            suggested = "tighten_relative_error_cap_after tier-1 MC; keep as decision gate candidate"
            confidence = "medium"
        elif "drift" in key:
            dist_note = "MAE ratio bands calibrated in drift_detection_runner; align to tier-1 percentiles"
            suggested = "calibrate severity bands from tier-1 post/pre ratio distribution"
            confidence = "low"
        elif "optimizer" in key or "replay" in key:
            dist_note = "Dedicated worlds pass at MVP; expand tier-1 corner cases"
            suggested = "retain until tier-1 n≥100"
            confidence = "low"

        recs.append(
            {
                "metric_key": key,
                "validation_id": val_id,
                "metric_class": mclass,
                "current_threshold": cur,
                "observed_distribution_summary": dist_note,
                "suggested_action": suggested,
                "supporting_evidence": evidence,
                "confidence_level": confidence,
                "approval_status": "recommendation_only_not_approved",
            }
        )
    return recs


def trust_report_calibration_pilot(
    behavioral: dict[str, Any],
    distributions: list[CapabilityDistribution],
) -> dict[str, Any]:
    structural = behavioral.get("structural_score")
    behavioral_score = behavioral.get("behavioral_score")
    return {
        "mapping_version": "trust_report_mc_calibration_v1",
        "empirical_basis": "Tier-0 pilot (10 behavioral cells + anchors)",
        "green_criteria": {
            "trust_grade": "high",
            "decision_reliability_score_gte": 0.85,
            "structural_reliability_score_gte": 0.9,
            "trust_modifier_status": "acceptable",
            "attribution_diagnostic": "informational_only",
            "observed_at_pilot": False,
            "note": "No pilot cell achieves green on full behavioral score due to coef diagnostic",
        },
        "yellow_criteria": {
            "trust_grade": "moderate",
            "decision_usable": True,
            "attribution_safe": False,
            "trust_modifier_status_in": ("acceptable", "caution"),
            "observed_at_pilot": behavioral_score is not None and float(behavioral_score) >= 0.5,
        },
        "red_criteria": {
            "trust_grade": "low_or_insufficient",
            "structural_reliability_score_lt": 0.75,
            "trust_modifier_status": "degraded",
            "decision_reliability_score_lt": 0.75,
            "optimization_blocked": True,
        },
        "pilot_observed_structural_score": structural,
        "pilot_observed_behavioral_score": behavioral_score,
        "recommendation": (
            "Default prod TrustReport should mirror yellow for typical Ridge BO: "
            "decision-usable, attribution-unsafe, until tier-2 MC approves green bounds"
        ),
    }


def build_pilot_characterization(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root)
    inv_path = root / "docs" / "05_validation" / "investigations" / "exact_recovery_findings.json"
    beh_path = root / "validation" / "reports" / "behavioral_lattice_sweep_mvp_report.json"
    lat_path = root / "validation" / "reports" / "lattice_sweep_mvp_report.json"

    findings = _load_json(inv_path) or {}
    behavioral_raw = _load_json(beh_path) or {}
    lattice_raw = _load_json(lat_path) or {}

    inv = _pilot_from_exact_recovery(findings if isinstance(findings, dict) else {})
    behavioral = _pilot_from_behavioral_report(
        behavioral_raw if isinstance(behavioral_raw, dict) else {}
    )
    distributions = characterize_capability_distributions(behavioral, inv)
    boundaries = reliability_boundary_analysis(inv)
    thresholds = threshold_calibration_framework(distributions, boundaries)
    trust_cal = trust_report_calibration_pilot(behavioral, distributions)

    return {
        "program_version": PROGRAM_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tier": "tier_0_pilot",
        "tier_targets": TIER_TARGETS,
        "coverage_axes": COVERAGE_AXES,
        "sampling_strategy": "structured_coverage_not_uniform",
        "evidence_sources": [
            str(beh_path.relative_to(root)) if beh_path.is_file() else str(beh_path),
            str(inv_path.relative_to(root)) if inv_path.is_file() else str(inv_path),
            str(lat_path.relative_to(root)) if lat_path.is_file() else str(lat_path),
        ],
        "behavioral_pilot": behavioral,
        "lattice_structural_score": (lattice_raw or {}).get("scorecard_summary", {}).get(
            "structural_score"
        )
        if isinstance(lattice_raw, dict)
        else None,
        "capability_distributions": [d.to_dict() for d in distributions],
        "reliability_boundaries": boundaries,
        "threshold_recommendations": thresholds,
        "trust_report_calibration": trust_cal,
        "current_thresholds": CURRENT_THRESHOLDS,
    }


def write_pilot_characterization(
    repo_root: str | Path,
    output_dir: str | Path | None = None,
) -> Path:
    root = Path(repo_root)
    out_dir = Path(output_dir) if output_dir else root / "docs" / "05_validation" / "investigations"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = build_pilot_characterization(root)
    path = out_dir / CHARACTERIZATION_ARTIFACT
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
