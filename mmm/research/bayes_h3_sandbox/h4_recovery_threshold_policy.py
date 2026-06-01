"""INV-071 Bayes-H4 true-effect recovery threshold policy (report-only, claim-specific)."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

from mmm.research.bayes_h3_sandbox.h4c_recovery_worlds import H4C_WORLD_IDS
from mmm.research.bayes_h3_sandbox.h5_validation_worlds import (
    WORLD_BAYES_H5_ADSTOCK_ALIGNED,
    WORLD_BAYES_H5_ADSTOCK_MISMATCH,
    WORLD_BAYES_H5_CORRELATED_CHANNELS,
    WORLD_BAYES_H5_SATURATION_ALIGNED,
    WORLD_BAYES_H5_SATURATION_MISMATCH,
    WORLD_BAYES_H5_SPARSE_RECOVERY,
    WORLD_BAYES_H5_WEAK_SIGNAL,
)
from mmm.research.bayes_h3_sandbox.recovery_worlds import (
    H4_WORLD_IDS,
    WORLD_BAYES_H4_CONFLICTING_EVIDENCE,
    WORLD_BAYES_H4_SPARSE_GEO,
)

POLICY_ID = "BAYES_H4_RECOVERY_THRESHOLD_POLICY_20260601"
POLICY_VERSION = "bayes_h4_recovery_threshold_policy_v1"
DEFAULT_POLICY_PATH = Path("docs/05_validation/archives/BAYES_H4_RECOVERY_THRESHOLD_POLICY_20260601.json")

ARCHIVE_DIR = Path("docs/05_validation/archives")
INPUT_ARTIFACTS = (
    "BAYES_H4_THRESHOLD_PILOT_20260601.json",
    "BAYES_H4_REPEATED_PILOT_PRIMARY_METRIC_20260601.json",
    "BAYES_H4C_EXTENDED_RECOVERY_PILOT_20260601.json",
    "BAYES_H4_SPARSE_VARIANT_SWEEP_20260601.json",
)

# Claim-specific world roles (not a single global threshold).
WORLD_ROLE_RECOVERY_CANDIDATE = "recovery_candidate"
WORLD_ROLE_STRESS_DIAGNOSTIC = "stress_diagnostic"
WORLD_ROLE_WEAK_IDENTIFICATION = "weak_identification"
WORLD_ROLE_TRANSFORM_MISMATCH = "transform_mismatch"
WORLD_ROLE_CONFLICT_DIAGNOSTIC = "conflict_diagnostic"
WORLD_ROLE_BASELINE_POOLING = "baseline_pooling"

RECOVERY_CANDIDATE_WORLDS: frozenset[str] = frozenset(
    {
        "WORLD-BAYES-H4C-CLEAN-RECOVERY",
        "WORLD-BAYES-H4C-SPARSE-RECOVERY",
        "WORLD-BAYES-H4-SIMPLE-POOLING",
        WORLD_BAYES_H5_ADSTOCK_ALIGNED,
        WORLD_BAYES_H5_SATURATION_ALIGNED,
        WORLD_BAYES_H5_SPARSE_RECOVERY,
    }
)

STRESS_DIAGNOSTIC_WORLDS: frozenset[str] = frozenset(
    {
        WORLD_BAYES_H4_SPARSE_GEO,
    }
)

WEAK_IDENTIFICATION_WORLDS: frozenset[str] = frozenset(
    {
        "WORLD-BAYES-H4C-CORRELATED-CHANNELS",
        "WORLD-BAYES-H4C-WEAK-SIGNAL",
        WORLD_BAYES_H5_CORRELATED_CHANNELS,
        WORLD_BAYES_H5_WEAK_SIGNAL,
    }
)

TRANSFORM_MISMATCH_WORLDS: frozenset[str] = frozenset(
    {
        "WORLD-BAYES-H4C-ADSTOCKED-MEDIA",
        "WORLD-BAYES-H4C-SATURATION",
        WORLD_BAYES_H5_ADSTOCK_MISMATCH,
        WORLD_BAYES_H5_SATURATION_MISMATCH,
    }
)

CONFLICT_WORLDS: frozenset[str] = frozenset({WORLD_BAYES_H4_CONFLICTING_EVIDENCE})


def world_policy_role(world_id: str) -> str:
    """Map world_id to INV-071 claim-specific role."""
    if world_id in RECOVERY_CANDIDATE_WORLDS:
        return WORLD_ROLE_RECOVERY_CANDIDATE
    if world_id in STRESS_DIAGNOSTIC_WORLDS:
        return WORLD_ROLE_STRESS_DIAGNOSTIC
    if world_id in WEAK_IDENTIFICATION_WORLDS:
        return WORLD_ROLE_WEAK_IDENTIFICATION
    if world_id in TRANSFORM_MISMATCH_WORLDS:
        return WORLD_ROLE_TRANSFORM_MISMATCH
    if world_id in CONFLICT_WORLDS:
        return WORLD_ROLE_CONFLICT_DIAGNOSTIC
    return "unclassified"


def gate_behavior_for_role(role: str) -> dict[str, Any]:
    """Claim-specific gate behavior — report-only; no production promotion."""
    behaviors: dict[str, dict[str, Any]] = {
        WORLD_ROLE_RECOVERY_CANDIDATE: {
            "default": "warn",
            "on_metric_exceed_report_warn": "warn",
            "on_metric_exceed_report_restricted": "restricted",
            "global_model_failure": False,
            "informs_future_thresholds": True,
            "note": "Only role used to calibrate true-effect report bands.",
        },
        WORLD_ROLE_STRESS_DIAGNOSTIC: {
            "default": "report_only",
            "on_any_metric": "report_only",
            "global_model_failure": False,
            "hard_gate": False,
            "note": "Stress world; never a global Bayes pass/fail.",
        },
        WORLD_ROLE_WEAK_IDENTIFICATION: {
            "default": "warn",
            "expected": "warning_or_restricted",
            "global_model_failure": False,
            "note": "Collinearity / weak SNR; not recovery failure.",
        },
        WORLD_ROLE_TRANSFORM_MISMATCH: {
            "default": "restricted",
            "expected": "transform_mismatch_warning",
            "global_model_failure": False,
            "note": "Generative transform differs from MVP; not recovery failure.",
        },
        WORLD_ROLE_CONFLICT_DIAGNOSTIC: {
            "default": "warn",
            "expected": "conflict_warning_required",
            "global_model_failure": False,
        },
    }
    return behaviors.get(
        role,
        {"default": "report_only", "global_model_failure": False},
    )


def _load_json(name: str, *, base: Path | None = None) -> dict[str, Any]:
    path = (base or ARCHIVE_DIR) / name
    return json.loads(path.read_text(encoding="utf-8"))


def _rows_from_artifact(artifact: dict[str, Any]) -> list[dict[str, Any]]:
    if "worlds" in artifact:
        return list(artifact["worlds"])
    if "per_run" in artifact:
        return list(artifact["per_run"])
    if "world_runs" in artifact:
        return list(artifact["world_runs"])
    return []


def collect_metric_observations(
    *,
    archive_dir: Path | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Gather per-world metric observations from committed pilot artifacts."""
    base = archive_dir or ARCHIVE_DIR
    by_world: dict[str, list[dict[str, Any]]] = {}

    for name in INPUT_ARTIFACTS:
        art = _load_json(name, base=base)
        pilot_id = str(art.get("pilot_id", name))
        for row in _rows_from_artifact(art):
            wid = str(row.get("world_id", ""))
            if not wid:
                continue
            by_world.setdefault(wid, []).append(
                {
                    "source_pilot": pilot_id,
                    "beta_gc_mae": row.get("beta_gc_mae"),
                    "mu_c_mae": row.get("mu_c_mae"),
                    "beta_gc_coverage_90": row.get("beta_gc_coverage_90"),
                    "beta_interval_width_90_mean": row.get("beta_interval_width_90_mean"),
                    "shrinkage_ratio_sparse": row.get("shrinkage_ratio_sparse"),
                    "shrinkage_ratio_sparse_vs_true_mu": row.get("shrinkage_ratio_sparse_vs_true_mu"),
                }
            )
    return by_world


def _band_from_values(values: list[float], *, margin: float = 1.15) -> dict[str, Any]:
    if not values:
        return {
            "n": 0,
            "min": None,
            "max": None,
            "median": None,
            "report_warn_above": None,
            "report_restricted_above": None,
        }
    vmax = max(values)
    med = statistics.median(values)
    return {
        "n": len(values),
        "min": float(min(values)),
        "max": float(vmax),
        "median": float(med),
        "report_warn_above": float(vmax * margin),
        "report_restricted_above": float(vmax * 1.35),
        "hard_fail_above": None,
        "mode": "report_only",
    }


def build_threshold_policy(
    *,
    archive_dir: Path | None = None,
    policy_id: str = POLICY_ID,
) -> dict[str, Any]:
    """Build claim-specific report-only threshold policy from H4/H4c artifacts."""
    base = archive_dir or ARCHIVE_DIR
    by_world = collect_metric_observations(archive_dir=base)

    recovery_beta: list[float] = []
    recovery_mu: list[float] = []
    recovery_cov: list[float] = []
    recovery_width: list[float] = []

    world_entries: list[dict[str, Any]] = []
    for wid in sorted(by_world):
        role = world_policy_role(wid)
        obs = by_world[wid]
        beta_vals = [float(o["beta_gc_mae"]) for o in obs if o.get("beta_gc_mae") is not None]
        mu_vals = [float(o["mu_c_mae"]) for o in obs if o.get("mu_c_mae") is not None]
        if role == WORLD_ROLE_RECOVERY_CANDIDATE:
            recovery_beta.extend(beta_vals)
            recovery_mu.extend(mu_vals)
            recovery_cov.extend(
                float(o["beta_gc_coverage_90"]) for o in obs if o.get("beta_gc_coverage_90") is not None
            )
            recovery_width.extend(
                float(o["beta_interval_width_90_mean"]) for o in obs if o.get("beta_interval_width_90_mean") is not None
            )
        world_entries.append(
            {
                "world_id": wid,
                "policy_role": role,
                "gate_behavior": gate_behavior_for_role(role),
                "observations": obs,
                "metric_summary": {
                    "beta_gc_mae": _band_from_values(beta_vals) if beta_vals else {"n": 0},
                    "mu_c_mae": _band_from_values(mu_vals) if mu_vals else {"n": 0},
                },
            }
        )

    metric_definitions = {
        "pooling_mechanics": {
            "shrinkage_ratio_sparse": {
                "role": "pooling_mechanics_only",
                "pool_center": "posterior_mu_c",
                "true_effect_recovery_gate": False,
                "note": "Per H4b-disposition C; not a truth-recovery gate.",
            },
            "shrinkage_ratio_sparse_vs_true_mu": {
                "role": "true_effect_recovery_diagnostic",
                "pool_center": "true_mu_c",
                "true_effect_recovery_gate": False,
                "note": "Legacy diagnostic only.",
            },
        },
        "true_effect_recovery": {
            "beta_gc_mae": {
                "role": "point_recovery",
                "applies_to_roles": [WORLD_ROLE_RECOVERY_CANDIDATE],
            },
            "mu_c_mae": {
                "role": "point_recovery",
                "applies_to_roles": [WORLD_ROLE_RECOVERY_CANDIDATE],
            },
            "beta_gc_coverage_90": {
                "role": "uncertainty_directional",
                "note": "Do not require exact 90% on toy panels.",
            },
            "beta_interval_width_90_mean": {
                "role": "uncertainty_sanity",
                "note": "Wide intervals expected under weak ID; not a global fail.",
            },
        },
        "reliability_diagnostics": {
            "h4c_classification": {"role": "reliability_map"},
            "h4c_diagnostic_warnings": {"role": "expected_on_mismatch_worlds"},
            "conflict_warnings": {"role": "expected_on_conflict_world"},
        },
    }

    provisional_thresholds = {
        "hard_gate": False,
        "production_promotion": False,
        "report_only": True,
        "calibrated_from_roles": [WORLD_ROLE_RECOVERY_CANDIDATE],
        "point_recovery": {
            "beta_gc_mae": _band_from_values(recovery_beta),
            "mu_c_mae": _band_from_values(recovery_mu),
        },
        "uncertainty": {
            "beta_gc_coverage_90": {
                "mode": "directional_only",
                "observed_range": [min(recovery_cov), max(recovery_cov)] if recovery_cov else None,
                "require_exact_90": False,
                "note": "Toy panels; coverage is trend-only.",
            },
            "beta_interval_width_90_mean": _band_from_values(recovery_width)
            if recovery_width
            else {
                "mode": "sanity_check",
                "report_restricted_above": 0.25,
                "hard_fail_above": None,
            },
        },
        "reliability_by_role": {
            WORLD_ROLE_RECOVERY_CANDIDATE: gate_behavior_for_role(WORLD_ROLE_RECOVERY_CANDIDATE),
            WORLD_ROLE_STRESS_DIAGNOSTIC: gate_behavior_for_role(WORLD_ROLE_STRESS_DIAGNOSTIC),
            WORLD_ROLE_WEAK_IDENTIFICATION: gate_behavior_for_role(WORLD_ROLE_WEAK_IDENTIFICATION),
            WORLD_ROLE_TRANSFORM_MISMATCH: gate_behavior_for_role(WORLD_ROLE_TRANSFORM_MISMATCH),
            WORLD_ROLE_CONFLICT_DIAGNOSTIC: gate_behavior_for_role(WORLD_ROLE_CONFLICT_DIAGNOSTIC),
        },
    }

    return {
        "policy_id": policy_id,
        "policy_version": POLICY_VERSION,
        "investigation_id": "INV-071",
        "label": "RESEARCH ONLY — NOT DECISION GRADE",
        "research_only": True,
        "approved_for_prod": False,
        "prod_decisioning_allowed": False,
        "production_promotion": False,
        "decision_grade": False,
        "hard_gate": False,
        "interpretation": {
            "purpose": "Claim-specific true-effect recovery thresholds for Bayes-H4 research sandbox.",
            "does_not_claim": "Bayesian MMM is production-ready or recovers truth globally.",
            "pooling_vs_recovery": "Pooling metrics (primary shrinkage) are separate from true-effect recovery.",
            "stress_worlds_never_global_fail": True,
            "promotion_blocked": True,
        },
        "input_artifacts": [str(base / n) for n in INPUT_ARTIFACTS],
        "world_catalog": {
            "h4_core": list(H4_WORLD_IDS),
            "h4c_extended": list(H4C_WORLD_IDS),
            "recovery_candidate": sorted(RECOVERY_CANDIDATE_WORLDS),
            "stress_diagnostic": sorted(STRESS_DIAGNOSTIC_WORLDS),
            "weak_identification": sorted(WEAK_IDENTIFICATION_WORLDS),
            "transform_mismatch": sorted(TRANSFORM_MISMATCH_WORLDS),
            "conflict_diagnostic": sorted(CONFLICT_WORLDS),
        },
        "metric_definitions": metric_definitions,
        "provisional_thresholds": provisional_thresholds,
        "worlds": world_entries,
    }


def write_threshold_policy_artifact(path: str | Path | None = None, policy: dict[str, Any] | None = None) -> Path:
    out = Path(path or DEFAULT_POLICY_PATH)
    payload = policy if policy is not None else build_threshold_policy()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def load_threshold_policy(path: str | Path | None = None) -> dict[str, Any]:
    p = Path(path or DEFAULT_POLICY_PATH)
    return json.loads(p.read_text(encoding="utf-8"))


def evaluate_world_against_policy(
    world_id: str,
    metrics: dict[str, Any],
    *,
    policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Report-only evaluation for one world (no production gate)."""
    pol = policy or load_threshold_policy()
    role = world_policy_role(world_id)
    behavior = gate_behavior_for_role(role)
    outcome = str(behavior.get("default", "report_only"))

    if role == WORLD_ROLE_RECOVERY_CANDIDATE:
        pt = pol["provisional_thresholds"]["point_recovery"]
        beta = metrics.get("beta_gc_mae")
        if beta is not None and pt["beta_gc_mae"].get("report_restricted_above") is not None:
            if float(beta) > float(pt["beta_gc_mae"]["report_restricted_above"]):
                outcome = "restricted"
            elif float(beta) > float(pt["beta_gc_mae"]["report_warn_above"]):
                outcome = "warn"
            else:
                outcome = "pass"
    elif role in (WORLD_ROLE_TRANSFORM_MISMATCH, WORLD_ROLE_WEAK_IDENTIFICATION):
        outcome = str(behavior.get("expected", behavior.get("default", "warn")))
    elif role == WORLD_ROLE_STRESS_DIAGNOSTIC:
        outcome = "report_only"

    return {
        "world_id": world_id,
        "policy_role": role,
        "outcome": outcome,
        "hard_gate": False,
        "fail_for_claim": False,
        "global_model_failure": bool(behavior.get("global_model_failure", False)),
        "production_promotion": False,
    }
