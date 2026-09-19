"""Tier-1 Monte Carlo batch runner — deterministic stratified N>=100 execution.

Composes existing Phase 3B/4A/4B/5A/5B entry points (ScenarioBuilder truth,
smoke + DGP materializers, structural certification with automatic recovery on
eligible worlds). Adds no new generative semantics, estimators, transforms, or
thresholds. All threshold output is report-only (provisional, DR-04 deferred).

World bundles materialize under a caller-supplied root (tests use tmp dirs);
only the manifest, characterization JSON, and recommendations note are
committed artifacts. Same-seed runs produce identical manifests.

Coupling note: behavioral truth/train-config/post steps reuse
``behavioral_lattice_sweep`` builders (same package, versioned together).
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mmm.validation.synthetic import behavioral_lattice_sweep as _behavioral
from mmm.validation.synthetic.behavioral_lattice_sweep import BehavioralWorldSpec
from mmm.validation.synthetic.certification_runner import run_world_certification
from mmm.validation.synthetic.dgp_materializer import materialize_dgp_world
from mmm.validation.synthetic.generators import write_world_truth
from mmm.validation.synthetic.materializer import materialize_world
from mmm.validation.synthetic.monte_carlo_reliability import (
    CURRENT_THRESHOLDS,
    CapabilityDistribution,
)
from mmm.validation.synthetic.scenario_builder import ScenarioSpec, write_scenario_world

TIER1_VERSION = "tier1_batch_runner_v1.0.0"
TIER1_MIN_WORLDS = 100
TIER1_WORLD_PREFIX = "T1-"
TIER1_SEED_BASE = 71_000

ANCHOR_WORLD_IDS = (
    "WORLD-008-exact-recovery",
    "WORLD-009-optimizer-recovery",
    "WORLD-010-replay-recovery",
    "WORLD-011-drift-recovery",
    "WORLD-012-identifiability-recovery",
)

RECOVERY_CAPABILITIES = (
    "coefficient_recovery",
    "transform_recovery",
    "delta_mu_recovery",
    "optimizer_recovery",
    "replay_recovery",
    "drift_recovery",
    "identifiability_recovery",
)


def _deterministic_seed(world_id: str, base: int = TIER1_SEED_BASE) -> int:
    digest = hashlib.md5(world_id.encode("utf-8")).hexdigest()
    return base + int(digest[:8], 16) % 10_000


def tier1_smoke_specs() -> tuple[ScenarioSpec, ...]:
    """Stratum S — structural smoke grid (no training; recovery honestly skipped)."""
    specs: list[ScenarioSpec] = []
    for family in ("baseline", "replay"):
        for noise in ("low", "medium", "high"):
            for corr in ("low", "medium", "severe"):
                for drift in (False, True):
                    for seasonality in ("none", "mild"):
                        eq = "none" if family == "baseline" else "medium"
                        channels = ("search", "social") if family == "baseline" else ("search",)
                        dr = "on" if drift else "off"
                        world_id = (
                            f"{TIER1_WORLD_PREFIX}smoke-{family}-noise-{noise}-"
                            f"corr-{corr}-drift-{dr}-seas-{seasonality}"
                        )
                        specs.append(
                            ScenarioSpec(
                                world_id=world_id,
                                family=family,
                                seed=_deterministic_seed(world_id),
                                n_geos=2,
                                n_periods=12,
                                channels=channels,
                                noise_level=noise,
                                correlation_level=corr,
                                seasonality=seasonality,
                                drift=drift,
                                experiment_quality=eq,
                                privacy_loss=False,
                                missingness="none",
                            )
                        )
    for noise in ("low", "high"):
        for corr in ("low", "severe"):
            for n_geos in (1, 4):
                world_id = f"{TIER1_WORLD_PREFIX}smoke-baseline-noise-{noise}-corr-{corr}-geos-{n_geos}"
                specs.append(
                    ScenarioSpec(
                        world_id=world_id,
                        family="baseline",
                        seed=_deterministic_seed(world_id),
                        n_geos=n_geos,
                        n_periods=12,
                        channels=("search", "social"),
                        noise_level=noise,
                        correlation_level=corr,
                        seasonality="none",
                        drift=False,
                        experiment_quality="none",
                        privacy_loss=False,
                        missingness="none",
                    )
                )
    return tuple(specs)


def tier1_behavioral_specs() -> tuple[BehavioralWorldSpec, ...]:
    """Stratum B — rich-DGP worlds; recovery auto-runs where eligible."""
    cells: list[tuple[str, str, str, bool, bool]] = []
    for noise in ("low", "medium", "high"):
        for corr in ("low", "severe"):
            for drift in (False, True):
                cells.append(("exact_recovery", noise, corr, drift, False))
    for noise in ("low", "high"):
        for corr in ("low", "severe"):
            cells.append(("optimizer", noise, corr, False, False))
            cells.append(("replay", noise, corr, False, True))
    for noise in ("low", "high"):
        cells.append(("drift", noise, "low", True, False))
    cells.append(("identifiability", "low", "severe", False, False))
    specs: list[BehavioralWorldSpec] = []
    for world_type, noise, corr, drift, replay in cells:
        base = _behavioral.behavioral_spec_from_cell(world_type, noise, corr, drift, replay)
        if base.behavioral_mode == "unsupported":
            continue
        world_id = f"{TIER1_WORLD_PREFIX}{base.world_id}"
        specs.append(
            BehavioralWorldSpec(
                world_id=world_id,
                world_type=base.world_type,
                noise_level=base.noise_level,
                correlation_level=base.correlation_level,
                drift=base.drift,
                replay=base.replay,
                seed=_deterministic_seed(world_id, base=58_000),
                behavioral_mode=base.behavioral_mode,
            )
        )
    return tuple(specs)


def tier1_manifest() -> dict[str, Any]:
    """Deterministic batch manifest (no execution). Same seed set, same manifest."""
    smoke = tier1_smoke_specs()
    behavioral = tier1_behavioral_specs()
    return {
        "tier1_version": TIER1_VERSION,
        "tier1_min_worlds": TIER1_MIN_WORLDS,
        "strata": {
            "smoke_structural": [s.to_dict() for s in smoke],
            "behavioral_rich_dgp": [
                {
                    "world_id": s.world_id,
                    "world_type": s.world_type,
                    "noise_level": s.noise_level,
                    "correlation_level": s.correlation_level,
                    "drift": s.drift,
                    "replay": s.replay,
                    "seed": s.seed,
                    "behavioral_mode": s.behavioral_mode,
                }
                for s in behavioral
            ],
            "anchors": list(ANCHOR_WORLD_IDS),
        },
        "world_count": len(smoke) + len(behavioral) + len(ANCHOR_WORLD_IDS),
    }


@dataclass
class Tier1WorldOutcome:
    world_id: str
    stratum: str
    axes: dict[str, str]
    structural_status: str
    contract_passed: bool | None
    recovery: dict[str, str | None]
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "world_id": self.world_id,
            "stratum": self.stratum,
            "axes": self.axes,
            "structural_status": self.structural_status,
            "contract_passed": self.contract_passed,
            "recovery": self.recovery,
            "error": self.error,
        }


def build_tier1_characterization(batch: dict[str, Any]) -> dict[str, Any]:
    """Aggregate batch outcomes into the committed characterization document."""
    dists = _capability_distributions(batch)
    recs = _tier1_recommendations(dists)
    return {
        "tier1_version": TIER1_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tier": "tier_1_calibration",
        "world_count": batch.get("world_count"),
        "capability_distributions": [d.to_dict() for d in dists],
        "axis_envelopes": _axis_envelopes(batch),
        "threshold_recommendations": recs,
        "recommendation_disclaimer": (
            "Report-only. No threshold is approved; thresholds remain provisional "
            "pending DR-04. No release-gate change."
        ),
        "limitations": [
            "Small-panel deterministic worlds; not production-panel geometry",
            "Smoke stratum is structural-only by design (recovery honestly unscored)",
            "Behavioral stratum inherits WORLD-008-template DGP assumptions",
            "Anchors re-certified read-only from committed fixtures",
        ],
    }


_SCORED_STATUSES = ("pass", "fail", "partial")


def _status_of(value: Any) -> str | None:
    if isinstance(value, str) and value in _SCORED_STATUSES:
        return value
    if isinstance(value, dict):
        status = value.get("status")
        if isinstance(status, str) and status in _SCORED_STATUSES:
            return status
    return None


def _recovery_statuses(report: dict[str, Any]) -> dict[str, str | None]:
    """Map recovery section statuses; unscored capabilities stay None (honest)."""
    rec_results = report.get("recovery_results") or {} if isinstance(report, dict) else {}
    out: dict[str, str | None] = {
        "coefficient_recovery": _status_of(rec_results.get("coefficient_recovery")),
        "delta_mu_recovery": _status_of(rec_results.get("delta_mu_recovery")),
        "transform_recovery": None,
        "optimizer_recovery": _status_of(rec_results.get("optimizer_recovery"))
        or _status_of(rec_results.get("optimizer_recovery_status")),
        "replay_recovery": _status_of(rec_results.get("replay_recovery"))
        or _status_of(rec_results.get("replay_recovery_status")),
        "drift_recovery": _status_of(rec_results.get("drift_recovery"))
        or _status_of(rec_results.get("drift_recovery_status")),
        "identifiability_recovery": _status_of(rec_results.get("identifiability_recovery"))
        or _status_of(rec_results.get("identifiability_recovery_status")),
    }
    transform = rec_results.get("transform_recovery")
    if isinstance(transform, dict) and transform:
        subs = [_status_of(v) for v in transform.values()]
        subs = [s for s in subs if s is not None]
        if subs:
            if "fail" in subs:
                out["transform_recovery"] = "fail"
            elif all(s == "pass" for s in subs):
                out["transform_recovery"] = "pass"
            else:
                out["transform_recovery"] = "partial"
    return out


def _run_smoke_world(bundle_dir: Path, spec: ScenarioSpec) -> Tier1WorldOutcome:
    outcome = Tier1WorldOutcome(
        world_id=spec.world_id,
        stratum="smoke_structural",
        axes={
            "family": spec.family,
            "noise_level": spec.noise_level,
            "correlation_level": spec.correlation_level,
            "drift": "true" if spec.drift else "false",
            "n_geos": str(spec.n_geos),
        },
        structural_status="error",
        contract_passed=None,
        recovery={cap: None for cap in RECOVERY_CAPABILITIES},
    )
    try:
        write_scenario_world(bundle_dir, spec)
        materialize_world(bundle_dir, overwrite=True)
        cert = run_world_certification(
            bundle_dir,
            write_report=False,
            include_recovery=False,
            include_deferred_registry_rows=True,
        )
        outcome.structural_status = str(cert.report.get("overall_status", cert.overall_status))
        contract = cert.report.get("contract_compatibility") or {}
        outcome.contract_passed = contract.get("passed") if "passed" in contract else None
    except Exception as exc:
        outcome.error = f"{type(exc).__name__}: {exc}"
    return outcome


def _run_behavioral_world(bundle_dir: Path, spec: BehavioralWorldSpec) -> Tier1WorldOutcome:
    outcome = Tier1WorldOutcome(
        world_id=spec.world_id,
        stratum="behavioral_rich_dgp",
        axes=spec.axis_dict(),
        structural_status="error",
        contract_passed=None,
        recovery={cap: None for cap in RECOVERY_CAPABILITIES},
    )
    try:
        truth = _behavioral.build_behavioral_world_truth(spec)
        write_world_truth(bundle_dir, truth)
        _behavioral._copy_train_config(bundle_dir, spec)
        materialize_dgp_world(bundle_dir, overwrite=True)
        _behavioral._post_materialize(bundle_dir, spec)
        cert = run_world_certification(
            bundle_dir,
            write_report=False,
            include_recovery=None,
            include_deferred_registry_rows=True,
        )
        outcome.structural_status = str(cert.report.get("overall_status", cert.overall_status))
        contract = cert.report.get("contract_compatibility") or {}
        outcome.contract_passed = contract.get("passed") if "passed" in contract else None
        outcome.recovery = _recovery_statuses(cert.report)
    except Exception as exc:
        outcome.error = f"{type(exc).__name__}: {exc}"
    return outcome


def _run_anchor_world(repo_root: Path, world_id: str) -> Tier1WorldOutcome:
    outcome = Tier1WorldOutcome(
        world_id=world_id,
        stratum="anchor",
        axes={"anchor": world_id},
        structural_status="error",
        contract_passed=None,
        recovery={cap: None for cap in RECOVERY_CAPABILITIES},
    )
    try:
        bundle = repo_root / "validation" / "worlds" / world_id
        cert = run_world_certification(
            bundle,
            write_report=False,
            include_recovery=None,
            include_deferred_registry_rows=True,
        )
        outcome.structural_status = str(cert.report.get("overall_status", cert.overall_status))
        contract = cert.report.get("contract_compatibility") or {}
        outcome.contract_passed = contract.get("passed") if "passed" in contract else None
        outcome.recovery = _recovery_statuses(cert.report)
    except Exception as exc:
        outcome.error = f"{type(exc).__name__}: {exc}"
    return outcome


def run_tier1_batch(
    repo_root: str | Path,
    work_root: str | Path,
    *,
    smoke: tuple[ScenarioSpec, ...] | None = None,
    behavioral: tuple[BehavioralWorldSpec, ...] | None = None,
    include_anchors: bool = True,
) -> dict[str, Any]:
    """Execute the full Tier-1 batch; bundles live under ``work_root`` (uncommitted)."""
    repo = Path(repo_root)
    work = Path(work_root)
    work.mkdir(parents=True, exist_ok=True)
    outcomes: list[Tier1WorldOutcome] = []
    for spec in smoke if smoke is not None else tier1_smoke_specs():
        outcomes.append(_run_smoke_world(work / "smoke" / spec.world_id, spec))
    for spec in behavioral if behavioral is not None else tier1_behavioral_specs():
        outcomes.append(_run_behavioral_world(work / "behavioral" / spec.world_id, spec))
    if include_anchors:
        for world_id in ANCHOR_WORLD_IDS:
            outcomes.append(_run_anchor_world(repo, world_id))
    return {
        "tier1_version": TIER1_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "world_count": len(outcomes),
        "outcomes": [o.to_dict() for o in outcomes],
    }


def _capability_distributions(batch: dict[str, Any]) -> list[CapabilityDistribution]:
    outcomes = batch.get("outcomes") or []
    dists: list[CapabilityDistribution] = []
    struct = [o for o in outcomes if o.get("structural_status") in ("pass", "fail")]
    struct_pass = sum(1 for o in struct if o.get("structural_status") == "pass")
    dists.append(
        CapabilityDistribution(
            capability="structural_integrity",
            metric_class="structural",
            n_observations=len(struct),
            pass_rate=(struct_pass / len(struct)) if struct else None,
            mean_score=(struct_pass / len(struct)) if struct else None,
            failure_regions=[],
            notes="Tier-1 batch structural overall_status",
        )
    )
    contracted = [o for o in outcomes if o.get("contract_passed") in (True, False)]
    cpass = sum(1 for o in contracted if o.get("contract_passed") is True)
    dists.append(
        CapabilityDistribution(
            capability="platform_contract_compatibility",
            metric_class="structural",
            n_observations=len(contracted),
            pass_rate=(cpass / len(contracted)) if contracted else None,
            mean_score=(cpass / len(contracted)) if contracted else None,
            failure_regions=[],
            notes="Tier-1 batch contract compatibility",
        )
    )
    metric_class = {
        "coefficient_recovery": "diagnostic_attribution",
        "transform_recovery": "diagnostic_attribution",
        "delta_mu_recovery": "decision_grade",
        "optimizer_recovery": "decision_grade",
        "replay_recovery": "decision_grade",
        "drift_recovery": "trust_modifier",
        "identifiability_recovery": "trust_modifier",
    }
    for cap in RECOVERY_CAPABILITIES:
        scored = [
            o for o in outcomes if (o.get("recovery") or {}).get(cap) in ("pass", "fail", "partial")
        ]
        npass = sum(1 for o in scored if (o.get("recovery") or {}).get(cap) == "pass")
        failures = sorted(
            {
                o.get("world_id", "")
                for o in scored
                if (o.get("recovery") or {}).get(cap) in ("fail", "partial")
            }
        )
        dists.append(
            CapabilityDistribution(
                capability=cap,
                metric_class=metric_class[cap],
                n_observations=len(scored),
                pass_rate=(npass / len(scored)) if scored else None,
                mean_score=(npass / len(scored)) if scored else None,
                failure_regions=failures[:25],
                notes="Tier-1 batch recovery sections (report-only)",
            )
        )
    return dists


def _axis_envelopes(batch: dict[str, Any]) -> dict[str, Any]:
    outcomes = batch.get("outcomes") or []
    envelopes: dict[str, dict[str, dict[str, int]]] = {}
    for axis in ("family", "noise_level", "correlation_level", "drift", "stratum"):
        groups: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "pass": 0, "fail": 0, "error": 0})
        for o in outcomes:
            val = str((o.get("axes") or {}).get(axis, o.get("stratum", "unknown")))
            if axis == "stratum":
                val = str(o.get("stratum", "unknown"))
            groups[val]["n"] += 1
            st = str(o.get("structural_status", "error"))
            groups[val]["pass" if st == "pass" else ("fail" if st == "fail" else "error")] += 1
        envelopes[axis] = dict(groups)
    return envelopes


def _tier1_recommendations(dists: list[CapabilityDistribution]) -> list[dict[str, Any]]:
    """Tier-1 observed-rate recommendations — report-only, never approved."""
    by_cap = {d.capability: d for d in dists}
    recs: list[dict[str, Any]] = []
    for key, cur in CURRENT_THRESHOLDS.items():
        val_id = key.split("_")[0]
        cap = {
            "VAL-001": "coefficient_recovery",
            "VAL-002": "transform_recovery",
            "VAL-003": "transform_recovery",
            "VAL-004": "delta_mu_recovery",
            "VAL-005": "optimizer_recovery",
            "VAL-006": "replay_recovery",
            "VAL-012": "drift_recovery",
        }.get(val_id, "")
        dist = by_cap.get(cap)
        rate = dist.pass_rate if dist and dist.pass_rate is not None else None
        n = dist.n_observations if dist else 0
        if cur["metric_class"] == "diagnostic_attribution":
            action = "report_only; must not gate releases"
        elif cur["metric_class"] == "trust_modifier":
            action = "calibrate severity bands from tier-1 distribution; DR-04 decides"
        elif rate is not None and rate >= 0.95 and (n or 0) >= 20:
            action = "candidate for DR-04 approval review; not approved"
        else:
            action = "retain_provisional; needs DR-04 review"
        recs.append(
            {
                "metric_key": key,
                "validation_id": val_id,
                "metric_class": cur["metric_class"],
                "current_threshold": cur,
                "tier1_pass_rate": rate,
                "tier1_n_scored": n,
                "suggested_action": action,
                "supporting_evidence": "tier1_batch_runner_v1.0.0 batch",
                "confidence_level": "medium" if (n or 0) >= 20 else "low",
                "approval_status": "recommendation_only_not_approved",
            }
        )
    return recs
    return {
        "tier1_version": TIER1_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tier": "tier_1_calibration",
        "world_count": batch.get("world_count"),
        "capability_distributions": [d.to_dict() for d in dists],
        "axis_envelopes": boundaries["reliability_envelopes"],
        "threshold_recommendations": recs,
        "recommendation_disclaimer": (
            "Report-only. No threshold is approved; thresholds remain provisional "
            "pending DR-04. No release-gate change."
        ),
        "limitations": [
            "Small-panel deterministic worlds; not production-panel geometry",
            "Smoke stratum is structural-only by design (recovery honestly unscored)",
            "Behavioral stratum inherits WORLD-008-template DGP assumptions",
            "Anchors re-certified read-only from committed fixtures",
        ],
    }


def write_tier1_characterization(
    characterization: dict[str, Any],
    output_path: str | Path,
) -> Path:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(characterization, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def write_tier1_recommendations_note(
    characterization: dict[str, Any],
    output_path: str | Path,
) -> Path:
    """Render the report-only recommendations note from a characterization document."""
    lines = [
        "# Monte Carlo Tier-1 recommendations (report-only)",
        "",
        "**Status:** recommendations only — not approved for production gates. "
        "Thresholds remain provisional pending DR-04.",
        "",
        f"Batch: `{characterization.get('tier1_version')}` · "
        f"tier `{characterization.get('tier')}` · "
        f"N={characterization.get('world_count')} worlds · "
        f"generated {characterization.get('generated_at')}.",
        "",
        "## Observed capability summary",
        "",
        "| Capability | Class | n scored | Pass rate |",
        "|------------|-------|----------|-----------|",
    ]
    for dist in characterization.get("capability_distributions") or []:
        rate = dist.get("pass_rate")
        lines.append(
            f"| {dist.get('capability')} | {dist.get('metric_class')} | "
            f"{dist.get('n_observations')} | "
            f"{(round(rate, 3) if rate is not None else 'unscored')} |"
        )
    lines += [
        "",
        "## Threshold recommendations (report-only)",
        "",
        "| Metric | VAL | Class | n | Pass rate | Suggested action | Confidence |",
        "|--------|-----|-------|---|-----------|------------------|------------|",
    ]
    for rec in characterization.get("threshold_recommendations") or []:
        lines.append(
            f"| {rec.get('metric_key')} | {rec.get('validation_id')} | "
            f"{rec.get('metric_class')} | {rec.get('tier1_n_scored')} | "
            f"{rec.get('tier1_pass_rate')} | {rec.get('suggested_action')} | "
            f"{rec.get('confidence_level')} |"
        )
    lines += [
        "",
        "## Disclaimer",
        "",
        characterization.get("recommendation_disclaimer", ""),
        "",
        "## Limitations",
        "",
    ]
    lines += [f"- {item}" for item in characterization.get("limitations") or []]
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out
