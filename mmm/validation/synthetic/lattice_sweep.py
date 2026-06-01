"""Phase 5A — deterministic ScenarioBuilder lattice sweep (MVP)."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mmm.validation.synthetic.certification_runner import run_world_certification
from mmm.validation.synthetic.materializer import materialize_world
from mmm.validation.synthetic.reliability_scorecard import build_scorecard_from_reports
from mmm.validation.synthetic.scenario_builder import ScenarioSpec, write_scenario_world

SWEEP_ID = "lattice_sweep_mvp"
SWEEP_VERSION = "lattice_sweep_v1.0.0"
LATTICE_REPORT_NAME = "lattice_sweep_mvp_report.json"
LATTICE_WORLD_PREFIX = "L5A-"

# MVP lattice axes (fixed grid, 12 worlds).
LATTICE_AXES: dict[str, tuple[str, ...]] = {
    "family": ("baseline", "replay"),
    "noise_level": ("low", "high"),
    "correlation_level": ("low", "severe"),
    "drift": ("false", "true"),
    "experiment_quality": ("none", "medium"),
}

# Baseline: 6 cells — noise x correlation x drift (subset); replay: 6 cells — same pattern.
_BASELINE_CELLS: tuple[tuple[str, str, bool], ...] = (
    ("low", "low", False),
    ("low", "severe", False),
    ("high", "low", False),
    ("high", "severe", False),
    ("low", "low", True),
    ("high", "severe", True),
)

_REPLAY_CELLS: tuple[tuple[str, str, bool], ...] = (
    ("low", "low", False),
    ("low", "severe", False),
    ("high", "low", False),
    ("high", "severe", False),
    ("low", "low", True),
    ("high", "severe", True),
)


def encode_world_id(
    *,
    family: str,
    noise_level: str,
    correlation_level: str,
    drift: bool,
    experiment_quality: str,
) -> str:
    """Deterministic world_id encoding scenario axes (no hand-editing)."""
    dr = "on" if drift else "off"
    return (
        f"{LATTICE_WORLD_PREFIX}{family}-noise-{noise_level}-"
        f"corr-{correlation_level}-drift-{dr}-eq-{experiment_quality}"
    )


def _deterministic_seed(world_id: str, base: int = 50_000) -> int:
    digest = hashlib.md5(world_id.encode("utf-8")).hexdigest()
    return base + int(digest[:8], 16) % 10_000


def scenario_spec_from_axes(
    *,
    family: str,
    noise_level: str,
    correlation_level: str,
    drift: bool,
    experiment_quality: str,
    n_geos: int = 2,
    n_periods: int = 12,
) -> ScenarioSpec:
    if family == "baseline":
        eq = "none"
        channels = ("search", "social")
    else:
        eq = experiment_quality if experiment_quality != "none" else "medium"
        channels = ("search",)
    world_id = encode_world_id(
        family=family,
        noise_level=noise_level,
        correlation_level=correlation_level,
        drift=drift,
        experiment_quality=eq,
    )
    return ScenarioSpec(
        world_id=world_id,
        family=family,
        seed=_deterministic_seed(world_id),
        n_geos=n_geos,
        n_periods=n_periods,
        channels=channels,
        noise_level=noise_level,
        correlation_level=correlation_level,
        seasonality="none",
        drift=drift,
        experiment_quality=eq,
        privacy_loss=False,
        missingness="none",
    )


def mvp_lattice_specs() -> tuple[ScenarioSpec, ...]:
    """Fixed 12-world MVP lattice (deterministic, 6 baseline + 6 replay)."""
    specs: list[ScenarioSpec] = []
    for noise, corr, drift in _BASELINE_CELLS:
        specs.append(
            scenario_spec_from_axes(
                family="baseline",
                noise_level=noise,
                correlation_level=corr,
                drift=drift,
                experiment_quality="none",
            )
        )
    for noise, corr, drift in _REPLAY_CELLS:
        specs.append(
            scenario_spec_from_axes(
                family="replay",
                noise_level=noise,
                correlation_level=corr,
                drift=drift,
                experiment_quality="medium",
            )
        )
    return tuple(specs)


def axis_metadata(spec: ScenarioSpec) -> dict[str, str]:
    return {
        "family": spec.family,
        "noise_level": spec.noise_level,
        "correlation_level": spec.correlation_level,
        "drift": "true" if spec.drift else "false",
        "experiment_quality": spec.experiment_quality,
    }


@dataclass
class WorldSweepOutcome:
    world_id: str
    bundle_dir: Path
    axis: dict[str, str]
    truth_written: bool
    materialized: bool
    certified: bool
    overall_status: str
    error: str | None = None
    report: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "world_id": self.world_id,
            "bundle_dir": str(self.bundle_dir.as_posix()),
            "axis": self.axis,
            "truth_written": self.truth_written,
            "materialized": self.materialized,
            "certified": self.certified,
            "overall_status": self.overall_status,
            "error": self.error,
        }


def _classify_failure_taxonomy(
    world_id: str,
    report: dict[str, Any] | None,
) -> dict[str, list[str]]:
    buckets: dict[str, list[str]] = {
        "structural_failures": [],
        "replay_failures": [],
        "contract_failures": [],
        "recovery_failures": [],
        "governance_failures": [],
        "unsupported_or_skipped": [],
    }
    if report is None:
        return buckets
    if not (report.get("contract_compatibility") or {}).get("passed", True):
        buckets["contract_failures"].append(f"{world_id}:contract_compatibility")
    for row in report.get("validation_results") or []:
        cid = str(row.get("check_id", ""))
        status = str(row.get("status", ""))
        tag = f"{world_id}:{cid}"
        if status == "skipped":
            buckets["unsupported_or_skipped"].append(tag)
            continue
        if status != "fail":
            continue
        if cid.startswith("CERT-4A-003") or cid.startswith("CERT-4A-008"):
            buckets["replay_failures"].append(tag)
        elif cid.startswith("CERT-4A-006") or cid.startswith("CERT-4A-012") or cid.startswith("CERT-4A-013"):
            buckets["governance_failures"].append(tag)
        elif cid.startswith("REC-4B"):
            buckets["recovery_failures"].append(tag)
        elif cid.startswith("CERT-4A-"):
            buckets["structural_failures"].append(tag)
        else:
            buckets["unsupported_or_skipped"].append(tag)
    return buckets


def _summarize_by_axis(
    outcomes: list[WorldSweepOutcome],
) -> dict[str, dict[str, Any]]:
    axis_names = ("family", "noise_level", "correlation_level", "drift", "experiment_quality")
    summary: dict[str, dict[str, Any]] = {ax: {} for ax in axis_names}
    groups: dict[str, dict[str, list[WorldSweepOutcome]]] = {ax: defaultdict(list) for ax in axis_names}

    for outcome in outcomes:
        for ax in axis_names:
            val = outcome.axis.get(ax, "unknown")
            groups[ax][val].append(outcome)

    for ax in axis_names:
        for val, rows in sorted(groups[ax].items()):
            n = len(rows)
            n_pass = sum(1 for r in rows if r.overall_status == "pass")
            n_fail = sum(1 for r in rows if r.overall_status == "fail")
            n_error = sum(1 for r in rows if r.overall_status == "error")
            severe_warn = sum(
                1
                for r in rows
                if r.report
                and any(
                    "identifiability_collinearity" in str(w)
                    for w in (r.report.get("warnings") or [])
                )
            )
            drift_partial = sum(
                1
                for r in rows
                if r.report
                and any(
                    "drift" in str(w).lower() or "VAL-012" in str(w)
                    for w in (r.report.get("partial_validations") or [])
                )
            )
            summary[ax][val] = {
                "world_count": n,
                "certification_pass_rate": float(n_pass / n) if n else 0.0,
                "certification_fail_rate": float(n_fail / n) if n else 0.0,
                "certification_error_rate": float(n_error / n) if n else 0.0,
                "world_ids": [r.world_id for r in rows],
                "collinearity_warning_worlds": severe_warn if ax == "correlation_level" else None,
                "drift_signal_worlds": drift_partial if ax == "drift" else None,
            }
    return summary


def _skipped_validation_summary(
    reports: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    by_reason: dict[str, int] = defaultdict(int)
    by_check: dict[str, int] = defaultdict(int)
    rows: list[dict[str, Any]] = []
    for world_id, report in reports.items():
        for skip in report.get("skipped_validations") or []:
            cid = str(skip.get("check_id", ""))
            reason = str(skip.get("skip_reason", "unknown"))
            by_reason[reason] += 1
            by_check[cid] += 1
            rows.append({"world_id": world_id, "check_id": cid, "skip_reason": reason})
    return {
        "total_skipped_rows": len(rows),
        "by_skip_reason": dict(by_reason),
        "by_check_id": dict(by_check),
        "entries": rows[:50],
    }


def run_single_world(
    bundle_dir: Path,
    spec: ScenarioSpec,
    *,
    overwrite: bool = True,
) -> WorldSweepOutcome:
    """ScenarioBuilder → truth → materialize → certify (failures preserved)."""
    outcome = WorldSweepOutcome(
        world_id=spec.world_id,
        bundle_dir=bundle_dir,
        axis=axis_metadata(spec),
        truth_written=False,
        materialized=False,
        certified=False,
        overall_status="error",
    )
    try:
        write_scenario_world(bundle_dir, spec)
        outcome.truth_written = True
        materialize_world(bundle_dir, overwrite=overwrite)
        outcome.materialized = True
        cert = run_world_certification(
            bundle_dir,
            write_report=True,
            include_recovery=False,
            include_deferred_registry_rows=True,
        )
        outcome.certified = True
        outcome.report = cert.report
        outcome.overall_status = str(cert.report.get("overall_status", cert.overall_status))
    except Exception as exc:
        outcome.error = str(exc)
        outcome.overall_status = "error"
    return outcome


def run_lattice_sweep(
    repo_root: str | Path,
    *,
    specs: tuple[ScenarioSpec, ...] | None = None,
    lattice_subdir: str = "lattice",
    overwrite: bool = True,
) -> dict[str, Any]:
    """
    Run full lattice pipeline and return sweep report dict.

    Worlds live under ``validation/worlds/{lattice_subdir}/<world_id>/``.
    """
    root = Path(repo_root)
    lattice_specs = specs or mvp_lattice_specs()
    worlds_root = root / "validation" / "worlds" / lattice_subdir
    worlds_root.mkdir(parents=True, exist_ok=True)

    outcomes: list[WorldSweepOutcome] = []
    reports: dict[str, dict[str, Any]] = {}
    taxonomy: dict[str, list[str]] = {
        "structural_failures": [],
        "replay_failures": [],
        "contract_failures": [],
        "recovery_failures": [],
        "governance_failures": [],
        "unsupported_or_skipped": [],
    }

    for spec in lattice_specs:
        bundle = worlds_root / spec.world_id
        result = run_single_world(bundle, spec, overwrite=overwrite)
        outcomes.append(result)
        if result.report is not None:
            reports[spec.world_id] = result.report
            for key, items in _classify_failure_taxonomy(spec.world_id, result.report).items():
                taxonomy[key].extend(items)

    scorecard = build_scorecard_from_reports(reports, mode="lattice_structural")

    per_world_status = {o.world_id: o.to_dict() for o in outcomes}
    per_axis_summary = _summarize_by_axis(outcomes)

    limitations = [
        "Phase 5A MVP — deterministic ScenarioBuilder lattice only (not Monte Carlo)",
        "Materialized panels use smoke materializer (constant KPI), not rich DGP",
        "Behavioral recovery (VAL-001–006) deferred/skipped — structural certification focus",
        "Small fixed grid (12 worlds); not statistically representative",
        "Does not certify production readiness or causal validity",
        "thresholds_TBD_v1_runtime — no automatic threshold learning",
    ]

    recommended_followups = [
        "Phase 5B — rich DGP lattice with recovery metrics per axis",
        "Dedicated drift_detection_runner for VAL-012 on drift axis",
        "Expand lattice with privacy_loss and missingness axes",
    ]

    return {
        "sweep_id": SWEEP_ID,
        "sweep_version": SWEEP_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "lattice_axes": {k: list(v) for k, v in LATTICE_AXES.items()},
        "world_count": len(lattice_specs),
        "world_ids": [s.world_id for s in lattice_specs],
        "per_world_status": per_world_status,
        "per_axis_summary": per_axis_summary,
        "capability_summary": scorecard.get("capability_summary", {}),
        "scorecard_summary": {
            "reliability_score": scorecard.get("reliability_score"),
            "reliability_score_method": scorecard.get("reliability_score_method"),
            "status_counts": scorecard.get("status_counts"),
            "coverage_ratio": scorecard.get("coverage_ratio"),
            "scored_capabilities": scorecard.get("scored_capabilities"),
            "release_readiness_interpretation": scorecard.get("release_readiness_interpretation"),
        },
        "failure_taxonomy": taxonomy,
        "skipped_validation_summary": _skipped_validation_summary(reports),
        "limitations": limitations,
        "required_warnings": list(scorecard.get("required_warnings", [])),
        "recommended_followups": recommended_followups,
    }


def write_lattice_sweep_report(
    repo_root: str | Path,
    output_path: str | Path | None = None,
    **kwargs: Any,
) -> Path:
    """Run lattice sweep and write ``lattice_sweep_mvp_report.json``."""
    root = Path(repo_root)
    report = run_lattice_sweep(root, **kwargs)
    out = (
        Path(output_path)
        if output_path is not None
        else root / "validation" / "reports" / LATTICE_REPORT_NAME
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out
