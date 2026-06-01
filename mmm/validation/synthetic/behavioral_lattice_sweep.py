"""Phase 5B — rich DGP behavioral lattice sweep (MVP)."""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import yaml

from mmm.data.schema import PanelSchema
from mmm.validation.synthetic._io import write_json
from mmm.validation.synthetic.certification_runner import run_world_certification
from mmm.validation.synthetic.dgp_materializer import materialize_dgp_world
from mmm.validation.synthetic.generators import write_world_truth
from mmm.validation.synthetic.optimizer_truth import (
    build_world_009_truth,
    enrich_world_009_decision_truth,
)
from mmm.validation.synthetic.recovery_certification import (
    build_recovery_mmm_config,
    is_coef_recovery_eligible,
    is_drift_recovery_eligible,
    is_identifiability_recovery_eligible,
    is_optimizer_recovery_eligible,
    is_replay_recovery_eligible,
)
from mmm.validation.synthetic.reliability_scorecard import (
    CAPABILITY_GROUPS,
    build_scorecard_from_reports,
)
from mmm.validation.synthetic.reliability_truth import build_world_011_truth, build_world_012_truth
from mmm.validation.synthetic.replay_truth import (
    build_world_010_truth,
    enrich_world_010_experiment_truth,
)

SWEEP_ID = "behavioral_lattice_sweep_mvp"
SWEEP_VERSION = "behavioral_lattice_sweep_v1.0.0"
BEHAVIORAL_REPORT_NAME = "behavioral_lattice_sweep_mvp_report.json"
BEHAVIORAL_WORLD_PREFIX = "L5B-"

_REPO_ROOT = Path(__file__).resolve().parents[3]
_WORLDS_ROOT = _REPO_ROOT / "validation" / "worlds"

WorldType = Literal[
    "exact_recovery",
    "optimizer",
    "replay",
    "drift",
    "identifiability",
]
BehavioralMode = Literal["behavioral", "structural_only", "unsupported"]

BEHAVIORAL_AXES: dict[str, tuple[str, ...]] = {
    "world_type": (
        "exact_recovery",
        "optimizer",
        "replay",
        "drift",
        "identifiability",
    ),
    "noise_level": ("zero", "low"),
    "correlation_level": ("low", "severe"),
    "drift": ("false", "true"),
    "replay": ("false", "true"),
}

# Fixed 10-world MVP grid (controlled, not full factorial).
_BEHAVIORAL_CELLS: tuple[tuple[str, str, str, bool, bool], ...] = (
    ("exact_recovery", "zero", "low", False, False),
    ("exact_recovery", "low", "low", False, False),
    ("exact_recovery", "low", "severe", False, False),
    ("optimizer", "zero", "low", False, False),
    ("optimizer", "low", "low", False, False),
    ("replay", "zero", "low", False, True),
    ("replay", "low", "low", False, True),
    ("drift", "zero", "low", True, False),
    ("drift", "low", "low", True, False),
    ("identifiability", "low", "severe", False, False),
)

_TRAIN_CONFIG_TEMPLATE: dict[str, str] = {
    "exact_recovery": "WORLD-008-exact-recovery",
    "optimizer": "WORLD-009-optimizer-recovery",
    "replay": "WORLD-010-replay-recovery",
    "drift": "WORLD-011-drift-recovery",
    "identifiability": "WORLD-012-identifiability-recovery",
}

_DGP_TAG_BY_TYPE: dict[str, str] = {
    "exact_recovery": "dgp:exact_recovery",
    "optimizer": "dgp:optimizer_recovery",
    "replay": "dgp:replay_recovery",
    "drift": "dgp:drift_recovery",
    "identifiability": "dgp:identifiability_recovery",
}

_CAPABILITY_SCOPE_BY_TYPE: dict[str, frozenset[str]] = {
    "exact_recovery": frozenset(
        {
            "structural_integrity",
            "transform_consistency",
            "coefficient_recovery",
            "delta_mu_recovery",
            "platform_contract_compatibility",
            "artifact_integrity",
        }
    ),
    "optimizer": frozenset(
        {
            "structural_integrity",
            "optimizer_recovery",
            "platform_contract_compatibility",
        }
    ),
    "replay": frozenset(
        {
            "structural_integrity",
            "replay_recovery",
            "platform_contract_compatibility",
            "artifact_integrity",
        }
    ),
    "drift": frozenset(
        {
            "structural_integrity",
            "drift_behavior",
            "governance_reaction",
            "platform_contract_compatibility",
        }
    ),
    "identifiability": frozenset(
        {
            "structural_integrity",
            "identifiability_behavior",
            "governance_reaction",
            "platform_contract_compatibility",
        }
    ),
}

_EXPECTED_SKIP_BY_TYPE: dict[str, frozenset[str]] = {
    "exact_recovery": frozenset({"VAL-005", "VAL-006", "VAL-007", "REC-4B2-006"}),
    "optimizer": frozenset({"VAL-001", "VAL-002", "VAL-003", "VAL-004", "VAL-006"}),
    "replay": frozenset({"VAL-001", "VAL-005"}),
    "drift": frozenset({"VAL-001", "VAL-004", "VAL-005", "REC-4B5-DRIFT-COEF"}),
    "identifiability": frozenset({"VAL-001", "VAL-004", "VAL-005", "REC-4B5-ID-COEF"}),
}

_RECOVERY_METRIC_KEYS = (
    "coefficient_recovery_status",
    "delta_mu_recovery_status",
    "optimizer_recovery_status",
    "replay_recovery_status",
    "drift_behavior_status",
    "identifiability_behavior_status",
    "contract_compatibility_status",
)


@dataclass(frozen=True)
class BehavioralWorldSpec:
    world_id: str
    world_type: str
    noise_level: str
    correlation_level: str
    drift: bool
    replay: bool
    seed: int
    behavioral_mode: BehavioralMode

    def axis_dict(self) -> dict[str, str]:
        return {
            "world_type": self.world_type,
            "noise_level": self.noise_level,
            "correlation_level": self.correlation_level,
            "drift": "true" if self.drift else "false",
            "replay": "true" if self.replay else "false",
        }


def encode_behavioral_world_id(
    *,
    world_type: str,
    noise_level: str,
    correlation_level: str,
    drift: bool,
    replay: bool,
) -> str:
    dr = "on" if drift else "off"
    rp = "on" if replay else "off"
    return (
        f"{BEHAVIORAL_WORLD_PREFIX}{world_type}-noise-{noise_level}-"
        f"corr-{correlation_level}-drift-{dr}-replay-{rp}"
    )


def _deterministic_seed(world_id: str, base: int = 58_000) -> int:
    digest = hashlib.md5(world_id.encode("utf-8")).hexdigest()
    return base + int(digest[:8], 16) % 10_000


def resolve_behavioral_mode(
    *,
    world_type: str,
    drift: bool,
    replay: bool,
    correlation_level: str,
) -> BehavioralMode:
    if world_type == "replay" and not replay:
        return "unsupported"
    if world_type == "drift" and not drift:
        return "unsupported"
    if world_type == "identifiability" and correlation_level != "severe":
        return "structural_only"
    if world_type in _DGP_TAG_BY_TYPE:
        return "behavioral"
    return "structural_only"


def behavioral_spec_from_cell(
    world_type: str,
    noise_level: str,
    correlation_level: str,
    drift: bool,
    replay: bool,
) -> BehavioralWorldSpec:
    world_id = encode_behavioral_world_id(
        world_type=world_type,
        noise_level=noise_level,
        correlation_level=correlation_level,
        drift=drift,
        replay=replay,
    )
    mode = resolve_behavioral_mode(
        world_type=world_type,
        drift=drift,
        replay=replay,
        correlation_level=correlation_level,
    )
    return BehavioralWorldSpec(
        world_id=world_id,
        world_type=world_type,
        noise_level=noise_level,
        correlation_level=correlation_level,
        drift=drift,
        replay=replay,
        seed=_deterministic_seed(world_id),
        behavioral_mode=mode,
    )


def mvp_behavioral_lattice_specs() -> tuple[BehavioralWorldSpec, ...]:
    return tuple(
        behavioral_spec_from_cell(wt, noise, corr, drift, replay)
        for wt, noise, corr, drift, replay in _BEHAVIORAL_CELLS
    )


def _load_world_008_template() -> dict[str, Any]:
    path = _WORLDS_ROOT / "WORLD-008-exact-recovery" / "world_truth.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _noise_std(noise_level: str) -> float:
    if noise_level == "zero":
        return 0.0
    if noise_level == "low":
        return 0.02
    return 0.0


def _apply_lattice_axes(truth: dict[str, Any], spec: BehavioralWorldSpec) -> dict[str, Any]:
    out = copy.deepcopy(truth)
    meta = out.setdefault("metadata", {})
    meta["world_id"] = spec.world_id
    meta["generation_seed"] = spec.seed
    tags = [t for t in (meta.get("scenario_tags") or []) if not str(t).startswith("noise:")]
    tags.append(f"noise:{spec.noise_level}")
    tags.append("lattice:behavioral_v1")
    tags.append(_DGP_TAG_BY_TYPE[spec.world_type])
    meta["scenario_tags"] = tags
    meta["lattice_axes"] = spec.axis_dict()
    meta["behavioral_mode"] = spec.behavioral_mode

    out["outcome_truth"]["observation_noise_std"] = _noise_std(spec.noise_level)
    out["outcome_truth"]["observation_noise_level"] = (
        "low" if spec.noise_level == "zero" else spec.noise_level
    )

    channels = list(out["media_truth"]["channels"])
    if spec.correlation_level == "severe" and len(channels) >= 2:
        primary, secondary = channels[0], channels[1]
        out["media_truth"]["spend_process_spec"] = {
            "correlation_level": "severe",
            "kind": "collinear_block",
            "primary_channel": primary,
            "secondary_channel": secondary,
            "scale": 0.98,
            "level": float(out["media_truth"]["baseline_spend_by_channel"].get(primary, 10.0)),
        }
        warnings = list((out.get("artifact_truth") or {}).get("expected_warnings") or [])
        if not any(w.get("warning_id") == "identifiability_collinearity" for w in warnings):
            warnings.append({"warning_id": "identifiability_collinearity", "severity": "high"})
        out.setdefault("artifact_truth", {})["expected_warnings"] = warnings
    else:
        spec_media = out["media_truth"].get("spend_process_spec") or {}
        if isinstance(spec_media, dict):
            spec_media["correlation_level"] = spec.correlation_level

    if spec.world_type != "drift":
        out["drift_truth"] = {
            "changepoints": [],
            "coefficient_drift": [],
            "policy_changes": [],
            "privacy_shifts": [],
        }

    return out


def build_behavioral_world_truth(spec: BehavioralWorldSpec) -> dict[str, Any]:
    """Authoritative rich-DGP truth for one lattice cell."""
    builders = {
        "exact_recovery": lambda: _apply_lattice_axes(_load_world_008_template(), spec),
        "optimizer": lambda: _apply_lattice_axes(build_world_009_truth(seed=spec.seed), spec),
        "replay": lambda: _apply_lattice_axes(build_world_010_truth(seed=spec.seed), spec),
        "drift": lambda: _apply_lattice_axes(build_world_011_truth(seed=spec.seed), spec),
        "identifiability": lambda: _apply_lattice_axes(build_world_012_truth(seed=spec.seed), spec),
    }
    return builders[spec.world_type]()


def _copy_train_config(bundle_dir: Path, spec: BehavioralWorldSpec) -> None:
    template_id = _TRAIN_CONFIG_TEMPLATE[spec.world_type]
    src = _WORLDS_ROOT / template_id / "train_config.yaml"
    dst = bundle_dir / "train_config.yaml"
    shutil.copy2(src, dst)
    data = yaml.safe_load(dst.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data.setdefault("data", {})["data_version_id"] = f"{spec.world_id}-recovery"
        dst.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def _panel_schema_from_truth(truth: dict[str, Any]) -> PanelSchema:
    geo_col = str(truth["geo_truth"]["geo_column_name"])
    week_col = str(truth["time_truth"]["week_column_name"])
    target = str(truth["outcome_truth"]["target_column"])
    channels = tuple(str(c) for c in truth["media_truth"]["channels"])
    return PanelSchema(geo_col, week_col, target, channels, ())


def _post_materialize(bundle_dir: Path, spec: BehavioralWorldSpec) -> None:
    if spec.world_type not in ("optimizer", "replay"):
        return
    truth = json.loads((bundle_dir / "world_truth.json").read_text(encoding="utf-8"))
    panel = pd.read_parquet(bundle_dir / "panel.parquet")
    config = build_recovery_mmm_config(truth, panel_path=bundle_dir / "panel.parquet")
    schema = _panel_schema_from_truth(truth)
    if spec.world_type == "optimizer":
        enriched = enrich_world_009_decision_truth(truth, panel, schema, config)
    else:
        enriched = enrich_world_010_experiment_truth(truth, panel, schema, config)
    write_json(bundle_dir / "world_truth.json", enriched)


def _row_status(section: dict[str, Any] | None) -> str | None:
    if not isinstance(section, dict):
        return None
    status = str(section.get("status", ""))
    if not status:
        return None
    details = section.get("details") if isinstance(section.get("details"), dict) else {}
    if str(details.get("registry_validation_id", "")) == "VAL-012":
        val_out = str(details.get("val_012_outcome", ""))
        if val_out == "warning":
            return "partial"
        if val_out == "severe":
            return "fail"
        if val_out == "pass":
            return "pass"
    return status


def extract_recovery_metrics(report: dict[str, Any]) -> dict[str, str | None]:
    contract = report.get("contract_compatibility") or {}
    contract_st: str | None
    if contract.get("passed") is True:
        contract_st = "pass"
    elif contract.get("passed") is False:
        contract_st = "fail"
    else:
        contract_st = None

    recovery_blob = report.get("recovery_results") if isinstance(report.get("recovery_results"), dict) else {}

    opt_st = _row_status(report.get("optimizer_recovery"))
    if opt_st is None:
        opt_st = report.get("optimizer_recovery_status") or recovery_blob.get("optimizer_recovery_status")

    replay_st = _row_status(report.get("replay_recovery"))
    if replay_st is None:
        replay_st = report.get("replay_recovery_status") or recovery_blob.get("replay_recovery_status")
        if replay_st is None and isinstance(report.get("replay_recovery"), dict):
            replay_st = report["replay_recovery"].get("replay_recovery_status")

    drift_st = _row_status(report.get("drift_recovery"))
    if drift_st is None:
        drift_st = report.get("drift_recovery_status") or recovery_blob.get("drift_recovery_status")

    id_st = _row_status(report.get("identifiability_recovery"))
    if id_st is None:
        id_st = report.get("identifiability_recovery_status") or recovery_blob.get(
            "identifiability_recovery_status"
        )

    coef_st = _row_status(report.get("coefficient_recovery"))
    delta_st = _row_status(report.get("delta_mu_recovery"))

    for row in report.get("recovery_validation_results") or []:
        cid = str(row.get("check_id", ""))
        st = str(row.get("status", ""))
        if cid == "REC-4B5-ID-COEF" and coef_st is None:
            coef_st = st
        if cid == "REC-4B5-DRIFT-COEF" and coef_st is None:
            coef_st = st
        if cid == "REC-4B5-DRIFT" and drift_st is None:
            details = row.get("details") if isinstance(row.get("details"), dict) else {}
            val_out = str(details.get("val_012_outcome", ""))
            if val_out == "warning":
                drift_st = "partial"
            elif val_out == "severe":
                drift_st = "fail"
            else:
                drift_st = st
        if cid == "REC-4B5-ID" and id_st is None:
            id_st = st
        if cid == "REC-4B4-REPLAY" and replay_st is None:
            replay_st = st
        if cid.startswith("REC-4B3") and "OPT" in cid and opt_st is None:
            opt_st = st

    return {
        "coefficient_recovery_status": coef_st,
        "delta_mu_recovery_status": delta_st,
        "optimizer_recovery_status": opt_st,
        "replay_recovery_status": replay_st,
        "drift_behavior_status": drift_st,
        "identifiability_behavior_status": id_st,
        "contract_compatibility_status": contract_st,
    }


def _metric_score(status: str | None) -> float | None:
    if status is None or status == "skipped":
        return None
    if status == "pass":
        return 1.0
    if status == "partial":
        return 0.5
    if status == "fail":
        return 0.0
    return None


def _metrics_in_scope(spec: BehavioralWorldSpec) -> tuple[str, ...]:
    if spec.behavioral_mode != "behavioral":
        return ()
    mapping: dict[str, tuple[str, ...]] = {
        "exact_recovery": (
            "coefficient_recovery_status",
            "delta_mu_recovery_status",
            "contract_compatibility_status",
        ),
        "optimizer": ("optimizer_recovery_status", "contract_compatibility_status"),
        "replay": ("replay_recovery_status", "contract_compatibility_status"),
        "drift": ("drift_behavior_status", "contract_compatibility_status"),
        "identifiability": (
            "identifiability_behavior_status",
            "contract_compatibility_status",
        ),
    }
    return mapping.get(spec.world_type, ())


def compute_behavioral_scores(
    reports: dict[str, dict[str, Any]],
    specs_by_id: dict[str, BehavioralWorldSpec],
) -> dict[str, Any]:
    behavioral_scores: list[float] = []
    per_world: dict[str, dict[str, Any]] = {}
    n_in_scope = 0
    n_scored = 0

    for world_id, report in reports.items():
        spec = specs_by_id[world_id]
        metrics = extract_recovery_metrics(report)
        in_scope = _metrics_in_scope(spec)
        world_scores: list[float] = []
        for key in in_scope:
            n_in_scope += 1
            st = metrics.get(key)
            if (
                key == "coefficient_recovery_status"
                and spec.world_type in ("identifiability", "drift")
                and st == "skipped"
            ):
                continue
            sc = _metric_score(st)
            if sc is not None:
                n_scored += 1
                world_scores.append(sc)
                behavioral_scores.append(sc)
        per_world[world_id] = {
            "behavioral_mode": spec.behavioral_mode,
            "recovery_metrics": metrics,
            "metrics_in_scope": list(in_scope),
            "world_behavioral_score": (
                float(sum(world_scores) / len(world_scores)) if world_scores else None
            ),
        }

    behavioral_score = (
        float(sum(behavioral_scores) / len(behavioral_scores)) if behavioral_scores else None
    )
    coverage_ratio = float(n_scored / n_in_scope) if n_in_scope else 0.0
    return {
        "behavioral_score": behavioral_score,
        "coverage_ratio": coverage_ratio,
        "per_world": per_world,
        "n_metrics_in_scope": n_in_scope,
        "n_metrics_scored": n_scored,
    }


def _scope_and_skip_overrides(
    specs: tuple[BehavioralWorldSpec, ...],
) -> tuple[dict[str, frozenset[str]], dict[str, frozenset[str]]]:
    scopes: dict[str, frozenset[str]] = {}
    skips: dict[str, frozenset[str]] = {}
    for spec in specs:
        scopes[spec.world_id] = _CAPABILITY_SCOPE_BY_TYPE[spec.world_type]
        skips[spec.world_id] = _EXPECTED_SKIP_BY_TYPE[spec.world_type]
    return scopes, skips


@dataclass
class BehavioralWorldOutcome:
    spec: BehavioralWorldSpec
    bundle_dir: Path
    truth_written: bool = False
    materialized: bool = False
    certified: bool = False
    overall_status: str = "error"
    error: str | None = None
    report: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "world_id": self.spec.world_id,
            "bundle_dir": str(self.bundle_dir.as_posix()),
            "axes": self.spec.axis_dict(),
            "behavioral_mode": self.spec.behavioral_mode,
            "truth_written": self.truth_written,
            "materialized": self.materialized,
            "certified": self.certified,
            "overall_status": self.overall_status,
            "recovery_eligible": self._recovery_eligible(),
            "error": self.error,
        }

    def _recovery_eligible(self) -> bool:
        if self.spec.behavioral_mode != "behavioral":
            return False
        truth = {
            "metadata": {
                "world_id": self.spec.world_id,
                "scenario_tags": [_DGP_TAG_BY_TYPE[self.spec.world_type]],
            }
        }
        return (
            is_coef_recovery_eligible(truth)
            or is_optimizer_recovery_eligible(truth)
            or is_replay_recovery_eligible(truth)
            or is_drift_recovery_eligible(truth)
            or is_identifiability_recovery_eligible(truth)
        )


def run_single_behavioral_world(
    bundle_dir: Path,
    spec: BehavioralWorldSpec,
    *,
    overwrite: bool = True,
) -> BehavioralWorldOutcome:
    outcome = BehavioralWorldOutcome(spec=spec, bundle_dir=bundle_dir)
    if spec.behavioral_mode == "unsupported":
        outcome.overall_status = "unsupported"
        outcome.error = "axis combination not behaviorally implemented"
        return outcome

    try:
        truth = build_behavioral_world_truth(spec)
        write_world_truth(bundle_dir, truth)
        outcome.truth_written = True
        _copy_train_config(bundle_dir, spec)
        materialize_dgp_world(bundle_dir, overwrite=overwrite)
        _post_materialize(bundle_dir, spec)
        outcome.materialized = True

        include_recovery = spec.behavioral_mode == "behavioral"
        cert = run_world_certification(
            bundle_dir,
            write_report=True,
            include_recovery=include_recovery,
            include_deferred_registry_rows=True,
        )
        outcome.certified = True
        outcome.report = cert.report
        outcome.overall_status = str(cert.report.get("overall_status", cert.overall_status))
    except Exception as exc:
        outcome.error = str(exc)
        outcome.overall_status = "error"
    return outcome


def _summarize_recovery_by_axis(
    outcomes: list[BehavioralWorldOutcome],
    *,
    axis_name: str,
) -> dict[str, Any]:
    groups: dict[str, list[BehavioralWorldOutcome]] = defaultdict(list)
    for outcome in outcomes:
        if outcome.report is None:
            continue
        val = outcome.spec.axis_dict().get(axis_name, "unknown")
        groups[val].append(outcome)

    summary: dict[str, Any] = {}
    for val, rows in sorted(groups.items()):
        scores: list[float] = []
        metric_pass: dict[str, int] = defaultdict(int)
        metric_total: dict[str, int] = defaultdict(int)
        for outcome in rows:
            metrics = extract_recovery_metrics(outcome.report or {})
            for key in _metrics_in_scope(outcome.spec):
                metric_total[key] += 1
                st = metrics.get(key)
                sc = _metric_score(st)
                if sc is not None:
                    scores.append(sc)
                    if sc >= 0.5:
                        metric_pass[key] += 1
        summary[val] = {
            "world_count": len(rows),
            "world_ids": [r.spec.world_id for r in rows],
            "mean_behavioral_metric_score": float(sum(scores) / len(scores)) if scores else None,
            "metric_pass_rates": {
                k: float(metric_pass[k] / metric_total[k]) if metric_total[k] else None
                for k in sorted(metric_total)
            },
        }
    return summary


def _collect_failures_partials_skips(
    reports: dict[str, dict[str, Any]],
) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    failures: list[str] = []
    partials: list[str] = []
    skips: list[dict[str, Any]] = []
    for world_id, report in reports.items():
        for row in report.get("validation_results") or []:
            cid = str(row.get("check_id", ""))
            st = str(row.get("status", ""))
            tag = f"{world_id}:{cid}"
            if st == "fail":
                failures.append(tag)
            elif st == "skipped":
                skips.append(
                    {
                        "world_id": world_id,
                        "check_id": cid,
                        "skip_reason": row.get("skip_reason"),
                    }
                )
        for row in report.get("recovery_validation_results") or []:
            cid = str(row.get("check_id", ""))
            st = str(row.get("status", ""))
            tag = f"{world_id}:{cid}"
            if st == "fail":
                failures.append(tag)
            elif st == "skipped":
                skips.append(
                    {
                        "world_id": world_id,
                        "check_id": cid,
                        "skip_reason": row.get("skip_reason"),
                    }
                )
            elif st == "pass":
                details = row.get("details") if isinstance(row.get("details"), dict) else {}
                if str(details.get("registry_validation_id", "")) == "VAL-012":
                    partials.append(tag)
        metrics = extract_recovery_metrics(report)
        for key, st in metrics.items():
            if st == "partial":
                partials.append(f"{world_id}:{key}")
            elif st == "fail":
                failures.append(f"{world_id}:{key}")
    return failures, partials, skips


def run_behavioral_lattice_sweep(
    repo_root: str | Path,
    *,
    specs: tuple[BehavioralWorldSpec, ...] | None = None,
    lattice_subdir: str = "behavioral-lattice",
    overwrite: bool = True,
) -> dict[str, Any]:
    root = Path(repo_root)
    lattice_specs = specs or mvp_behavioral_lattice_specs()
    specs_by_id = {s.world_id: s for s in lattice_specs}
    worlds_root = root / "validation" / "worlds" / lattice_subdir
    worlds_root.mkdir(parents=True, exist_ok=True)

    outcomes: list[BehavioralWorldOutcome] = []
    reports: dict[str, dict[str, Any]] = {}

    for spec in lattice_specs:
        bundle = worlds_root / spec.world_id
        result = run_single_behavioral_world(bundle, spec, overwrite=overwrite)
        outcomes.append(result)
        if result.report is not None:
            reports[spec.world_id] = result.report

    scope_overrides, skip_overrides = _scope_and_skip_overrides(lattice_specs)
    structural_scorecard = build_scorecard_from_reports(
        reports,
        mode="lattice_structural",
    )
    behavioral_scorecard = build_scorecard_from_reports(
        reports,
        mode="recovery",
        world_scope_overrides=scope_overrides,
        expected_skip_overrides=skip_overrides,
    )
    behavioral_scores = compute_behavioral_scores(reports, specs_by_id)

    failures, partials, skips = _collect_failures_partials_skips(reports)

    per_axis_recovery: dict[str, Any] = {
        "world_type": _summarize_recovery_by_axis(outcomes, axis_name="world_type"),
        "noise_level": _summarize_recovery_by_axis(outcomes, axis_name="noise_level"),
        "correlation_level": _summarize_recovery_by_axis(outcomes, axis_name="correlation_level"),
        "drift": _summarize_recovery_by_axis(outcomes, axis_name="drift"),
        "replay": _summarize_recovery_by_axis(outcomes, axis_name="replay"),
    }

    capability_recovery: dict[str, Any] = {}
    for cap in CAPABILITY_GROUPS:
        if cap in behavioral_scorecard.get("capability_summary", {}):
            capability_recovery[cap] = behavioral_scorecard["capability_summary"][cap]

    limitations = [
        "Phase 5B MVP — small fixed behavioral lattice (10 rich DGP worlds)",
        "Tolerances are TBD_v1_runtime provisional — not production gates",
        "VAL-012 executed via drift_detection_runner on drift lattice cells",
        "Identifiability worlds skip coefficient recovery by design (REC-4B5-ID-COEF)",
        "No causal incrementality claims — synthetic recovery only",
        "No Monte Carlo, Bayesian validation, or automatic threshold learning",
        "Does not certify production release readiness",
    ]

    return {
        "sweep_id": SWEEP_ID,
        "sweep_version": SWEEP_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "world_count": len(lattice_specs),
        "world_ids": [s.world_id for s in lattice_specs],
        "axes": {k: list(v) for k, v in BEHAVIORAL_AXES.items()},
        "per_world_certification_status": {o.spec.world_id: o.to_dict() for o in outcomes},
        "per_axis_recovery_summary": per_axis_recovery,
        "capability_recovery_summary": capability_recovery,
        "scorecard_summary": {
            "behavioral_score": behavioral_scores["behavioral_score"],
            "structural_score": structural_scorecard.get("reliability_score"),
            "behavioral_scorecard_reliability": behavioral_scorecard.get("reliability_score"),
            "coverage_ratio": behavioral_scores["coverage_ratio"],
            "reliability_score_method": behavioral_scorecard.get("reliability_score_method"),
            "status_counts": behavioral_scorecard.get("status_counts"),
            "release_readiness_interpretation": behavioral_scorecard.get(
                "release_readiness_interpretation"
            ),
        },
        "per_world_recovery": behavioral_scores["per_world"],
        "failures": sorted(set(failures)),
        "partials": sorted(set(partials)),
        "skips": skips,
        "limitations": limitations,
        "recommended_followups": [
            "Phase 5F — Monte Carlo threshold calibration (INV-060)",
            "ReliabilityScorecard release-review role (DR-06)",
            "Expand behavioral lattice after threshold calibration pilot",
        ],
    }


def write_behavioral_lattice_sweep_report(
    repo_root: str | Path,
    output_path: str | Path | None = None,
    **kwargs: Any,
) -> Path:
    root = Path(repo_root)
    report = run_behavioral_lattice_sweep(root, **kwargs)
    out = (
        Path(output_path)
        if output_path is not None
        else root / "validation" / "reports" / BEHAVIORAL_REPORT_NAME
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out
