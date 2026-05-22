"""Lightweight local readers for continuous / decision validation (no remote registry)."""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from mmm.evaluation.experiment_evidence_extension import load_evidence_from_path
from mmm.experiments.evidence import ExperimentEvidence

ACCEPTED_RUN_REGISTRY_VERSION = "mmm_accepted_run_registry_v1"
DECISION_VALIDATION_REGISTRY_VERSION = "mmm_decision_validation_registry_v1"


def _parse_date(value: date | str | datetime | None) -> date | None:
    if value is None:
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    s = str(value).strip()[:10]
    try:
        return date.fromisoformat(s)
    except ValueError:
        return None


def load_json_file(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(str(p))
    raw = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"expected JSON object at {p}")
    return raw


def load_extension_report(path: str | Path) -> dict[str, Any]:
    raw = load_json_file(path)
    if "extension_report" in raw and isinstance(raw["extension_report"], dict):
        return raw["extension_report"]
    return raw


def load_decision_bundle(path: str | Path) -> dict[str, Any]:
    raw = load_json_file(path)
    if "decision_bundle" in raw and isinstance(raw["decision_bundle"], dict):
        return raw["decision_bundle"]
    return raw


def load_accepted_run_registry(registry_dir: str | Path) -> list[dict[str, Any]]:
    """
    Load ``accepted_runs.json`` or merge ``runs/*.json`` from a directory.
    """
    root = Path(registry_dir)
    if not root.exists():
        return []
    single = root / "accepted_runs.json"
    if single.is_file():
        data = load_json_file(single)
        runs = data.get("runs") or []
        return [r for r in runs if isinstance(r, dict)]
    runs_out: list[dict[str, Any]] = []
    for p in sorted(root.glob("**/*.json")):
        if p.name == "accepted_runs.json":
            continue
        try:
            blob = load_json_file(p)
        except (OSError, json.JSONDecodeError, ValueError):
            continue
        if "run_id" in blob:
            runs_out.append(blob)
        elif isinstance(blob.get("runs"), list):
            runs_out.extend(x for x in blob["runs"] if isinstance(x, dict))
    return runs_out


def load_decision_registry(decision_registry_dir: str | Path) -> list[dict[str, Any]]:
    root = Path(decision_registry_dir)
    if not root.exists():
        return []
    single = root / "decisions.json"
    if single.is_file():
        data = load_json_file(single)
        decs = data.get("decisions") or []
        return [d for d in decs if isinstance(d, dict)]
    out: list[dict[str, Any]] = []
    for p in sorted(root.glob("**/*.json")):
        if p.name == "decisions.json":
            continue
        try:
            blob = load_json_file(p)
        except (OSError, json.JSONDecodeError, ValueError):
            continue
        if "decision_id" in blob:
            out.append(blob)
        elif isinstance(blob.get("decisions"), list):
            out.extend(x for x in blob["decisions"] if isinstance(x, dict))
    return out


def load_experiment_evidence_list(
    path: str | Path | None,
    *,
    calibration_evidence_path: str | None = None,
) -> list[ExperimentEvidence]:
    p = path or calibration_evidence_path
    if not p:
        return []
    return load_evidence_from_path(p)


def filter_runs_before(
    runs: list[dict[str, Any]],
    *,
    before: date,
    lookback_days: int,
) -> list[dict[str, Any]]:
    window_start = before - timedelta(days=max(lookback_days, 1))
    kept: list[dict[str, Any]] = []
    for run in runs:
        completed = _parse_date(run.get("completed_at") or run.get("run_completed_at"))
        if completed is None:
            continue
        if window_start <= completed < before:
            kept.append(run)
    return sorted(kept, key=lambda r: str(r.get("completed_at") or r.get("run_completed_at")))


def extract_predicted_lift_for_experiment(
    experiment_id: str,
    *,
    run_record: dict[str, Any] | None = None,
    extension_report: dict[str, Any] | None = None,
) -> float | None:
    """Retrieve stored or extension-embedded predicted lift for an experiment id."""
    if run_record:
        for item in run_record.get("predicted_lifts") or []:
            if isinstance(item, dict) and item.get("experiment_id") == experiment_id:
                v = item.get("predicted_lift")
                if v is not None:
                    return float(v)
    if not isinstance(extension_report, dict):
        return None
    bel = extension_report.get("bayesian_experiment_likelihood_report")
    if isinstance(bel, dict):
        fit = bel.get("posterior_experiment_fit") or {}
        if experiment_id in fit and isinstance(fit[experiment_id], dict):
            m = fit[experiment_id].get("implied_lift_mean")
            if m is not None:
                return float(m)
    ew = extension_report.get("evidence_weighted_replay_summary")
    if isinstance(ew, dict):
        for unit in ew.get("units") or []:
            if isinstance(unit, dict) and unit.get("experiment_id") == experiment_id:
                imp = unit.get("implied_delta")
                if imp is not None:
                    return float(imp)
    return None


def extract_channel_ranking_from_bundle(bundle: dict[str, Any]) -> list[str]:
    """Infer channel priority order from decision / simulation artifacts."""
    sim = bundle.get("simulation_at_recommendation") or bundle.get("simulation_json")
    if isinstance(sim, dict):
        spends = sim.get("channel_spends") or sim.get("recommended_spends")
        if isinstance(spends, dict):
            return sorted(spends.keys(), key=lambda c: -float(spends[c]))
    rob = bundle.get("robust_optimization_research")
    if isinstance(rob, dict) and rob.get("candidate_details"):
        best = max(
            rob["candidate_details"],
            key=lambda r: float(r.get("expected_delta_mu", -1e18)),
        )
        alloc = best.get("allocation") if isinstance(best, dict) else {}
        if isinstance(alloc, dict):
            return sorted(alloc.keys(), key=lambda c: -float(alloc[c]))
    rs = bundle.get("resolved_config_snapshot") or {}
    data = rs.get("data") if isinstance(rs, dict) else {}
    chans = data.get("channel_columns") if isinstance(data, dict) else []
    if isinstance(chans, list):
        return [str(c) for c in chans]
    return []


def extract_recommended_allocation(bundle: dict[str, Any]) -> dict[str, float]:
    sim = bundle.get("simulation_at_recommendation") or bundle.get("simulation_json")
    if isinstance(sim, dict):
        spends = sim.get("channel_spends") or sim.get("recommended_spends")
        if isinstance(spends, dict):
            return {str(k): float(v) for k, v in spends.items()}
    opt = bundle.get("optimization_result")
    if isinstance(opt, dict):
        spends = opt.get("channel_spends") or opt.get("allocation")
        if isinstance(spends, dict):
            return {str(k): float(v) for k, v in spends.items()}
    return {}
