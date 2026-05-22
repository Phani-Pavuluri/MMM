"""Experiment evidence registry — store, validate, query; no experiment execution or auto-calibration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from mmm.experiments.evidence import (
    ApprovalStatus,
    ExperimentEvidence,
    validate_evidence_for_registry,
)

EVIDENCE_REGISTRY_VERSION = "mmm_experiment_evidence_registry_v1"
DEFAULT_STALE_DAYS = 365


@dataclass
class RegistryCoverage:
    """Freshness and coverage summary for operator dashboards."""

    n_total: int
    n_accepted: int
    n_rejected: int
    n_expired: int
    n_draft: int
    channels: list[str]
    kpis: list[str]
    geo_granularities: list[str]
    oldest_freshness: str | None
    newest_freshness: str | None
    stale_count: int

    def to_json(self) -> dict[str, Any]:
        return {
            "n_total": self.n_total,
            "n_accepted": self.n_accepted,
            "n_rejected": self.n_rejected,
            "n_expired": self.n_expired,
            "n_draft": self.n_draft,
            "channels": self.channels,
            "kpis": self.kpis,
            "geo_granularities": self.geo_granularities,
            "oldest_freshness": self.oldest_freshness,
            "newest_freshness": self.newest_freshness,
            "stale_count": self.stale_count,
        }


class ExperimentEvidenceRegistry:
    """
    In-memory evidence registry.

    Guardrails: does not run experiments, does not auto-calibrate, only stores/validates evidence.
    """

    def __init__(self, *, stale_after_days: int = DEFAULT_STALE_DAYS) -> None:
        self._records: dict[str, ExperimentEvidence] = {}
        self._stale_after_days = stale_after_days

    def register(self, evidence: ExperimentEvidence, *, allow_duplicate: bool = False) -> None:
        violations = validate_evidence_for_registry(evidence)
        if violations:
            raise ValueError(f"evidence failed registry validation: {violations}")
        eid = evidence.experiment_id
        if eid in self._records and not allow_duplicate:
            raise ValueError(f"experiment_id already registered: {eid!r}")
        self._records[eid] = evidence

    def get(self, experiment_id: str) -> ExperimentEvidence | None:
        return self._records.get(experiment_id)

    def list_all(self) -> list[ExperimentEvidence]:
        return list(self._records.values())

    def mark_accepted(self, experiment_id: str) -> None:
        self._set_approval(experiment_id, ApprovalStatus.ACCEPTED)

    def mark_rejected(self, experiment_id: str) -> None:
        self._set_approval(experiment_id, ApprovalStatus.REJECTED)

    def mark_expired(self, experiment_id: str) -> None:
        self._set_approval(experiment_id, ApprovalStatus.EXPIRED)

    def _set_approval(self, experiment_id: str, status: ApprovalStatus) -> None:
        rec = self.get(experiment_id)
        if rec is None:
            raise KeyError(f"unknown experiment_id: {experiment_id!r}")
        updated = rec.model_copy(update={"approval_status": status})
        self._records[experiment_id] = updated

    def retrieve(
        self,
        *,
        channel: str | None = None,
        kpi: str | None = None,
        geo_scope_contains: str | None = None,
        time_start: str | None = None,
        time_end: str | None = None,
        approval_only: bool = False,
        exclude_expired: bool = True,
    ) -> list[ExperimentEvidence]:
        out: list[ExperimentEvidence] = []
        for ev in self._records.values():
            if approval_only and ev.approval_status != ApprovalStatus.ACCEPTED:
                continue
            if exclude_expired and ev.approval_status == ApprovalStatus.EXPIRED:
                continue
            if channel is not None and ev.channel != channel:
                continue
            if kpi is not None and ev.kpi != kpi:
                continue
            if geo_scope_contains is not None:
                scope = {str(g) for g in ev.geo_scope}
                if geo_scope_contains not in scope and not (
                    not scope and geo_scope_contains.lower() in {"national", "us", "all"}
                ):
                    continue
            if time_start is not None and ev.time_window.end < time_start:
                continue
            if time_end is not None and ev.time_window.start > time_end:
                continue
            out.append(ev)
        return out

    def is_stale(self, evidence: ExperimentEvidence, *, as_of: date | None = None) -> bool:
        ref = as_of or date.today()
        fd = evidence.freshness_as_date()
        return (ref - fd).days > self._stale_after_days

    def coverage(self, *, as_of: date | None = None) -> RegistryCoverage:
        ref = as_of or date.today()
        recs = list(self._records.values())
        stale = sum(1 for r in recs if self.is_stale(r, as_of=ref))
        dates = sorted(r.freshness_as_date().isoformat() for r in recs) if recs else []
        by_status = {s: 0 for s in ApprovalStatus}
        for r in recs:
            by_status[r.approval_status] = by_status.get(r.approval_status, 0) + 1
        return RegistryCoverage(
            n_total=len(recs),
            n_accepted=by_status.get(ApprovalStatus.ACCEPTED, 0),
            n_rejected=by_status.get(ApprovalStatus.REJECTED, 0),
            n_expired=by_status.get(ApprovalStatus.EXPIRED, 0),
            n_draft=by_status.get(ApprovalStatus.DRAFT, 0)
            + by_status.get(ApprovalStatus.PENDING, 0),
            channels=sorted({r.channel for r in recs}),
            kpis=sorted({r.kpi for r in recs}),
            geo_granularities=sorted({r.geo_granularity.value for r in recs}),
            oldest_freshness=dates[0] if dates else None,
            newest_freshness=dates[-1] if dates else None,
            stale_count=stale,
        )


def load_evidence_registry(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"evidence registry not found: {p}")
    raw = json.loads(p.read_text(encoding="utf-8"))
    if raw.get("registry_version") != EVIDENCE_REGISTRY_VERSION:
        raise ValueError(
            f"unsupported registry_version {raw.get('registry_version')!r}; "
            f"expected {EVIDENCE_REGISTRY_VERSION!r}"
        )
    return raw


def save_evidence_registry(path: str | Path, registry: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(registry, indent=2, sort_keys=True, default=str), encoding="utf-8")
    tmp.replace(p)


def empty_evidence_registry() -> dict[str, Any]:
    return {"registry_version": EVIDENCE_REGISTRY_VERSION, "experiments": {}}


def registry_from_dict(data: dict[str, Any]) -> ExperimentEvidenceRegistry:
    reg = ExperimentEvidenceRegistry()
    ex = data.get("experiments") or {}
    if not isinstance(ex, dict):
        raise ValueError("registry.experiments must be an object")
    for _eid, blob in ex.items():
        ev = ExperimentEvidence.model_validate(blob)
        reg.register(ev, allow_duplicate=True)
    return reg


def registry_to_dict(reg: ExperimentEvidenceRegistry) -> dict[str, Any]:
    return {
        "registry_version": EVIDENCE_REGISTRY_VERSION,
        "experiments": {e.experiment_id: e.to_registry_dict() for e in reg.list_all()},
    }


def upsert_evidence(path: str | Path, evidence: ExperimentEvidence) -> dict[str, Any]:
    p = Path(path)
    data = load_evidence_registry(p) if p.exists() else empty_evidence_registry()
    reg = registry_from_dict(data)
    violations = validate_evidence_for_registry(evidence)
    if violations:
        raise ValueError(f"evidence failed registry validation: {violations}")
    reg._records[evidence.experiment_id] = evidence
    out = registry_to_dict(reg)
    save_evidence_registry(p, out)
    return out
