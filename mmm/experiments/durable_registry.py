"""Durable on-disk experiment registry (JSON) — lineage for calibration / replay gating."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mmm.experiments.registry import ApprovalState, ExperimentRecord

REGISTRY_VERSION = "mmm_experiment_registry_v1"


def _canonical_record_blob(rec: dict[str, Any]) -> bytes:
    return json.dumps(rec, sort_keys=True, separators=(",", ":")).encode("utf-8")


def load_experiment_registry(path: str | Path) -> dict[str, Any]:
    """Load registry JSON; returns dict with ``registry_version`` and ``experiments`` mapping id → record."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"experiment registry not found: {p}")
    raw = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("registry root must be a JSON object")
    if raw.get("registry_version") != REGISTRY_VERSION:
        raise ValueError(f"unsupported registry_version {raw.get('registry_version')!r}; expected {REGISTRY_VERSION!r}")
    ex = raw.get("experiments")
    if not isinstance(ex, dict):
        raise ValueError("registry.experiments must be an object mapping experiment_id → record")
    return raw


def save_experiment_registry(path: str | Path, registry: dict[str, Any]) -> None:
    """Atomically write registry (caller supplies full object including ``registry_version``)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    blob = json.dumps(registry, indent=2, sort_keys=True, default=str)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(blob, encoding="utf-8")
    tmp.replace(p)


def empty_experiment_registry() -> dict[str, Any]:
    return {"registry_version": REGISTRY_VERSION, "experiments": {}}


def upsert_experiment_record(path: str | Path, record: ExperimentRecord) -> dict[str, Any]:
    """Load (or create) registry, upsert one :class:`ExperimentRecord`, persist, return full registry."""
    p = Path(path)
    reg = load_experiment_registry(p) if p.exists() else empty_experiment_registry()
    ex: dict[str, Any] = reg.setdefault("experiments", {})
    rid = record.experiment_id
    ex[rid] = {
        "experiment_id": rid,
        "approval": record.approval.value,
        "calibration_artifact_ref": record.calibration_artifact_ref,
        "payload_signature": record.payload_signature,
        "calibration_version": record.calibration_version,
        "metadata": dict(record.metadata),
    }
    save_experiment_registry(p, reg)
    return reg


def experiment_record_from_registry_dict(d: dict[str, Any]) -> ExperimentRecord:
    """Hydrate :class:`ExperimentRecord` from a durable JSON entry."""
    return ExperimentRecord(
        experiment_id=str(d["experiment_id"]),
        approval=ApprovalState(str(d.get("approval", "draft"))),
        calibration_artifact_ref=d.get("calibration_artifact_ref"),
        payload_signature=d.get("payload_signature"),
        calibration_version=d.get("calibration_version"),
        metadata=dict(d.get("metadata") or {}),
    )


def get_experiment_from_registry(registry: dict[str, Any], experiment_id: str) -> ExperimentRecord | None:
    ex = registry.get("experiments") or {}
    raw = ex.get(experiment_id)
    if not isinstance(raw, dict):
        return None
    return experiment_record_from_registry_dict(raw)
