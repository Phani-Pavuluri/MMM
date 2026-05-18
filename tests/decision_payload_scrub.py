"""Shared scrubbing for decide CLI/API/shim JSON parity tests."""

from __future__ import annotations

import json
from typing import Any


def _scrub_lineage_metadata(sl: dict[str, Any] | None) -> None:
    if not isinstance(sl, dict):
        return
    meta = sl.get("metadata")
    if isinstance(meta, dict):
        meta.pop("created_at", None)
        meta.pop("source_path", None)


def strip_volatile_decision_payload(d: dict[str, Any]) -> dict[str, Any]:
    """Remove wall-clock / path fields; keep planning contract fields intact."""
    out: dict[str, Any] = json.loads(json.dumps(d, default=str))
    db = out.get("decision_bundle")
    if isinstance(db, dict):
        db.pop("created_at", None)
        db.pop("python_version", None)
        _scrub_lineage_metadata(db.get("scenario_lineage") if isinstance(db.get("scenario_lineage"), dict) else None)
    _scrub_lineage_metadata(out.get("scenario_lineage") if isinstance(out.get("scenario_lineage"), dict) else None)
    sim = out.get("simulation")
    if isinstance(sim, dict):
        _scrub_lineage_metadata(sim.get("scenario_lineage") if isinstance(sim.get("scenario_lineage"), dict) else None)
    dr = out.get("decision_result")
    if isinstance(dr, dict):
        lr = dr.get("lineage_refs")
        if isinstance(lr, dict):
            _scrub_lineage_metadata(
                lr.get("scenario_lineage") if isinstance(lr.get("scenario_lineage"), dict) else None
            )
    return out
