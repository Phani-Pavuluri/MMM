"""Accepted-run registry for historical drift and lineage (no auto-retrain)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mmm.evaluation.drift_history import load_historical_reference


@dataclass
class AcceptedRunRecord:
    run_id: str
    run_dir: str
    registered_at: str
    fingerprint_sha: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AcceptedRunRegistry:
    """
    Append-only registry of operator-accepted training runs.

    Used for cross-run drift comparison; never triggers retraining.
    """

    def __init__(self, registry_dir: str | Path) -> None:
        self._dir = Path(registry_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self._dir / "accepted_runs.jsonl"

    def register_accepted_run(
        self,
        run_dir: str | Path,
        *,
        run_id: str | None = None,
        fingerprint_sha: str | None = None,
    ) -> AcceptedRunRecord:
        run_path = Path(run_dir).resolve()
        rid = run_id or run_path.name
        ref = load_historical_reference(run_path)
        fp = (ref or {}).get("panel_fingerprint") or {}
        sha = fingerprint_sha or fp.get("sha256_combined") or fp.get("sha256_panel_keycols_sorted_csv")
        rec = AcceptedRunRecord(
            run_id=rid,
            run_dir=str(run_path),
            registered_at=datetime.now(timezone.utc).isoformat(),
            fingerprint_sha=sha,
        )
        with self._index_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec.to_dict(), default=str) + "\n")
        return rec

    def list_accepted_runs(self) -> list[AcceptedRunRecord]:
        if not self._index_path.is_file():
            return []
        records: list[AcceptedRunRecord] = []
        for line in self._index_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            records.append(AcceptedRunRecord(**row))
        return records

    def latest_accepted(
        self,
        *,
        exclude_run_id: str | None = None,
        exclude_run_dir: str | Path | None = None,
    ) -> AcceptedRunRecord | None:
        ex_dir = str(Path(exclude_run_dir).resolve()) if exclude_run_dir else None
        candidates = self.list_accepted_runs()
        for rec in reversed(candidates):
            if exclude_run_id and rec.run_id == exclude_run_id:
                continue
            if ex_dir and str(Path(rec.run_dir).resolve()) == ex_dir:
                continue
            if Path(rec.run_dir).is_dir():
                return rec
        return None

    def historical_reference_for_drift(
        self,
        *,
        exclude_run_id: str | None = None,
        exclude_run_dir: str | Path | None = None,
    ) -> dict[str, Any] | None:
        rec = self.latest_accepted(exclude_run_id=exclude_run_id, exclude_run_dir=exclude_run_dir)
        if rec is None:
            return None
        ref = load_historical_reference(rec.run_dir)
        if ref is None:
            return None
        ref["registry_run_id"] = rec.run_id
        ref["registry_registered_at"] = rec.registered_at
        return ref
