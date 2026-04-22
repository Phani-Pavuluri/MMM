"""Filesystem-backed artifact store (always available)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from mmm.artifacts.base import ArtifactStoreBase


class LocalArtifactStore(ArtifactStoreBase):
    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)
        self._run_id: str | None = None
        self._run_path: Path | None = None

    @property
    def run_path(self) -> Path:
        if self._run_path is None:
            raise RuntimeError("start_run not called")
        return self._run_path

    def start_run(self, run_id: str, metadata: dict[str, Any] | None = None) -> None:
        self._run_id = run_id
        self._run_path = self._root / run_id
        self._run_path.mkdir(parents=True, exist_ok=True)
        meta = {"run_id": run_id, **(metadata or {})}
        (self._run_path / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        path = self.run_path / "metrics.jsonl"
        row = {"step": step, **metrics}
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    def log_artifact(self, name: str, path: str | Path) -> None:
        src = Path(path)
        dest_dir = self.run_path / "artifacts" / name
        dest_dir.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            if dest_dir.exists():
                shutil.rmtree(dest_dir)
            shutil.copytree(src, dest_dir)
        else:
            dest_dir.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest_dir)

    def log_dict(self, name: str, payload: dict[str, Any]) -> None:
        out = self.run_path / f"{name}.json"
        out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    def end_run(self, status: str = "FINISHED") -> None:
        (self.run_path / "status.txt").write_text(status, encoding="utf-8")
