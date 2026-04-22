"""Pluggable artifact / tracking interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class ArtifactStoreBase(ABC):
    """Every run writes resolved config, metrics, diagnostics, model blobs."""

    @abstractmethod
    def start_run(self, run_id: str, metadata: dict[str, Any] | None = None) -> None:
        raise NotImplementedError

    @abstractmethod
    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        raise NotImplementedError

    @abstractmethod
    def log_artifact(self, name: str, path: str | Path) -> None:
        raise NotImplementedError

    @abstractmethod
    def log_dict(self, name: str, payload: dict[str, Any]) -> None:
        raise NotImplementedError

    @abstractmethod
    def end_run(self, status: str = "FINISHED") -> None:
        raise NotImplementedError

    @property
    @abstractmethod
    def run_path(self) -> Path:
        raise NotImplementedError
