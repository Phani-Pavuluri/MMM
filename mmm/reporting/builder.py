"""Machine-readable reports + optional markdown."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class ReporterBase(ABC):
    @abstractmethod
    def write(self, path: Path) -> None:
        raise NotImplementedError


@dataclass
class ReportBuilder(ReporterBase):
    sections: dict[str, Any] = field(default_factory=dict)

    def add(self, name: str, payload: Any) -> None:
        self.sections[name] = payload

    def write(self, path: Path) -> None:
        import json

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.sections, indent=2, default=str), encoding="utf-8")
