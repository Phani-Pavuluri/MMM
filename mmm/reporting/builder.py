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
        envelope = {
            "report_contract_version": "mmm_json_report_v2",
            "artifact_tier": "diagnostic",
            "not_for_budgeting": True,
            "surface": "report_builder_cli",
            "sections": self.sections,
        }
        path.write_text(json.dumps(envelope, indent=2, default=str), encoding="utf-8")
