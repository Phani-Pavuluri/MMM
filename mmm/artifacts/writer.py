"""Helpers for writing versioned decision artifacts (thin layer over JSON)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_decision_artifact_json(payload: dict[str, Any], path: Path) -> None:
    """Write decision JSON with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
