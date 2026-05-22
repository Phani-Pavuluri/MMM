"""Load persisted run artifacts from a local run directory."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mmm.artifacts.compatibility import LIFECYCLE_ARTIFACT_NAMES, MODEL_CARD_FILENAME, verify_run_roundtrip


def load_run_json(run_path: Path, name: str) -> dict[str, Any]:
    path = run_path / f"{name}.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing artifact {name}.json under {run_path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{name}.json root must be an object")
    return data


def load_model_card(run_path: Path) -> str:
    path = run_path / MODEL_CARD_FILENAME
    if not path.is_file():
        raise FileNotFoundError(f"Missing {MODEL_CARD_FILENAME} under {run_path}")
    return path.read_text(encoding="utf-8")


def load_run_lineage(run_path: Path) -> dict[str, Any]:
    """Load fingerprint + seed + config lineage blobs when present."""
    out: dict[str, Any] = {"run_path": str(run_path)}
    for name in LIFECYCLE_ARTIFACT_NAMES:
        p = run_path / f"{name}.json"
        if p.is_file():
            out[name] = json.loads(p.read_text(encoding="utf-8"))
    if (run_path / MODEL_CARD_FILENAME).is_file():
        out["model_card_md"] = load_model_card(run_path)
    out["roundtrip"] = verify_run_roundtrip(run_path)
    return out
