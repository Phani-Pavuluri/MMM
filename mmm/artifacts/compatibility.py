"""Artifact backend classification and compatibility checks."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal

from mmm.artifacts.base import ArtifactStoreBase
from mmm.config.schema import ArtifactBackend, MMMConfig

SupportTier = Literal["supported", "experimental", "unsupported"]

LIFECYCLE_ARTIFACT_NAMES = (
    "data_fingerprint",
    "extension_report",
    "decision_bundle",
    "seed_resolution",
    "config_fingerprint",
)

MODEL_CARD_FILENAME = "model_card.md"


def classify_backend(config: MMMConfig) -> dict[str, Any]:
    """Return support tier and CI posture for the configured backend."""
    backend = config.artifacts.backend
    if backend == ArtifactBackend.LOCAL:
        return {
            "backend": backend.value,
            "support_tier": "supported",
            "ci_contract_tested": True,
            "remote_tracking": False,
            "notice": "Local file-store is the supported production artifact path.",
        }
    uri = os.environ.get("MLFLOW_TRACKING_URI", "")
    remote = bool(uri) and not uri.startswith("file:")
    return {
        "backend": backend.value,
        "support_tier": "experimental",
        "ci_contract_tested": not remote,
        "remote_tracking": remote,
        "notice": (
            "MLflow remote tracking is experimental and not integration-guaranteed. "
            "File-based MLflow tracking is contract-tested in CI; use local for production."
            if remote
            else "MLflow file-store tracking is experimental but contract-tested in CI."
        ),
    }


def validate_backend_config(config: MMMConfig) -> None:
    """Reject unsupported backend values (only ``local`` and ``mlflow`` exist)."""
    backend = config.artifacts.backend
    if backend not in (ArtifactBackend.LOCAL, ArtifactBackend.MLFLOW):
        raise ValueError(f"Unsupported artifacts.backend: {backend!r}")


def check_store_capabilities(store: ArtifactStoreBase) -> dict[str, bool]:
    """Runtime capability matrix for a store instance."""
    return {
        "start_run": callable(getattr(store, "start_run", None)),
        "log_dict": callable(getattr(store, "log_dict", None)),
        "log_metrics": callable(getattr(store, "log_metrics", None)),
        "log_artifact": callable(getattr(store, "log_artifact", None)),
        "end_run": callable(getattr(store, "end_run", None)),
        "run_path": isinstance(getattr(type(store), "run_path", None), property),
    }


def verify_run_roundtrip(run_path: Path) -> dict[str, Any]:
    """
    Verify required lifecycle artifacts exist and JSON payloads load.

    Returns summary with ``ok`` and ``missing`` / ``errors`` lists.
    """
    missing: list[str] = []
    errors: list[str] = []
    for name in LIFECYCLE_ARTIFACT_NAMES:
        p = run_path / f"{name}.json"
        if not p.is_file():
            missing.append(name)
            continue
        try:
            json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            errors.append(f"{name}: {e}")
    model_card = run_path / MODEL_CARD_FILENAME
    if not model_card.is_file():
        missing.append(MODEL_CARD_FILENAME)
    status = run_path / "status.txt"
    if not status.is_file():
        missing.append("status.txt")
    ok = not missing and not errors
    return {
        "ok": ok,
        "run_path": str(run_path),
        "missing": missing,
        "errors": errors,
        "artifacts_present": [n for n in LIFECYCLE_ARTIFACT_NAMES if (run_path / f"{n}.json").is_file()],
    }
