"""Centralized decision-tier lineage (fail-closed in prod for decision artifacts)."""

from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path
from typing import Any

import mmm.version as mmm_version
from mmm.config.schema import MMMConfig, RunEnvironment
from mmm.governance.policy import PolicyError

# Canonical names written on decision-tier bundles (aliases kept for backward readers).
DECISION_TIER_LINEAGE_CANONICAL_KEYS: tuple[str, ...] = (
    "git_sha",
    "package_version",
    "dependency_lock_digest",
    "config_hash",
    "panel_fingerprint",
    "dataset_snapshot_id",
    "model_release_id",
    "runtime_policy_hash",
    "artifact_schema_version",
)

# Non-prod escape hatch: tests / sandboxes without a real lockfile (must never be default).
_RELAX_LINEAGE_ENV = "MMM_DECISION_LINEAGE_RELAXED_TEST_ONLY"


def _git_sha_best_effort() -> str | None:
    if os.environ.get("MMM_GIT_SHA"):
        return os.environ["MMM_GIT_SHA"].strip()
    if os.environ.get("GITHUB_SHA"):
        return os.environ["GITHUB_SHA"].strip()
    try:
        root = Path(__file__).resolve().parents[2]
        cp = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if cp.returncode == 0 and (sha := cp.stdout.strip()):
            return sha
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def _dependency_runtime_digest() -> str:
    """Runtime fingerprint (Python + installed mmm version) — not a full lockfile."""
    import sys

    try:
        import importlib.metadata as imd

        v = imd.version("mmm")
    except Exception:
        v = "unknown"
    blob = f"{sys.version_info.major}.{sys.version_info.minor}|{v}".encode()
    return hashlib.sha256(blob).hexdigest()


def resolve_dependency_lock_digest(*, strict: bool) -> str:
    """
    Lockfile-backed digest when available; else env ``MMM_DEPENDENCY_LOCK_DIGEST``.

    ``dataset_snapshot_id`` contract: operational version id, not a substitute for lock provenance.
    """
    env = os.environ.get("MMM_DEPENDENCY_LOCK_DIGEST")
    if env and str(env).strip():
        return str(env).strip()
    root = Path(__file__).resolve().parents[2]
    for name in ("uv.lock", "poetry.lock", "requirements.lock.txt"):
        p = root / name
        if p.is_file():
            return hashlib.sha256(p.read_bytes()).hexdigest()
    if os.environ.get(_RELAX_LINEAGE_ENV, "").strip() == "1" and not strict:
        return "relaxed_nonprod_dependency_lock_digest"
    if strict:
        raise PolicyError(
            "decision-tier lineage requires dependency_lock_digest: provide MMM_DEPENDENCY_LOCK_DIGEST "
            "or place uv.lock/poetry.lock at the package root."
        )
    return _dependency_runtime_digest()


def resolve_dataset_snapshot_id(config: MMMConfig) -> str:
    """
    Operational dataset snapshot id: config ``data.data_version_id`` or ``MMM_DATA_VERSION_ID``.

    This is a durable registry / lakehouse snapshot handle — not a panel content hash.
    Panel identity is carried separately in ``panel_fingerprint``.
    """
    import os as _os

    raw = (config.data.data_version_id or _os.environ.get("MMM_DATA_VERSION_ID") or "").strip()
    if not raw:
        raise PolicyError(
            "dataset_snapshot_id requires data.data_version_id or environment MMM_DATA_VERSION_ID "
            "(operational dataset version, not a substitute for panel_fingerprint)."
        )
    return raw


def collect_lineage_fields(
    *,
    data_version_id: str | None = None,
    runtime_policy_hash: str | None = None,
) -> dict[str, Any]:
    """Backward-compatible minimal lineage (extensions / non-decision paths)."""
    import sys
    from datetime import datetime, timezone

    dvid = data_version_id or os.environ.get("MMM_DATA_VERSION_ID")
    return {
        "git_sha": _git_sha_best_effort(),
        "dependency_digest": _dependency_runtime_digest(),
        "data_version_id": dvid,
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "runtime_policy_hash": runtime_policy_hash,
    }


def build_decision_tier_lineage_payload(
    *,
    config: MMMConfig,
    config_fingerprint_sha256: str,
    panel_fingerprint: dict[str, Any],
    runtime_policy_hash: str | None,
    model_release_id: str | None,
    artifact_schema_version: str,
    package_version: str | None,
    strict: bool,
) -> dict[str, Any]:
    """
    Single writer for canonical decision-tier lineage fields.

    When ``strict`` is True (prod + decision-tier bundle), missing mandatory inputs raise ``PolicyError``.
    """
    if strict and not panel_fingerprint:
        raise PolicyError("decision-tier lineage requires non-empty panel_fingerprint dict.")
    mid = str(model_release_id or "").strip()
    if strict and not mid:
        raise PolicyError(
            "decision-tier lineage requires model_release_id (tie decision artifacts to an explicit release record)."
        )
    git_sha = _git_sha_best_effort()
    if strict and not git_sha:
        raise PolicyError(
            "decision-tier lineage requires git_sha (set MMM_GIT_SHA or run from a git checkout). "
            f"Non-prod test escape: set {_RELAX_LINEAGE_ENV}=1 only in controlled tests."
        )
    if strict and not runtime_policy_hash:
        raise PolicyError("decision-tier lineage requires runtime_policy_hash (resolved runtime policy fingerprint).")
    pkg = (package_version or str(mmm_version.__version__)).strip()
    lock_digest = resolve_dependency_lock_digest(strict=strict)
    d_snap = resolve_dataset_snapshot_id(config) if strict else (
        (config.data.data_version_id or os.environ.get("MMM_DATA_VERSION_ID") or "") or "unspecified_dataset_snapshot"
    )
    if strict and d_snap == "unspecified_dataset_snapshot":
        raise PolicyError(
            "decision-tier lineage requires dataset_snapshot_id via data.data_version_id or MMM_DATA_VERSION_ID."
        )

    base = collect_lineage_fields(data_version_id=config.data.data_version_id, runtime_policy_hash=runtime_policy_hash)
    cfg_hash = str(config_fingerprint_sha256)
    out = {
        **base,
        "package_version": pkg,
        "dependency_lock_digest": lock_digest,
        "config_hash": cfg_hash,
        "panel_fingerprint": panel_fingerprint if isinstance(panel_fingerprint, dict) else {},
        "dataset_snapshot_id": d_snap,
        "model_release_id": mid if strict else (mid or "unversioned_extension_bundle"),
        "artifact_schema_version": str(artifact_schema_version),
        # Aliases / supplemental (existing consumers)
        "config_sha": cfg_hash,
        "config_fingerprint_sha256": cfg_hash,
        "data_version_id": base.get("data_version_id"),
    }
    return out


def assert_decision_tier_lineage_complete(
    bundle: dict[str, Any],
    *,
    _run_environment: RunEnvironment | None = None,
) -> None:
    """Fail closed: decision-tier bundles must carry every canonical lineage field (read or write)."""
    tier = bundle.get("artifact_tier") or bundle.get("tier")
    if str(tier) != "decision":
        return
    if os.environ.get(_RELAX_LINEAGE_ENV, "").strip() == "1":
        return
    missing: list[str] = []
    for k in DECISION_TIER_LINEAGE_CANONICAL_KEYS:
        v = bundle.get(k)
        if v is None or v == "":
            missing.append(k)
            continue
        if k == "panel_fingerprint" and (not isinstance(v, dict) or not v):
            missing.append("panel_fingerprint(empty)")
    if missing:
        raise PolicyError(
            "decision-tier artifact rejected: incomplete lineage for prod decision bundle. "
            f"Missing or empty: {', '.join(missing)}. "
            "Populate via build_decision_tier_lineage_payload(strict=True) and bundle assembly."
        )
