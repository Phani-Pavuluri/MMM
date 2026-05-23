"""Append-only promotion registry (immutable records)."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from mmm.governance.promotion import PromotionRecord, build_promotion_record, validate_promotion_eligibility


class PromotionRegistryError(ValueError):
    """Registry read/write failures."""


def _registry_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def load_promotion_registry(path: str | Path) -> list[PromotionRecord]:
    p = _registry_path(path)
    if not p.is_file():
        return []
    out: list[PromotionRecord] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(PromotionRecord.from_dict(json.loads(line)))
    return out


def append_promotion_record(path: str | Path, record: PromotionRecord) -> None:
    """Append one immutable JSON line; never overwrite existing lines."""
    p = _registry_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    existing = load_promotion_registry(p)
    if any(r.promotion_id == record.promotion_id for r in existing):
        raise PromotionRegistryError(f"promotion_id already exists: {record.promotion_id!r}")
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record.to_dict(), sort_keys=True, default=str) + "\n")


def get_promotion_by_id(path: str | Path, promotion_id: str) -> PromotionRecord | None:
    for rec in load_promotion_registry(path):
        if rec.promotion_id == promotion_id:
            return rec
    return None


def promote_run(
    *,
    registry_path: str | Path,
    config: Any,
    extension_report: dict[str, Any],
    artifact_uri: str,
    data_fingerprint: dict[str, Any],
    config_fingerprint: str,
    model_fingerprint: str,
    seed_resolution: dict[str, Any],
    promoted_by: str,
    run_id: str | None = None,
    model_id: str | None = None,
    approval_notes: str = "",
    allowed_surfaces: list[str] | None = None,
    expiration_date: str | None = None,
    unsupported_questions: list[str] | None = None,
) -> PromotionRecord:
    """Validate eligibility and append a new promotion record."""
    validate_promotion_eligibility(
        config=config,
        extension_report=extension_report,
        data_fingerprint=data_fingerprint,
        config_fingerprint=config_fingerprint,
    )
    promotion_id = str(uuid.uuid4())
    record = build_promotion_record(
        promotion_id=promotion_id,
        run_id=run_id or str(config.run_id or promotion_id),
        model_id=model_id or promotion_id,
        artifact_uri=artifact_uri,
        data_fingerprint=data_fingerprint,
        config_fingerprint=config_fingerprint,
        model_fingerprint=model_fingerprint,
        seed_resolution=seed_resolution,
        promoted_by=promoted_by,
        governance_summary=dict(extension_report.get("governance") or {}),
        calibration_summary=dict(extension_report.get("calibration_summary") or {}),
        unsupported_questions=list(unsupported_questions or []),
        allowed_surfaces=allowed_surfaces,
        approval_notes=approval_notes,
        expiration_date=expiration_date,
    )
    append_promotion_record(registry_path, record)
    return record


def rollback_promotion(
    *,
    registry_path: str | Path,
    prior_promotion_id: str,
    config: Any,
    extension_report: dict[str, Any],
    artifact_uri: str,
    data_fingerprint: dict[str, Any],
    config_fingerprint: str,
    model_fingerprint: str,
    seed_resolution: dict[str, Any],
    promoted_by: str,
    approval_notes: str = "",
) -> PromotionRecord:
    """Create a new promotion record pointing at a prior valid promotion (rollback lineage)."""
    prior = get_promotion_by_id(registry_path, prior_promotion_id)
    if prior is None:
        raise PromotionRegistryError(f"prior promotion not found: {prior_promotion_id!r}")
    validate_promotion_eligibility(
        config=config,
        extension_report=extension_report,
        data_fingerprint=data_fingerprint,
        config_fingerprint=config_fingerprint,
    )
    promotion_id = str(uuid.uuid4())
    record = build_promotion_record(
        promotion_id=promotion_id,
        run_id=str(config.run_id or promotion_id),
        model_id=prior.model_id,
        artifact_uri=artifact_uri,
        data_fingerprint=data_fingerprint,
        config_fingerprint=config_fingerprint,
        model_fingerprint=model_fingerprint,
        seed_resolution=seed_resolution,
        promoted_by=promoted_by,
        governance_summary=dict(extension_report.get("governance") or {}),
        calibration_summary=dict(extension_report.get("calibration_summary") or {}),
        unsupported_questions=[],
        allowed_surfaces=list(prior.allowed_surfaces),
        approval_notes=approval_notes or f"rollback to {prior_promotion_id}",
        rollback_of=prior_promotion_id,
        parent_promotion_id=prior.parent_promotion_id or prior_promotion_id,
    )
    append_promotion_record(registry_path, record)
    return record
