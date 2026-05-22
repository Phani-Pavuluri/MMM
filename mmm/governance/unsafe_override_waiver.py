"""Signed waiver for ``override_unsafe`` in production (narrow escape hatch)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator


class UnsafeOverrideWaiverArtifact(BaseModel):
    waiver_id: str = Field(min_length=4)
    created_at: str
    reason: str = Field(min_length=8)
    owner: str | None = None
    expires_at: str | None = None
    acknowledged_risks: list[str] = Field(default_factory=list)

    @field_validator("created_at")
    @classmethod
    def _parse_created(cls, v: str) -> str:
        datetime.fromisoformat(v.replace("Z", "+00:00"))
        return v


def load_unsafe_override_waiver(path: str | Path) -> UnsafeOverrideWaiverArtifact:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    waiver = UnsafeOverrideWaiverArtifact.model_validate(raw)
    if waiver.expires_at:
        exp = datetime.fromisoformat(waiver.expires_at.replace("Z", "+00:00"))
        if exp.tzinfo is None:
            exp = exp.replace(tzinfo=timezone.utc)
        if datetime.now(timezone.utc) >= exp:
            raise ValueError(f"unsafe_override_waiver expired at {waiver.expires_at!r}")
    return waiver


def waiver_summary_for_artifacts(waiver: UnsafeOverrideWaiverArtifact) -> dict[str, Any]:
    return {
        "waiver_id": waiver.waiver_id,
        "owner": waiver.owner,
        "reason": waiver.reason,
        "expires_at": waiver.expires_at,
    }
