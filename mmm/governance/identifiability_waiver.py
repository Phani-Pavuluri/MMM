"""Narrow, machine-validated waiver flow for severe identifiability limits (prod decision paths)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from mmm.governance.policy import PolicyError


class IdentifiabilityWaiverArtifact(BaseModel):
    """Formal waiver artifact — not a generic policy bypass."""

    waiver_id: str = Field(min_length=4)
    created_at: str = Field(description="ISO-8601 UTC timestamp")
    reason: str = Field(min_length=8)
    owner: str | None = Field(default=None, description="Human/service owner accepting accountability")
    model_release_id: str | None = None
    config_fingerprint_sha256: str | None = Field(
        default=None,
        description="When set, waiver applies only if bundle/config fingerprint matches.",
    )
    max_identifiability_score_waived: float = Field(ge=0.0, le=1.0)
    expires_at: str | None = Field(default=None, description="ISO-8601 UTC; waiver invalid after this instant")
    waived_severity: str = Field(default="identifiability_exceeds_governance_cap")
    # Optional operational traceability (recommended for audits; not all environments populate every field).
    affected_artifact_ids: list[str] = Field(default_factory=list)
    affected_artifact_tier: str | None = None
    affected_run_ids: list[str] = Field(default_factory=list)
    affected_model_release_id: str | None = Field(
        default=None,
        description="Optional explicit release id; when set with model_release_id both must match each other.",
    )
    affected_dataset_snapshot_id: str | None = None
    affected_decision_bundle_id: str | None = None
    affected_surfaces: list[str] = Field(default_factory=list)
    waived_identifiability_conditions: list[str] = Field(
        default_factory=list,
        description="Machine-oriented labels for what severity/condition is waived (e.g. governance_cap_exceeded).",
    )

    @field_validator("created_at")
    @classmethod
    def _parse_created(cls, v: str) -> str:
        datetime.fromisoformat(v.replace("Z", "+00:00"))
        return v

    @model_validator(mode="after")
    def _expires_after_created(self) -> IdentifiabilityWaiverArtifact:
        if self.expires_at:
            exp = datetime.fromisoformat(self.expires_at.replace("Z", "+00:00"))
            cre = datetime.fromisoformat(self.created_at.replace("Z", "+00:00"))
            if exp <= cre:
                raise ValueError("expires_at must be strictly after created_at")
        return self


def parse_waiver_from_extension(er: dict[str, Any] | None) -> IdentifiabilityWaiverArtifact | None:
    raw = er.get("identifiability_waiver") if isinstance(er, dict) else None
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise PolicyError("identifiability_waiver must be a dict when present")
    return IdentifiabilityWaiverArtifact.model_validate(raw)


def _parse_utc(s: str) -> datetime:
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def validate_waiver_for_run(
    waiver: IdentifiabilityWaiverArtifact,
    *,
    identifiability_score: float,
    model_release_id: str | None,
    config_fingerprint_sha256: str | None,
    dataset_snapshot_id: str | None = None,
) -> None:
    """Reject scope-mismatched, expired, or insufficient waivers."""
    now = datetime.now(timezone.utc)
    if waiver.expires_at:
        exp = _parse_utc(waiver.expires_at)
        if now > exp:
            raise PolicyError(f"identifiability waiver {waiver.waiver_id!r} expired at {waiver.expires_at!r}")
    if float(identifiability_score) > float(waiver.max_identifiability_score_waived) + 1e-9:
        raise PolicyError(
            f"identifiability_score={identifiability_score:.4f} exceeds waiver max "
            f"{waiver.max_identifiability_score_waived:.4f} (waiver {waiver.waiver_id!r})"
        )
    if (
        waiver.config_fingerprint_sha256
        and config_fingerprint_sha256
        and str(waiver.config_fingerprint_sha256) != str(config_fingerprint_sha256)
    ):
        raise PolicyError("identifiability waiver config_fingerprint_sha256 does not match current run")
    if (
        waiver.model_release_id
        and model_release_id
        and str(waiver.model_release_id) != str(model_release_id)
    ):
        raise PolicyError("identifiability waiver model_release_id does not match current extension release id")
    if (
        waiver.affected_model_release_id
        and waiver.model_release_id
        and str(waiver.affected_model_release_id) != str(waiver.model_release_id)
    ):
        raise PolicyError(
            "identifiability waiver affected_model_release_id must match model_release_id when both set"
        )
    if (
        waiver.affected_model_release_id
        and model_release_id
        and str(waiver.affected_model_release_id) != str(model_release_id)
    ):
        raise PolicyError("identifiability waiver affected_model_release_id does not match current run release id")
    if (
        waiver.affected_dataset_snapshot_id
        and dataset_snapshot_id
        and str(waiver.affected_dataset_snapshot_id) != str(dataset_snapshot_id)
    ):
        raise PolicyError(
            "identifiability waiver affected_dataset_snapshot_id does not match operational dataset_snapshot_id"
        )


def waiver_allows_identifiability_block(
    *,
    waiver: IdentifiabilityWaiverArtifact | None,
    score: float,
    threshold: float,
    allow_waiver_policy: bool,
    model_release_id: str | None,
    config_fingerprint_sha256: str | None,
    dataset_snapshot_id: str | None = None,
) -> tuple[bool, IdentifiabilityWaiverArtifact | None]:
    """
    Returns (allowed, waiver_used).

    When score <= threshold, always allowed with no waiver.
    When score > threshold, requires allow_waiver_policy and valid waiver covering score.
    """
    if score <= threshold + 1e-12:
        return True, None
    if not allow_waiver_policy or waiver is None:
        return False, None
    validate_waiver_for_run(
        waiver,
        identifiability_score=score,
        model_release_id=model_release_id,
        config_fingerprint_sha256=config_fingerprint_sha256,
        dataset_snapshot_id=dataset_snapshot_id,
    )
    return True, waiver
