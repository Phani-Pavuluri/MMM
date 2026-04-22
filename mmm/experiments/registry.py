"""In-memory experiment registry with stable UUID ids and approval state."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class ApprovalState(StrEnum):
    DRAFT = "draft"
    APPROVED = "approved"
    REVOKED = "revoked"


def new_experiment_id() -> str:
    """Return a new immutable experiment identifier (UUID4 string)."""
    return str(uuid.uuid4())


@dataclass
class ExperimentRecord:
    """
    One registered experiment / calibration lineage unit.

    ``experiment_id`` must be treated as immutable once published to downstream systems.
    """

    experiment_id: str
    approval: ApprovalState = ApprovalState.DRAFT
    calibration_artifact_ref: str | None = None
    payload_signature: str | None = None
    calibration_version: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ExperimentRegistry:
    """Process-local registry (swap for DB / artifact store in production)."""

    def __init__(self) -> None:
        self._records: dict[str, ExperimentRecord] = {}

    def register(self, record: ExperimentRecord) -> None:
        if record.experiment_id in self._records:
            raise ValueError(f"experiment_id already registered: {record.experiment_id!r}")
        self._records[record.experiment_id] = record

    def get(self, experiment_id: str) -> ExperimentRecord | None:
        return self._records.get(experiment_id)

    def require_approved(self, experiment_id: str) -> ExperimentRecord:
        rec = self.get(experiment_id)
        if rec is None:
            raise KeyError(f"unknown experiment_id: {experiment_id!r}")
        if rec.approval != ApprovalState.APPROVED:
            raise PermissionError(f"experiment {experiment_id!r} is not approved (state={rec.approval})")
        return rec

    def set_approval(self, experiment_id: str, state: ApprovalState) -> None:
        rec = self.get(experiment_id)
        if rec is None:
            raise KeyError(experiment_id)
        rec.approval = state
