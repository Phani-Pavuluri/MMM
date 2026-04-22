"""Programmatic readiness checks for replay / decision surfaces."""

from __future__ import annotations

from typing import Any

from mmm.experiments.registry import ApprovalState, ExperimentRecord


def experiment_readiness(record: ExperimentRecord) -> dict[str, Any]:
    """
    Return ``{"ready": bool, "reasons": [...]}`` for gating services and CLIs.

    Stricter deployments can treat missing ``payload_signature`` or calibration ref as blocking.
    """
    reasons: list[str] = []
    if record.approval != ApprovalState.APPROVED:
        reasons.append("approval_not_approved")
    if not record.payload_signature:
        reasons.append("missing_payload_signature")
    if not record.calibration_artifact_ref:
        reasons.append("missing_calibration_artifact_ref")
    return {"ready": len(reasons) == 0, "reasons": reasons, "approval": record.approval.value}
