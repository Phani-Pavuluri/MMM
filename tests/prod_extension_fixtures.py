"""Shared prod extension_report fragments for decide-path tests."""

from __future__ import annotations

from typing import Any


def prod_replay_evidence_block() -> dict[str, Any]:
    """Satisfies ``require_replay_calibration`` on prod decide paths."""
    return {"calibration_summary": {"replay_calibration_active": True, "replay_loss": 0.5}}


def merge_prod_extension(base: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    out.update(prod_replay_evidence_block())
    return out
