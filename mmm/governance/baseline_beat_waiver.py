"""Visibility helpers when baseline-beat approval is waived via governance config."""

from __future__ import annotations

from typing import Any

BASELINE_BEAT_WAIVER_MESSAGE = (
    "BASELINE_BEAT_WAIVER: require_beats_baselines_for_approval=false — optimization/reporting "
    "approval does not require beating simple baselines. Exceptional / fixture use only; not default prod posture."
)


def baseline_beat_waiver_active(governance_config: Any) -> bool:
    return not bool(getattr(governance_config, "require_beats_baselines_for_approval", True))


def baseline_beat_waiver_warnings() -> list[str]:
    return [BASELINE_BEAT_WAIVER_MESSAGE]


def governance_summary_with_baseline_waiver(
    existing: dict[str, Any] | None,
    *,
    waiver_active: bool,
) -> dict[str, Any]:
    out = dict(existing or {})
    release = list(out.get("release_review_warnings") or [])
    opt = list(out.get("optimization_use_warnings") or [])
    if waiver_active:
        for msg in baseline_beat_waiver_warnings():
            if msg not in release:
                release.append(msg)
            if msg not in opt:
                opt.append(msg)
    out["release_review_warnings"] = release
    out["optimization_use_warnings"] = opt
    out["baseline_beat_waiver_active"] = bool(waiver_active)
    return out
