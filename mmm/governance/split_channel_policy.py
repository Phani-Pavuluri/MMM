"""Block split-level ROI / optimization claims when separability is insufficient."""

from __future__ import annotations

from typing import Any


def apply_split_channel_governance(extension_report: dict[str, Any]) -> None:
    """
    Mutates ``extension_report["governance"]`` when feature separability blocks split-level claims.

    Low separability + material business importance ⇒ no optimization approval for split attribution.
    """
    sep = extension_report.get("feature_separability_report")
    gov = extension_report.get("governance")
    if not isinstance(sep, dict) or sep.get("skipped"):
        return
    if not isinstance(gov, dict):
        return

    notes = list(gov.get("notes") or [])
    blocked = False
    blocked_groups: list[str] = []

    for claim in sep.get("unsupported_split_level_claims") or []:
        if not isinstance(claim, dict):
            continue
        fg = str(claim.get("feature_group", ""))
        if fg:
            blocked_groups.append(fg)
        blocked = True

    for g in sep.get("feature_groups") or []:
        if not isinstance(g, dict):
            continue
        if str(g.get("separability_classification")) != "low":
            continue
        biz = g.get("business_importance") if isinstance(g.get("business_importance"), dict) else {}
        high = str(biz.get("importance_band", "")) == "high" or bool(biz.get("material_spend_share"))
        if high:
            fg = str(g.get("feature_group", ""))
            if fg and fg not in blocked_groups:
                blocked_groups.append(fg)
            blocked = True

    if not blocked:
        return

    gov["split_level_claims_supported"] = False
    gov["split_level_optimization_blocked"] = True
    gov["approved_for_optimization"] = False
    note = (
        "split_level_attribution_blocked: low feature separability with material spend/contribution — "
        "do not use split-channel ROI or budget interpretation; roll up or run experiments. "
        f"groups={sorted(set(blocked_groups))}"
    )
    if note not in notes:
        notes.append(note)
    gov["notes"] = notes
    extension_report["governance"] = gov
