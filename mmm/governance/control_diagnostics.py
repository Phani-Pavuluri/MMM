"""Control-variable governance diagnostics (guidance only — never mutates training)."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

from mmm.config.schema import MMMConfig
from mmm.data.schema import PanelSchema
from mmm.governance.control_policy_packs import all_pack_summaries, list_policy_pack_ids
from mmm.helpers.control_templates import ControlDomain, list_template_controls

_POST_TREATMENT_PATTERNS = (
    r"conversion",
    r"revenue_attributed",
    r"orders?_post",
    r"outcome",
    r"sales_after",
)

_PROXY_PATTERNS = (
    r"^spend_",
    r"_spend$",
    r"media_",
    r"impressions?",
)


def _post_treatment_flags(columns: list[str]) -> list[dict[str, str]]:
    flagged: list[dict[str, str]] = []
    for col in columns:
        low = col.lower()
        for pat in _POST_TREATMENT_PATTERNS:
            if re.search(pat, low):
                flagged.append({"column": col, "reason": f"matches_post_treatment_pattern:{pat}"})
                break
    return flagged


def _proxy_flags(columns: list[str], channel_columns: list[str]) -> list[dict[str, str]]:
    flagged: list[dict[str, str]] = []
    ch_lower = {c.lower() for c in channel_columns}
    for col in columns:
        low = col.lower()
        if low in ch_lower or any(low.startswith(ch) or low.endswith(ch) for ch in ch_lower):
            flagged.append({"column": col, "reason": "name_overlaps_channel"})
            continue
        for pat in _PROXY_PATTERNS:
            if re.search(pat, low):
                flagged.append({"column": col, "reason": f"matches_proxy_pattern:{pat}"})
                break
    return flagged


def _duplicate_controls(columns: list[str]) -> list[str]:
    seen: dict[str, str] = {}
    dups: list[str] = []
    for col in columns:
        key = col.lower().strip()
        if key in seen and col not in dups:
            dups.append(col)
        seen[key] = col
    return dups


def build_control_governance_report(
    *,
    config: MMMConfig,
    schema: PanelSchema,
    panel: pd.DataFrame | None = None,
    domain: ControlDomain = ControlDomain.GENERIC,
) -> dict[str, Any]:
    """
    Emit control guidance for operators.

    Does **not** insert, drop, or reweight controls in the design matrix.
    """
    configured = list(schema.control_columns)
    template_cols = list_template_controls(domain)
    missing = [c for c in template_cols if c not in configured]
    recommended = [c for c in template_cols if c not in configured][:8]

    policy_pack = {
        "domain": domain.value,
        "template_version": "control_template_v1",
        "description": "Illustrative onboarding pack — not applied to training automatically.",
    }

    report: dict[str, Any] = {
        "policy_note": "Guidance only; controls are never auto-inserted into training.",
        "configured_controls": configured,
        "recommended_controls": recommended,
        "missing_from_template_pack": missing,
        "potentially_post_treatment": _post_treatment_flags(configured),
        "duplicate_controls": _duplicate_controls(configured),
        "proxy_controls": _proxy_flags(configured, list(schema.channel_columns)),
        "policy_pack": policy_pack,
        "policy_packs": all_pack_summaries(configured_controls=configured),
        "available_policy_pack_ids": list_policy_pack_ids(),
    }

    if panel is not None and configured:
        present = [c for c in configured if c in panel.columns]
        report["present_in_panel"] = present
        report["configured_but_absent"] = [c for c in configured if c not in panel.columns]

    return report
