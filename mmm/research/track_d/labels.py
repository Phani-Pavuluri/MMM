"""Research-only labels for Track D validation artifacts."""

from __future__ import annotations

from typing import Any

RESEARCH_ONLY_LABEL = "RESEARCH ONLY — NOT DECISION GRADE"


def research_only_governance() -> dict[str, Any]:
    return {
        "label": RESEARCH_ONLY_LABEL,
        "research_only": True,
        "decision_grade": False,
        "approved_for_prod": False,
        "prod_decisioning_allowed": False,
        "production_promotion": False,
        "hard_gate": False,
        "track_d_research_lane": True,
    }
