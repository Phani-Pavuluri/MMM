"""Research-only labeling for Bayes-H3 sandbox artifacts."""

from __future__ import annotations

from typing import Any

RESEARCH_ONLY_LABEL = "RESEARCH ONLY — NOT DECISION GRADE"

REQUIRED_RESEARCH_KEYS: frozenset[str] = frozenset(
    {
        "label",
        "research_only",
        "decision_grade",
        "approved_for_prod",
        "prod_decisioning_allowed",
    }
)

FORBIDDEN_PROD_VALUES: dict[str, object] = {
    "research_only": False,
    "decision_grade": True,
    "approved_for_prod": True,
    "prod_decisioning_allowed": True,
}


class ResearchOnlyLabelError(ValueError):
    """Raised when sandbox artifact labels are missing or contradict research-only posture."""


def research_only_fields() -> dict[str, Any]:
    return {
        "label": RESEARCH_ONLY_LABEL,
        "research_only": True,
        "decision_grade": False,
        "approved_for_prod": False,
        "prod_decisioning_allowed": False,
        "bayes_h3_sandbox": True,
    }


def apply_research_only_envelope(payload: dict[str, Any]) -> dict[str, Any]:
    """Merge required research-only fields into a fit or sandbox artifact dict."""
    out = dict(payload)
    out.update(research_only_fields())
    return out


def validate_research_only_artifact(artifact: dict[str, Any]) -> None:
    """Fail closed if required labels are missing or imply production decisioning."""
    missing = REQUIRED_RESEARCH_KEYS - set(artifact.keys())
    if missing:
        raise ResearchOnlyLabelError(f"missing required research-only keys: {sorted(missing)}")

    if artifact.get("label") != RESEARCH_ONLY_LABEL:
        raise ResearchOnlyLabelError(
            f"label must be {RESEARCH_ONLY_LABEL!r}, got {artifact.get('label')!r}"
        )

    for key, forbidden in FORBIDDEN_PROD_VALUES.items():
        if artifact.get(key) is forbidden:
            raise ResearchOnlyLabelError(f"{key} must not be {forbidden!r} for sandbox artifacts")

    if artifact.get("production_decision_surface") is True:
        raise ResearchOnlyLabelError("production_decision_surface must not be true")

    if artifact.get("production_recommendation") is True:
        raise ResearchOnlyLabelError("production_recommendation must not be true")


def assert_diagnostic_only_outputs(artifact: dict[str, Any]) -> None:
    """Posterior/coef/decomposition paths must remain diagnostic, not decision-grade."""
    validate_research_only_artifact(artifact)
    if artifact.get("outputs_are_diagnostic_only") is False:
        raise ResearchOnlyLabelError("outputs_are_diagnostic_only must not be false")
