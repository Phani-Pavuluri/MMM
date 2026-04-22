"""Model release / invalidation state machine for operator and CLI surfaces."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from mmm.config.schema import MMMConfig, RunEnvironment


class ModelReleaseState(StrEnum):
    """Coarse lifecycle for what decision surfaces a trained artifact may drive."""

    RESEARCH_ONLY = "research_only"
    REPORTING_ALLOWED = "reporting_allowed"
    PLANNING_ALLOWED = "planning_allowed"
    INVALIDATED = "invalidated"


def infer_model_release_state(
    *,
    config: MMMConfig,
    panel_qa_max_severity: str,
    governance_approved_for_optimization: bool,
    governance_approved_for_reporting: bool,
    ridge_fit_summary_present: bool,
    invalidation_reasons: list[str] | None = None,
) -> dict[str, Any]:
    """
    Infer a single release state from panel QA, governance flags, and optional explicit invalidations.

    Advisory unless enforced by callers; persisted on extension reports for traceability.
    """
    inv = list(invalidation_reasons or [])
    triggers: dict[str, Any] = {"panel_qa_max_severity": panel_qa_max_severity}

    if panel_qa_max_severity == "block":
        inv.append("panel_qa_block_severity")
    if inv:
        return {"state": ModelReleaseState.INVALIDATED.value, "reasons": inv, "triggers": triggers}

    if not ridge_fit_summary_present:
        return {
            "state": ModelReleaseState.RESEARCH_ONLY.value,
            "reasons": ["missing_ridge_fit_summary_for_full_model_planner"],
            "triggers": triggers,
        }

    prod = config.run_environment == RunEnvironment.PROD
    if prod and panel_qa_max_severity == "warn":
        if governance_approved_for_optimization:
            return {
                "state": ModelReleaseState.REPORTING_ALLOWED.value,
                "reasons": ["prod_downgrade_panel_qa_warn_blocks_planning"],
                "triggers": triggers,
            }
        if governance_approved_for_reporting:
            return {
                "state": ModelReleaseState.REPORTING_ALLOWED.value,
                "reasons": ["prod_panel_qa_warn_without_optimization_approval"],
                "triggers": triggers,
            }
        return {
            "state": ModelReleaseState.RESEARCH_ONLY.value,
            "reasons": ["prod_panel_qa_warn_without_governance_approvals"],
            "triggers": triggers,
        }

    if governance_approved_for_optimization:
        return {"state": ModelReleaseState.PLANNING_ALLOWED.value, "reasons": [], "triggers": triggers}

    if governance_approved_for_reporting:
        return {
            "state": ModelReleaseState.REPORTING_ALLOWED.value,
            "reasons": [],
            "triggers": triggers,
        }

    return {
        "state": ModelReleaseState.RESEARCH_ONLY.value,
        "reasons": ["governance_not_approved_for_reporting_or_optimization"],
        "triggers": triggers,
    }
