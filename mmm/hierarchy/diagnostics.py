"""Post-fit hierarchy diagnostics and governance warnings."""

from __future__ import annotations

from typing import Any

import numpy as np

from mmm.config.schema import HierarchyConfig, MMMConfig
from mmm.hierarchy.hierarchy_definition import HierarchyDefinition, HierarchyValidationReport
from mmm.hierarchy.penalty import HierarchyCoefPair, hierarchical_penalty

HIERARCHY_GOVERNANCE_WARNINGS: tuple[str, ...] = (
    "Hierarchical borrowing stabilizes estimates but does not create causal evidence.",
    "Parent evidence does not imply child causal validity.",
    "Aggregate experiment evidence cannot justify local claims.",
    "Hierarchy is regularization, not identification.",
)

HIERARCHY_UNSUPPORTED_QUESTIONS: tuple[str, ...] = (
    "Local causal claims from parent-only evidence.",
    "Hidden borrowing interpretation (hierarchy applies only to explicitly mapped effects).",
    "Child-level experiment conclusions from parent evidence.",
)


def hierarchy_enabled(config: MMMConfig) -> bool:
    return bool(config.hierarchy.enabled)


def build_hierarchy_effect_summary(
    coef: np.ndarray,
    pairs: list[HierarchyCoefPair],
    *,
    regularization_strength: float,
    coef_before: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """Per-child shrinkage vs parent (``coef_before`` = pre-hierarchy trial coef if available)."""
    media = np.asarray(coef, dtype=float).ravel()
    before = np.asarray(coef_before, dtype=float).ravel() if coef_before is not None else media
    rows: list[dict[str, Any]] = []
    for p in pairs:
        local_after = float(media[p.child_index])
        local_before = float(before[p.child_index])
        parent_est = float(media[p.parent_index])
        rows.append(
            {
                "child": p.child_name,
                "parent": p.parent_name,
                "local_estimate_before": local_before,
                "local_estimate_after": local_after,
                "parent_estimate": parent_est,
                "shrinkage_delta": local_after - local_before,
            }
        )
    _ = regularization_strength
    return rows


def build_hierarchy_diagnostics(
    config: MMMConfig,
    definition: HierarchyDefinition,
    validation: HierarchyValidationReport,
    pairs: list[HierarchyCoefPair],
    coef: np.ndarray,
    *,
    coef_before: np.ndarray | None = None,
    extra_warnings: list[str] | None = None,
) -> dict[str, Any]:
    hcfg: HierarchyConfig = config.hierarchy
    pen, pen_meta = hierarchical_penalty(coef, pairs, regularization_strength=hcfg.regularization_strength)
    media = np.asarray(coef, dtype=float).ravel()
    local_effects = {p.child_name: float(media[p.child_index]) for p in pairs}
    parent_effects = {p.parent_name: float(media[p.parent_index]) for p in pairs}
    shrinkage_amount: dict[str, float] = {}
    shrinkage_ratio: dict[str, float] = {}
    unstable_children: list[str] = []
    if coef_before is not None:
        before = np.asarray(coef_before, dtype=float).ravel()
        for p in pairs:
            b = float(before[p.child_index])
            a = float(media[p.child_index])
            par = float(media[p.parent_index])
            shrinkage_amount[p.child_name] = a - b
            denom = abs(b - par) + 1e-9
            shrinkage_ratio[p.child_name] = (a - par) / denom
            if abs(b) > 3.0 * (abs(par) + 1e-6):
                unstable_children.append(p.child_name)

    warnings = list(validation.warnings) + list(HIERARCHY_GOVERNANCE_WARNINGS)
    if extra_warnings:
        warnings.extend(extra_warnings)

    structure = {
        "hierarchy_id": definition.hierarchy_id,
        "hierarchy_type": definition.hierarchy_type,
        "version": definition.version,
        "parent_nodes": list(definition.parent_nodes),
        "child_nodes": list(definition.child_nodes),
    }

    return {
        "hierarchy_enabled": True,
        "hierarchy_type": definition.hierarchy_type,
        "hierarchy_structure": structure,
        "parent_child_mapping": dict(definition.node_mapping),
        "regularization_strength": float(hcfg.regularization_strength),
        "local_effects": local_effects,
        "parent_effects": parent_effects,
        "shrinkage_amount": shrinkage_amount,
        "shrinkage_ratio": shrinkage_ratio,
        "unstable_children": unstable_children,
        "warnings": warnings,
        "validation": validation.model_dump(),
        "hierarchical_penalty_at_fit": pen,
        "penalty_meta": pen_meta,
        "n_coef_pairs": len(pairs),
    }


def build_hierarchy_extension_reports(
    config: MMMConfig,
    definition: HierarchyDefinition,
    validation: HierarchyValidationReport,
    pairs: list[HierarchyCoefPair],
    coef: np.ndarray,
    *,
    coef_before: np.ndarray | None = None,
    extra_warnings: list[str] | None = None,
) -> dict[str, Any]:
    if not hierarchy_enabled(config):
        return {"skipped": True, "reason": "hierarchy_disabled"}
    diag = build_hierarchy_diagnostics(
        config,
        definition,
        validation,
        pairs,
        coef,
        coef_before=coef_before,
        extra_warnings=extra_warnings,
    )
    summary = build_hierarchy_effect_summary(
        coef,
        pairs,
        regularization_strength=config.hierarchy.regularization_strength,
        coef_before=coef_before,
    )
    return {
        "hierarchy_diagnostics": diag,
        "hierarchy_effect_summary": summary,
        "hierarchy_validation_report": validation.model_dump(),
        "governance_unsupported_claims": list(HIERARCHY_UNSUPPORTED_QUESTIONS),
        "hierarchy_governance_warnings": list(HIERARCHY_GOVERNANCE_WARNINGS),
    }
