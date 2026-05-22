"""Ridge hierarchical borrowing penalty (explicit mapping only)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from mmm.hierarchy.hierarchy_definition import HierarchyDefinition, HierarchyValidationReport
from mmm.hierarchy.validator import HierarchyValidator


@dataclass(frozen=True)
class HierarchyCoefPair:
    child_name: str
    parent_name: str
    child_index: int
    parent_index: int


def resolve_hierarchy_coef_pairs(
    definition: HierarchyDefinition,
    channel_columns: tuple[str, ...] | list[str],
    *,
    validation: HierarchyValidationReport | None = None,
) -> tuple[list[HierarchyCoefPair], list[str]]:
    """
    Map hierarchy edges to media coefficient indices (first ``len(channel_columns)`` Ridge coefs).

    Geography hierarchies require explicit ``metadata["ridge_effect_pairs"]``; geo ``node_mapping``
    alone does not imply coefficient pooling.
    """
    channels = list(channel_columns)
    idx = {c: i for i, c in enumerate(channels)}
    warnings: list[str] = []
    pairs: list[HierarchyCoefPair] = []

    def add_pair(child: str, parent: str) -> None:
        if child not in idx or parent not in idx:
            warnings.append(f"coef_pair_not_in_channel_columns: {child}->{parent}")
            return
        if child == parent:
            warnings.append(f"self_parent_pair: {child}")
            return
        pairs.append(
            HierarchyCoefPair(
                child_name=child,
                parent_name=parent,
                child_index=idx[child],
                parent_index=idx[parent],
            )
        )

    if definition.hierarchy_type in ("channel", "campaign"):
        for child, parent in definition.node_mapping.items():
            add_pair(child, parent)
    elif definition.hierarchy_type == "geography":
        raw_pairs = definition.metadata.get("ridge_effect_pairs") or []
        if not raw_pairs:
            warnings.append(
                "geography_hierarchy: no metadata.ridge_effect_pairs; ridge penalty not applied "
                "(geo structure validated separately)"
            )
        for item in raw_pairs:
            if not isinstance(item, dict):
                continue
            add_pair(str(item.get("child", "")), str(item.get("parent", "")))
    else:
        warnings.append(f"unsupported_hierarchy_type: {definition.hierarchy_type}")

    if validation is not None and not validation.valid:
        warnings.append("hierarchy_validation_failed_penalty_skipped")

    dedup: dict[tuple[int, int], HierarchyCoefPair] = {}
    for p in pairs:
        dedup[(p.child_index, p.parent_index)] = p
    return list(dedup.values()), warnings


def hierarchical_penalty(
    coef: np.ndarray,
    pairs: list[HierarchyCoefPair],
    *,
    regularization_strength: float,
) -> tuple[float, dict[str, Any]]:
    """
    ``lambda_hier * sum((local - parent)^2)`` on explicitly mapped media coefficients.
    """
    lam = float(regularization_strength)
    if lam <= 0.0 or not pairs:
        return 0.0, {"n_pairs": 0, "lambda": lam, "per_pair_sq": []}
    media = np.asarray(coef, dtype=float).ravel()
    sq_terms: list[dict[str, Any]] = []
    total = 0.0
    for p in pairs:
        local = float(media[p.child_index])
        parent = float(media[p.parent_index])
        diff = local - parent
        sq = diff * diff
        total += sq
        sq_terms.append(
            {
                "child": p.child_name,
                "parent": p.parent_name,
                "local": local,
                "parent_effect": parent,
                "squared_diff": sq,
            }
        )
    return lam * total, {"n_pairs": len(pairs), "lambda": lam, "per_pair_sq": sq_terms}


def prepare_hierarchy_for_ridge(
    definition: HierarchyDefinition,
    channel_columns: tuple[str, ...] | list[str],
    *,
    panel_geos: set[str] | None,
    min_children_per_parent: int,
    allow_cross_branch_pooling: bool,
) -> tuple[list[HierarchyCoefPair], HierarchyValidationReport, list[str]]:
    validator = HierarchyValidator(
        min_children_per_parent=min_children_per_parent,
        allow_cross_branch_pooling=allow_cross_branch_pooling,
    )
    report = validator.validate(
        definition,
        panel_entities=panel_geos,
        model_entities=set(channel_columns),
    )
    pairs, pen_warnings = resolve_hierarchy_coef_pairs(
        definition, channel_columns, validation=report
    )
    if not report.valid:
        return [], report, pen_warnings
    return pairs, report, pen_warnings
