"""Validate explicit hierarchy definitions before Ridge hierarchical borrowing."""

from __future__ import annotations

from collections import defaultdict

from mmm.hierarchy.hierarchy_definition import HierarchyDefinition, HierarchyValidationReport


class HierarchyValidator:
    """Structural validation only; does not infer hierarchy from data."""

    def __init__(
        self,
        *,
        min_children_per_parent: int = 2,
        allow_cross_branch_pooling: bool = False,
    ) -> None:
        self.min_children_per_parent = max(1, int(min_children_per_parent))
        self.allow_cross_branch_pooling = bool(allow_cross_branch_pooling)

    def validate(
        self,
        definition: HierarchyDefinition,
        *,
        panel_entities: set[str] | None = None,
        model_entities: set[str] | None = None,
    ) -> HierarchyValidationReport:
        """
        ``panel_entities``: geo ids present in the training panel (geography type).
        ``model_entities``: channel (or campaign) names in ``data.channel_columns``.
        """
        warnings: list[str] = []
        mapping = dict(definition.node_mapping)
        all_declared = set(definition.parent_nodes) | set(definition.child_nodes) | set(mapping.keys()) | set(
            mapping.values()
        )

        duplicate_assignments: list[str] = []
        seen_child: set[str] = set()
        for child, _parent in mapping.items():
            if child in seen_child:
                duplicate_assignments.append(child)
            seen_child.add(child)
            if not self.allow_cross_branch_pooling:
                parents_for_child = [p for c, p in mapping.items() if c == child]
                if len(set(parents_for_child)) > 1:
                    duplicate_assignments.append(child)

        cycle_detected = _has_cycle(mapping)
        if cycle_detected:
            warnings.append("cycle_detected_in_node_mapping")

        parent_cannot_descend_violations = _parent_is_descendant_violations(mapping)
        if parent_cannot_descend_violations:
            warnings.append(
                f"parent_is_descendant_of_child: {parent_cannot_descend_violations[:5]}"
            )

        missing_nodes: list[str] = []
        if definition.hierarchy_type == "geography":
            geo_universe = panel_entities or set()
            for node in all_declared:
                if node not in geo_universe:
                    missing_nodes.append(node)
        else:
            chan_universe = model_entities or set()
            for node in all_declared:
                if node not in chan_universe:
                    missing_nodes.append(node)
        if missing_nodes:
            warnings.append(f"entities_missing_from_panel_or_model: {sorted(set(missing_nodes))[:10]}")

        children_by_parent: dict[str, list[str]] = defaultdict(list)
        for child, parent in mapping.items():
            children_by_parent[parent].append(child)

        min_child_violations: list[str] = []
        for parent in definition.parent_nodes:
            n = len(children_by_parent.get(parent, []))
            if n < self.min_children_per_parent:
                min_child_violations.append(parent)
        if min_child_violations:
            warnings.append(
                f"min_children_per_parent={self.min_children_per_parent} violated for: "
                f"{min_child_violations[:10]}"
            )

        orphan_nodes = _orphan_nodes(definition, mapping)
        disconnected_nodes = _disconnected_nodes(all_declared, mapping)

        depth = 0 if cycle_detected else _hierarchy_depth(mapping)

        valid = (
            not cycle_detected
            and not duplicate_assignments
            and not missing_nodes
            and not min_child_violations
            and not parent_cannot_descend_violations
            and not orphan_nodes
            and not disconnected_nodes
        )

        return HierarchyValidationReport(
            valid=valid,
            warnings=warnings,
            node_counts={
                "parent_nodes": len(definition.parent_nodes),
                "child_nodes": len(definition.child_nodes),
                "mapped_children": len(mapping),
                "unique_nodes": len(all_declared),
            },
            hierarchy_depth=depth,
            orphan_nodes=sorted(orphan_nodes),
            cycle_detected=cycle_detected,
            duplicate_assignments=sorted(set(duplicate_assignments)),
            disconnected_nodes=sorted(disconnected_nodes),
        )


def _has_cycle(mapping: dict[str, str]) -> bool:
    visited: set[str] = set()
    stack: set[str] = set()

    def dfs(node: str) -> bool:
        if node in stack:
            return True
        if node in visited:
            return False
        visited.add(node)
        stack.add(node)
        parent = mapping.get(node)
        if parent is not None and dfs(parent):
            return True
        stack.discard(node)
        return False

    return any(dfs(start) for start in list(mapping.keys()) + list(mapping.values()))


def _parent_is_descendant_violations(mapping: dict[str, str]) -> list[str]:
    """Parent assigned to a child must not lie in the child's descendant set."""
    children_of: dict[str, list[str]] = defaultdict(list)
    for c, p in mapping.items():
        children_of[p].append(c)

    def descendants(node: str) -> set[str]:
        out: set[str] = set()
        stack = list(children_of.get(node, []))
        while stack:
            n = stack.pop()
            if n in out:
                continue
            out.add(n)
            stack.extend(children_of.get(n, []))
        return out

    bad: list[str] = []
    for child, parent in mapping.items():
        if parent in descendants(child):
            bad.append(f"{child}->{parent}")
    return bad


def _orphan_nodes(definition: HierarchyDefinition, mapping: dict[str, str]) -> list[str]:
    declared = set(definition.parent_nodes) | set(definition.child_nodes)
    touched = set(mapping.keys()) | set(mapping.values())
    roots = {p for p in definition.parent_nodes if p not in mapping}
    orphans = []
    for node in declared:
        if node not in touched and node not in roots:
            orphans.append(node)
    for node in set(mapping.keys()) | set(mapping.values()):
        if node not in declared and node not in roots:
            orphans.append(node)
    return orphans


def _disconnected_nodes(all_nodes: set[str], mapping: dict[str, str]) -> list[str]:
    """Nodes with no edge in the explicit mapping (isolated declarations are invalid)."""
    if not all_nodes:
        return []
    if not mapping:
        return sorted(all_nodes)
    adj: dict[str, set[str]] = defaultdict(set)
    for child, parent in mapping.items():
        adj[child].add(parent)
        adj[parent].add(child)
    isolated = [n for n in all_nodes if not adj.get(n)]
    return sorted(isolated)


def _hierarchy_depth(mapping: dict[str, str]) -> int:
    if not mapping:
        return 0
    depth_cache: dict[str, int] = {}

    def depth(node: str) -> int:
        if node in depth_cache:
            return depth_cache[node]
        parent = mapping.get(node)
        if parent is None:
            depth_cache[node] = 0
            return 0
        d = 1 + depth(parent)
        depth_cache[node] = d
        return d

    return max(depth(c) for c in mapping) if mapping else 0
