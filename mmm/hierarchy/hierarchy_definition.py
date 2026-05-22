"""Explicit hierarchy definitions for Ridge hierarchical borrowing (opt-in)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

HierarchyType = Literal["geography", "channel", "campaign"]


class HierarchyDefinition(BaseModel):
    """
    User-supplied hierarchy; never inferred from panel data.

    For ``channel`` / ``campaign``, ``node_mapping`` keys are child effect entities and values
    are parent effect entities (typically ``data.channel_columns`` names).

    For ``geography``, ``node_mapping`` keys are child geo ids and values are parent geo ids
    (panel ``geo_column`` values). Ridge penalty applies only to pairs listed in
    ``metadata["ridge_effect_pairs"]`` when ``hierarchy_type=geography`` (each entry:
    ``{"child": "<channel>", "parent": "<channel>"}``).
    """

    hierarchy_id: str
    hierarchy_type: HierarchyType
    parent_nodes: list[str] = Field(default_factory=list)
    child_nodes: list[str] = Field(default_factory=list)
    node_mapping: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    version: str = "1"


class HierarchyValidationReport(BaseModel):
    valid: bool = False
    warnings: list[str] = Field(default_factory=list)
    node_counts: dict[str, int] = Field(default_factory=dict)
    hierarchy_depth: int = 0
    orphan_nodes: list[str] = Field(default_factory=list)
    cycle_detected: bool = False
    duplicate_assignments: list[str] = Field(default_factory=list)
    disconnected_nodes: list[str] = Field(default_factory=list)


def load_hierarchy_definition(path: str | Path) -> HierarchyDefinition:
    p = Path(path)
    raw = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "hierarchy_definition" in raw:
        raw = raw["hierarchy_definition"]
    return HierarchyDefinition.model_validate(raw)
