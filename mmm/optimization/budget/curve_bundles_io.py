"""Load per-channel curve bundles from extension JSON or standalone files (Sprint 5)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mmm.decomposition.curve_export_gate import validate_curve_bundle_typed_curve_quantity


def _valid_bundle(d: dict[str, Any]) -> bool:
    g = d.get("spend_grid")
    r = d.get("response_on_modeling_scale")
    return isinstance(g, list) and isinstance(r, list) and len(g) >= 2 and len(g) == len(r)


def gather_curve_bundles_from_dict(
    data: dict[str, Any],
    *,
    require_typed_curve_quantity: bool = False,
) -> tuple[list[str], list[dict[str, Any]]] | None:
    """
    Parse ``curve_bundles`` (preferred) or a single ``curve_bundle`` dict.

    Returns ``(channel_names, bundles)`` in aligned order, or ``None`` if no usable curves.
    """
    raw = data.get("curve_bundles")
    if isinstance(raw, list) and raw:
        names: list[str] = []
        bundles: list[dict[str, Any]] = []
        for item in raw:
            if not isinstance(item, dict) or not _valid_bundle(item):
                continue
            ch = item.get("channel")
            if not isinstance(ch, str) or not ch:
                continue
            if require_typed_curve_quantity:
                validate_curve_bundle_typed_curve_quantity(item, context=f"gather_curve_bundles[{ch}]")
            names.append(ch)
            bundles.append(item)
        if names:
            return (names, bundles)

    one = data.get("curve_bundle")
    if isinstance(one, dict) and _valid_bundle(one):
        ch = one.get("channel")
        if isinstance(ch, str) and ch:
            if require_typed_curve_quantity:
                validate_curve_bundle_typed_curve_quantity(one, context=f"gather_curve_bundles[{ch}]")
            return ([ch], [one])
    return None


def load_curve_bundles_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def gather_curve_bundles_from_path(
    path: Path,
    *,
    require_typed_curve_quantity: bool = False,
) -> tuple[list[str], list[dict[str, Any]]] | None:
    """Load JSON file; body may be a full extension report or ``{\"curve_bundles\": [...]}``."""
    data = load_curve_bundles_json(path)
    if not isinstance(data, dict):
        return None
    return gather_curve_bundles_from_dict(data, require_typed_curve_quantity=require_typed_curve_quantity)
