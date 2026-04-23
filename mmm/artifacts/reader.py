"""Surface-aware artifact loading (reader-side tier enforcement)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mmm.contracts.lineage import assert_decision_tier_lineage_complete
from mmm.governance.policy import PolicyError, require_surface_compatible
from mmm.governance.semantics import ArtifactTier, Surface
from mmm.governance.validation import validate_decision_surface_bundle


def _infer_tier(payload: dict[str, Any]) -> ArtifactTier:
    raw = payload.get("artifact_tier") or payload.get("tier")
    if raw is None and "decision_bundle" in payload:
        raw = (payload.get("decision_bundle") or {}).get("artifact_tier") or (payload.get("decision_bundle") or {}).get(
            "tier"
        )
    if raw is None:
        raise PolicyError("artifact missing artifact_tier / tier metadata")
    s = str(raw).lower()
    for t in ArtifactTier:
        if t.value == s:
            return t
    raise PolicyError(f"unknown artifact tier: {raw!r}")


def load_artifact(path: Path, *, surface: Surface) -> dict[str, Any]:
    """
    Load JSON artifact and enforce tier vs intended consumer surface.

    Decision surfaces reject diagnostic/research artifacts.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise PolicyError("artifact root must be a JSON object")
    tier = _infer_tier(data)
    require_surface_compatible(tier, surface)
    bundle = data if "artifact_tier" in data else data.get("decision_bundle") or {}
    validate_decision_surface_bundle(bundle, surface=surface)
    if surface == Surface.DECISION:
        assert_decision_tier_lineage_complete(data if "artifact_tier" in data else bundle)
    return data
