"""Load training extension reports vs decision-tier artifacts (tier enforcement)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mmm.artifacts.reader import load_artifact
from mmm.governance.policy import PolicyError
from mmm.governance.semantics import ArtifactTier, Surface


def load_training_extension_report(path: str | Path) -> dict[str, Any]:
    """
    Load a post-train ``extension_report.json`` for ``mmm decide`` inputs.

    Training reports are **research/diagnostic** tier by design; this loader rejects
    decision-tier bundles mistaken as extension reports.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"extension report not found: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise PolicyError("extension_report root must be a JSON object")
    tier = str(data.get("artifact_tier") or data.get("tier") or "").lower()
    if tier == ArtifactTier.DECISION.value and "ridge_fit_summary" not in data:
        raise PolicyError(
            "artifact_tier=decision without ridge_fit_summary: file looks like a CLI decision bundle, "
            "not a training extension_report.json"
        )
    rs = data.get("ridge_fit_summary")
    if not isinstance(rs, dict) or not rs.get("coef"):
        raise PolicyError("training extension_report must contain ridge_fit_summary.coef")
    nested = data.get("decision_bundle")
    if isinstance(nested, dict):
        nested_tier = str(nested.get("artifact_tier") or nested.get("tier") or "").lower()
        if nested_tier == ArtifactTier.DECISION.value:
            raise PolicyError(
                "nested decision_bundle has artifact_tier=decision inside extension_report; "
                "use the training extension_report path from the run directory, not a CLI optimize output"
            )
    return data


def load_decision_tier_artifact(path: str | Path) -> dict[str, Any]:
    """Load a persisted decision bundle or simulate output (decision tier required)."""
    return load_artifact(Path(path), surface=Surface.DECISION)
