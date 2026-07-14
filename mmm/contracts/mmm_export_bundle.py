"""Conservative MIP-side parser for externally produced MMM export bundles.

This module intentionally does not import the MMM producer schema.  Its job is
to normalize an untrusted interchange payload for answerability checks; absent
or malformed safety fields therefore become blocked values instead of producer
defaults.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _text(value: Any, default: str = "unknown") -> str:
    return value.strip() if isinstance(value, str) and value.strip() else default


def _explicit_true(value: Any) -> bool:
    return value is True


def _string_set(value: Any) -> frozenset[str]:
    if not isinstance(value, list):
        return frozenset()
    return frozenset(item.strip() for item in value if isinstance(item, str) and item.strip())


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


@dataclass(frozen=True)
class ParsedMMMExportArtifact:
    """Normalized safety fields for one artifact in an MMM export bundle."""

    artifact_type: str = "unknown"
    llm_exposure_allowed: bool = False
    demo_fixture_allowed: bool = False
    recommendation_allowed: bool = False
    planning_allowed: bool = False
    production_claim_allowed: bool = False
    allowed_claims: frozenset[str] = field(default_factory=frozenset)
    forbidden_claims: frozenset[str] = field(default_factory=frozenset)
    diagnostic_status: str = "unknown"
    promotion_status: str = "unknown"
    uncertainty_status: str = "unknown"
    artifact_safety_status: str = "blocked"
    lineage: Mapping[str, Any] = field(default_factory=dict)
    source_optimizer_artifact_id: str = ""
    trust_report_refs: tuple[str, ...] = ()
    proposed_budget_shifts: tuple[Mapping[str, Any], ...] = ()
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False)


@dataclass(frozen=True)
class ParsedMMMExportBundle:
    """Normalized bundle envelope; every safety switch defaults to false."""

    schema_version: str = "unknown"
    bundle_id: str = ""
    model_run_id: str = ""
    artifacts: tuple[ParsedMMMExportArtifact, ...] = ()
    llm_exposure_allowed: bool = False
    demo_fixture_allowed: bool = False
    recommendation_allowed: bool = False
    planning_allowed: bool = False
    production_claim_allowed: bool = False
    allowed_claims: frozenset[str] = field(default_factory=frozenset)
    forbidden_claims: frozenset[str] = field(default_factory=frozenset)
    diagnostic_status: str = "unknown"
    promotion_status: str = "unknown"
    uncertainty_status: str = "unknown"
    artifact_safety_status: str = "blocked"
    lineage: Mapping[str, Any] = field(default_factory=dict)
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False)


def _parse_artifact(value: Any) -> ParsedMMMExportArtifact:
    data = _mapping(value)
    refs = data.get("trust_report_refs")
    shifts = data.get("proposed_budget_shifts")
    return ParsedMMMExportArtifact(
        artifact_type=_text(data.get("artifact_type")),
        llm_exposure_allowed=_explicit_true(data.get("llm_exposure_allowed")),
        demo_fixture_allowed=_explicit_true(data.get("demo_fixture_allowed")),
        recommendation_allowed=_explicit_true(data.get("recommendation_allowed")),
        planning_allowed=_explicit_true(data.get("planning_allowed")),
        production_claim_allowed=_explicit_true(data.get("production_claim_allowed")),
        allowed_claims=_string_set(data.get("allowed_claims")),
        forbidden_claims=_string_set(data.get("forbidden_claims")),
        diagnostic_status=_text(data.get("diagnostic_status")),
        promotion_status=_text(data.get("promotion_status")),
        uncertainty_status=_text(data.get("uncertainty_status")),
        artifact_safety_status=_text(data.get("artifact_safety_status"), "blocked"),
        lineage=_mapping(data.get("lineage")),
        source_optimizer_artifact_id=_text(data.get("source_optimizer_artifact_id"), ""),
        trust_report_refs=tuple(x for x in refs if isinstance(x, str) and x.strip())
        if isinstance(refs, list)
        else (),
        proposed_budget_shifts=tuple(dict(x) for x in shifts if isinstance(x, Mapping))
        if isinstance(shifts, list)
        else (),
        raw=data,
    )


def parse_mmm_export_bundle(payload: Mapping[str, Any]) -> ParsedMMMExportBundle:
    """Parse an external bundle without borrowing producer-side defaults."""

    data = _mapping(payload)
    raw_artifacts = data.get("artifacts")
    artifacts = tuple(_parse_artifact(item) for item in raw_artifacts) if isinstance(raw_artifacts, list) else ()
    lineage = _mapping(data.get("lineage"))
    for key in (
        "training_data_fingerprint",
        "model_artifact_fingerprint",
        "source_artifacts",
        "generated_at",
        "package_version",
        "git_commit",
    ):
        if key in data and key not in lineage:
            lineage[key] = data[key]
    return ParsedMMMExportBundle(
        schema_version=_text(data.get("schema_version")),
        bundle_id=_text(data.get("bundle_id"), ""),
        model_run_id=_text(data.get("model_run_id"), ""),
        artifacts=artifacts,
        llm_exposure_allowed=_explicit_true(data.get("llm_exposure_allowed")),
        demo_fixture_allowed=_explicit_true(data.get("demo_fixture_allowed")),
        recommendation_allowed=_explicit_true(data.get("recommendation_allowed")),
        planning_allowed=_explicit_true(data.get("planning_allowed")),
        production_claim_allowed=_explicit_true(data.get("production_claim_allowed")),
        allowed_claims=_string_set(data.get("allowed_claims")),
        forbidden_claims=_string_set(data.get("forbidden_claims")),
        diagnostic_status=_text(data.get("diagnostic_status")),
        promotion_status=_text(data.get("promotion_status")),
        uncertainty_status=_text(data.get("uncertainty_status")),
        artifact_safety_status=_text(data.get("artifact_safety_status"), "blocked"),
        lineage=lineage,
        raw=data,
    )


def load_mmm_export_bundle(path: str | Path) -> ParsedMMMExportBundle:
    """Load and parse a JSON bundle from a local interchange file."""

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError("MMMExportBundle JSON must contain an object")
    return parse_mmm_export_bundle(data)
