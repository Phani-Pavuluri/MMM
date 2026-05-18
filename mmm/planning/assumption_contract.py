"""Canonical planning assumption contract — single source of truth for allowed values and validation."""

from __future__ import annotations

from typing import Any, Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, field_validator

# --- Allowed values (add new enums here only) ---

ControlsAssumption = Literal["observed", "overlay", "frozen_scenario"]
MediaAssumption = Literal["constant", "geo_channel", "piecewise_path", "optimized"]
WorldAssumption = Literal["historical_panel", "explicit_scenario", "multi_world"]

CONTROLS_ASSUMPTION_VALUES: frozenset[str] = frozenset(get_args(ControlsAssumption))
MEDIA_ASSUMPTION_VALUES: frozenset[str] = frozenset(get_args(MediaAssumption))
WORLD_ASSUMPTION_VALUES: frozenset[str] = frozenset(get_args(WorldAssumption))

REQUIRED_PLANNING_ASSUMPTION_KEYS: frozenset[str] = frozenset(
    {
        "controls_assumption",
        "media_assumption",
        "world_assumption",
    }
)


class PlanningAssumptionsContract(BaseModel):
    """Typed planning metadata attached to decision-tier artifacts and simulate/optimize JSON."""

    model_config = ConfigDict(extra="allow")

    controls_assumption: ControlsAssumption
    media_assumption: MediaAssumption
    world_assumption: WorldAssumption
    seasonality_assumption: str = "observed_panel"
    promo_assumption: str = "observed_panel_unless_overlay"
    macro_assumption: str = "observed_panel_unless_overlay"
    pricing_assumption: str = "observed_panel_unless_overlay"
    planning_disclosures: list[str] = Field(default_factory=list)
    controls_disclosure: str | None = None

    @field_validator("controls_assumption", "media_assumption", "world_assumption", mode="before")
    @classmethod
    def _reject_non_str_enums(cls, v: Any) -> Any:
        if v is not None and not isinstance(v, str):
            raise ValueError("planning assumption enum fields must be strings")
        return v

    def to_artifact_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=True)


def parse_planning_assumptions(raw: Any) -> PlanningAssumptionsContract:
    """Parse and validate enum fields. Raises ``ValidationError`` on unknown values."""
    if not isinstance(raw, dict):
        raise TypeError("planning_assumptions must be a dict")
    return PlanningAssumptionsContract.model_validate(raw)


def planning_assumptions_enum_errors(raw: Any) -> list[str]:
    """Return machine-readable issues for invalid structure or enum values (no combination rules)."""
    issues: list[str] = []
    if raw is None:
        return ["planning_assumptions_missing"]
    if not isinstance(raw, dict):
        return ["planning_assumptions_must_be_dict"]
    for key in REQUIRED_PLANNING_ASSUMPTION_KEYS:
        if key not in raw:
            issues.append(f"planning_assumptions_missing_field:{key}")
    if issues:
        return issues
    for key, allowed in (
        ("controls_assumption", CONTROLS_ASSUMPTION_VALUES),
        ("media_assumption", MEDIA_ASSUMPTION_VALUES),
        ("world_assumption", WORLD_ASSUMPTION_VALUES),
    ):
        val = raw.get(key)
        if val is None or (isinstance(val, str) and not val.strip()):
            issues.append(f"planning_assumptions_empty:{key}")
        elif not isinstance(val, str):
            issues.append(f"planning_assumptions_invalid_type:{key}")
        elif val not in allowed:
            issues.append(f"planning_assumptions_invalid_enum:{key}={val!r}")
    return issues


def _lineage_has_scenario_identity(sl: dict[str, Any]) -> bool:
    sid = sl.get("scenario_id")
    sh = sl.get("scenario_hash")
    return bool(sid and isinstance(sid, str) and sh and isinstance(sh, str))


def _lineage_supports_frozen_scenario(sl: dict[str, Any] | None) -> bool:
    if not isinstance(sl, dict) or not sl:
        return False
    return (
        _lineage_has_scenario_identity(sl)
        or sl.get("non_media_overlay_applied") is True
        or (sl.get("non_media_overlay_supplied") is True and bool(sl.get("scenario_hash")))
    )


def _bundle_is_optimize_context(bundle: dict[str, Any] | None) -> bool:
    if bundle is None:
        return False
    if bundle.get("optimizer_success") is not None:
        return True
    sc = bundle.get("simulation_contract")
    if isinstance(sc, dict):
        src = str(sc.get("source") or "")
        if "optim" in src.lower() or "slsqp" in src.lower():
            return True
    ms = bundle.get("model_summary")
    return isinstance(ms, dict) and ms.get("optimizer_success") is not None


def validate_planning_assumptions_semantics(
    raw: Any,
    *,
    scenario_lineage: dict[str, Any] | None = None,
    bundle: dict[str, Any] | None = None,
    strict: bool = True,
) -> list[str]:
    """
    Validate planning assumptions structure, enums, and required combinations.

    When ``strict`` is False, enum/structure errors are still reported; combination rules
    that are prod-only (e.g. ``multi_world``) may be omitted by the caller.
    """
    issues = planning_assumptions_enum_errors(raw)
    if issues:
        return issues

    assert isinstance(raw, dict)
    controls = raw["controls_assumption"]
    media = raw["media_assumption"]
    world = raw["world_assumption"]
    sl = scenario_lineage
    if sl is None and bundle is not None:
        sl_b = bundle.get("scenario_lineage")
        if isinstance(sl_b, dict):
            sl = sl_b

    if world == "explicit_scenario" and (
        not isinstance(sl, dict) or not _lineage_has_scenario_identity(sl)
    ):
        issues.append("explicit_scenario_requires_scenario_lineage_id_and_hash")

    if world == "multi_world" and strict:
        issues.append("multi_world_not_implemented_for_decision_bundles")

    sl_dict = sl if isinstance(sl, dict) else None
    if controls == "frozen_scenario" and not _lineage_supports_frozen_scenario(sl_dict):
        issues.append("frozen_scenario_requires_scenario_lineage_or_overlay_metadata")

    if controls == "overlay" and isinstance(sl, dict) and sl:
            summary = sl.get("control_overlay_summary")
            if isinstance(summary, dict):
                cols = summary.get("all_overlay_columns") or []
                if not cols and not sl.get("plan_overlay_spec_sha256"):
                    issues.append("overlay_controls_assumption_requires_overlay_columns_or_sha256")
            elif not sl.get("plan_overlay_spec_sha256") and not sl.get("control_overlay_spec_sha256"):
                issues.append("overlay_controls_assumption_requires_scenario_lineage_overlay_evidence")

    if (
        media == "optimized"
        and strict
        and bundle is not None
        and not _bundle_is_optimize_context(bundle)
    ):
        issues.append("optimized_media_assumption_requires_optimize_bundle_context")

    if media != "optimized" and bundle is not None and _bundle_is_optimize_context(bundle):
        issues.append("optimize_bundle_requires_media_assumption_optimized")

    return issues


def coerce_validated_planning_assumptions(raw: dict[str, Any]) -> dict[str, Any]:
    """Return a normalized dict that passes enum validation (for builders)."""
    return parse_planning_assumptions(raw).to_artifact_dict()
