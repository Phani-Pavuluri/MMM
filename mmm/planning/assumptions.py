"""Planning contract assumptions — explicit, auditable fields for decision outputs."""

from __future__ import annotations

from typing import Any

from mmm.planning.assumption_contract import (
    ControlsAssumption,
    MediaAssumption,
    PlanningAssumptionsContract,
    WorldAssumption,
)

CONTROLS_DISCLOSURE = (
    "Non-media controls are held at observed historical panel values unless an explicit "
    "PlanningScenario or control overlay is supplied."
)
OPTIMIZE_MEDIA_DISCLOSURE = (
    "optimize-budget optimizes media spend only under one fixed non-media world; "
    "it does not optimize promos, pricing, macro, or other control variables."
)
SIMULATE_PARTIAL_WORLD_DISCLOSURE = (
    "decide simulate changes media spend on the training panel; non-media variables default to "
    "observed historical values unless overlays or a PlanningScenario are supplied."
)


def infer_media_assumption(
    *,
    optimized: bool = False,
    spend_plan_geo: bool = False,
    spend_path: bool = False,
) -> MediaAssumption:
    if optimized:
        return "optimized"
    if spend_path:
        return "piecewise_path"
    if spend_plan_geo:
        return "geo_channel"
    return "constant"


def infer_controls_assumption(
    *,
    has_baseline_overlay: bool,
    has_plan_overlay: bool,
    frozen_non_media: bool = False,
) -> ControlsAssumption:
    if frozen_non_media:
        return "frozen_scenario"
    if has_baseline_overlay or has_plan_overlay:
        return "overlay"
    return "observed"


def infer_world_assumption(
    *,
    scenario_id: str | None,
    explicit_scenario: bool,
) -> WorldAssumption:
    if explicit_scenario and scenario_id:
        return "explicit_scenario"
    return "historical_panel"


def build_planning_assumptions(
    *,
    controls_assumption: ControlsAssumption,
    media_assumption: MediaAssumption,
    world_assumption: WorldAssumption,
    seasonality_assumption: str = "observed_panel",
    promo_assumption: str = "observed_panel_unless_overlay",
    macro_assumption: str = "observed_panel_unless_overlay",
    pricing_assumption: str = "observed_panel_unless_overlay",
) -> dict[str, Any]:
    disclosures = [CONTROLS_DISCLOSURE]
    if media_assumption == "optimized":
        disclosures.append(OPTIMIZE_MEDIA_DISCLOSURE)
    else:
        disclosures.append(SIMULATE_PARTIAL_WORLD_DISCLOSURE)
    contract = PlanningAssumptionsContract(
        controls_assumption=controls_assumption,
        media_assumption=media_assumption,
        world_assumption=world_assumption,
        seasonality_assumption=seasonality_assumption,
        promo_assumption=promo_assumption,
        macro_assumption=macro_assumption,
        pricing_assumption=pricing_assumption,
        planning_disclosures=disclosures,
        controls_disclosure=CONTROLS_DISCLOSURE,
    )
    return contract.to_artifact_dict()


def minimal_planning_assumptions_simulate(
    *,
    controls_assumption: ControlsAssumption = "observed",
    media_assumption: MediaAssumption = "constant",
    world_assumption: WorldAssumption = "historical_panel",
) -> dict[str, Any]:
    """Test / stub helper for decision bundles."""
    return build_planning_assumptions(
        controls_assumption=controls_assumption,
        media_assumption=media_assumption,
        world_assumption=world_assumption,
    )


def minimal_planning_assumptions_optimize(
    *,
    controls_assumption: ControlsAssumption = "observed",
    world_assumption: WorldAssumption = "historical_panel",
) -> dict[str, Any]:
    return build_planning_assumptions(
        controls_assumption=controls_assumption,
        media_assumption="optimized",
        world_assumption=world_assumption,
    )


def merge_disclosure(existing: str, assumptions: dict[str, Any]) -> str:
    parts = [p for p in [existing.strip(), assumptions.get("controls_disclosure", "")] if p]
    if assumptions.get("media_assumption") == "optimized":
        parts.append(OPTIMIZE_MEDIA_DISCLOSURE)
    return " ".join(dict.fromkeys(parts))
