"""Fixed non-media context for media-only budget optimization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mmm.planning.control_overlay import ControlOverlaySpec


@dataclass(frozen=True)
class OptimizeNonMediaContext:
    """
    Non-media controls held fixed while SLSQP varies media spend.

    When ``frozen_non_media`` and only ``control_overlay_plan`` is set, that overlay applies
    to **both** baseline and plan μ (single fixed world).
    """

    control_overlay_baseline: ControlOverlaySpec | None = None
    control_overlay_plan: ControlOverlaySpec | None = None
    frozen_non_media: bool = False
    scenario_lineage: dict[str, Any] | None = None

    def resolved_overlays(self) -> tuple[ControlOverlaySpec | None, ControlOverlaySpec | None]:
        if self.frozen_non_media and self.control_overlay_plan and not self.control_overlay_baseline:
            return self.control_overlay_plan, self.control_overlay_plan
        return self.control_overlay_baseline, self.control_overlay_plan
