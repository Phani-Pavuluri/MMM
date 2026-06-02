"""Bayes-H6 production-shaped synthetic validation lane (research only)."""

from mmm.research.h6_synthetic.benchmark_harness import (
    build_h6_benchmark_artifact,
    run_h6_benchmark_pair,
)
from mmm.research.h6_synthetic.production_shapes import (
    H6_PILOT_WORLD_IDS,
    ProductionShapedWorldSpec,
    get_h6_world,
    list_h6_world_ids,
    materialize_h6_panel,
)
from mmm.research.h6_synthetic.vertical_controls import (
    VERTICAL_PROFILES,
    VerticalControlProfile,
    get_vertical_profile,
)

__all__ = [
    "H6_PILOT_WORLD_IDS",
    "ProductionShapedWorldSpec",
    "VERTICAL_PROFILES",
    "VerticalControlProfile",
    "build_h6_benchmark_artifact",
    "get_h6_world",
    "get_vertical_profile",
    "list_h6_world_ids",
    "materialize_h6_panel",
    "run_h6_benchmark_pair",
]
