"""H6b — vertical-specific control profiles (re-export from config)."""

from mmm.config.vertical_control_profiles import (
    VERTICAL_AUTO,
    VERTICAL_CPG,
    VERTICAL_PROFILES,
    VERTICAL_RETAIL,
    VerticalControlProfile,
    control_truth_for_profile,
    get_vertical_profile,
)

__all__ = [
    "VERTICAL_AUTO",
    "VERTICAL_CPG",
    "VERTICAL_PROFILES",
    "VERTICAL_RETAIL",
    "VerticalControlProfile",
    "control_truth_for_profile",
    "get_vertical_profile",
]
