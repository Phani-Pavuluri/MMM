"""Research-only and deprecated statistical helpers — not imported by decision or training paths."""

from mmm.research_legacy.calibration_coef import calibration_mismatch_loss, implied_channel_weights_from_coef

__all__ = ["calibration_mismatch_loss", "implied_channel_weights_from_coef"]
