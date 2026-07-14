"""Cross-cutting contracts and canonical semantics."""

from mmm.contracts.semantics import (
    CalibrationEstimandSpec,
    ContributionInterpretation,
    ModelingTargetSpec,
    OptimizationSafetySpec,
)
from mmm.contracts.mip_failure import (
    MMMExportOutcome,
    MMMFailureCode,
    MMMFailurePacket,
    MMMFailureStage,
    MMMRemediationAction,
    MMMRetryDisposition,
)

__all__ = [
    "ModelingTargetSpec",
    "CalibrationEstimandSpec",
    "ContributionInterpretation",
    "OptimizationSafetySpec",
    "MMMFailurePacket",
    "MMMFailureCode",
    "MMMFailureStage",
    "MMMRetryDisposition",
    "MMMRemediationAction",
    "MMMExportOutcome",
]
