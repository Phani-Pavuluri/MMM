"""Cross-cutting contracts and canonical semantics."""

from typing import TYPE_CHECKING, Any

from mmm.contracts.calibration_treatment import (
    MMMCalibrationApplicationRole,
    MMMCalibrationCompatibilityStatus,
    MMMCalibrationFreshnessStatus,
    MMMCalibrationTransformationStep,
    MMMCalibrationTreatmentDisposition,
    MMMCalibrationTreatmentLineage,
    MMMCalibrationTreatmentRecord,
)
from mmm.contracts.diagnostics_limitations import (
    MMMAffectedScope,
    MMMDiagnosticCategory,
    MMMDiagnosticRecord,
    MMMDiagnosticSeverity,
    MMMDiagnosticsLimitations,
    MMMDiagnosticStatus,
    MMMLimitationRecord,
    MMMTechnicalClaim,
    MMMTechnicalClaimDisposition,
)
from mmm.contracts.mip_failure import (
    MMMExportOutcome,
    MMMFailureCode,
    MMMFailurePacket,
    MMMFailureStage,
    MMMRemediationAction,
    MMMRetryDisposition,
)
from mmm.contracts.run_manifest import (
    MMMAnalyticalArtifactOutcome,
    MMMArtifactReference,
    MMMExportManifestOutcome,
    MMMRunManifest,
    MMMRunStatus,
    MMMRunStep,
    MMMRunStepStatus,
)
from mmm.contracts.semantics import (
    CalibrationEstimandSpec,
    ContributionInterpretation,
    ModelingTargetSpec,
    OptimizationSafetySpec,
)

if TYPE_CHECKING:
    from mmm.contracts.public_simulation import (
        MMM_PUBLIC_SIMULATION_ARTIFACT_KIND,
        MMM_PUBLIC_SIMULATION_SCHEMA_VERSION,
        MMMPublicSimulationExport,
        parse_mmm_public_simulation_export,
    )


_PUBLIC_SIMULATION_EXPORTS = frozenset(
    {
        "MMMPublicSimulationExport",
        "MMM_PUBLIC_SIMULATION_ARTIFACT_KIND",
        "MMM_PUBLIC_SIMULATION_SCHEMA_VERSION",
        "parse_mmm_public_simulation_export",
    }
)


def __getattr__(name: str) -> Any:
    """Load the planning-dependent public simulation contract only on demand."""
    if name not in _PUBLIC_SIMULATION_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from mmm.contracts import public_simulation

    value = getattr(public_simulation, name)
    globals()[name] = value
    return value


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
    "MMMPublicSimulationExport",
    "MMM_PUBLIC_SIMULATION_ARTIFACT_KIND",
    "MMM_PUBLIC_SIMULATION_SCHEMA_VERSION",
    "parse_mmm_public_simulation_export",
    "MMMRunManifest",
    "MMMRunStatus",
    "MMMRunStep",
    "MMMRunStepStatus",
    "MMMArtifactReference",
    "MMMAnalyticalArtifactOutcome",
    "MMMExportManifestOutcome",
    "MMMCalibrationTreatmentLineage",
    "MMMCalibrationTreatmentRecord",
    "MMMCalibrationTreatmentDisposition",
    "MMMCalibrationApplicationRole",
    "MMMCalibrationCompatibilityStatus",
    "MMMCalibrationFreshnessStatus",
    "MMMCalibrationTransformationStep",
    "MMMDiagnosticsLimitations",
    "MMMDiagnosticRecord",
    "MMMDiagnosticStatus",
    "MMMDiagnosticSeverity",
    "MMMDiagnosticCategory",
    "MMMLimitationRecord",
    "MMMTechnicalClaim",
    "MMMTechnicalClaimDisposition",
    "MMMAffectedScope",
]
