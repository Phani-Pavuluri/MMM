"""Curve vs decision alignment extension registration."""

from mmm.evaluation.extensions.handlers.reporting_governance import _run_curve_decision_alignment
from mmm.evaluation.extensions.registry import ExtensionSpec, register_extension


def register() -> None:
    register_extension(
        ExtensionSpec(
            name="curve_decision_alignment",
            artifact_key="curve_decision_alignment",
            priority=240,
            dependencies=("curves",),
            run=_run_curve_decision_alignment,
            report_keys=("curve_decision_alignment",),
            artifact_tier="diagnostic",
        )
    )
