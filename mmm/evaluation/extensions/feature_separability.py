"""Feature separability extension registration."""

from mmm.evaluation.extensions.builtin import _run_feature_separability
from mmm.evaluation.extensions.registry import ExtensionSpec, register_extension


def register() -> None:
    register_extension(
        ExtensionSpec(
            name="feature_separability",
            artifact_key="feature_separability_report",
            priority=120,
            dependencies=("governance", "core_diagnostics"),
            run=_run_feature_separability,
            report_keys=("feature_separability_report",),
        )
    )
