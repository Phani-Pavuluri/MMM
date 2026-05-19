"""Experiment scheduler extension registration."""

from mmm.evaluation.extensions.builtin import _run_experiment_scheduler
from mmm.evaluation.extensions.registry import ExtensionSpec, register_extension


def register() -> None:
    register_extension(
        ExtensionSpec(
            name="experiment_scheduler",
            artifact_key="experiment_scheduler_report",
            priority=220,
            dependencies=("optimization_gate", "feature_separability"),
            run=_run_experiment_scheduler,
            report_keys=("experiment_scheduler_report",),
        )
    )
