"""Governance extension registration."""

from mmm.evaluation.extensions.builtin import _run_governance
from mmm.evaluation.extensions.registry import ExtensionSpec, register_extension


def register() -> None:
    register_extension(
        ExtensionSpec(
            name="governance",
            artifact_key="governance",
            priority=110,
            dependencies=("baselines", "core_diagnostics", "replay_calibration"),
            run=_run_governance,
            report_keys=("governance",),
        )
    )
