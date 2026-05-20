"""Control governance diagnostics extension (guidance only)."""

from mmm.evaluation.extensions.handlers.reporting_governance import _run_control_governance
from mmm.evaluation.extensions.registry import ExtensionSpec, register_extension


def register() -> None:
    register_extension(
        ExtensionSpec(
            name="control_governance",
            artifact_key="control_governance",
            priority=115,
            dependencies=("panel_qa",),
            run=_run_control_governance,
            report_keys=("control_governance",),
        )
    )
