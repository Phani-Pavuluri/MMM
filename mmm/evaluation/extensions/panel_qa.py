"""Panel QA extension registration."""

from mmm.evaluation.extensions.builtin import _run_panel_qa
from mmm.evaluation.extensions.registry import ExtensionSpec, register_extension


def register() -> None:
    register_extension(
        ExtensionSpec(
            name="panel_qa",
            artifact_key="panel_qa",
            priority=20,
            run=_run_panel_qa,
            report_keys=("panel_qa",),
        )
    )
