"""Bootstrap economics / transform / decision uncertainty extension."""

from mmm.evaluation.extensions.builtin import _run_bootstrap
from mmm.evaluation.extensions.registry import ExtensionSpec, register_extension


def register() -> None:
    register_extension(
        ExtensionSpec(
            name="bootstrap",
            artifact_key="bootstrap",
            priority=10,
            run=_run_bootstrap,
            report_keys=("economics_contract", "transform_policy", "decision_uncertainty"),
            artifact_tier="research",
        )
    )
