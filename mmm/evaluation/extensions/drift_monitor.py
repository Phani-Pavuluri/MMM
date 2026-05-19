"""Data fingerprint and drift monitor extension registration."""

from mmm.evaluation.extensions.builtin import _run_fingerprint_and_drift
from mmm.evaluation.extensions.registry import ExtensionSpec, register_extension


def register() -> None:
    register_extension(
        ExtensionSpec(
            name="fingerprint_drift",
            artifact_key="drift_report",
            priority=230,
            dependencies=("replay_calibration",),
            run=_run_fingerprint_and_drift,
            report_keys=("data_fingerprint", "drift_report"),
        )
    )
