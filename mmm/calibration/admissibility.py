"""Hard admissibility rules for experiment rows before calibration matching weights apply."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mmm.calibration.schema import ExperimentObservation

if TYPE_CHECKING:
    from mmm.config.schema import RunEnvironment


def _is_prod_run_environment(run_environment: object | None) -> bool:
    if run_environment is None:
        return False
    v = getattr(run_environment, "value", run_environment)
    return str(v) == "prod"


def experiment_admissibility_violations(
    ex: ExperimentObservation,
    *,
    run_environment: RunEnvironment | None,
) -> list[str]:
    """
    Return human-readable violation codes (empty if admissible).

    In prod, inadmissible rows must never enter the matched calibration set.
    """
    bad: list[str] = []
    if not str(ex.experiment_id or "").strip():
        bad.append("missing_experiment_id")
    if not str(ex.channel or "").strip():
        bad.append("missing_channel")
    if ex.lift is None or not isinstance(ex.lift, (int, float)):
        bad.append("lift_not_numeric")
    elif ex.lift != ex.lift:  # NaN
        bad.append("lift_nan")
    if _is_prod_run_environment(run_environment):
        if ex.lift_se is None or (isinstance(ex.lift_se, (int, float)) and float(ex.lift_se) <= 0):
            bad.append("prod_requires_positive_finite_lift_se")
        elif isinstance(ex.lift_se, float) and ex.lift_se != ex.lift_se:
            bad.append("lift_se_nan")
    return bad
