"""Fail-closed production checks for replay calibration units."""

from __future__ import annotations

from mmm.calibration.contracts import CalibrationUnit
from mmm.calibration.replay_estimand import ReplayEstimandSpec
from mmm.config.schema import MMMConfig, RunEnvironment
from mmm.data.schema import PanelSchema
from mmm.economics.canonical import REPLAY_LIFT_SCALES_KPI_LEVEL


def validate_replay_units_economics_alignment(
    config: MMMConfig,
    schema: PanelSchema,
    units: list[CalibrationUnit],
) -> None:
    """Replay estimand KPI, lift scale, and channels must match training economics contract."""
    target = schema.target_column
    exp = (config.calibration.experiment_target_kpi or "").strip()
    if exp and exp != target:
        raise ValueError(
            f"calibration.experiment_target_kpi {exp!r} must match schema.target_column {target!r} "
            "for replay economics alignment"
        )
    channels = set(schema.channel_columns)
    for u in units:
        if not u.replay_estimand:
            continue
        spec = ReplayEstimandSpec.from_dict(u.replay_estimand)
        if spec.target_kpi_column != target:
            raise ValueError(
                f"replay unit {u.unit_id!r}: replay_estimand.target_kpi_column {spec.target_kpi_column!r} "
                f"must match schema.target_column / economics contract KPI {target!r}"
            )
        if (u.target_kpi or "").strip() and u.target_kpi.strip() != target:
            raise ValueError(
                f"replay unit {u.unit_id!r}: unit.target_kpi {u.target_kpi!r} must match "
                f"schema.target_column {target!r}"
            )
        if u.lift_scale and u.lift_scale not in REPLAY_LIFT_SCALES_KPI_LEVEL:
            raise ValueError(
                f"replay unit {u.unit_id!r}: lift_scale {u.lift_scale!r} not in supported KPI-level scales "
                f"{sorted(REPLAY_LIFT_SCALES_KPI_LEVEL)} (align with predict_fn level outputs)"
            )
        if spec.lift_scale not in REPLAY_LIFT_SCALES_KPI_LEVEL:
            raise ValueError(
                f"replay unit {u.unit_id!r}: replay_estimand.lift_scale {spec.lift_scale!r} not in "
                f"{sorted(REPLAY_LIFT_SCALES_KPI_LEVEL)}"
            )
        for ch in u.treated_channel_names:
            if ch not in channels:
                raise ValueError(
                    f"replay unit {u.unit_id!r}: treated channel {ch!r} not in schema.channel_columns "
                    f"for this model (economics scope mismatch)"
                )


def assert_replay_production_ready(
    config: MMMConfig,
    units: list[CalibrationUnit],
    *,
    schema: PanelSchema | None = None,
) -> None:
    """
    In ``run_environment=prod``, replay units must carry estimand, lift scale, positive SE, frames,
    and **economics-aligned** replay_estimand (requires ``schema`` when units are non-empty).
    """
    if config.run_environment != RunEnvironment.PROD:
        return
    if not config.calibration.use_replay_calibration or not units:
        return
    if schema is None:
        raise ValueError(
            "schema is required for production replay validation (KPI / channel alignment vs economics contract)"
        )
    for u in units:
        if not (u.estimand or "").strip():
            raise ValueError(f"replay unit {u.unit_id!r}: estimand is required in prod")
        if not (u.lift_scale or "").strip():
            raise ValueError(f"replay unit {u.unit_id!r}: lift_scale is required in prod (e.g. mean_kpi_level_delta)")
        if u.observed_lift is None:
            raise ValueError(f"replay unit {u.unit_id!r}: observed_lift required in prod")
        if u.lift_se is None or float(u.lift_se) <= 0.0:
            raise ValueError(f"replay unit {u.unit_id!r}: positive lift_se required in prod")
        if u.observed_spend_frame is None or u.counterfactual_spend_frame is None:
            raise ValueError(f"replay unit {u.unit_id!r}: observed and counterfactual spend frames required in prod")
        if not u.replay_estimand:
            raise ValueError(f"replay unit {u.unit_id!r}: replay_estimand object required in prod")
        try:
            ReplayEstimandSpec.from_dict(u.replay_estimand)
        except ValueError as e:
            raise ValueError(f"replay unit {u.unit_id!r}: invalid replay_estimand: {e}") from e
    validate_replay_units_economics_alignment(config, schema, units)
