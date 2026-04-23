"""Fail-closed production checks for replay calibration units."""

from __future__ import annotations

import numpy as np

from mmm.calibration.contracts import CalibrationUnit
from mmm.calibration.replay_estimand import ReplayEstimandSpec
from mmm.config.schema import MMMConfig, RunEnvironment
from mmm.data.schema import PanelSchema
from mmm.economics.canonical import REPLAY_LIFT_SCALES_KPI_LEVEL
from mmm.experiments.durable_registry import get_experiment_from_registry, load_experiment_registry
from mmm.experiments.registry import ApprovalState


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
    _assert_replay_inverse_se_concentration(config, units)
    _assert_replay_experiment_registry_gate(config, units)


def _assert_replay_inverse_se_concentration(config: MMMConfig, units: list[CalibrationUnit]) -> None:
    if config.run_environment != RunEnvironment.PROD or len(units) < 2:
        return
    inv: list[float] = []
    for u in units:
        se = float(u.lift_se) if u.lift_se is not None and float(u.lift_se) > 0 else 1.0
        se = max(se, 1e-4)
        inv.append(1.0 / se)
    arr = np.asarray(inv, dtype=float)
    s = float(arr.sum())
    if s <= 0:
        return
    mx = float(arr.max() / s)
    cap = float(config.extensions.governance.replay_max_unit_inverse_se_influence_share)
    if mx > cap:
        raise ValueError(
            f"replay calibration inverse-SE concentration too high in prod (max share {mx:.3f} > {cap}); "
            "add experiments or fix underestimated lift_se values."
        )


def _assert_replay_experiment_registry_gate(config: MMMConfig, units: list[CalibrationUnit]) -> None:
    if config.run_environment != RunEnvironment.PROD:
        return
    if not config.calibration.require_approved_experiment_registry:
        return
    path = (config.calibration.experiment_registry_path or "").strip()
    if not path:
        raise ValueError(
            "calibration.require_approved_experiment_registry requires calibration.experiment_registry_path"
        )
    reg = load_experiment_registry(path)
    for u in units:
        eid = (u.experiment_id or "").strip()
        if not eid:
            raise ValueError(
                f"replay unit {u.unit_id!r}: experiment_id required when "
                "calibration.require_approved_experiment_registry is true in prod"
            )
        rec = get_experiment_from_registry(reg, eid)
        if rec is None:
            raise ValueError(f"replay unit {u.unit_id!r}: experiment_id {eid!r} not found in registry {path!r}")
        if rec.approval != ApprovalState.APPROVED:
            raise PermissionError(
                f"replay unit {u.unit_id!r}: experiment_id {eid!r} has approval={rec.approval.value!r}; "
                "expected approved in prod registry gate"
            )
