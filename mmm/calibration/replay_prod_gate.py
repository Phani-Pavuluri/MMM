"""Fail-closed production checks for replay calibration (legacy units and evidence registry)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from mmm.calibration.contracts import CalibrationUnit
from mmm.calibration.replay_estimand import ReplayEstimandSpec
from mmm.config.schema import MMMConfig, RunEnvironment
from mmm.data.schema import PanelSchema
from mmm.economics.canonical import REPLAY_LIFT_SCALES_KPI_LEVEL
from mmm.evaluation.experiment_evidence_extension import load_evidence_from_path
from mmm.experiments.durable_registry import get_experiment_from_registry, load_experiment_registry
from mmm.experiments.readiness import experiment_readiness
from mmm.experiments.registry import ApprovalState

_ACCEPTABLE_QUALITY_TIERS = frozenset({"high", "medium"})
_FORBIDDEN_COMPAT_FOR_OBJECTIVE = frozenset({"rejected", "diagnostic_only"})
_AGGREGATE_COMPAT_STATUSES = frozenset({"aggregate_only", "allocation_required"})
_BRIDGE_ROLE = "computational_bridge_only"


def _uses_evidence_registry_replay(config: MMMConfig) -> bool:
    cal = config.calibration
    return bool(cal.use_replay_calibration and cal.replay_mode == "evidence_registry")


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
    evidence_summary: dict[str, Any] | None = None,
) -> None:
    """
    In ``run_environment=prod``, validate replay readiness.

    - ``replay_mode=legacy``: per-unit replay contract (existing rules).
    - ``replay_mode=evidence_registry``: requires ``evidence_weighted_replay_summary`` + evidence rules.
    """
    if config.run_environment != RunEnvironment.PROD:
        return
    if not config.calibration.use_replay_calibration:
        return
    if _uses_evidence_registry_replay(config):
        assert_evidence_registry_replay_production_ready(
            config,
            evidence_summary,
            schema=schema,
            units=units,
        )
        return
    if not units:
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


def assert_evidence_registry_replay_production_ready(
    config: MMMConfig,
    evidence_summary: dict[str, Any] | None,
    *,
    schema: PanelSchema | None,
    units: list[CalibrationUnit] | None = None,
) -> None:
    """
    Prod fail-closed gate for evidence-registry replay (calibration evidence, not causal proof).

    Requires ``evidence_weighted_replay_summary`` on the extension report or an equivalent dict
    passed at train time after ``prepare_evidence_replay``.
    """
    if config.run_environment != RunEnvironment.PROD:
        return
    if not _uses_evidence_registry_replay(config):
        return
    if schema is None:
        raise ValueError(
            "schema is required for prod evidence-registry replay validation "
            "(KPI / channel alignment vs economics contract)"
        )
    if not isinstance(evidence_summary, dict) or not evidence_summary:
        raise ValueError(
            "prod evidence-registry replay requires non-empty evidence_weighted_replay_summary "
            "on extension_report (or evidence_summary argument at train time)"
        )
    n_used = int(evidence_summary.get("n_evidence_units_used") or 0)
    if n_used < 1:
        raise ValueError(
            "prod evidence-registry replay requires n_evidence_units_used >= 1; "
            f"got {n_used} (rejected={evidence_summary.get('rejected_evidence_reasons')!r})"
        )
    if str(evidence_summary.get("replay_mode_used", "")) != "evidence_registry":
        raise ValueError(
            "evidence_weighted_replay_summary.replay_mode_used must be 'evidence_registry' in prod; "
            f"got {evidence_summary.get('replay_mode_used')!r}"
        )

    unit_rows: list[dict[str, Any]] = list(evidence_summary.get("unit_governance") or [])
    if not unit_rows and units:
        unit_rows = [
            {
                "experiment_id": u.experiment_id or u.unit_id,
                "channel": (u.treated_channel_names or [""])[0],
                "quality_tier": "unknown",
                "compatibility_status": "unknown",
                "supports_subgeo_claims": True,
                "allocation_role": "",
                "allocation_required": False,
                "lift_se": u.lift_se,
            }
            for u in units
        ]

    has_acceptable_quality = False
    for row in unit_rows:
        eid = str(row.get("experiment_id", ""))
        tier = str(row.get("quality_tier", "")).lower()
        compat = str(row.get("compatibility_status", "")).lower()
        if tier in _ACCEPTABLE_QUALITY_TIERS:
            has_acceptable_quality = True
        if compat in _FORBIDDEN_COMPAT_FOR_OBJECTIVE:
            raise ValueError(
                f"prod evidence-registry replay: unit {eid!r} has forbidden compatibility_status={compat!r} "
                "for objective-eligible evidence"
            )
        if compat in _AGGREGATE_COMPAT_STATUSES and bool(row.get("supports_subgeo_claims")):
            raise ValueError(
                f"prod evidence-registry replay: aggregate-only unit {eid!r} must declare "
                "supports_subgeo_claims=false (no DMA/subgeo claims from national or allocated evidence)"
            )
        if bool(row.get("allocation_required")) or (
            compat == "allocation_required"
            and str(row.get("allocation_method", "none")) not in {"", "none"}
        ):
            role = str(row.get("allocation_role", ""))
            if role != _BRIDGE_ROLE:
                raise ValueError(
                    f"prod evidence-registry replay: allocated shock for {eid!r} must declare "
                    f"allocation_role={_BRIDGE_ROLE!r}; got {role!r}"
                )
        lift_se = row.get("lift_se")
        allow_missing = bool(config.calibration.allow_missing_se_in_prod_evidence_replay)
        if not allow_missing and (lift_se is None or float(lift_se) <= 0):
            raise ValueError(
                f"prod evidence-registry replay: unit {eid!r} requires positive lift_se "
                "(set calibration.allow_missing_se_in_prod_evidence_replay=true to override)"
            )

    if not has_acceptable_quality:
        raise ValueError(
            "prod evidence-registry replay requires at least one used unit with quality_tier "
            f"in {sorted(_ACCEPTABLE_QUALITY_TIERS)}; got unit_governance={unit_rows!r}"
        )

    _assert_required_channels_have_usable_evidence(config, schema, evidence_summary, unit_rows)
    if units:
        validate_replay_units_economics_alignment(config, schema, units)
        _assert_replay_inverse_se_concentration(config, units)


def _assert_required_channels_have_usable_evidence(
    config: MMMConfig,
    schema: PanelSchema,
    summary: dict[str, Any],
    unit_rows: list[dict[str, Any]],
) -> None:
    """Channels with loaded registry evidence must not be critically rejected without a used unit."""
    path = (config.calibration.evidence_registry_path or "").strip()
    if not path or not Path(path).exists():
        return
    ch_map = dict(config.calibration.channel_mapping or {})
    loaded_by_channel: dict[str, list[str]] = {}
    for ev in load_evidence_from_path(path):
        raw = ev.channel
        mapped = ch_map.get(raw, raw)
        if mapped in schema.channel_columns:
            loaded_by_channel.setdefault(mapped, []).append(ev.experiment_id)

    used_channels = {str(r.get("channel", "")) for r in unit_rows}
    rejected = summary.get("rejected_evidence_reasons") or []
    critical_reasons = frozenset(
        {
            "channel_not_in_model",
            "kpi_mismatch",
            "geo_scope_no_overlap",
            "time_window_no_overlap",
            "missing_or_invalid_standard_error",
            "quality_rejected",
            "shock_plan_rejected",
        }
    )
    for ch, eids in loaded_by_channel.items():
        if ch in used_channels:
            continue
        critical = [
            r
            for r in rejected
            if isinstance(r, dict)
            and str(r.get("experiment_id", "")) in eids
            and str(r.get("reason", "")) in critical_reasons
        ]
        if critical:
            raise ValueError(
                f"prod evidence-registry replay: channel {ch!r} has loaded evidence but no used unit "
                f"and critical rejections {critical!r}"
            )


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
        ready = experiment_readiness(rec)
        if not ready.get("ready"):
            raise PermissionError(
                f"replay unit {u.unit_id!r}: experiment_id {eid!r} failed experiment_readiness: "
                f"{ready.get('reasons')}; prod requires approved + payload_signature + calibration_artifact_ref"
            )
