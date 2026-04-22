"""Hard validation of experiment / replay scope vs MMM panel (Tier 1 decision safety)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from mmm.calibration.contracts import CalibrationUnit
from mmm.calibration.replay_etl import SpendShiftSpec
from mmm.calibration.schema import ExperimentObservation
from mmm.data.schema import PanelSchema


@dataclass
class ValidationIssue:
    code: str
    message: str
    severity: str  # "error" | "warning"


@dataclass
class ExperimentValidationReport:
    """If any ``error`` issues, the experiment must not drive replay or calibration."""

    experiment_id: str
    accepted: bool
    issues: list[ValidationIssue] = field(default_factory=list)

    def add_error(self, code: str, message: str) -> None:
        self.issues.append(ValidationIssue(code, message, "error"))
        self.accepted = False

    def add_warning(self, code: str, message: str) -> None:
        self.issues.append(ValidationIssue(code, message, "warning"))

    def to_json(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "accepted": self.accepted,
            "issues": [{"code": i.code, "message": i.message, "severity": i.severity} for i in self.issues],
        }


def _norm_kpi(a: str | None, b: str | None) -> bool:
    if not a or not b:
        return True
    return str(a).strip().lower() == str(b).strip().lower()


def _panel_week_bounds(panel: pd.DataFrame, schema: PanelSchema) -> tuple[Any, Any]:
    w = panel[schema.week_column]
    if pd.api.types.is_numeric_dtype(w):
        return float(w.min()), float(w.max())
    wt = pd.to_datetime(w, errors="coerce")
    return wt.min(), wt.max()


def _interval_overlaps_panel(
    start: Any, end: Any, lo: Any, hi: Any, panel: pd.DataFrame, schema: PanelSchema
) -> bool:
    """True if experiment [start,end] overlaps panel week range [lo, hi]."""
    wcol = panel[schema.week_column]
    if pd.api.types.is_numeric_dtype(wcol):
        try:
            a, b = float(start), float(end)
            return a <= float(hi) and b >= float(lo)
        except (TypeError, ValueError):
            return False
    a0 = pd.to_datetime(start, errors="coerce")
    a1 = pd.to_datetime(end, errors="coerce")
    return bool(a0 <= hi and a1 >= lo)


def validate_spend_shift_against_panel(
    sp: SpendShiftSpec,
    panel: pd.DataFrame,
    schema: PanelSchema,
    *,
    expected_target_kpi: str | None = None,
    unit_kpi: str | None = None,
) -> ExperimentValidationReport:
    rep = ExperimentValidationReport(experiment_id=sp.unit_id, accepted=True)
    gcol = schema.geo_column
    if not sp.geo_ids:
        rep.add_error("geo_missing", "experiment must list at least one geo_id for replay scope")
    panel_geos = set(panel[gcol].astype(str).unique())
    for g in sp.geo_ids:
        if str(g) not in panel_geos:
            rep.add_error("geo_mismatch", f"geo_id {g!r} not in panel {gcol} universe")
    if sp.channel not in schema.channel_columns:
        rep.add_error("channel_mismatch", f"channel {sp.channel!r} not in MMM channel_columns")
    if sp.channel not in panel.columns:
        rep.add_error("channel_missing_in_panel", f"column {sp.channel!r} missing from panel")
    lo, hi = _panel_week_bounds(panel, schema)
    if not _interval_overlaps_panel(sp.week_start, sp.week_end, lo, hi, panel, schema):
        rep.add_error(
            "time_window_mismatch",
            f"experiment window [{sp.week_start!r}, {sp.week_end!r}] does not overlap panel range [{lo}, {hi}]",
        )
    if expected_target_kpi and unit_kpi and not _norm_kpi(unit_kpi, expected_target_kpi):
        rep.add_error(
            "kpi_mismatch",
            f"experiment target_kpi {unit_kpi!r} != MMM target {expected_target_kpi!r}",
        )
    return rep


def validate_calibration_unit_against_panel(
    unit: CalibrationUnit,
    panel: pd.DataFrame,
    schema: PanelSchema,
    *,
    expected_target_kpi: str | None = None,
) -> ExperimentValidationReport:
    rep = ExperimentValidationReport(experiment_id=unit.unit_id, accepted=True)
    if expected_target_kpi and unit.target_kpi and not _norm_kpi(unit.target_kpi, expected_target_kpi):
        rep.add_error(
            "kpi_mismatch",
            f"unit.target_kpi {unit.target_kpi!r} != expected {expected_target_kpi!r}",
        )
    if unit.observed_spend_frame is None or unit.counterfactual_spend_frame is None:
        rep.add_error("missing_frames", "CalibrationUnit missing observed or counterfactual spend frames")
        return rep
    obs = unit.observed_spend_frame
    for g in unit.geo_ids:
        if g not in set(panel[schema.geo_column].astype(str).unique()):
            rep.add_error("geo_mismatch", f"geo_id {g!r} not in training panel")
    for ch in unit.treated_channel_names:
        if ch not in schema.channel_columns:
            rep.add_error("channel_mismatch", f"treated channel {ch!r} not in MMM channel_columns")
    if schema.geo_column not in obs.columns or schema.week_column not in obs.columns:
        rep.add_error("frame_schema", "observed_spend_frame missing geo_column or week_column")
    return rep


def validate_experiment_observation_against_panel(
    obs: ExperimentObservation,
    panel: pd.DataFrame,
    schema: PanelSchema,
    *,
    expected_target_kpi: str | None = None,
) -> ExperimentValidationReport:
    """Validate a lift-table row for scope (geo/time/channel); does not check lift magnitudes."""
    rep = ExperimentValidationReport(experiment_id=obs.experiment_id, accepted=True)
    if obs.geo_id and str(obs.geo_id) not in set(panel[schema.geo_column].astype(str).unique()):
        rep.add_error("geo_mismatch", f"geo_id {obs.geo_id!r} not in panel")
    if obs.channel not in schema.channel_columns:
        rep.add_error("channel_mismatch", f"channel {obs.channel!r} not in MMM channel_columns")
    lo, hi = _panel_week_bounds(panel, schema)
    if obs.start_week and obs.end_week and not _interval_overlaps_panel(
        obs.start_week, obs.end_week, lo, hi, panel, schema
    ):
        rep.add_error(
            "time_window_mismatch",
            f"experiment window does not overlap panel range [{lo}, {hi}]",
        )
    if expected_target_kpi and getattr(obs, "metadata", None):
        mk = (obs.metadata or {}).get("target_kpi")
        if mk and not _norm_kpi(mk, expected_target_kpi):
            rep.add_error("kpi_mismatch", f"metadata.target_kpi {mk!r} != {expected_target_kpi!r}")
    return rep
