"""First-class planning scenario contract (media + sparse control overlays)."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from pydantic import BaseModel, Field, field_validator

from mmm.data.panel_order import sort_panel_for_modeling
from mmm.data.schema import PanelSchema
from mmm.planning.control_overlay import (
    ControlOverlaySpec,
    overlay_rows_canonical,
    overlay_spec_sha256,
    summarize_scenario_overlays,
)
from mmm.planning.spend_path import PiecewiseSpendPath


class PlanningScenarioMedia(BaseModel):
    baseline_spend: dict[str, float] | None = None
    baseline_spend_by_geo: dict[str, dict[str, float]] | None = None
    candidate_spend: dict[str, float] | None = None
    candidate_spend_by_geo: dict[str, dict[str, float]] | None = None
    candidate_spend_path: dict[str, Any] | None = None


class PlanningScenarioControls(BaseModel):
    control_overlay: dict[str, Any] | None = None
    control_overlay_baseline: dict[str, Any] | None = None
    control_overlay_plan: dict[str, Any] | None = None


class PlanningScenarioAssumptions(BaseModel):
    controls_assumption: str | None = None
    seasonality_assumption: str = "observed_panel"
    promo_assumption: str = "observed_panel_unless_overlay"
    macro_assumption: str = "observed_panel_unless_overlay"
    pricing_assumption: str = "observed_panel_unless_overlay"


class PlanningScenarioMetadata(BaseModel):
    created_at: str | None = None
    source: str | None = None
    owner: str | None = None
    source_path: str | None = None


class PlanningScenario(BaseModel):
    """
    Typed planning world for ``decide simulate`` / ``decide optimize-budget``.

  Wraps legacy YAML keys; does not replace sparse ``ControlOverlaySpec`` mechanics.
    """

    scenario_id: str
    scenario_version: str | None = None
    description: str | None = None
    media: PlanningScenarioMedia = Field(default_factory=PlanningScenarioMedia)
    controls: PlanningScenarioControls = Field(default_factory=PlanningScenarioControls)
    assumptions: PlanningScenarioAssumptions = Field(default_factory=PlanningScenarioAssumptions)
    metadata: PlanningScenarioMetadata = Field(default_factory=PlanningScenarioMetadata)
    #: Reserved for future multi-world planning (not implemented).
    scenarios: list[dict[str, Any]] | None = None
    scenario_weights: list[float] | None = None

    @field_validator("scenario_id")
    @classmethod
    def _non_empty_id(cls, v: str) -> str:
        s = str(v).strip()
        if not s:
            raise ValueError("scenario_id must be non-empty")
        return s

    def canonical_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=True)

    def scenario_hash(self) -> str:
        blob = json.dumps(self.canonical_dict(), sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()

    def overlay_specs(self) -> tuple[ControlOverlaySpec | None, ControlOverlaySpec | None, ControlOverlaySpec | None]:
        c = self.controls
        single = ControlOverlaySpec.from_dict(c.control_overlay) if c.control_overlay else None
        base = ControlOverlaySpec.from_dict(c.control_overlay_baseline) if c.control_overlay_baseline else None
        plan = ControlOverlaySpec.from_dict(c.control_overlay_plan) if c.control_overlay_plan else None
        return single, base, plan

    def spend_path(self) -> PiecewiseSpendPath | None:
        raw = self.media.candidate_spend_path
        if raw is None:
            return None
        return PiecewiseSpendPath.from_dict(raw)

    def validate_against_panel(
        self,
        panel: pd.DataFrame,
        schema: PanelSchema,
        *,
        control_columns: tuple[str, ...] | list[str],
        require_overlay_coverage: bool = False,
    ) -> list[str]:
        """Return warnings (empty if OK). Raises ValueError on hard failures."""
        issues: list[str] = []
        panel_s = sort_panel_for_modeling(panel, schema)
        ctrl_set = set(control_columns)
        single, base_ov, plan_ov = self.overlay_specs()
        for label, ov in (("control_overlay", single), ("control_overlay_baseline", base_ov), ("control_overlay_plan", plan_ov)):
            if ov is None:
                continue
            for r in ov.rows:
                col = str(r["column"])
                if col not in panel_s.columns:
                    raise ValueError(f"{label}: column {col!r} not in panel")
                if col not in ctrl_set and col not in schema.channel_columns:
                    issues.append(f"{label}: column {col!r} not listed in data.control_columns")
                geo = str(r["geo"])
                week = r["week"]
                gcol, wcol = schema.geo_column, schema.week_column
                m = panel_s[gcol].astype(str) == geo
                wseries = panel_s[wcol]
                if pd.api.types.is_numeric_dtype(wseries):
                    m_w = wseries.astype(float) == float(week)
                else:
                    m_w = pd.to_datetime(wseries, errors="coerce") == pd.to_datetime(week, errors="coerce")
                if not bool((m & m_w).any()):
                    raise ValueError(f"{label}: no panel row for geo={geo!r} week={week!r} column={col!r}")
        if require_overlay_coverage and not any(
            ov is not None and ov.rows for ov in (single, base_ov, plan_ov)
        ):
            raise ValueError("explicit_scenario requires at least one control overlay when strict policy is enabled")
        return issues

    def overlay_lineage(self, *, store_full_overlays: bool = False) -> dict[str, Any]:
        single, b, p = self.overlay_specs()
        summary = summarize_scenario_overlays(b, p)
        plan_ov = p or single
        out: dict[str, Any] = {
            "scenario_id": self.scenario_id,
            "scenario_version": self.scenario_version,
            "scenario_hash": self.scenario_hash(),
            "scenario_source_path": self.metadata.source_path,
            "control_overlay_summary": summary,
            "baseline_control_overlay_summary": summarize_scenario_overlays(b, None),
            "plan_control_overlay_summary": summarize_scenario_overlays(None, plan_ov),
            "overlay_row_count_baseline": len(b.rows) if b else 0,
            "overlay_row_count_plan": len(plan_ov.rows) if plan_ov else 0,
            "baseline_overlay_spec_sha256": overlay_spec_sha256(b),
            "plan_overlay_spec_sha256": overlay_spec_sha256(plan_ov),
            "control_overlay_spec_sha256": overlay_spec_sha256(single),
        }
        if store_full_overlays:
            if b and b.rows:
                out["baseline_control_overlay_spec"] = overlay_rows_canonical(b)
            if plan_ov and plan_ov.rows:
                out["plan_control_overlay_spec"] = overlay_rows_canonical(plan_ov)
            if single and single.rows and single is not plan_ov:
                out["control_overlay_spec"] = overlay_rows_canonical(single)
        return out

    def spend_summary(self) -> dict[str, Any]:
        m = self.media
        out: dict[str, Any] = {
            "has_candidate_spend": m.candidate_spend is not None,
            "has_candidate_spend_by_geo": m.candidate_spend_by_geo is not None,
            "has_candidate_spend_path": m.candidate_spend_path is not None,
            "has_baseline_spend": m.baseline_spend is not None,
            "has_baseline_spend_by_geo": m.baseline_spend_by_geo is not None,
        }
        if m.candidate_spend_path is not None:
            path_blob = json.dumps(m.candidate_spend_path, sort_keys=True, default=str)
            out["candidate_spend_path_hash"] = hashlib.sha256(path_blob.encode()).hexdigest()
        return out

    def lineage_payload(self, *, store_full_overlays: bool = False) -> dict[str, Any]:
        return {
            **self.overlay_lineage(store_full_overlays=store_full_overlays),
            "spend_scenario_summary": self.spend_summary(),
            "description": self.description,
            "metadata": self.metadata.model_dump(mode="json", exclude_none=True),
        }


def planning_scenario_from_yaml(path: str | Path) -> PlanningScenario:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("scenario YAML must be a mapping")
    return planning_scenario_from_dict(raw, source_path=str(path))


def planning_scenario_from_dict(raw: dict[str, Any], *, source_path: str | None = None) -> PlanningScenario:
    """Accept ``planning_scenario`` block or legacy flat keys."""
    if "planning_scenario" in raw and isinstance(raw["planning_scenario"], dict):
        block = dict(raw["planning_scenario"])
    else:
        block = _legacy_dict_to_scenario_block(raw)
    if source_path and "metadata" not in block:
        block["metadata"] = {}
    if source_path:
        meta = block.setdefault("metadata", {})
        if isinstance(meta, dict):
            meta.setdefault("source_path", source_path)
            meta.setdefault("source", "yaml")
    if not block.get("scenario_id"):
        block["scenario_id"] = f"legacy_{hashlib.sha256(json.dumps(raw, sort_keys=True, default=str).encode()).hexdigest()[:12]}"
    if not block.get("metadata", {}).get("created_at"):
        block.setdefault("metadata", {})["created_at"] = datetime.now(timezone.utc).isoformat()
    return PlanningScenario.model_validate(block)


def _legacy_dict_to_scenario_block(raw: dict[str, Any]) -> dict[str, Any]:
    media: dict[str, Any] = {}
    for k in (
        "baseline_spend",
        "baseline_spend_by_geo",
        "candidate_spend",
        "candidate_spend_by_geo",
        "candidate_spend_path",
    ):
        if k in raw:
            media[k] = raw[k]
    controls: dict[str, Any] = {}
    for k in ("control_overlay", "control_overlay_baseline", "control_overlay_plan", "controls_plan"):
        if k in raw:
            controls[k if k != "controls_plan" else "control_overlay"] = raw[k]
    sid = raw.get("scenario_id") or raw.get("id")
    return {
        "scenario_id": sid or "legacy_inline",
        "scenario_version": raw.get("scenario_version"),
        "description": raw.get("description"),
        "media": media,
        "controls": controls,
        "assumptions": raw.get("assumptions") or {},
        "metadata": raw.get("metadata") or {},
    }


def resolve_simulate_inputs(
    scenario: PlanningScenario,
    panel: pd.DataFrame,
    schema: PanelSchema,
) -> dict[str, Any]:
    """Map PlanningScenario → kwargs for decision_simulate / _scenario_simulate."""
    single, co_b, co_p = scenario.overlay_specs()
    m = scenario.media
    spend_path = scenario.spend_path()
    has_geo = m.candidate_spend_by_geo is not None
    has_scalar = m.candidate_spend is not None
    if spend_path is not None and has_geo:
        raise ValueError("candidate_spend_path cannot combine with candidate_spend_by_geo")
    if has_geo and has_scalar:
        raise ValueError("use either candidate_spend or candidate_spend_by_geo, not both")
    if spend_path is None and not has_scalar and not has_geo:
        raise ValueError("scenario must include candidate_spend, candidate_spend_by_geo, and/or candidate_spend_path")
    geos = sorted({str(x) for x in panel[schema.geo_column].unique()})
    from mmm.planning.baseline import channel_means_from_geo_plan, locked_geo_plan_baseline, locked_plan_baseline
    from mmm.planning.spend_path import counterfactual_piecewise_spend_panel, time_mean_spend_by_channel

    spend_plan_geo = None
    if spend_path is not None:
        if has_scalar and m.candidate_spend:
            cand = {str(k): float(v) for k, v in m.candidate_spend.items()}
        else:
            tmp_df = counterfactual_piecewise_spend_panel(panel, schema, spend_path)
            cand = time_mean_spend_by_channel(tmp_df, schema)
    elif has_geo:
        raw_geo = m.candidate_spend_by_geo or {}
        spend_plan_geo = {
            str(g): {str(c): float(v) for c, v in row.items()}
            for g, row in raw_geo.items()
            if isinstance(row, dict)
        }
        cand = channel_means_from_geo_plan(spend_plan_geo, schema, geos)
    else:
        cs = m.candidate_spend
        if not isinstance(cs, dict):
            raise ValueError("candidate_spend required")
        cand = {str(k): float(v) for k, v in cs.items()}
    base_plan = None
    if isinstance(m.baseline_spend, dict):
        base_plan = locked_plan_baseline(
            {str(k): float(v) for k, v in m.baseline_spend.items()},
            source="planning_scenario:baseline_spend",
            notes="baseline_spend from PlanningScenario.",
        )
    elif isinstance(m.baseline_spend_by_geo, dict):
        by_geo = {
            str(g): {str(c): float(v) for c, v in row.items()}
            for g, row in m.baseline_spend_by_geo.items()
            if isinstance(row, dict)
        }
        base_plan = locked_geo_plan_baseline(
            by_geo,
            source="planning_scenario:baseline_spend_by_geo",
            notes="baseline_spend_by_geo from PlanningScenario.",
        )
    co_plan = co_p or single
    return {
        "cand": cand,
        "base_plan": base_plan,
        "spend_path": spend_path,
        "spend_plan_geo": spend_plan_geo,
        "control_overlay_baseline": co_b,
        "control_overlay_plan": co_plan,
        "control_overlay": None,
        "controls_plan": None,
    }
