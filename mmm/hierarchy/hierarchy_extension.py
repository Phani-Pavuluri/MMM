"""Load and validate hierarchy definitions for Ridge extensions."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from mmm.config.schema import MMMConfig
from mmm.data.schema import PanelSchema
from mmm.governance.model_form_policy import assert_log_log_hierarchy_blocked
from mmm.hierarchy.diagnostics import build_hierarchy_extension_reports, hierarchy_enabled
from mmm.hierarchy.hierarchy_definition import HierarchyDefinition, load_hierarchy_definition
from mmm.hierarchy.penalty import prepare_hierarchy_for_ridge


def load_and_validate_hierarchy(
    config: MMMConfig,
    schema: PanelSchema,
    panel: pd.DataFrame,
) -> tuple[HierarchyDefinition | None, Any, list[Any], list[str]]:
    """
    Returns ``(definition, validation_report, coef_pairs, warnings)``.

    ``validation_report`` is a :class:`HierarchyValidationReport` or ``None`` when disabled.
    """
    if not hierarchy_enabled(config):
        return None, None, [], []
    assert_log_log_hierarchy_blocked(config)
    path = config.hierarchy.hierarchy_definition_path
    if not path:
        raise ValueError(
            "hierarchy.enabled requires hierarchy.hierarchy_definition_path (explicit JSON; "
            "hierarchy is never inferred from panel data)"
        )
    definition = load_hierarchy_definition(path)
    if config.hierarchy.hierarchy_type and definition.hierarchy_type != config.hierarchy.hierarchy_type:
        pass  # config type is advisory default; definition file is source of truth
    geos = {str(x) for x in panel[schema.geo_column].unique()}
    pairs, report, warnings = prepare_hierarchy_for_ridge(
        definition,
        list(schema.channel_columns),
        panel_geos=geos,
        min_children_per_parent=config.hierarchy.min_children_per_parent,
        allow_cross_branch_pooling=config.hierarchy.allow_cross_branch_pooling,
    )
    if not report.valid:
        raise ValueError(
            f"hierarchy definition failed validation: {report.model_dump()}"
        )
    return definition, report, pairs, warnings


def build_hierarchy_reports_for_fit(
    config: MMMConfig,
    schema: PanelSchema,
    panel: pd.DataFrame,
    coef: np.ndarray,
    *,
    coef_before: np.ndarray | None = None,
) -> dict[str, Any]:
    if not hierarchy_enabled(config):
        return {"skipped": True, "reason": "hierarchy_disabled"}
    definition, report, pairs, warnings = load_and_validate_hierarchy(config, schema, panel)
    assert definition is not None and report is not None
    return build_hierarchy_extension_reports(
        config,
        definition,
        report,
        pairs,
        coef,
        coef_before=coef_before,
        extra_warnings=warnings,
    )
