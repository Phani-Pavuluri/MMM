"""Shared prod extension_report fragments for decide-path tests."""

from __future__ import annotations

from typing import Any

import pandas as pd

from mmm.config.schema import CalibrationConfig, MMMConfig
from mmm.config.transform_policy import TRANSFORM_POLICY_VERSION, build_transform_policy_manifest
from mmm.data.fingerprint import fingerprint_panel
from mmm.data.schema import PanelSchema


def prod_replay_calibration_config(**kwargs: Any) -> CalibrationConfig:
    """Prod-safe replay calibration defaults (fold-aligned, not optimistic full-panel refit)."""
    base = {
        "use_replay_calibration": True,
        "replay_refit_mode": "fold_aligned",
    }
    base.update(kwargs)
    return CalibrationConfig(**base)


def prod_replay_evidence_block() -> dict[str, Any]:
    """Satisfies ``require_replay_calibration`` on prod decide paths."""
    return {"calibration_summary": {"replay_calibration_active": True, "replay_loss": 0.5}}


def merge_prod_extension(base: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    out.update(prod_replay_evidence_block())
    return out


def enrich_prod_ridge_decide_extension(
    base: dict[str, Any],
    *,
    cfg: MMMConfig,
    panel: pd.DataFrame,
    schema: PanelSchema,
) -> dict[str, Any]:
    """
    Add transform_policy, data_fingerprint, and ridge_fit_summary fields required for prod decide.
    """
    from mmm.data.loader import DatasetBuilder
    from mmm.data.panel_order import sort_panel_for_modeling
    from mmm.data.schema import validate_panel

    if cfg.data.path:
        builder = DatasetBuilder(cfg.data)
        schema = builder.schema()
        panel = sort_panel_for_modeling(validate_panel(builder.build(), schema), schema)

    out = merge_prod_extension(dict(base))
    tp = build_transform_policy_manifest(cfg)
    from mmm.contracts.seed_resolution import resolve_seed_contract

    seed_resolution = resolve_seed_contract(cfg)
    out["transform_policy"] = tp
    out["data_fingerprint"] = fingerprint_panel(panel, schema, config=cfg, seed_resolution=seed_resolution)
    rfs = out.get("ridge_fit_summary")
    if isinstance(rfs, dict):
        rfs = dict(rfs)
        rfs.setdefault("model_form", cfg.model_form.value)
        rfs.setdefault(
            "best_params",
            {"decay": 0.5, "hill_half": 1.0, "hill_slope": 2.0},
        )
        rfs.setdefault("intercept", [0.0])
        rfs.setdefault("transform_policy", tp)
        rfs.setdefault("data_fingerprint", out["data_fingerprint"])
        rfs.setdefault("model_form", cfg.model_form.value)
        out["ridge_fit_summary"] = rfs
    else:
        out["ridge_fit_summary"] = {
            "coef": [0.1],
            "intercept": [0.0],
            "model_form": cfg.model_form.value,
            "best_params": {"decay": 0.5, "hill_half": 1.0, "hill_slope": 2.0},
            "transform_policy": tp,
            "data_fingerprint": out["data_fingerprint"],
        }
    assert str(tp.get("policy_version") or TRANSFORM_POLICY_VERSION)
    return out
