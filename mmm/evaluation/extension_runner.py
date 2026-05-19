"""Post-fit diagnostics — orchestrates registered extensions only."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from mmm.artifacts.base import ArtifactStoreBase
from mmm.config.schema import MMMConfig
from mmm.contracts.seed_resolution import resolve_seed_contract
from mmm.data.panel_order import sort_panel_for_modeling
from mmm.data.schema import PanelSchema
from mmm.evaluation.extensions.context import ExtensionContext
from mmm.evaluation.extensions.registry import get_extension_registry


def run_post_fit_extensions(
    *,
    panel: pd.DataFrame,
    schema: PanelSchema,
    config: MMMConfig,
    fit_out: dict[str, Any],
    yhat: np.ndarray,
    store: ArtifactStoreBase | None,
) -> dict[str, Any]:
    """Run all registered post-fit extensions in deterministic order."""
    seed_resolution = resolve_seed_contract(config)
    panel_s = sort_panel_for_modeling(panel, schema)
    ctx = ExtensionContext(
        panel=panel,
        panel_s=panel_s,
        schema=schema,
        config=config,
        fit_out=fit_out,
        yhat=yhat,
        store=store,
        out={"seed_resolution": seed_resolution},
        rng=np.random.default_rng(int(config.extension_seed)),
        ext=config.extensions,
        seed_resolution=seed_resolution,
    )
    get_extension_registry().run_all(ctx)
    if store:
        store.log_dict("extension_report", ctx.out)
    return ctx.out
