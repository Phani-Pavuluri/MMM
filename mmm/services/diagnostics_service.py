from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from mmm.config.schema import MMMConfig
from mmm.data.quality import DataQualityEngine
from mmm.data.schema import PanelSchema
from mmm.diagnostics.geo_spillover import run_geo_spillover_diagnostics
from mmm.diagnostics.lag import LagDiagnostics
from mmm.falsification.engine import FalsificationEngine
from mmm.identifiability.engine import IdentifiabilityEngine
from mmm.utils.math import safe_log


def run_core_diagnostics(
    *,
    panel: pd.DataFrame,
    schema: PanelSchema,
    config: MMMConfig,
    X_media: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    out["data_quality"] = DataQualityEngine().run(panel, schema).to_json()
    y_log = safe_log(panel[schema.target_column].to_numpy(dtype=float))
    ext = config.extensions
    if ext.identifiability.enabled:
        id_eng = IdentifiabilityEngine(ext.identifiability)
        out["identifiability"] = id_eng.analyze(X_media, list(schema.channel_columns), y_log, rng).to_json()
    else:
        out["identifiability"] = {"skipped": True, "identifiability_score": 0.0, "instability_score": 0.0}
    out["lag_diagnostics"] = LagDiagnostics(schema).run(panel).to_json()
    out["geo_spillover"] = run_geo_spillover_diagnostics(panel, schema).to_json()
    out["falsification"] = FalsificationEngine(schema, ext.falsification).run(X_media, y_log, rng).to_json()
    return out
