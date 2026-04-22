"""E9: weak geo signal / pseudo-spillover warnings."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from mmm.data.schema import PanelSchema


@dataclass
class GeoSpilloverReport:
    cv_spend_by_geo: dict[str, float]
    warnings: list[str] = field(default_factory=list)

    def to_json(self) -> dict:
        return {"cv_spend_by_geo": self.cv_spend_by_geo, "warnings": self.warnings}


def run_geo_spillover_diagnostics(df: pd.DataFrame, schema: PanelSchema) -> GeoSpilloverReport:
    warnings: list[str] = []
    agg = df.groupby(schema.geo_column)[list(schema.channel_columns)].sum()
    cv = agg.std(axis=0) / (agg.mean(axis=0) + 1e-9)
    cv_map = {c: float(cv[c]) for c in schema.channel_columns}
    if agg.shape[0] >= 2 and float(np.mean(list(cv_map.values()))) < 0.05:
        warnings.append("low_geo_variation: geos may be too homogeneous for geo MMM")
    return GeoSpilloverReport(cv_spend_by_geo=cv_map, warnings=warnings)
