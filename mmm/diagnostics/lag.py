"""E4: lag / alignment diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from mmm.data.schema import PanelSchema


@dataclass
class LagDiagnosticsReport:
    mean_abs_corr_lag1_spend: dict[str, float]
    warnings: list[str] = field(default_factory=list)

    def to_json(self) -> dict:
        return {"mean_abs_corr_lag1_spend": self.mean_abs_corr_lag1_spend, "warnings": self.warnings}


class LagDiagnostics:
    """Warn if lag-1 spend correlates unusually with KPI (adstock may be masking misalignment)."""

    def __init__(self, schema: PanelSchema) -> None:
        self.schema = schema

    def run(self, df: pd.DataFrame) -> LagDiagnosticsReport:
        warnings: list[str] = []
        corrs: dict[str, float] = {}
        y = df[self.schema.target_column].to_numpy(dtype=float)
        for ch in self.schema.channel_columns:
            lag_s = df.groupby(self.schema.geo_column)[ch].shift(1).fillna(0).to_numpy(dtype=float)
            c = float(np.corrcoef(y, lag_s)[0, 1]) if np.std(lag_s) > 1e-12 else 0.0
            corrs[ch] = abs(c)
        if np.mean(list(corrs.values())) > 0.35:
            warnings.append("heavy_lag1_spend_kpi_correlation: check measurement lag vs adstock compensation")
        return LagDiagnosticsReport(mean_abs_corr_lag1_spend=corrs, warnings=warnings)
