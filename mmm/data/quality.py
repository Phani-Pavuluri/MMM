"""E16: tabular data quality checks."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from mmm.data.schema import PanelSchema


@dataclass
class DataQualityReport:
    missing_rate: dict[str, float]
    warnings: list[str] = field(default_factory=list)
    fail: bool = False

    def to_json(self) -> dict:
        return {"missing_rate": self.missing_rate, "warnings": self.warnings, "fail": self.fail}


class DataQualityEngine:
    def run(self, df: pd.DataFrame, schema: PanelSchema) -> DataQualityReport:
        warns: list[str] = []
        miss = {c: float(df[c].isna().mean()) for c in df.columns}
        for ch in schema.channel_columns:
            if (df[ch] < 0).any():
                warns.append(f"negative_spend:{ch}")
        if miss.get(schema.target_column, 0) > 0:
            warns.append("target_has_missing")
        fail = miss.get(schema.target_column, 0) > 0.01
        return DataQualityReport(missing_rate=miss, warnings=warns, fail=fail)
