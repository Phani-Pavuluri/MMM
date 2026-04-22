"""DataLoader / DatasetBuilder — load CSV/Parquet and enforce schema."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from mmm.config.schema import DataConfig
from mmm.data.schema import PanelSchema, validate_panel


@dataclass
class DataLoader:
    """Load tabular inputs into DataFrame."""

    config: DataConfig

    def load(self) -> pd.DataFrame:
        if not self.config.path:
            raise ValueError("data.path is required for DataLoader.load()")
        p = Path(self.config.path)
        if not p.exists():
            raise FileNotFoundError(p)
        if p.suffix.lower() in {".parquet", ".pq"}:
            return pd.read_parquet(p)
        if p.suffix.lower() in {".csv"}:
            return pd.read_csv(p, parse_dates=[self.config.week_column] if self._week_is_date() else None)
        raise ValueError(f"Unsupported file type: {p.suffix}")

    def _week_is_date(self) -> bool:
        # Heuristic: column name suggests date; actual dtype checked post-load
        return "date" in self.config.week_column.lower() or "week" in self.config.week_column.lower()


class DatasetBuilder:
    """Build validated internal panel from config + raw frame."""

    def __init__(self, data_cfg: DataConfig, schema: PanelSchema | None = None) -> None:
        self.data_cfg = data_cfg
        self._schema = schema

    def schema(self) -> PanelSchema:
        if self._schema is not None:
            return self._schema
        return PanelSchema(
            geo_column=self.data_cfg.geo_column,
            week_column=self.data_cfg.week_column,
            target_column=self.data_cfg.target_column,
            channel_columns=tuple(self.data_cfg.channel_columns),
            control_columns=tuple(self.data_cfg.control_columns),
        )

    def build(self, df: pd.DataFrame | None = None) -> pd.DataFrame:
        if df is None:
            df = DataLoader(self.data_cfg).load()
        return validate_panel(df, self.schema())
