"""CmdStanPy backend — optional; raises informative error until model assets exist."""

from __future__ import annotations

import pandas as pd

from mmm.config.schema import BayesianBackend, Framework, MMMConfig
from mmm.data.schema import PanelSchema
from mmm.models.base import BayesianMMMBase


class StanMMMTrainer(BayesianMMMBase):
    def __init__(self, config: MMMConfig, schema: PanelSchema) -> None:
        if config.framework != Framework.BAYESIAN:
            raise ValueError("framework must be bayesian")
        if config.bayesian.backend != BayesianBackend.STAN:
            raise ValueError("backend must be stan")
        self.config = config
        self.schema = schema

    def fit(self, df: pd.DataFrame) -> dict:
        try:
            import cmdstanpy  # noqa: F401
        except ImportError as e:  # pragma: no cover
            raise ImportError("Install stan extra: pip install mmm[stan]") from e
        raise NotImplementedError(
            "Stan backend requires checked-in .stan programs and packaging; use PyMC backend for now."
        )

    def predict(self, df: pd.DataFrame):
        raise NotImplementedError
