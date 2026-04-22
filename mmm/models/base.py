"""Model interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd

from mmm.config.schema import MMMConfig
from mmm.data.schema import PanelSchema


class MMMModelBase(ABC):
    config: MMMConfig
    schema: PanelSchema

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError


class RidgeBOMMMBase(MMMModelBase):
    pass


class BayesianMMMBase(MMMModelBase):
    pass
