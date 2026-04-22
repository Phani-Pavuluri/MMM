"""Transform plugin protocol."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class TransformBase(ABC):
    name: str

    @abstractmethod
    def fit(self, x: np.ndarray, **kwargs: Any) -> TransformBase:
        raise NotImplementedError

    @abstractmethod
    def transform(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def inverse(self, y: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def parameter_metadata(self) -> dict[str, Any]:
        return {}


class AdstockBase(TransformBase):
    """1D adstock along time within a single geo series."""

    pass


class SaturationBase(TransformBase):
    """Pointwise saturation."""

    pass
