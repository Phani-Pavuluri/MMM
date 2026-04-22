"""Deterministic RNG helpers."""

from __future__ import annotations

import numpy as np


class SeededRNG:
    """Centralized RNG for reproducible runs."""

    def __init__(self, seed: int) -> None:
        self.seed = int(seed)
        self._rng = np.random.default_rng(self.seed)

    def numpy(self) -> np.random.Generator:
        return self._rng

    def integers(self, low: int, high: int, size: int) -> np.ndarray:
        return self._rng.integers(low, high, size=size)
