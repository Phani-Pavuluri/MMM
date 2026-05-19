"""Shared execution context for post-fit extension plugins."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from mmm.artifacts.base import ArtifactStoreBase
from mmm.config.extensions import ExtensionSuiteConfig
from mmm.config.schema import MMMConfig
from mmm.data.schema import PanelSchema


@dataclass
class ExtensionContext:
    """Mutable pipeline state passed to every registered extension."""

    panel: pd.DataFrame
    panel_s: pd.DataFrame
    schema: PanelSchema
    config: MMMConfig
    fit_out: dict[str, Any]
    yhat: np.ndarray
    store: ArtifactStoreBase | None
    out: dict[str, Any]
    rng: np.random.Generator
    ext: ExtensionSuiteConfig
    seed_resolution: dict[str, Any]
    X_media: np.ndarray | None = None
    x_shuf: np.ndarray | None = None
    curve_bundle: dict[str, Any] = field(default_factory=dict)
    baselines_result: Any = None
    gate_result: Any = None
    fingerprint: dict[str, Any] | None = None
    is_replay: bool = False
    replay_meta: Any = None
    cal_loss: float | None = None
