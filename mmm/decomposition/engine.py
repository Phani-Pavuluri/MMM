"""Contribution decomposition on transformed media features."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

from mmm.config.schema import MMMConfig, ModelForm
from mmm.data.schema import PanelSchema
from mmm.economics.canonical import economics_output_metadata, validate_business_economics_metadata
from mmm.features.design_matrix import build_design_matrix


@dataclass
class DecompositionResult:
    channel_contributions: pd.DataFrame  # rows aligned to df.index
    total_media: np.ndarray
    scale: Literal["log_surrogate", "level_approx"] = "log_surrogate"
    is_exact_additive: bool = False
    safe_for_budgeting: bool = False
    notes: list[str] = field(default_factory=list)
    economics_output_metadata: dict[str, Any] | None = None


class DecompositionEngine:
    """Default: additive on modeling scale (log for semi-log -> optional exp for reporting)."""

    def __init__(self, schema: PanelSchema, model_form: ModelForm) -> None:
        self.schema = schema
        self.model_form = model_form

    def ridge_decompose(
        self,
        df: pd.DataFrame,
        coef: np.ndarray,
        intercept: float,
        config: MMMConfig,
        *,
        decay: float,
        hill_half: float,
        hill_slope: float,
    ) -> DecompositionResult:
        bundle = build_design_matrix(
            df,
            self.schema,
            config,
            decay=decay,
            hill_half=hill_half,
            hill_slope=hill_slope,
        )
        X = bundle.X
        p = len(self.schema.channel_columns)
        media_part = X[:, :p] * coef[:p]
        cols = {f"contrib__{c}": media_part[:, i] for i, c in enumerate(self.schema.channel_columns)}
        frame = pd.DataFrame(cols, index=bundle.df_aligned.index)
        total = media_part.sum(axis=1) + float(intercept)
        notes = [
            "Contributions are on the modeling (log) scale for semi-log/log-log; "
            "not literal incremental dollars without an explicit level map."
        ]
        econ = economics_output_metadata(
            config,
            uncertainty_mode="point",
            surface="decomposition",
            baseline_type="training_fit_reference",
            decision_safe=False,
        )
        validate_business_economics_metadata(
            econ,
            require_specific_baseline=False,
            require_decision_safe_bool=False,
        )
        econ.update(
            {
                "artifact_tier": "diagnostic",
                "approximate": True,
                "not_for_budgeting": True,
                "is_proxy_metric": True,
                "not_exact_business_value": True,
            }
        )
        return DecompositionResult(
            channel_contributions=frame,
            total_media=total,
            scale="log_surrogate",
            is_exact_additive=False,
            safe_for_budgeting=False,
            notes=notes,
            economics_output_metadata=econ,
        )
