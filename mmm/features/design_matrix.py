"""Single entry: full-panel media transforms, target scaling, masks, and feature lineage."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from mmm.config.schema import ModelForm, MMMConfig
from mmm.data.panel_order import sort_panel_for_modeling
from mmm.data.schema import PanelSchema
from mmm.transforms.stack import build_channel_features_from_params
from mmm.utils.math import safe_log


@dataclass
class DesignMatrixMasks:
    """Causal contract: who supplies state recursion vs who enters which loss."""

    history_mask: np.ndarray  # rows used for building recursive state (typically all True)
    train_loss_mask: np.ndarray
    val_loss_mask: np.ndarray
    calibration_mask: np.ndarray

    def validate(self, n: int) -> None:
        for name, m in self.__dict__.items():
            if len(m) != n:
                raise ValueError(f"{name} length {len(m)} != n={n}")


@dataclass
class DesignMatrixBundle:
    """X and y on modeling scale; masks and lineage for artifacts and CV."""

    X: np.ndarray
    y_modeling: np.ndarray
    df_aligned: pd.DataFrame
    masks: DesignMatrixMasks
    feature_lineage: dict[str, Any]
    meta: dict[str, Any] = field(default_factory=dict)

    def to_lineage_json(self) -> dict[str, Any]:
        return {"lineage": self.feature_lineage, "mask_contract": self.meta.get("mask_contract", {})}


def _default_masks(n: int) -> DesignMatrixMasks:
    return DesignMatrixMasks(
        history_mask=np.ones(n, dtype=bool),
        train_loss_mask=np.ones(n, dtype=bool),
        val_loss_mask=np.zeros(n, dtype=bool),
        calibration_mask=np.zeros(n, dtype=bool),
    )


def design_masks_from_cv_split(
    n_rows: int,
    train_loss_mask: np.ndarray,
    val_loss_mask: np.ndarray,
    *,
    history_mask: np.ndarray | None = None,
    calibration_mask: np.ndarray | None = None,
) -> DesignMatrixMasks:
    """
    Build a :class:`DesignMatrixMasks` contract from CV row masks (Sprint 1.2).

    ``history_mask`` defaults to all-True (full series for recursive adstock). ``calibration_mask``
    defaults to all-False unless replay windows are supplied.
    """
    tr = np.asarray(train_loss_mask, dtype=bool)
    va = np.asarray(val_loss_mask, dtype=bool)
    if tr.shape[0] != n_rows or va.shape[0] != n_rows:
        raise ValueError("train_loss_mask and val_loss_mask must have length n_rows")
    hist = np.ones(n_rows, dtype=bool) if history_mask is None else np.asarray(history_mask, dtype=bool)
    cal = np.zeros(n_rows, dtype=bool) if calibration_mask is None else np.asarray(calibration_mask, dtype=bool)
    if hist.shape[0] != n_rows or cal.shape[0] != n_rows:
        raise ValueError("history_mask and calibration_mask must have length n_rows")
    return DesignMatrixMasks(
        history_mask=hist,
        train_loss_mask=tr,
        val_loss_mask=va,
        calibration_mask=cal,
    )


def build_design_matrix(
    df: pd.DataFrame,
    schema: PanelSchema,
    config: MMMConfig,
    *,
    decay: float,
    hill_half: float,
    hill_slope: float,
    masks: DesignMatrixMasks | None = None,
) -> DesignMatrixBundle:
    """
    Build X on **full** sorted panel so adstock carryover is causal within each geo.

    Rows are sorted by (geo, week). Callers must use the returned ``df_aligned`` for y/masks.
    """
    df_aligned = sort_panel_for_modeling(df, schema)
    n = len(df_aligned)
    m = masks if masks is not None else _default_masks(n)
    m.validate(n)

    X_media = build_channel_features_from_params(
        df_aligned,
        schema,
        config.transforms,
        decay=decay,
        hill_half=hill_half,
        hill_slope=hill_slope,
        modeling_config=config,
    )
    y = df_aligned[schema.target_column].to_numpy(dtype=float)
    if config.model_form == ModelForm.SEMI_LOG:
        y_modeling = safe_log(y)
    else:
        X_media = safe_log(np.maximum(X_media, 1e-9))
        y_modeling = safe_log(y)

    if schema.control_columns:
        ctrl = np.column_stack([df_aligned[c].to_numpy(dtype=float) for c in schema.control_columns])
        X = np.column_stack([X_media, ctrl])
    else:
        X = X_media

    lineage = {
        "media": [
            {
                "stage": "raw_spend",
                "columns": list(schema.channel_columns),
            },
            {
                "stage": "adstock",
                "kind": config.transforms.adstock,
                "params": {"decay": decay},
            },
            {
                "stage": "saturation",
                "kind": config.transforms.saturation,
                "params": {"half_max": hill_half, "slope": hill_slope},
            },
        ],
        "target": {
            "column": schema.target_column,
            "form": config.model_form.value,
            "scale": "log" if config.model_form == ModelForm.SEMI_LOG else "log_log_media",
        },
    }
    if schema.control_columns:
        lineage["controls"] = {"columns": list(schema.control_columns), "stage": "as_provided"}

    meta = {
        "mask_contract": (
            "history_mask: rows eligible for recursive state (full series per geo); "
            "train_loss_mask / val_loss_mask: which rows enter each loss; "
            "calibration_mask: replay windows. "
            "X is always built on full df_aligned so adstock is causal."
        ),
        "n_rows": n,
    }
    return DesignMatrixBundle(
        X=X,
        y_modeling=y_modeling,
        df_aligned=df_aligned,
        masks=m,
        feature_lineage=lineage,
        meta=meta,
    )


def apply_masks_for_fit(bundle: DesignMatrixBundle, train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Subset design matrix for fitting (training loss rows only)."""
    X = bundle.X[train]
    y = bundle.y_modeling[train]
    return X, y


def media_design_matrix(bundle: DesignMatrixBundle, schema: PanelSchema) -> np.ndarray:
    """Media columns only (first ``len(channel_columns)``), same transform path as full ``X``."""
    p = len(schema.channel_columns)
    return np.asarray(bundle.X[:, :p], dtype=float, order="C")
