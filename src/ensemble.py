"""Simple ensemble utilities for Kaggle experiments."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


def load_predictions(path: str | Path, *, prefix: str = "pred") -> np.ndarray:
    """Load prediction columns from parquet/csv and return as ndarray."""
    path = Path(path)
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    pred_cols = [col for col in df.columns if col == prefix or col.startswith(f"{prefix}_")]
    if not pred_cols:
        raise ValueError(f"No prediction columns found in {path}")
    return df[pred_cols].to_numpy()


def blend(predictions_list: Sequence[np.ndarray], weights: Sequence[float]) -> np.ndarray:
    """Weighted average of prediction arrays."""
    if len(predictions_list) != len(weights):
        raise ValueError("predictions_list and weights must be the same length")
    weights_array = np.asarray(weights, dtype=float)
    if weights_array.sum() == 0:
        raise ValueError("weights must sum to a non-zero value")
    weights_array = weights_array / weights_array.sum()

    blended = None
    for preds, weight in zip(predictions_list, weights_array):
        preds_array = np.asarray(preds, dtype=float)
        if blended is None:
            blended = preds_array * weight
        else:
            blended += preds_array * weight
    if blended is None:
        raise ValueError("predictions_list must not be empty")
    return blended


def correlation_matrix(predictions_list: Sequence[np.ndarray]) -> pd.DataFrame:
    """Return correlation matrix for 1D predictions (OOF or test)."""
    if not predictions_list:
        raise ValueError("predictions_list must not be empty")
    stacked = []
    for preds in predictions_list:
        preds_array = np.asarray(preds)
        if preds_array.ndim != 1:
            raise ValueError("correlation_matrix only supports 1D predictions")
        stacked.append(preds_array)
    matrix = np.corrcoef(np.vstack(stacked))
    return pd.DataFrame(matrix)
