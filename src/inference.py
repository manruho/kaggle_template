"""Inference helpers."""
from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd


def predict(model: Any, X: pd.DataFrame, task_type: str) -> np.ndarray:
    """Return numpy predictions for ``X`` depending on ``task_type``."""

    if task_type in {"binary", "multiclass"}:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
            if task_type == "binary":
                return np.asarray(proba)[:, 1]
            return np.asarray(proba)
        preds = model.predict(X)
        return np.asarray(preds)
    if task_type in {"multilabel", "multitarget", "multioutput_regression"}:
        preds = model.predict(X)
        return np.asarray(preds)
    preds = model.predict(X)
    return np.asarray(preds)


def predict_ensemble(models: Iterable[Any], X: pd.DataFrame, task_type: str) -> np.ndarray:
    predictions = [predict(model, X, task_type) for model in models]
    if not predictions:
        raise ValueError("No models provided for inference")
    stacked = np.asarray(predictions)
    return stacked.mean(axis=0)
