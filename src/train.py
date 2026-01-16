"""Training utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error, mean_squared_error, roc_auc_score

from .config import Config
from .inference import predict
from .models import create_model
from .split import build_splitter


@dataclass
class TrainingResult:
    models: List[Any]
    oof: np.ndarray
    predictions_test: np.ndarray
    scores: List[float]
    folds: np.ndarray


def train_cv(
    config: Config,
    X_train: pd.DataFrame,
    y: Sequence,
    X_test: pd.DataFrame,
    groups: Sequence | None = None,
) -> TrainingResult:
    y_series = pd.Series(y).reset_index(drop=True)
    splitter = build_splitter(config, y_series, groups)
    oof = np.zeros(len(X_train))
    fold_ids = np.full(len(X_train), -1, dtype=int)
    test_pred_accumulator = None
    models: List[Any] = []
    scores: List[float] = []

    for fold, (train_idx, valid_idx) in enumerate(splitter):
        X_tr = X_train.iloc[train_idx]
        X_val = X_train.iloc[valid_idx]
        y_tr = y_series.iloc[train_idx]
        y_val = y_series.iloc[valid_idx]

        model = create_model(config)
        model.fit(X_tr, y_tr)
        models.append(model)

        pred_val = predict(model, X_val, config.task_type)
        oof[valid_idx] = _flatten_binary_predictions(pred_val)
        fold_ids[valid_idx] = fold

        fold_score = _compute_metric(config.metric, y_val, pred_val, config.task_type)
        scores.append(fold_score)

        pred_test = predict(model, X_test, config.task_type)
        if test_pred_accumulator is None:
            test_pred_accumulator = np.zeros_like(pred_test, dtype=float)
        test_pred_accumulator += pred_test

    if test_pred_accumulator is None:
        raise RuntimeError("CV splitter produced no folds")
    test_pred_avg = test_pred_accumulator / config.n_splits
    return TrainingResult(
        models=models,
        oof=oof,
        predictions_test=test_pred_avg,
        scores=scores,
        folds=fold_ids,
    )


def _flatten_binary_predictions(predictions: np.ndarray) -> np.ndarray:
    if predictions.ndim == 1:
        return predictions
    if predictions.ndim == 2 and predictions.shape[1] == 1:
        return predictions.squeeze(1)
    return predictions.argmax(axis=1)


def _compute_metric(metric_name: str, y_true: Sequence, preds, task_type: str) -> float:
    metric = metric_name.lower()
    y_true_array = np.asarray(y_true)
    preds_array = np.asarray(preds)

    if metric in {"roc_auc", "auc"}:
        if preds_array.ndim == 1:
            return roc_auc_score(y_true_array, preds_array)
        return roc_auc_score(y_true_array, preds_array, multi_class="ovr")

    if metric in {"log_loss", "cross_entropy"}:
        if preds_array.ndim == 1:
            proba = np.vstack([1 - preds_array, preds_array]).T
        else:
            proba = preds_array
        return log_loss(y_true_array, proba)

    if metric in {"accuracy", "acc"}:
        if task_type == "binary":
            labels = (preds_array >= 0.5).astype(int)
        elif preds_array.ndim == 2:
            labels = preds_array.argmax(axis=1)
        else:
            labels = preds_array
        return accuracy_score(y_true_array, labels)

    if metric == "rmse":
        return float(np.sqrt(mean_squared_error(y_true_array, preds_array)))

    if metric == "mae":
        return float(mean_absolute_error(y_true_array, preds_array))

    if metric == "mse":
        return float(mean_squared_error(y_true_array, preds_array))

    raise ValueError(f"Unsupported metric: {metric_name}")
