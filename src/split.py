"""Cross-validation helpers."""
from __future__ import annotations

from typing import Iterable, Iterator, Sequence, Tuple

import numpy as np
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    RepeatedKFold,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    TimeSeriesSplit,
)

from .config import Config


FoldIterator = Iterator[Tuple[np.ndarray, np.ndarray]]


def build_splitter(
    config: Config,
    y: Sequence,
    groups: Sequence | None = None,
    time_values: Sequence | None = None,
) -> FoldIterator:
    """Return generator yielding train/valid indices."""

    custom_splitter = config.get("cv_splitter")
    if custom_splitter is not None:
        return custom_splitter.split(np.zeros(len(y)), y, groups)

    method = config.get_cv_method()

    if method in {"stratified", "stratifiedkfold", "strat"} and config.task_type in {"binary", "multiclass"}:
        splitter = StratifiedKFold(n_splits=config.n_splits, shuffle=True, random_state=config.seed)
        return splitter.split(np.zeros(len(y)), y)
    if method in {"repeated_stratified_kfold", "repeated_stratified"} and config.task_type in {
        "binary",
        "multiclass",
    }:
        params = dict(config.cv_params or {})
        splitter = RepeatedStratifiedKFold(
            n_splits=config.n_splits,
            random_state=config.seed,
            **params,
        )
        return splitter.split(np.zeros(len(y)), y)
    if method in {"group", "groupkfold"}:
        if groups is None:
            raise ValueError("Group CV requested but no groups were provided")
        splitter = GroupKFold(n_splits=config.n_splits)
        return splitter.split(np.zeros(len(y)), y, groups)
    if method in {"time", "timeseries", "time_series", "timeseriessplit"}:
        if time_values is None:
            raise ValueError("TimeSeries CV requested but no time values were provided")
        params = dict(config.cv_params or {})
        splitter = TimeSeriesSplit(n_splits=config.n_splits, **params)
        order = np.argsort(np.asarray(time_values))
        return _time_series_split(splitter, order)
    if method in {"purged", "purged_timeseries"}:
        if time_values is None:
            raise ValueError("Purged CV requested but no time values were provided")
        params = dict(config.cv_params or {})
        purge_gap = int(params.pop("purge_gap", 0))
        splitter = TimeSeriesSplit(n_splits=config.n_splits, **params)
        order = np.argsort(np.asarray(time_values))
        return _purged_time_series_split(splitter, order, purge_gap=purge_gap)
    if method in {"repeated_kfold", "repeated"}:
        params = dict(config.cv_params or {})
        splitter = RepeatedKFold(
            n_splits=config.n_splits,
            random_state=config.seed,
            **params,
        )
        return splitter.split(np.zeros(len(y)))
    splitter = KFold(n_splits=config.n_splits, shuffle=True, random_state=config.seed)
    return splitter.split(np.zeros(len(y)))


def _time_series_split(splitter: TimeSeriesSplit, order: np.ndarray) -> FoldIterator:
    for train_pos, valid_pos in splitter.split(order):
        train_idx = order[train_pos]
        valid_idx = order[valid_pos]
        yield train_idx, valid_idx


def _purged_time_series_split(
    splitter: TimeSeriesSplit,
    order: np.ndarray,
    *,
    purge_gap: int,
) -> FoldIterator:
    for train_pos, valid_pos in splitter.split(order):
        train_idx = order[train_pos]
        valid_idx = order[valid_pos]
        if purge_gap > 0:
            first_valid = valid_pos[0]
            last_valid = valid_pos[-1]
            purge_start = max(0, first_valid - purge_gap)
            purge_end = min(len(order) - 1, last_valid + purge_gap)
            mask = (train_pos < purge_start) | (train_pos > purge_end)
            train_idx = order[train_pos[mask]]
        yield train_idx, valid_idx
