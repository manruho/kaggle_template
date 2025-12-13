"""Cross-validation helpers."""
from __future__ import annotations

from typing import Iterable, Iterator, Sequence, Tuple

import numpy as np
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold

from .config import Config


FoldIterator = Iterator[Tuple[np.ndarray, np.ndarray]]


def build_splitter(config: Config, y: Sequence, groups: Sequence | None = None) -> FoldIterator:
    """Return generator yielding train/valid indices."""

    if config.cv_type == "stratified" and config.task_type in {"binary", "multiclass"}:
        splitter = StratifiedKFold(n_splits=config.n_splits, shuffle=True, random_state=config.seed)
        return splitter.split(np.zeros(len(y)), y)
    if config.cv_type == "group":
        group_col = groups
        if group_col is None:
            raise ValueError("Group CV requested but no groups were provided")
        splitter = GroupKFold(n_splits=config.n_splits)
        return splitter.split(np.zeros(len(y)), y, group_col)
    splitter = KFold(n_splits=config.n_splits, shuffle=True, random_state=config.seed)
    return splitter.split(np.zeros(len(y)))
