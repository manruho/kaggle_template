"""Data loading utilities."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd

from .config import Config


def load_datasets(config: Config) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train, test, and sample submission DataFrames."""

    train_df = pd.read_csv(config.train_path)
    test_df = pd.read_csv(config.test_path)
    sample_sub = pd.read_csv(config.sample_sub_path)

    if config.debug:
        n_samples = config.sample_size or min(1000, len(train_df))
        train_df = train_df.sample(n=min(n_samples, len(train_df)), random_state=config.seed)
        test_df = test_df.sample(n=min(n_samples, len(test_df)), random_state=config.seed)
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True), sample_sub


def cache_dataset(df: pd.DataFrame, path: Path) -> None:
    """Persist DataFrame to ``path`` (used when sharing artifacts)."""

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
