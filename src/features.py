"""Feature preparation helpers."""
from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import pandas as pd

from .config import Config


def _select_columns(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in dataframe: {missing}")
    return df.loc[:, columns]


def build_feature_frames(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: Config,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return aligned feature matrices for train/test."""

    config.with_train_columns(train_df.columns)
    if config.features:
        feature_cols = list(config.features)
        X_train = _select_columns(train_df, feature_cols)
        X_test = _select_columns(test_df, feature_cols)
    else:
        excluded = set([config.id_col, config.target_col])
        if config.drop_cols:
            excluded.update(config.drop_cols)
        feature_cols = [c for c in train_df.columns if c not in excluded]
        X_train = train_df[feature_cols]
        X_test = test_df[feature_cols]

    X_train_enc, X_test_enc = _align_dummies(X_train, X_test)
    return X_train_enc, X_test_enc


def _align_dummies(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_encoded = pd.get_dummies(train_df, dummy_na=True)
    test_encoded = pd.get_dummies(test_df, dummy_na=True)
    train_aligned, test_aligned = train_encoded.align(test_encoded, join="outer", axis=1, fill_value=0)
    return train_aligned, test_aligned
