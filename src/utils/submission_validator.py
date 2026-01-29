"""提出ファイルの検証ユーティリティ."""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def validate_submission(
    submission_df: pd.DataFrame,
    sample_submission_df: pd.DataFrame,
    *,
    id_col: str,
    target_cols: list[str],
    task_type: str | None = None,
) -> None:
    """提出ファイルを検証し、失敗したら例外を送出する."""
    _require_columns(submission_df, sample_submission_df.columns.tolist())
    _require_rows(submission_df, sample_submission_df)
    _require_id_match(submission_df, sample_submission_df, id_col=id_col)
    _require_no_nan_inf(submission_df, target_cols)
    if task_type == "binary":
        _require_binary_range(submission_df, target_cols)
    if task_type == "multiclass":
        _require_multiclass_prob(submission_df, target_cols)


def _require_columns(df: pd.DataFrame, expected_cols: list[str]) -> None:
    missing = [col for col in expected_cols if col not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    extra = [col for col in df.columns if col not in expected_cols]
    if extra:
        raise ValueError(f"unexpected columns: {extra}")


def _require_rows(df: pd.DataFrame, sample_df: pd.DataFrame) -> None:
    if len(df) != len(sample_df):
        raise ValueError("row count mismatch")


def _require_id_match(df: pd.DataFrame, sample_df: pd.DataFrame, *, id_col: str) -> None:
    if id_col not in df.columns or id_col not in sample_df.columns:
        raise ValueError("id column missing")
    if not sample_df[id_col].equals(df[id_col]):
        raise ValueError("id mismatch or order mismatch")


def _require_no_nan_inf(df: pd.DataFrame, target_cols: Iterable[str]) -> None:
    values = df[list(target_cols)].to_numpy()
    if np.isnan(values).any():
        raise ValueError("submission contains NaN")
    if np.isinf(values).any():
        raise ValueError("submission contains inf")


def _require_binary_range(df: pd.DataFrame, target_cols: Iterable[str]) -> None:
    values = df[list(target_cols)].to_numpy()
    if (values < 0).any() or (values > 1).any():
        raise ValueError("binary predictions out of [0,1]")


def _require_multiclass_prob(df: pd.DataFrame, target_cols: Iterable[str]) -> None:
    values = df[list(target_cols)].to_numpy()
    row_sums = values.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-3):
        raise ValueError("multiclass probabilities do not sum to 1")
