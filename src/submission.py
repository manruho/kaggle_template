"""Submission building and validation helpers."""
from __future__ import annotations

from typing import Any, Dict, Sequence

import pandas as pd
from pandas.api.types import is_numeric_dtype


def validate_submission(
    submission: pd.DataFrame,
    sample_sub: pd.DataFrame,
    *,
    id_col: str,
    strict: bool = True,
) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "row_count_ok": len(submission) == len(sample_sub),
        "columns_ok": list(sample_sub.columns) == list(submission.columns),
        "missing_columns": [],
        "extra_columns": [],
        "id_alignment_ok": True,
        "nan_count": int(submission.isna().sum().sum()),
        "dtype_ok": True,
    }

    expected_cols = list(sample_sub.columns)
    actual_cols = list(submission.columns)
    report["missing_columns"] = [col for col in expected_cols if col not in actual_cols]
    report["extra_columns"] = [col for col in actual_cols if col not in expected_cols]

    if id_col in sample_sub.columns and id_col in submission.columns:
        report["id_alignment_ok"] = bool(sample_sub[id_col].equals(submission[id_col]))
    else:
        report["id_alignment_ok"] = False

    target_cols = [col for col in expected_cols if col != id_col]
    for col in target_cols:
        if col in submission.columns and not is_numeric_dtype(submission[col]):
            report["dtype_ok"] = False
            break

    report["columns_ok"] = not report["missing_columns"] and not report["extra_columns"]

    if strict:
        errors = []
        if not report["row_count_ok"]:
            errors.append("row count mismatch")
        if not report["columns_ok"]:
            errors.append("submission columns mismatch")
        if not report["id_alignment_ok"]:
            errors.append("id column mismatch or misordered")
        if report["nan_count"] > 0:
            errors.append("submission contains NaN")
        if not report["dtype_ok"]:
            errors.append("submission dtype mismatch")
        if errors:
            raise ValueError("Submission validation failed: " + ", ".join(errors))

    return report
