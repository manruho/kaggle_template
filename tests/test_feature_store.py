from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import Config
from src.feature_store import FeatureStore


def _write_csv(df: pd.DataFrame, path: Path) -> str:
    df.to_csv(path, index=False)
    return str(path)


def test_feature_store_cache_hit_and_miss(tmp_path: Path) -> None:
    train_df = pd.DataFrame({"id": [1, 2, 3], "target": [0, 1, 0], "f1": [0.1, 0.2, 0.3]})
    test_df = pd.DataFrame({"id": [10, 11], "f1": [0.5, 0.7]})
    sample_sub = pd.DataFrame({"id": [10, 11], "target": [0.0, 0.0]})

    train_path = _write_csv(train_df, tmp_path / "train.csv")
    test_path = _write_csv(test_df, tmp_path / "test.csv")
    sample_path = _write_csv(sample_sub, tmp_path / "sample_submission.csv")

    config = Config(
        train_path=train_path,
        test_path=test_path,
        sample_sub_path=sample_path,
        id_col="id",
        target_col="target",
        feature_version="cache_test",
    )

    store = FeatureStore(cache_dir=tmp_path / "cache", format="auto")
    result_first = store.load_or_build(
        config,
        builder=lambda: (train_df[["f1"]], test_df[["f1"]]),
    )
    assert result_first.cache_hit is False

    result_second = store.load_or_build(
        config,
        builder=lambda: (train_df[["f1"]], test_df[["f1"]]),
    )
    assert result_second.cache_hit is True
