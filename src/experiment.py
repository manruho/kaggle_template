"""Experiment orchestration entry points."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .config import Config
from .config_io import save_config
from .data import cache_dataset, load_datasets
from .features import build_feature_frames
from .train import TrainingResult, train_cv
from .utils import ArtifactPaths, seed_everything


@dataclass
class ExperimentResult:
    config: Config
    training_result: TrainingResult
    submission: pd.DataFrame

    @property
    def oof(self) -> np.ndarray:
        return self.training_result.oof

    @property
    def predictions_test(self) -> np.ndarray:
        return self.training_result.predictions_test

    @property
    def scores(self) -> Sequence[float]:
        return self.training_result.scores


def run(config: Config) -> ExperimentResult:
    """High level pipeline: load data -> train -> save artifacts."""

    seed_everything(config.seed)
    train_df, test_df, sample_sub = load_datasets(config)
    group_col_name = config.get("group_col")
    groups = train_df[group_col_name] if group_col_name and group_col_name in train_df.columns else None

    X_train, X_test = build_feature_frames(train_df, test_df, config)
    y = train_df[config.target_col]

    training_result = train_cv(config, X_train, y, X_test, groups)
    submission = _build_submission(config, sample_sub, training_result.predictions_test)

    _write_artifacts(config, train_df, submission, training_result)
    return ExperimentResult(config=config, training_result=training_result, submission=submission)


def _build_submission(config: Config, sample_sub: pd.DataFrame, predictions: np.ndarray) -> pd.DataFrame:
    submission = sample_sub.copy()
    target_columns = [col for col in submission.columns if col != config.id_col]

    if predictions.ndim == 1:
        if not target_columns:
            raise ValueError("Sample submission does not contain target columns")
        submission[target_columns[0]] = predictions
    else:
        if len(target_columns) != predictions.shape[1]:
            raise ValueError("Number of prediction columns does not match submission template")
        for idx, column in enumerate(target_columns):
            submission[column] = predictions[:, idx]
    return submission


def _write_artifacts(config: Config, train_df: pd.DataFrame, submission: pd.DataFrame, result: TrainingResult) -> None:
    output_dir = config.resolve_output_dir()
    paths = ArtifactPaths.from_root(str(output_dir))

    submission.to_csv(paths.submission_path, index=False)

    oof_df = pd.DataFrame({config.id_col: train_df[config.id_col], f"oof_{config.target_col}": result.oof})
    oof_df.to_csv(paths.oof_path, index=False)

    Path(paths.metrics_path).write_text(json.dumps({"scores": result.scores}, indent=2), encoding="utf-8")
    save_config(config, paths.config_copy_path)

    if config.dataset_name:
        cache_dataset(submission, Path(output_dir) / f"submission_{config.dataset_name}.csv")
