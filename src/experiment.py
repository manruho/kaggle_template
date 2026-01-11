"""Experiment orchestration entry points."""
from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
import pandas as pd

from .config import Config
from .config_io import save_config
from .data import cache_dataset, load_datasets
from .features import build_feature_frames
from .feature_store import FeatureStore
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
    config.with_train_columns(train_df.columns)
    group_col_name = config.get("group_col")
    groups = train_df[group_col_name] if group_col_name and group_col_name in train_df.columns else None

    feature_cache: Dict[str, Any] | None = None
    if config.get("use_feature_cache", False):
        cache_dir = Path(config.get("feature_cache_dir", Path(config.output_dir) / "_cache" / "features"))
        store = FeatureStore(cache_dir=cache_dir, format=str(config.get("feature_cache_format", "auto")))
        store_result = store.load_or_build(
            config,
            lambda: build_feature_frames(train_df, test_df, config),
            force_recompute=bool(config.get("feature_cache_force_recompute", False)),
        )
        X_train, X_test = store_result.X_train, store_result.X_test
        feature_cache = {
            "enabled": True,
            "cache_dir": str(store_result.cache_dir),
            "cache_key": store_result.cache_key,
            "cache_hit": store_result.cache_hit,
            "format": store_result.format,
        }
    else:
        X_train, X_test = build_feature_frames(train_df, test_df, config)
        feature_cache = {"enabled": False}
    y = train_df[config.target_col]

    training_result = train_cv(config, X_train, y, X_test, groups)
    submission = _build_submission(config, sample_sub, training_result.predictions_test)

    _write_artifacts(config, train_df, X_train, submission, training_result, feature_cache=feature_cache)
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


def _write_artifacts(
    config: Config,
    train_df: pd.DataFrame,
    X_train: pd.DataFrame,
    submission: pd.DataFrame,
    result: TrainingResult,
    *,
    feature_cache: Dict[str, Any] | None = None,
) -> None:
    output_dir = config.resolve_output_dir()
    paths = ArtifactPaths.from_root(str(output_dir))

    submission.to_csv(paths.submission_path, index=False)

    oof_df = pd.DataFrame({config.id_col: train_df[config.id_col], f"oof_{config.target_col}": result.oof})
    oof_df.to_csv(paths.oof_path, index=False)

    Path(paths.metrics_path).write_text(json.dumps({"scores": result.scores}, indent=2), encoding="utf-8")
    save_config(config, paths.config_copy_path)
    Path(paths.meta_path).write_text(
        json.dumps(
            _build_meta(config, train_df, X_train, submission, result, feature_cache=feature_cache),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    if config.dataset_name:
        cache_dataset(submission, Path(output_dir) / f"submission_{config.dataset_name}.csv")


def _build_meta(
    config: Config,
    train_df: pd.DataFrame,
    X_train: pd.DataFrame,
    submission: pd.DataFrame,
    result: TrainingResult,
    *,
    feature_cache: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    scores = [float(score) for score in result.scores]
    meta: Dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment_name": config.experiment_name,
        "cv_mean": float(np.mean(scores)) if scores else None,
        "cv_std": float(np.std(scores)) if scores else None,
        "fold_scores": scores,
        "seed": config.seed,
        "cv_type": config.cv_type,
        "n_splits": config.n_splits,
        "task_type": config.task_type,
        "metric": config.metric,
        "model_name": config.model_name,
        "model_params": config.model_params,
        "features_version": config.get("features_version", "default"),
        "n_train": int(len(train_df)),
        "n_test": int(len(submission)),
        "raw_feature_count": int(len(config.feature_columns)),
        "encoded_feature_count": int(X_train.shape[1]),
        "git_commit": _get_git_commit(),
        "python": platform.python_version(),
        "packages": _get_package_versions(
            (
                "numpy",
                "pandas",
                "scikit-learn",
                "lightgbm",
                "xgboost",
                "catboost",
                "torch",
                "tensorflow",
            )
        ),
    }
    if feature_cache is not None:
        meta["feature_cache"] = feature_cache
    return meta


def _get_git_commit() -> str | None:
    repo_root = _find_git_root(Path(__file__).resolve())
    if repo_root is None:
        return None
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root).decode("utf-8").strip()
    except Exception:
        return None
    return commit or None


def _find_git_root(start: Path) -> Path | None:
    for path in (start, *start.parents):
        if (path / ".git").exists():
            return path
    return None


def _get_package_versions(package_names: Sequence[str]) -> Dict[str, str]:
    versions: Dict[str, str] = {}
    for name in package_names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
        except Exception:
            continue
    return versions
