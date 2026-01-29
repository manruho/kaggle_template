"""Experiment orchestration entry points."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
import pandas as pd

from .config import Config
from .data import cache_dataset, load_datasets
from .features import build_feature_frames
from .feature_store import FeatureStore
from .inference import predict_ensemble
from .model_io import load_models, save_models
from .train import TrainingResult, train_cv
from .utils import ArtifactPaths, seed_everything
from .utils.experiment_id import generate_experiment_name, is_convention_name
from .utils.metadata import (
    build_run_summary,
    save_config_snapshot,
    save_cv_scores,
    save_env_metadata,
    save_git_metadata,
    save_run_summary,
)
from .utils.registry import append_experiment_record
from .utils.submission_validator import validate_submission


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

    start_time = datetime.now(timezone.utc)
    _ensure_experiment_name(config)
    seed_everything(config.seed)
    train_df, test_df, sample_sub = load_datasets(config)
    config.with_train_columns(train_df.columns)

    group_col_name = config.cv_group_col or config.get("group_col")
    groups = train_df[group_col_name] if group_col_name and group_col_name in train_df.columns else None
    time_col_name = config.cv_time_col or config.get("time_col")
    time_values = train_df[time_col_name] if time_col_name and time_col_name in train_df.columns else None

    X_train, X_test, feature_cache = _build_features(config, train_df, test_df)
    y = train_df[config.target_col]

    training_result = train_cv(config, X_train, y, X_test, groups, time_values)
    model_paths: Sequence[str] | None = None
    if config.save_models:
        model_paths = save_models(training_result.models, config.resolve_output_dir(), config)
    submission = _build_submission(config, sample_sub, training_result.predictions_test)

    end_time = datetime.now(timezone.utc)
    _write_artifacts(
        config,
        train_df,
        test_df,
        X_train,
        submission,
        training_result,
        sample_sub=sample_sub,
        feature_cache=feature_cache,
        start_time=start_time,
        end_time=end_time,
        model_paths=model_paths,
    )
    return ExperimentResult(config=config, training_result=training_result, submission=submission)


def train_only(config: Config) -> TrainingResult:
    start_time = datetime.now(timezone.utc)
    _ensure_experiment_name(config)
    seed_everything(config.seed)
    train_df, test_df, _sample_sub = load_datasets(config)
    config.with_train_columns(train_df.columns)

    group_col_name = config.cv_group_col or config.get("group_col")
    groups = train_df[group_col_name] if group_col_name and group_col_name in train_df.columns else None
    time_col_name = config.cv_time_col or config.get("time_col")
    time_values = train_df[time_col_name] if time_col_name and time_col_name in train_df.columns else None

    X_train, X_test, feature_cache = _build_features(config, train_df, test_df)
    y = train_df[config.target_col]
    training_result = train_cv(config, X_train, y, X_test, groups, time_values)
    model_paths: Sequence[str] | None = None
    if config.save_models:
        model_paths = save_models(training_result.models, config.resolve_output_dir(), config)

    end_time = datetime.now(timezone.utc)
    _write_training_artifacts(
        config,
        train_df,
        test_df,
        X_train,
        training_result,
        feature_cache=feature_cache,
        start_time=start_time,
        end_time=end_time,
        model_paths=model_paths,
    )
    return training_result


def infer_only(config: Config) -> np.ndarray:
    _ensure_experiment_name(config)
    seed_everything(config.seed)
    train_df, test_df, _sample_sub = load_datasets(config)
    config.with_train_columns(train_df.columns)

    _X_train, X_test, feature_cache = _build_features(config, train_df, test_df)
    models = load_models(config.resolve_output_dir())
    predictions = predict_ensemble(models, X_test, config.task_type)
    _write_inference_artifacts(
        config,
        test_df,
        predictions,
        feature_cache=feature_cache,
    )
    return predictions


def make_submission(config: Config, *, predictions: np.ndarray | None = None) -> pd.DataFrame:
    _ensure_experiment_name(config)
    _train_df, _test_df, sample_sub = load_datasets(config)
    output_dir = config.resolve_output_dir()
    if predictions is None:
        pred_path = Path(ArtifactPaths.from_root(str(output_dir)).predictions_path)
        if not pred_path.exists():
            raise FileNotFoundError(f"predictions not found: {pred_path}")
        predictions = np.load(pred_path)
    submission = _build_submission(config, sample_sub, predictions)
    validate_submission(
        submission,
        sample_sub,
        id_col=config.id_col,
        target_cols=[col for col in sample_sub.columns if col != config.id_col],
        task_type=config.task_type,
    )
    _write_submission_artifacts(
        config,
        submission,
    )
    return submission


def _build_features(
    config: Config,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any] | None]:
    if config.use_feature_cache:
        cache_dir = Path(config.feature_cache_dir or (Path(config.output_dir) / "_cache" / "features"))
        store = FeatureStore(cache_dir=cache_dir, format=str(config.feature_cache_format))
        store_result = store.load_or_build(
            config,
            lambda: build_feature_frames(train_df, test_df, config),
            force_recompute=bool(config.feature_cache_force_recompute),
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
    return X_train, X_test, feature_cache


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
    test_df: pd.DataFrame,
    X_train: pd.DataFrame,
    submission: pd.DataFrame,
    result: TrainingResult,
    *,
    sample_sub: pd.DataFrame,
    feature_cache: Dict[str, Any] | None = None,
    start_time: datetime,
    end_time: datetime,
    model_paths: Sequence[str] | None = None,
) -> None:
    output_dir = config.resolve_output_dir()
    paths = ArtifactPaths.from_root(str(output_dir))

    meta_dir = _prepare_meta_dir(output_dir)
    save_config_snapshot(meta_dir, config.as_dict())
    save_git_metadata(meta_dir, _find_repo_root(Path(__file__).resolve()))
    save_env_metadata(
        meta_dir,
        package_names=["numpy", "pandas", "scikit-learn", "lightgbm", "xgboost", "catboost", "torch"],
    )

    _write_cv_scores(meta_dir, config, result)

    validate_submission(
        submission,
        sample_sub,
        id_col=config.id_col,
        target_cols=[col for col in sample_sub.columns if col != config.id_col],
        task_type=config.task_type,
    )

    Path(paths.submission_path).parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(paths.submission_path, index=False)
    _write_oof_paths(paths, train_df, result, config)
    _write_pred_test_paths(paths, test_df, result.predictions_test, id_col=config.id_col)
    np.save(paths.predictions_path, result.predictions_test)
    _write_folds(paths.folds_path, train_df[config.id_col], result.folds)
    _write_run_summary_meta(meta_dir, config, result, start_time=start_time, end_time=end_time)
    _append_registry(config, result)

    if config.dataset_name:
        cache_dataset(submission, Path(output_dir) / f"submission_{config.dataset_name}.csv")


def _write_training_artifacts(
    config: Config,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    X_train: pd.DataFrame,
    result: TrainingResult,
    *,
    feature_cache: Dict[str, Any] | None = None,
    start_time: datetime,
    end_time: datetime,
    model_paths: Sequence[str] | None = None,
) -> None:
    output_dir = config.resolve_output_dir()
    paths = ArtifactPaths.from_root(str(output_dir))

    meta_dir = _prepare_meta_dir(output_dir)
    save_config_snapshot(meta_dir, config.as_dict())
    save_git_metadata(meta_dir, _find_repo_root(Path(__file__).resolve()))
    save_env_metadata(
        meta_dir,
        package_names=["numpy", "pandas", "scikit-learn", "lightgbm", "xgboost", "catboost", "torch"],
    )
    _write_cv_scores(meta_dir, config, result)

    _write_oof_paths(paths, train_df, result, config)
    _write_pred_test_paths(paths, test_df, result.predictions_test, id_col=config.id_col)
    np.save(paths.predictions_path, result.predictions_test)
    _write_folds(paths.folds_path, train_df[config.id_col], result.folds)
    _write_run_summary_meta(meta_dir, config, result, start_time=start_time, end_time=end_time)
    _append_registry(config, result)


def _write_inference_artifacts(
    config: Config,
    test_df: pd.DataFrame,
    predictions: np.ndarray,
    *,
    feature_cache: Dict[str, Any] | None = None,
) -> None:
    output_dir = config.resolve_output_dir()
    paths = ArtifactPaths.from_root(str(output_dir))
    _write_pred_test_paths(paths, test_df, predictions, id_col=config.id_col)
    np.save(paths.predictions_path, predictions)


def _write_submission_artifacts(
    config: Config,
    submission: pd.DataFrame,
) -> None:
    output_dir = config.resolve_output_dir()
    paths = ArtifactPaths.from_root(str(output_dir))
    Path(paths.submission_path).parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(paths.submission_path, index=False)
    if config.dataset_name:
        cache_dataset(submission, Path(output_dir) / f"submission_{config.dataset_name}.csv")


def _write_oof_paths(
    paths: ArtifactPaths,
    train_df: pd.DataFrame,
    result: TrainingResult,
    config: Config,
) -> None:
    oof_df = pd.DataFrame({config.id_col: train_df[config.id_col], f"oof_{config.target_col}": result.oof})
    oof_df.to_csv(paths.oof_path, index=False)
    _write_oof_parquet(paths.oof_parquet_path, train_df, result, config)


def _write_pred_test_paths(
    paths: ArtifactPaths,
    test_df: pd.DataFrame,
    predictions: np.ndarray,
    *,
    id_col: str,
) -> None:
    _write_pred_test_parquet(paths.pred_test_parquet_path, test_df[id_col], predictions)


def _write_oof_parquet(
    path: str,
    train_df: pd.DataFrame,
    result: TrainingResult,
    config: Config,
) -> None:
    pred_df = _build_pred_frame(result.oof, prefix="pred")
    oof_df = pd.concat(
        [
            train_df[[config.id_col, config.target_col]].reset_index(drop=True),
            pred_df.reset_index(drop=True),
            pd.DataFrame({"fold": result.folds}),
        ],
        axis=1,
    )
    _safe_to_parquet(oof_df, path)


def _write_pred_test_parquet(
    path: str,
    id_values: pd.Series,
    predictions: np.ndarray,
) -> None:
    pred_df = _build_pred_frame(predictions, prefix="pred")
    pred_test_df = pd.concat(
        [
            id_values.reset_index(drop=True).to_frame(name=id_values.name),
            pred_df.reset_index(drop=True),
        ],
        axis=1,
    )
    _safe_to_parquet(pred_test_df, path)


def _build_pred_frame(predictions: np.ndarray, *, prefix: str) -> pd.DataFrame:
    preds = np.asarray(predictions)
    if preds.ndim == 1:
        return pd.DataFrame({prefix: preds})
    return pd.DataFrame({f"{prefix}_{idx}": preds[:, idx] for idx in range(preds.shape[1])})


def _safe_to_parquet(df: pd.DataFrame, path: str) -> None:
    try:
        df.to_parquet(path, index=False)
    except Exception:
        fallback = str(Path(path).with_suffix(".pkl"))
        df.to_pickle(fallback)


def _prepare_meta_dir(output_dir: Path) -> Path:
    meta_dir = output_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    return meta_dir


def _write_folds(path: str, ids: pd.Series, folds: np.ndarray) -> None:
    fold_df = pd.DataFrame({ids.name: ids.values, "fold": folds})
    fold_df.to_csv(path, index=False)


def _write_run_summary_meta(
    meta_dir: Path,
    config: Config,
    result: TrainingResult,
    *,
    start_time: datetime,
    end_time: datetime,
) -> None:
    summary = build_run_summary(
        experiment_name=config.experiment_name or "unknown",
        metric=config.metric,
        scores=result.scores,
        model_name=config.model_name,
        seed=config.seed,
        start_time=start_time,
        end_time=end_time,
    )
    save_run_summary(meta_dir, summary)


def _write_cv_scores(meta_dir: Path, config: Config, result: TrainingResult) -> None:
    save_cv_scores(meta_dir, scores=result.scores, metric=config.metric)


def _append_registry(config: Config, result: TrainingResult) -> None:
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "experiment_name": config.experiment_name,
        "main_cv_score": float(sum(result.scores) / len(result.scores)) if result.scores else None,
        "cv_std": float(np.std(result.scores)) if result.scores else None,
        "metric": config.metric,
    }
    append_experiment_record(Path(config.output_dir) / "experiments.jsonl", record)


def _ensure_experiment_name(config: Config) -> None:
    name = config.experiment_name
    if name and str(name).strip().lower() != "auto":
        if not is_convention_name(name):
            print(f"[warn] experiment_name looks non-standard: {name}")
        return
    config.experiment_name = generate_experiment_name(config.as_dict())


def _find_repo_root(start: Path) -> Path:
    for path in (start, *start.parents):
        if (path / ".git").exists():
            return path
    return start
