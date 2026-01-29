"""Experiment orchestration entry points."""
from __future__ import annotations

import json
import platform
import subprocess
import sys
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
from .inference import predict_ensemble
from .model_io import load_models, save_models
from .submission import validate_submission
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

    start_time = datetime.now(timezone.utc)
    config.ensure_experiment_name()
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

    validation_report = None
    if config.validate_submission:
        validation_report = validate_submission(submission, sample_sub, id_col=config.id_col, strict=True)

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
        validation_report=validation_report,
        start_time=start_time,
        end_time=end_time,
        model_paths=model_paths,
    )
    return ExperimentResult(config=config, training_result=training_result, submission=submission)


def train_only(config: Config) -> TrainingResult:
    start_time = datetime.now(timezone.utc)
    config.ensure_experiment_name()
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
    config.ensure_experiment_name()
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
    config.ensure_experiment_name()
    _train_df, _test_df, sample_sub = load_datasets(config)
    output_dir = config.resolve_output_dir()
    if predictions is None:
        pred_path = Path(ArtifactPaths.from_root(str(output_dir)).predictions_path)
        if not pred_path.exists():
            raise FileNotFoundError(f"predictions not found: {pred_path}")
        predictions = np.load(pred_path)
    submission = _build_submission(config, sample_sub, predictions)
    validation_report = None
    if config.validate_submission:
        validation_report = validate_submission(submission, sample_sub, id_col=config.id_col, strict=True)
    _write_submission_artifacts(
        config,
        submission,
        validation_report=validation_report,
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
    validation_report: Dict[str, Any] | None = None,
    start_time: datetime,
    end_time: datetime,
    model_paths: Sequence[str] | None = None,
) -> None:
    output_dir = config.resolve_output_dir()
    paths = ArtifactPaths.from_root(str(output_dir))

    submission.to_csv(paths.submission_path, index=False)
    _write_oof_paths(paths, train_df, result, config)
    _write_pred_test_paths(paths, test_df, result.predictions_test, id_col=config.id_col)
    np.save(paths.predictions_path, result.predictions_test)
    _write_folds(paths.folds_path, train_df[config.id_col], result.folds)
    _write_metrics(paths.metrics_path, config, result)
    save_config(config, paths.config_copy_path)
    _write_meta(
        paths.meta_path,
        config,
        train_df,
        X_train,
        submission,
        result,
        feature_cache=feature_cache,
        start_time=start_time,
        end_time=end_time,
        model_paths=model_paths,
    )
    _write_env_files(paths)
    if validation_report is not None:
        Path(paths.submission_validation_path).write_text(
            json.dumps(validation_report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    _write_run_summary(
        paths.run_summary_path,
        config,
        result,
        start_time=start_time,
        end_time=end_time,
        validation_report=validation_report,
    )
    _update_experiment_registry(config, result)

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

    _write_oof_paths(paths, train_df, result, config)
    _write_pred_test_paths(paths, test_df, result.predictions_test, id_col=config.id_col)
    np.save(paths.predictions_path, result.predictions_test)
    _write_folds(paths.folds_path, train_df[config.id_col], result.folds)
    _write_metrics(paths.metrics_path, config, result)
    save_config(config, paths.config_copy_path)
    _write_meta(
        paths.meta_path,
        config,
        train_df,
        X_train,
        None,
        result,
        feature_cache=feature_cache,
        start_time=start_time,
        end_time=end_time,
        model_paths=model_paths,
    )
    _write_env_files(paths)
    _write_run_summary(
        paths.run_summary_path,
        config,
        result,
        start_time=start_time,
        end_time=end_time,
        validation_report=None,
    )
    _update_experiment_registry(config, result)


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
    if feature_cache is not None:
        Path(paths.meta_path).write_text(
            json.dumps({"feature_cache": feature_cache}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


def _write_submission_artifacts(
    config: Config,
    submission: pd.DataFrame,
    *,
    validation_report: Dict[str, Any] | None = None,
) -> None:
    output_dir = config.resolve_output_dir()
    paths = ArtifactPaths.from_root(str(output_dir))
    submission.to_csv(paths.submission_path, index=False)
    if validation_report is not None:
        Path(paths.submission_validation_path).write_text(
            json.dumps(validation_report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
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


def _write_meta(
    path: str,
    config: Config,
    train_df: pd.DataFrame,
    X_train: pd.DataFrame,
    submission: pd.DataFrame | None,
    result: TrainingResult,
    *,
    feature_cache: Dict[str, Any] | None = None,
    start_time: datetime,
    end_time: datetime,
    model_paths: Sequence[str] | None = None,
) -> None:
    scores = [float(score) for score in result.scores]
    meta: Dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_started_at": start_time.isoformat(),
        "run_finished_at": end_time.isoformat(),
        "run_duration_seconds": (end_time - start_time).total_seconds(),
        "experiment_name": config.experiment_name,
        "cv_mean": float(np.mean(scores)) if scores else None,
        "cv_std": float(np.std(scores)) if scores else None,
        "fold_scores": scores,
        "oof_score": result.oof_score,
        "seed": config.seed,
        "cv_method": config.get_cv_method(),
        "n_splits": config.n_splits,
        "cv_group_col": config.cv_group_col,
        "cv_time_col": config.cv_time_col,
        "task_type": config.task_type,
        "metric": config.metric,
        "model_name": config.model_name,
        "model_params": config.model_params,
        "feature_version": config.get_feature_version(),
        "n_train": int(len(train_df)),
        "n_test": int(len(submission)) if submission is not None else None,
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
        "tags": list(config.experiment_tags) if config.experiment_tags else [],
        "note": config.experiment_note,
        "parent": config.experiment_parent,
        "model_paths": list(model_paths) if model_paths else [],
    }
    if feature_cache is not None:
        meta["feature_cache"] = feature_cache
    Path(path).write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_metrics(path: str, config: Config, result: TrainingResult) -> None:
    Path(path).write_text(
        json.dumps(
            {
                "metric": config.metric,
                "fold_scores": result.scores,
                "mean": float(np.mean(result.scores)) if result.scores else None,
                "std": float(np.std(result.scores)) if result.scores else None,
                "oof_score": result.oof_score,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


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


def _write_env_files(paths: ArtifactPaths) -> None:
    Path(paths.env_path).write_text(
        f"python={platform.python_version()}\nplatform={platform.platform()}\n",
        encoding="utf-8",
    )
    pip_freeze = _get_pip_freeze()
    if pip_freeze is not None:
        Path(paths.pip_freeze_path).write_text(pip_freeze, encoding="utf-8")


def _get_pip_freeze() -> str | None:
    try:
        output = subprocess.check_output([sys.executable, "-m", "pip", "freeze"])
    except Exception:
        return None
    return output.decode("utf-8")


def _write_folds(path: str, ids: pd.Series, folds: np.ndarray) -> None:
    fold_df = pd.DataFrame({ids.name: ids.values, "fold": folds})
    fold_df.to_csv(path, index=False)


def _write_run_summary(
    path: str,
    config: Config,
    result: TrainingResult,
    *,
    start_time: datetime,
    end_time: datetime,
    validation_report: Dict[str, Any] | None,
) -> None:
    summary = {
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "duration_seconds": (end_time - start_time).total_seconds(),
        "experiment_name": config.experiment_name,
        "config_summary": _config_summary(config),
        "fold_scores": result.scores,
        "oof_score": result.oof_score,
        "metric": config.metric,
        "artifacts_dir": str(config.resolve_output_dir()),
        "validation": validation_report,
    }
    Path(path).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def _config_summary(config: Config) -> Dict[str, Any]:
    return {
        "model_name": config.model_name,
        "task_type": config.task_type,
        "feature_version": config.get_feature_version(),
        "cv_method": config.get_cv_method(),
        "n_splits": config.n_splits,
        "seed": config.seed,
        "debug": config.debug,
        "features": list(config.features) if config.features else None,
        "drop_cols": list(config.drop_cols) if config.drop_cols else None,
    }


def _update_experiment_registry(config: Config, result: TrainingResult) -> None:
    registry_path = Path(config.output_dir) / "experiments.csv"
    row = pd.DataFrame(
        [
            {
                "experiment_name": config.experiment_name,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "model_name": config.model_name,
                "feature_version": config.get_feature_version(),
                "cv_method": config.get_cv_method(),
                "n_splits": config.n_splits,
                "seed": config.seed,
                "metric": config.metric,
                "cv_mean": float(np.mean(result.scores)) if result.scores else None,
                "cv_std": float(np.std(result.scores)) if result.scores else None,
                "oof_score": result.oof_score,
                "tags": ",".join(config.experiment_tags or []),
                "note": config.experiment_note,
                "parent": config.experiment_parent,
            }
        ]
    )
    if registry_path.exists():
        existing = pd.read_csv(registry_path)
        registry = pd.concat([existing, row], ignore_index=True)
    else:
        registry = row
    registry.to_csv(registry_path, index=False)
