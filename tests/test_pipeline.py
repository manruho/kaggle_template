import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from src.config import Config
from src.experiment import run


def _write_csv(df: pd.DataFrame, path: Path) -> str:
    df.to_csv(path, index=False)
    return str(path)


def test_run_pipeline(tmp_path: Path) -> None:
    X, y = make_classification(
        n_samples=200,
        n_features=5,
        n_informative=3,
        random_state=42,
    )
    train_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    train_df["id"] = np.arange(len(train_df))
    train_df["target"] = y

    X_test, _ = make_classification(n_samples=100, n_features=5, random_state=0)
    test_df = pd.DataFrame(X_test, columns=[f"f{i}" for i in range(X_test.shape[1])])
    test_df["id"] = np.arange(len(test_df)) + 10_000

    sample_sub = pd.DataFrame({"id": test_df["id"], "target": 0.0})

    train_path = _write_csv(train_df, tmp_path / "train.csv")
    test_path = _write_csv(test_df, tmp_path / "test.csv")
    sample_path = _write_csv(sample_sub, tmp_path / "sample_submission.csv")

    config = Config(
        train_path=train_path,
        test_path=test_path,
        sample_sub_path=sample_path,
        id_col="id",
        target_col="target",
        task_type="binary",
        metric="roc_auc",
        cv_type="stratified",
        n_splits=3,
        seed=0,
        debug=False,
        model_name="logistic_regression",
        model_params={"max_iter": 200},
        output_dir=str(tmp_path / "outputs"),
        experiment_name="test_run",
        use_feature_cache=True,
        feature_version="test_v1",
        validate_submission=True,
        save_policy="all",
    )

    result = run(config)

    assert result.submission.shape[0] == len(test_df)
    assert len(result.scores) == config.n_splits
    assert Path(config.output_dir).exists()

    artifact_dir = Path(config.output_dir) / str(config.experiment_name)
    assert (artifact_dir / "submission" / "submission.csv").exists()
    assert (artifact_dir / "oof.csv").exists()
    assert (artifact_dir / "oof.parquet").exists() or (artifact_dir / "oof.pkl").exists()
    assert (artifact_dir / "pred_test.parquet").exists() or (artifact_dir / "pred_test.pkl").exists()
    assert (artifact_dir / "meta" / "cv_scores.json").exists()
    assert (artifact_dir / "meta" / "config.snapshot.json").exists()
    assert (artifact_dir / "meta" / "git.txt").exists()
    assert (artifact_dir / "meta" / "run_summary.json").exists()
    assert (artifact_dir / "meta" / "env.txt").exists()
    assert (artifact_dir / "folds.csv").exists()
    assert (artifact_dir / "pred_test.npy").exists()
    assert (artifact_dir / "models").exists()

    meta_first = json.loads((artifact_dir / "meta" / "cv_scores.json").read_text(encoding="utf-8"))
    assert meta_first["metric"] == config.metric

    run(config)
    meta_second = json.loads((artifact_dir / "meta" / "cv_scores.json").read_text(encoding="utf-8"))
    assert "fold_scores" in meta_second

    registry_path = Path(config.output_dir) / "experiments.jsonl"
    assert registry_path.exists()
