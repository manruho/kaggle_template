from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

pydantic = pytest.importorskip("pydantic")

from src.utils.experiment_id import generate_experiment_name
from src.utils.metadata import save_env_metadata, save_git_metadata
from src.utils.submission_validator import validate_submission
from src.model_io import save_models
from src.config_schema import ConfigSchema
from src.config import Config


def test_generate_experiment_name_contains_required_parts() -> None:
    name = generate_experiment_name(
        {
            "model_name": "lgb",
            "feature_version": "005",
            "cv_method": "kfold",
            "n_splits": 5,
            "seed": 42,
        }
    )
    assert "lgb" in name
    assert "fe005" in name
    assert "cv5" in name
    assert "seed42" in name


def test_validate_submission_failures() -> None:
    sample = pd.DataFrame({"id": [1, 2], "target": [0.0, 0.0]})
    ok = pd.DataFrame({"id": [1, 2], "target": [0.1, 0.9]})
    validate_submission(ok, sample, id_col="id", target_cols=["target"], task_type="binary")

    missing = pd.DataFrame({"id": [1, 2]})
    with pytest.raises(ValueError):
        validate_submission(missing, sample, id_col="id", target_cols=["target"], task_type="binary")

    bad_rows = pd.DataFrame({"id": [1], "target": [0.1]})
    with pytest.raises(ValueError):
        validate_submission(bad_rows, sample, id_col="id", target_cols=["target"], task_type="binary")

    bad_id = pd.DataFrame({"id": [2, 1], "target": [0.1, 0.9]})
    with pytest.raises(ValueError):
        validate_submission(bad_id, sample, id_col="id", target_cols=["target"], task_type="binary")

    bad_nan = pd.DataFrame({"id": [1, 2], "target": [0.1, float("nan")]})
    with pytest.raises(ValueError):
        validate_submission(bad_nan, sample, id_col="id", target_cols=["target"], task_type="binary")


def test_metadata_saves_even_when_git_unavailable(tmp_path: Path) -> None:
    meta_dir = tmp_path / "meta"
    save_git_metadata(meta_dir, tmp_path / "nonexistent_repo")
    save_env_metadata(meta_dir, package_names=["numpy"], include_pip_freeze=False)
    assert (meta_dir / "git.txt").exists()
    assert (meta_dir / "env.txt").exists()


def test_save_models_invalid_policy(tmp_path: Path) -> None:
    config = Config(
        train_path=str(tmp_path / "train.csv"),
        test_path=str(tmp_path / "test.csv"),
        sample_sub_path=str(tmp_path / "sample.csv"),
        id_col="id",
        target_col="target",
    )
    with pytest.raises(ValueError):
        save_models([{"a": 1}], tmp_path, config, policy="bad", scores=[0.1])


def test_save_models_keep_top_k(tmp_path: Path) -> None:
    config = Config(
        train_path=str(tmp_path / "train.csv"),
        test_path=str(tmp_path / "test.csv"),
        sample_sub_path=str(tmp_path / "sample.csv"),
        id_col="id",
        target_col="target",
    )
    models = [{"a": 1}, {"b": 2}, {"c": 3}]
    scores = [0.1, 0.9, 0.5]
    paths = save_models(
        models,
        tmp_path,
        config,
        policy="keep_top_k",
        scores=scores,
        top_k=2,
    )
    assert len(paths) == 2


def test_config_validation_group_requires_column() -> None:
    with pytest.raises(ValueError):
        ConfigSchema.model_validate(
            {
                "train_path": "train.csv",
                "test_path": "test.csv",
                "sample_sub_path": "sample.csv",
                "id_col": "id",
                "target_col": "target",
                "cv_method": "group",
            }
        )
