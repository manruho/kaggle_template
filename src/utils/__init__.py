"""ユーティリティ関数."""
from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np


def seed_everything(seed: int) -> None:
    """乱数シードを固定する."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        return


@dataclass
class ArtifactPaths:
    output_dir: str
    submission_path: str
    oof_path: str
    oof_parquet_path: str
    pred_test_parquet_path: str
    metrics_path: str
    config_copy_path: str
    meta_path: str
    run_summary_path: str
    env_path: str
    pip_freeze_path: str
    folds_path: str
    submission_validation_path: str
    models_dir: str
    predictions_path: str

    @classmethod
    def from_root(cls, root: str) -> "ArtifactPaths":
        return cls(
            output_dir=root,
            submission_path=os.path.join(root, "submission", "submission.csv"),
            oof_path=os.path.join(root, "oof.csv"),
            oof_parquet_path=os.path.join(root, "oof.parquet"),
            pred_test_parquet_path=os.path.join(root, "pred_test.parquet"),
            metrics_path=os.path.join(root, "cv_scores.json"),
            config_copy_path=os.path.join(root, "config_used.json"),
            meta_path=os.path.join(root, "meta.json"),
            run_summary_path=os.path.join(root, "run_summary.json"),
            env_path=os.path.join(root, "env.txt"),
            pip_freeze_path=os.path.join(root, "pip_freeze.txt"),
            folds_path=os.path.join(root, "folds.csv"),
            submission_validation_path=os.path.join(root, "submission_validation.json"),
            models_dir=os.path.join(root, "models"),
            predictions_path=os.path.join(root, "pred_test.npy"),
        )
