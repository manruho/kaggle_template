"""Utility functions."""
from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


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

    @classmethod
    def from_root(cls, root: str) -> "ArtifactPaths":
        return cls(
            output_dir=root,
            submission_path=os.path.join(root, "submission.csv"),
            oof_path=os.path.join(root, "oof.csv"),
            oof_parquet_path=os.path.join(root, "oof.parquet"),
            pred_test_parquet_path=os.path.join(root, "pred_test.parquet"),
            metrics_path=os.path.join(root, "cv_scores.json"),
            config_copy_path=os.path.join(root, "config_used.json"),
            meta_path=os.path.join(root, "meta.json"),
        )
