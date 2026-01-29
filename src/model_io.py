"""Model persistence helpers."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, List

import joblib

from .config import Config


def save_models(models: Iterable[Any], output_dir: str | Path, config: Config) -> List[str]:
    root = Path(output_dir) / "models"
    root.mkdir(parents=True, exist_ok=True)

    paths: List[str] = []
    for idx, model in enumerate(models):
        path = root / f"model_fold_{idx}.joblib"
        joblib.dump(model, path)
        paths.append(str(path))

    meta = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_name": config.model_name,
        "model_params": config.model_params,
        "n_models": len(paths),
        "paths": paths,
    }
    (root / "meta.json").write_text(
        json_dumps(meta),
        encoding="utf-8",
    )
    return paths


def load_models(output_dir: str | Path) -> List[Any]:
    root = Path(output_dir) / "models"
    if not root.exists():
        raise FileNotFoundError(f"Model directory not found: {root}")
    model_paths = sorted(root.glob("model_fold_*.joblib"))
    if not model_paths:
        raise FileNotFoundError(f"No model files found under: {root}")
    return [joblib.load(path) for path in model_paths]


def json_dumps(payload: Any) -> str:
    import json

    return json.dumps(payload, indent=2, ensure_ascii=False)
