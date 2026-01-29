"""Model persistence helpers."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, List, Sequence

import joblib

from .config import Config


def save_models(
    models: Sequence[Any],
    output_dir: str | Path,
    config: Config,
    *,
    policy: str,
    scores: Sequence[float] | None = None,
    top_k: int = 1,
    models_dir: str | Path | None = None,
) -> List[str]:
    """モデルを保存する."""
    root = Path(models_dir) if models_dir else Path(output_dir) / "models"
    root.mkdir(parents=True, exist_ok=True)

    indices = _select_indices(models, policy=policy, scores=scores, top_k=top_k)
    paths: List[str] = []
    for idx in indices:
        path = root / f"model_fold_{idx}.joblib"
        joblib.dump(models[idx], path)
        paths.append(str(path))

    meta = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_name": config.model_name,
        "model_params": config.model_params,
        "n_models": len(paths),
        "policy": policy,
        "indices": list(indices),
        "paths": paths,
        "models_dir": str(root),
    }
    (root / "meta.json").write_text(json_dumps(meta), encoding="utf-8")
    return paths


def load_models(output_dir: str | Path, *, models_dir: str | Path | None = None) -> List[Any]:
    root = Path(models_dir) if models_dir else Path(output_dir) / "models"
    if not root.exists():
        raise FileNotFoundError(f"Model directory not found: {root}")
    model_paths = sorted(root.glob("model_fold_*.joblib"))
    if not model_paths:
        raise FileNotFoundError(f"No model files found under: {root}")
    return [joblib.load(path) for path in model_paths]


def json_dumps(payload: Any) -> str:
    import json

    return json.dumps(payload, indent=2, ensure_ascii=False)


def _select_indices(
    models: Sequence[Any],
    *,
    policy: str,
    scores: Sequence[float] | None,
    top_k: int,
) -> List[int]:
    if policy == "none":
        return []
    if policy == "all":
        return list(range(len(models)))
    if policy == "best":
        if not scores:
            raise ValueError("scores are required when save_policy='best'")
        best_idx = int(max(range(len(scores)), key=lambda i: scores[i]))
        return [best_idx]
    if policy in {"keep_top_k", "top_k"}:
        if not scores:
            raise ValueError("scores are required when save_policy='keep_top_k'")
        if top_k <= 0:
            raise ValueError("save_top_k must be positive")
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return ranked[: min(top_k, len(ranked))]
    raise ValueError(f"unsupported save_policy: {policy}")
