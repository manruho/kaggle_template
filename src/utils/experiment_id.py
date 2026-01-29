"""実験IDの自動生成ユーティリティ."""
from __future__ import annotations

from typing import Any, Mapping


def generate_experiment_name(cfg: Mapping[str, Any]) -> str:
    """設定から実験名を自動生成する."""
    model = _token(cfg.get("model_name", "model"))
    feature = _token(cfg.get("feature_version", cfg.get("features_version", "default")))
    cv = _cv_tag(cfg)
    seed = f"seed{cfg.get('seed', 'na')}"
    version = _token(cfg.get("experiment_version", "v1"))
    return "__".join([f"{model}_{version}", f"fe{feature}", cv, seed])


def is_convention_name(name: str) -> bool:
    """規約に沿った実験名かを緩く判定する."""
    return "__" in name and "fe" in name and "seed" in name


def _cv_tag(cfg: Mapping[str, Any]) -> str:
    method = str(cfg.get("cv_method") or cfg.get("cv_type") or "kfold").lower()
    n_splits = cfg.get("n_splits", 5)
    if method in {"group", "groupkfold"}:
        group_col = _token(cfg.get("cv_group_col", "group"))
        return f"groupcv_{group_col}"
    if method in {"time", "timeseries", "time_series", "timeseriessplit"}:
        return f"timecv{n_splits}"
    if method in {"stratified", "stratifiedkfold", "strat"}:
        return f"cv{n_splits}"
    return f"cv{n_splits}"


def _token(value: Any) -> str:
    raw = str(value).strip().lower()
    if not raw:
        return "na"
    raw = raw.replace(" ", "-")
    return "".join(ch for ch in raw if ch.isalnum() or ch in {"-", "_", "."})
