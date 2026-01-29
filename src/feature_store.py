"""Feature caching helpers (FeatureStore)."""
from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Sequence, Tuple

import pandas as pd

from .config import Config

FEATURE_STORE_VERSION = 1


@dataclass(frozen=True)
class FeatureStoreResult:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    cache_hit: bool
    cache_key: str
    cache_dir: Path
    format: str


class FeatureStore:
    """Cache feature matrices on disk to avoid recomputation."""

    def __init__(self, cache_dir: str | Path, format: str = "auto") -> None:
        self.cache_dir = Path(cache_dir)
        self.format = format.lower()

    def load_or_build(
        self,
        config: Config,
        builder: Callable[[], Tuple[pd.DataFrame, pd.DataFrame]],
        *,
        force_recompute: bool = False,
    ) -> FeatureStoreResult:
        cache_key = self.build_cache_key(config)

        if not force_recompute:
            cached = self._load(cache_key)
            if cached is not None:
                X_train, X_test, fmt = cached
                return FeatureStoreResult(
                    X_train=X_train,
                    X_test=X_test,
                    cache_hit=True,
                    cache_key=cache_key,
                    cache_dir=self.cache_dir,
                    format=fmt,
                )

        X_train, X_test = builder()
        fmt = self._save(cache_key, X_train, X_test, config=config)
        return FeatureStoreResult(
            X_train=X_train,
            X_test=X_test,
            cache_hit=False,
            cache_key=cache_key,
            cache_dir=self.cache_dir,
            format=fmt,
        )

    def build_cache_key(self, config: Config) -> str:
        """Compute a stable cache key from feature-related config + data stats."""

        payload: Dict[str, Any] = {
            "feature_store_version": FEATURE_STORE_VERSION,
            "features_version": config.get_feature_version(),
            "data_version": config.data_version,
            "id_col": config.id_col,
            "target_col": config.target_col,
            "features": list(config.features) if config.features else None,
            "drop_cols": list(config.drop_cols) if config.drop_cols else None,
            "debug": config.debug,
            "sample_size": config.sample_size,
            "seed": config.seed,
            "train_path": config.train_path,
            "test_path": config.test_path,
            "train_stat": _safe_path_stat(config.train_path),
            "test_stat": _safe_path_stat(config.test_path),
            "feature_code_hash": _safe_file_hash(Path(__file__).resolve().parent / "features.py"),
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "config_fingerprint": _config_fingerprint(config),
        }
        extra_params = config.feature_cache_params or config.get("feature_cache_params")
        if isinstance(extra_params, Mapping):
            payload["feature_cache_params"] = dict(extra_params)

        raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def _load(self, cache_key: str) -> Tuple[pd.DataFrame, pd.DataFrame, str] | None:
        root = self.cache_dir / cache_key
        meta_path = root / "meta.json"
        if not meta_path.exists():
            return None

        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            return None

        fmt = str(meta.get("format", "parquet")).lower()
        train_path, test_path = _feature_paths(root, fmt)
        if not train_path.exists() or not test_path.exists():
            return None

        try:
            if fmt == "parquet":
                X_train = pd.read_parquet(train_path)
                X_test = pd.read_parquet(test_path)
            else:
                X_train = pd.read_pickle(train_path)
                X_test = pd.read_pickle(test_path)
        except Exception:
            return None

        return X_train.reset_index(drop=True), X_test.reset_index(drop=True), fmt

    def _save(self, cache_key: str, X_train: pd.DataFrame, X_test: pd.DataFrame, *, config: Config) -> str:
        root = self.cache_dir / cache_key
        root.mkdir(parents=True, exist_ok=True)

        created_at = datetime.now(timezone.utc).isoformat()
        meta: Dict[str, Any] = {
            "created_at": created_at,
            "cache_key": cache_key,
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "n_features": int(X_train.shape[1]),
            "features_version": config.get_feature_version(),
            "format": None,
        }

        preferred = self.format
        if preferred not in {"auto", "parquet", "pickle"}:
            raise ValueError(f"Unsupported feature cache format: {self.format}")

        if preferred in {"auto", "parquet"}:
            try:
                train_path, test_path = _feature_paths(root, "parquet")
                X_train.to_parquet(train_path, index=False)
                X_test.to_parquet(test_path, index=False)
                meta["format"] = "parquet"
                (root / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
                return "parquet"
            except Exception:
                if preferred == "parquet":
                    raise
                train_path, test_path = _feature_paths(root, "parquet")
                _safe_unlink(train_path)
                _safe_unlink(test_path)

        train_path, test_path = _feature_paths(root, "pickle")
        X_train.to_pickle(train_path)
        X_test.to_pickle(test_path)
        meta["format"] = "pickle"
        (root / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        return "pickle"


def _feature_paths(root: Path, fmt: str) -> Tuple[Path, Path]:
    ext = "parquet" if fmt == "parquet" else "pkl"
    return root / f"X_train.{ext}", root / f"X_test.{ext}"


def _safe_unlink(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return
    except Exception:
        return


def _safe_path_stat(path_value: str) -> Dict[str, int] | None:
    try:
        path = Path(path_value)
        stat = path.stat()
    except Exception:
        return None
    return {"size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


def _safe_file_hash(path: Path) -> str | None:
    try:
        data = path.read_bytes()
    except Exception:
        return None
    return hashlib.sha256(data).hexdigest()


def _config_fingerprint(config: Config) -> str:
    payload = {
        "feature_version": config.get_feature_version(),
        "data_version": config.data_version,
        "features": list(config.features) if config.features else None,
        "drop_cols": list(config.drop_cols) if config.drop_cols else None,
        "task_type": config.task_type,
        "model_name": config.model_name,
        "model_params": config.model_params,
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
