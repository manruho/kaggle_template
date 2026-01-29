"""Configuration objects used throughout the template."""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional, Sequence


@dataclass
class Config:
    """Container for experiment settings loaded from ``configs/*.json``."""

    train_path: str
    test_path: str
    sample_sub_path: str
    id_col: str
    target_col: str
    task_type: str = "binary"
    metric: str = "roc_auc"
    cv_type: str = "stratified"
    cv_method: Optional[str] = None
    cv_group_col: Optional[str] = None
    cv_time_col: Optional[str] = None
    cv_params: Dict[str, Any] = field(default_factory=dict)
    n_splits: int = 5
    seed: int = 42
    debug: bool = False
    sample_size: Optional[int] = None
    features: Optional[Sequence[str]] = None
    drop_cols: Optional[Sequence[str]] = None
    model_name: str = "logistic_regression"
    model_params: Dict[str, Any] = field(default_factory=dict)
    output_dir: str = "outputs"
    experiment_name: Optional[str] = None
    experiment_auto_name: bool = True
    experiment_tags: Optional[Sequence[str]] = None
    experiment_note: Optional[str] = None
    dataset_name: Optional[str] = None
    feature_version: Optional[str] = None
    use_feature_cache: bool = False
    feature_cache_dir: Optional[str] = None
    feature_cache_format: str = "auto"
    feature_cache_force_recompute: bool = False
    feature_cache_params: Dict[str, Any] = field(default_factory=dict)
    save_models: bool = False
    save_policy: str = "none"
    validate_submission: bool = True
    env_packages: Optional[Sequence[str]] = None
    include_pip_freeze: bool = False
    extras: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Config":
        """Create ``Config`` while storing unknown keys under ``extras``."""

        known_fields = {f.name for f in fields(cls)}
        kwargs: Dict[str, Any] = {}
        extras: Dict[str, Any] = {}
        for key, value in payload.items():
            if key in known_fields:
                kwargs[key] = value
            else:
                extras[key] = value
        merged_extras = {**extras, **kwargs.get("extras", {})}
        kwargs["extras"] = merged_extras
        return cls(**kwargs)

    def resolve_output_dir(self) -> Path:
        """Create/return the directory used for artifacts (OOF/submission/etc)."""

        base = Path(self.output_dir)
        final = base / self.experiment_name if self.experiment_name else base
        final.mkdir(parents=True, exist_ok=True)
        return final

    def resolve_path(self, path_value: str) -> Path:
        """Return a normalized :class:`~pathlib.Path`."""

        return Path(path_value).expanduser().resolve()

    def as_dict(self) -> Dict[str, Any]:
        """Return a shallow dict representation including extras."""

        data: Dict[str, Any] = {}
        for field_info in fields(self):
            if field_info.name.startswith("_"):
                continue
            data[field_info.name] = getattr(self, field_info.name)
        return data

    def get(self, key: str, default: Any = None) -> Any:
        """Lookup helper for ``extras`` dictionary."""

        if hasattr(self, key):
            return getattr(self, key)
        return self.extras.get(key, default)

    def get_cv_method(self) -> str:
        method = self.cv_method or self.cv_type or "kfold"
        return str(method).lower()

    def get_feature_version(self) -> str:
        if self.feature_version:
            return _normalize_token(self.feature_version)
        fallback = self.get("features_version", "default")
        return _normalize_token(str(fallback))

    @property
    def feature_columns(self) -> Sequence[str]:
        """Columns used for modeling once ``train`` and ``test`` are loaded."""

        if self.features:
            return list(self.features)
        excluded = {self.id_col, self.target_col}
        if self.drop_cols:
            excluded.update(self.drop_cols)
        return [col for col in self._cached_train_columns if col not in excluded]

    # Attribute set lazily when ``experiment.run`` loads train/test once.
    _cached_train_columns: Sequence[str] = field(
        default_factory=list, init=False, repr=False, compare=False
    )

    def with_train_columns(self, columns: Sequence[str]) -> "Config":
        self._cached_train_columns = list(columns)
        return self


def _normalize_token(value: str) -> str:
    cleaned = str(value).strip().lower()
    if not cleaned:
        return "na"
    cleaned = cleaned.replace(" ", "-")
    return "".join(ch for ch in cleaned if ch.isalnum() or ch in {"-", "_", "."})
