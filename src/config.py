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
    experiment_prefix: Optional[str] = None
    experiment_tags: Optional[Sequence[str]] = None
    experiment_note: Optional[str] = None
    experiment_parent: Optional[str] = None
    dataset_name: Optional[str] = None
    feature_version: Optional[str] = None
    use_feature_cache: bool = False
    feature_cache_dir: Optional[str] = None
    feature_cache_format: str = "auto"
    feature_cache_force_recompute: bool = False
    feature_cache_params: Dict[str, Any] = field(default_factory=dict)
    save_models: bool = True
    validate_submission: bool = True
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

    def ensure_experiment_name(self) -> "Config":
        if self.experiment_name and str(self.experiment_name).lower() != "auto":
            return self
        if not self.experiment_auto_name:
            return self
        self.experiment_name = self.build_experiment_name()
        return self

    def build_experiment_name(self) -> str:
        model = _normalize_token(self.model_name)
        feature_version = self.get_feature_version()
        cv_method = self.get_cv_method()
        cv_tag = _cv_tag(cv_method, self.n_splits)
        seed_tag = f"seed{self.seed}"

        parts = [model, f"fe{feature_version}", cv_tag, seed_tag]
        if self.experiment_prefix:
            parts.insert(0, _normalize_token(self.experiment_prefix))
        return "__".join(parts)

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


def _cv_tag(method: str, n_splits: int) -> str:
    method = method.lower()
    if method in {"kfold", "cv"}:
        return f"cv{n_splits}"
    if method in {"stratified", "stratifiedkfold", "strat"}:
        return f"stratcv{n_splits}"
    if method in {"group", "groupkfold"}:
        return f"groupcv{n_splits}"
    if method in {"time", "timeseries", "time_series", "timeseriessplit"}:
        return f"timecv{n_splits}"
    return f"{_normalize_token(method)}cv{n_splits}"
