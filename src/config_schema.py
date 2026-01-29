"""Configの検証スキーマ."""
from __future__ import annotations

from typing import Any, Optional, Sequence

try:
    from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
except Exception:  # pragma: no cover - optional dependency
    BaseModel = object  # type: ignore
    ConfigDict = dict  # type: ignore
    Field = lambda **_: None  # type: ignore

    def field_validator(*_args: object, **_kwargs: object):  # type: ignore
        def decorator(func):
            return func

        return decorator

    def model_validator(*_args: object, **_kwargs: object):  # type: ignore
        def decorator(func):
            return func

        return decorator


class ConfigSchema(BaseModel):
    """Configバリデーション用モデル."""

    model_config = ConfigDict(extra="allow")

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
    cv_params: dict[str, Any] = Field(default_factory=dict)
    n_splits: int = 5
    seed: int = 42
    debug: bool = False
    sample_size: Optional[int] = None
    features: Optional[Sequence[str]] = None
    drop_cols: Optional[Sequence[str]] = None
    model_name: str = "logistic_regression"
    model_params: dict[str, Any] = Field(default_factory=dict)
    output_dir: str = "outputs"
    experiment_name: Optional[str] = None
    experiment_auto_name: bool = True
    experiment_tags: Optional[Sequence[str]] = None
    experiment_note: Optional[str] = None
    experiment_version: str = "v1"
    dataset_name: Optional[str] = None
    feature_version: Optional[str] = None
    data_version: Optional[str] = None
    use_feature_cache: bool = False
    feature_cache_dir: Optional[str] = None
    feature_cache_format: str = "auto"
    feature_cache_force_recompute: bool = False
    feature_cache_params: dict[str, Any] = Field(default_factory=dict)
    save_models: bool = False
    save_policy: str = "none"
    save_top_k: int = 1
    models_dir: Optional[str] = None
    validate_submission: bool = True
    env_packages: Optional[Sequence[str]] = None
    include_pip_freeze: bool = False

    @field_validator("task_type")
    @classmethod
    def _task_type(cls, value: str) -> str:
        allowed = {
            "binary",
            "multiclass",
            "regression",
            "multilabel",
            "multitarget",
            "multioutput_regression",
        }
        if value not in allowed:
            raise ValueError(f"unsupported task_type: {value}")
        return value

    @field_validator("save_policy")
    @classmethod
    def _save_policy(cls, value: str) -> str:
        allowed = {"none", "best", "all", "keep_top_k", "top_k"}
        if value not in allowed:
            raise ValueError(f"unsupported save_policy: {value}")
        return value

    @field_validator("save_top_k")
    @classmethod
    def _save_top_k(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("save_top_k must be positive")
        return value

    @model_validator(mode="after")
    def _validate_cv_dependencies(self) -> "ConfigSchema":
        method = (self.cv_method or self.cv_type or "kfold").lower()
        if method in {"group", "groupkfold"} and not self.cv_group_col:
            raise ValueError("cv_group_col is required for group CV")
        if method in {"time", "timeseries", "time_series", "timeseriessplit"} and not self.cv_time_col:
            raise ValueError("cv_time_col is required for time CV")
        return self

    @classmethod
    def model_validate(cls, payload: dict[str, Any]) -> "ConfigSchema":  # type: ignore[override]
        if cls is BaseModel:
            raise ImportError("pydantic is required for config validation")
        return super().model_validate(payload)  # type: ignore[misc]
