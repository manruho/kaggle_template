"""Model factory and wrappers."""
from __future__ import annotations

from typing import Any

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression

from .config import Config


class MissingDependencyError(ImportError):
    """Raised when a requested optional dependency is unavailable."""


def create_model(config: Config) -> Any:
    """Return a model instance based on ``config.model_name``."""

    name = config.model_name.lower()
    params = dict(config.model_params)
    task = config.task_type

    if name in {"lgbm", "lightgbm"}:
        try:
            import lightgbm as lgb  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise MissingDependencyError("Install lightgbm to use the LightGBM models") from exc
        params.setdefault("random_state", config.seed)
        if task == "regression":
            return lgb.LGBMRegressor(**params)
        return lgb.LGBMClassifier(**params)

    if name in {"xgb", "xgboost"}:
        try:
            import xgboost as xgb  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise MissingDependencyError("Install xgboost to use the XGBoost models") from exc
        params.setdefault("random_state", config.seed)
        if task == "regression":
            return xgb.XGBRegressor(**params)
        return xgb.XGBClassifier(**params)

    if name == "random_forest":
        if task == "regression":
            return RandomForestRegressor(random_state=config.seed, **params)
        return RandomForestClassifier(random_state=config.seed, **params)

    if name == "logistic_regression":
        params.setdefault("max_iter", 200)
        params.setdefault("n_jobs", 1)
        params.setdefault("random_state", config.seed)
        return LogisticRegression(**params)

    if name == "linear_regression":
        return LinearRegression(**params)

    raise ValueError(f"Unsupported model name: {config.model_name}")
