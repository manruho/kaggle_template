"""Helpers for loading and saving configs."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import Config


def load_config(path: str | Path) -> Config:
    """Read JSON then convert it into :class:`Config`."""

    target = Path(path)
    payload = _load_payload(target)
    return Config.from_dict(payload)


def save_config(config: Config, path: str | Path) -> None:
    """Write ``config`` to ``path`` in JSON format."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = config.as_dict()
    if target.suffix.lower() in {".yaml", ".yml"}:
        _dump_yaml(payload, target)
    else:
        target.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


def _load_payload(path: Path) -> Any:
    if path.suffix.lower() in {".yaml", ".yml"}:
        return _load_yaml(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _load_yaml(path: Path) -> Any:
    try:
        import yaml  # type: ignore
    except Exception as exc:
        raise ImportError("PyYAML is required to load YAML configs") from exc
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _dump_yaml(payload: Any, path: Path) -> None:
    try:
        import yaml  # type: ignore
    except Exception as exc:
        raise ImportError("PyYAML is required to save YAML configs") from exc
    path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
