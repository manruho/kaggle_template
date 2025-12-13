"""Helpers for loading and saving configs."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import Config


def load_config(path: str | Path) -> Config:
    """Read JSON then convert it into :class:`Config`."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return Config.from_dict(payload)


def save_config(config: Config, path: str | Path) -> None:
    """Write ``config`` to ``path`` in JSON format."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(config.as_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
