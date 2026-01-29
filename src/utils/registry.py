"""実験レジストリの追記ユーティリティ."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


def append_experiment_record(path: Path, record: Mapping[str, Any]) -> None:
    """実験レコードをjsonl形式で追記する."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(record), ensure_ascii=False) + "\n")
