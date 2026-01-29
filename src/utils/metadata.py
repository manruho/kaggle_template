"""再現性メタデータの保存ユーティリティ."""
from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Mapping, Sequence


def save_config_snapshot(meta_dir: Path, cfg: Mapping[str, Any]) -> None:
    """設定スナップショットを保存する."""
    _ensure_dir(meta_dir)
    path = meta_dir / "config.snapshot.json"
    path.write_text(json.dumps(dict(cfg), indent=2, ensure_ascii=False), encoding="utf-8")


def save_git_metadata(meta_dir: Path, repo_root: Path) -> dict[str, str]:
    """Gitのメタ情報を保存する."""
    _ensure_dir(meta_dir)
    path = meta_dir / "git.txt"
    try:
        commit = _git(repo_root, "rev-parse", "HEAD")
        branch = _git(repo_root, "rev-parse", "--abbrev-ref", "HEAD")
        dirty = "dirty" if _is_dirty(repo_root) else "clean"
        content = f"commit={commit}\nbranch={branch}\nstatus={dirty}\n"
        info = {"commit": commit, "branch": branch, "status": dirty}
    except Exception:
        content = "unavailable\n"
        info = {"commit": "unavailable", "branch": "unavailable", "status": "unavailable"}
    path.write_text(content, encoding="utf-8")
    return info


def save_env_metadata(
    meta_dir: Path,
    package_names: Sequence[str],
    *,
    include_pip_freeze: bool = False,
) -> None:
    """実行環境のメタ情報を保存する."""
    _ensure_dir(meta_dir)
    path = meta_dir / "env.txt"
    try:
        lines = [
            f"python={platform.python_version()}",
            f"platform={platform.platform()}",
        ]
        for name in package_names:
            try:
                version = metadata.version(name)
                lines.append(f"{name}={version}")
            except metadata.PackageNotFoundError:
                continue
            except Exception:
                continue
        if include_pip_freeze:
            try:
                freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"]).decode("utf-8")
                lines.append("")
                lines.append("[pip_freeze]")
                lines.append(freeze.strip())
            except Exception:
                pass
        content = "\n".join(lines) + "\n"
    except Exception:
        content = "unavailable\n"
    path.write_text(content, encoding="utf-8")


def save_cv_scores(meta_dir: Path, scores: list[float], metric: str) -> None:
    """CVスコアを保存する."""
    _ensure_dir(meta_dir)
    path = meta_dir / "cv_scores.json"
    payload = {
        "metric": metric,
        "fold_scores": scores,
        "mean": float(sum(scores) / len(scores)) if scores else None,
        "std": float(_std(scores)) if scores else None,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def save_run_summary(meta_dir: Path, summary: Mapping[str, Any]) -> None:
    """実行サマリを保存する."""
    _ensure_dir(meta_dir)
    path = meta_dir / "run_summary.json"
    path.write_text(json.dumps(dict(summary), indent=2, ensure_ascii=False), encoding="utf-8")


def save_command(meta_dir: Path, argv: Sequence[str]) -> None:
    """実行コマンドを保存する."""
    _ensure_dir(meta_dir)
    path = meta_dir / "command.txt"
    path.write_text(" ".join(argv) + "\n", encoding="utf-8")


def save_seed(meta_dir: Path, seed: int) -> None:
    """乱数シードを保存する."""
    _ensure_dir(meta_dir)
    path = meta_dir / "seed.txt"
    path.write_text(str(seed) + "\n", encoding="utf-8")


def build_run_summary(
    experiment_name: str,
    metric: str,
    scores: list[float],
    *,
    model_name: str,
    seed: int,
    start_time: datetime,
    end_time: datetime,
) -> dict[str, Any]:
    """実行サマリの辞書を作る."""
    return {
        "experiment_name": experiment_name,
        "metric": metric,
        "main_cv_score": float(sum(scores) / len(scores)) if scores else None,
        "cv_std": float(_std(scores)) if scores else None,
        "model_name": model_name,
        "seed": seed,
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "duration_sec": (end_time - start_time).total_seconds(),
    }


def _git(repo_root: Path, *args: str) -> str:
    return (
        subprocess.check_output(["git", *args], cwd=repo_root)
        .decode("utf-8")
        .strip()
    )


def _is_dirty(repo_root: Path) -> bool:
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=repo_root).decode("utf-8")
    return bool(status.strip())


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _std(values: list[float]) -> float:
    mean = sum(values) / len(values)
    return (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5
