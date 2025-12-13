from __future__ import annotations

from pathlib import Path
import sys
import subprocess


def setup_template_repo(repo_url: str, work_root: Path) -> Path:
    repo_dir = work_root / "Project"

    if not repo_dir.exists():
        subprocess.run(["git", "clone", repo_url, str(repo_dir)], check=True)
    else:
        subprocess.run(["git", "-C", str(repo_dir), "pull"], check=False)  # Why not

    sys.path.append(str(repo_dir))
    return repo_dir


REPO_URL: str = "https://github.com/manruho/kaggle_template"
REPO_DIR: Path = setup_template_repo(REPO_URL, Path("/kaggle/working"))
