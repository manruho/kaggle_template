"""Create a zip archive that can be uploaded as a Kaggle Dataset."""
from __future__ import annotations

import argparse
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ITEMS = ("src", "configs", "README.md")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "kaggle_template_lib.zip",
        help="Destination zip file",
    )
    parser.add_argument(
        "--items",
        nargs="*",
        default=DEFAULT_ITEMS,
        help="Relative paths to include in the archive",
    )
    args = parser.parse_args()

    archive_path = args.output
    archive_path.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for item in args.items:
            _add_path(zf, ROOT / item, item)
    print(f"Created {archive_path}")


def _add_path(zf: zipfile.ZipFile, path: Path, arc_prefix: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Path not found: {path}")
    if path.is_file():
        zf.write(path, arc_prefix)
        return
    for file_path in path.rglob("*"):
        if file_path.is_file():
            relative = Path(arc_prefix) / file_path.relative_to(path)
            zf.write(file_path, relative)


if __name__ == "__main__":
    main()
